#import libraries for OCR + plate detection + image processing + data manipulation
import easyocr as ocr
from datetime import datetime, timedelta
import time
import cv2
from pandas import read_csv
import os
from detecto.core import Model
import pymysql

"""
@Author: xxdevilscloverxx
@Date:   2021-05-10
@Description: This class is used to read the plates from live video + upload the results to a database after x minutes
@Params:    Model Path - path to the model used to detect the plate
            Filter - list of states to filter out
            Language - language to use for OCR
            GPU - whether to use GPU or not
            Video Path - path to the video to process, 0 for webcam
"""
class plate_reader:
    
    #initialize the reader, database, and model
    def __init__(self, lang='en', gpu=True, filter=None, model_path=None, host='localhost', user='root', password='password', db='database'):
        self.reader = ocr.Reader([lang], gpu=gpu)
        self.filter = filter
        self.time = datetime.now()  #get the current time @ initialization
        self.buffer = {}  #initialize the buffer
        self.time_buffer = {}  #initialize the frame buffer -> prevent duplicate uploads
        try:
            self.connection = pymysql.connect(host=host, user=user, password=password, db=db)  #connect to the database
            self.connected = True
        except:
            self.connected = False
            print('Database not found! Please check the database credentials and try again! Printing results to console instead...')
        finally:
            #load the model
            if model_path is not None:
                self.platemodel = Model.load(file=model_path, classes=['licence'])
            else:
                raise Exception('Model path not provided!')
    
    #define a function to read the plate from an image
    def read_plate(self, plate_img):
        #read the plate from image
        results = self.reader.readtext(plate_img, detail=0, text_threshold=.9)  #play with threshold to get better results
        #apply a filter to the results to remove unwanted strings and states, if a filter is provided
        if self.filter is not None:
            #filter the results for the plate + remove useless strings and states
            results = [_ for _ in results if len(_) > 5 and len(_) < 9 and not _.isdigit() and _.upper() not in self.filter]
        else:
            #filter the results for the plate + remove useless strings and states
            results = [_ for _ in results if len(_) > 5 and len(_) < 9 and not _.isdigit()]
        
        #if no results are found, return an empty string
        if len(results) == 0:
            return ""
        
        #clean the text
        plate_text = [char for char in results[0] if char.isalnum()]
        plate_text = "".join(plate_text)
        plate_text = plate_text.upper()

        #check if the plate is valid
        if len(plate_text) < 5 or len(plate_text) > 8 or plate_text.isdigit():
            return ""
        
        return plate_text
    
    #define a function to read the plate from a video
    def process_video(self, video_path=0, frame_width=640, frame_height=480, fps=30):
        #initialize the video reader
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        cap.set(cv2.CAP_PROP_FPS, fps)

        #wait for the camera to initialize
        time.sleep(5)
        #loop over the frames
        while True:
            #read the frame
            ret, frame = cap.read()
            if not ret:
                raise Exception('Video not found')
            
            #process the frame
            plates = self.predict_plates(frame)

            #loop over the plates
            for plate in plates:
                #get the plate text
                plate_text = plates[plate][0]
                #get the plate score
                plate_score = plates[plate][1]
                plate_score = str(plate_score)
                plate_score = [char for char in plate_score if char.isnumeric()]
                plate_score = "".join(plate_score)
                #get the plate coordinates
                x1, y1, x2, y2 = plates[plate][2]
                
                #load the plate into the buffer + the latest time it was scanned
                self.buffer.update({plate_text: str(datetime.date(datetime.now())) + " " + str(datetime.time(datetime.now()))})

                #draw the text
                cv2.putText(frame, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                #draw the score
                cv2.putText(frame, plate_score, (x1+20, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                #draw the box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            #upload the results to a database
            self.upload_results()

            #display the frame
            cv2.imshow('Frame', frame)
            #exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                if self.connected:
                    self.connection.close()
                break
        
        #release the camera
        cap.release()
        cv2.destroyAllWindows()
    
    #define a function to apply model and predict the plates
    def predict_plates(self, frame):
        #predict the plate
        labels, boxes, scores = self.platemodel.predict_top(frame)

        #initialize the dictionary
        plates = {}
        
        #loop over the results
        for index, (label, box, score) in enumerate(zip(labels, boxes, scores)):
            #get the box coordinates
            x1, y1, x2, y2 = box
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            #get the plate image
            plate_img = frame[y1:y2, x1:x2]
            plate_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            plate_img = cv2.medianBlur(plate_img, 3)
            
            #read the plate
            plate_text = self.read_plate(plate_img)
            #update the dictionary
            if plate_text != "":
                plates[f"{label}{index}"] = (plate_text, score, (x1, y1, x2, y2))

        return plates

    #define a function to upload the results to a database
    def upload_results(self):
        #check if 5 seconds have passed
        if datetime.now() - self.time > timedelta(seconds=10):
            i = 0  #initialize the counter
            self.time = datetime.now()   #reset the timer
            if self.connected:
                print("Uploading results to database...")
                #loop over the buffer
                for plate in self.buffer.keys():
                    if plate not in self.time_buffer.keys():
                        with self.connection.cursor() as cursor:
                            #create the sql query
                            sql = "INSERT INTO `plates` (`Plate`, `Time`) VALUES (%s, %s)"
                            #execute the query
                            cursor.execute(sql, (plate, self.buffer[plate]))
                            #commit the changes
                            self.connection.commit()
                        i += 1
                        
                        #upload the results to the database
                        print(f"{plate}: {self.buffer[plate]} uploaded to database!")
            else:
                print("Database connection failed! Printing results...")
                #loop over the buffer
                for plate in self.buffer.keys():
                    if plate not in self.time_buffer.keys():
                        print(f"{plate}: {self.buffer[plate]}")
                        i += 1
            print(f"{i} results uploaded to database!")

            #update the time buffer
            self.time_buffer.clear()              #clear the time buffer before updating
            self.time_buffer.update(self.buffer)  #update the time buffer

            print(self.time_buffer)               #print the time buffer -> show plate when empty!

            #clear the buffer
            self.buffer.clear()
            print('Upload completed!')
        return None

#run test script
if __name__ == '__main__':

    #define the sql connection
    host = 'localhost'
    user = None
    password = None
    db = None

    model_path = '/home/xxdevilscloverxx/Documents/models/plates.pth'
    # define a filter path
    filter_path = os.path.join(os.path.dirname(__file__), 'states.csv')
    #define a filter
    filter = read_csv(filter_path)
    filter = filter['State'].tolist()

    #initialize the reader
    detector = plate_reader(filter=filter, model_path=model_path,
                             gpu=True, host=host, user=user, password=password, db=db)
    #process the video
    detector.process_video()