#import libraries for OCR + plate detection + image processing + data manipulation
import easyocr as ocr
from datetime import datetime, timedelta
import time
import cv2
from pandas import read_csv
import os
import pymysql
import tensorflow as tf
import numpy as np
import sys
import glob
import random
import importlib.util
from tensorflow.lite.python.interpreter import Interpreter
import matplotlib.pyplot as plt


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
    
    """
    @Author: xdevilscloverx
    @Description: This function is used to initialize the class
    @Params:    Model Path - path to the model used to detect the plate (default: None) -> required parameter to run the class
    """
    def __init__(self, lang='en', gpu=False, filter=None, model_path=None, host='localhost', user='root', password='password', db='database'):
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
                self.platemodel = Interpreter(model_path=model_path)
            else:
                raise Exception('Model path not provided!')
    
    """
    @Author: xdevilscloverx
    @Description: This function is used to read the plates from a frame and return a string of the most likely plate
    """
    def __read_plate(self, plate_img):
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
    
    """
    @Author: xdevilscloverx
    @Description: This function is the primary handler for the class. It is used to process the video and upload the results to a database
    """
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
            plates = self.__predict_plates(frame)

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
            self.__upload_results()

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
    
    """
    @Author: xdevilscloverx
    @Description: This function uploads the results to a database
    """
    def __upload_results(self):
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
    
    """
    @Author xdevilscloverx
    @Description: This function defines a bounding box for the plate
    """
    def tflite_detect_images(self, frame, min_conf=0.5):

        # Load the Tensorflow Lite model into memory
        self.platemodel.allocate_tensors()

        # Get model details
        input_details = self.platemodel.get_input_details()
        output_details = self.platemodel.get_output_details()
        height = input_details[0]['shape'][1]
        width = input_details[0]['shape'][2]

        float_input = (input_details[0]['dtype'] == np.float32)

        input_mean = 127.5
        input_std = 127.5

        imH, imW, _ = frame.shape  # image size

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if float_input:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform the actual detection by running the model with the image as input
        self.platemodel.set_tensor(input_details[0]['index'],input_data)
        self.platemodel.invoke()

        # Retrieve detection results
        boxes = self.platemodel.get_tensor(output_details[1]['index'])[0] # Bounding box coordinates of detected objects
        classes = self.platemodel.get_tensor(output_details[3]['index'])[0] # Class index of detected objects
        scores = self.platemodel.get_tensor(output_details[0]['index'])[0] # Confidence of detected objects

        detections = []

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > min_conf) and (scores[i] <= 1.0)):

                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
                    
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                # Draw label
                object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

                detections.append([object_name, scores[i], xmin, ymin, xmax, ymax])

            return

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