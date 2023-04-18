#import libraries for OCR + plate detection + image processing + data manipulation
from datetime import datetime, timedelta
import time
import cv2
from pandas import read_csv
import pymysql
import tensorflow
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter
import argparse
from PIL import Image
from io import BytesIO  #used to convert the image to a byte stream
#from picamera2 import Picamera2
import pytesseract

"""
@Author: xxdevilscloverxx
@Date:   2023-04-16
@Description: This class is used to read the plates from live video + upload the results to a database after x seconds
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
    def __init__(self, filter=None, model_path=None, usbwebcam=False,host='localhost', user='root', password='password', db='database'):
        self.filter = filter
        self.time = datetime.now()  #get the current time @ initialization
        self.buffer = {}  #initialize the buffer
        self.time_buffer = {}  #initialize the frame buffer -> prevent duplicate uploads
        self.usbwebcam = usbwebcam #define what camera object to use
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
    def read_plate(self, plate_img):
        # read the plate
        results = pytesseract.image_to_data(plate_img, lang = "en", config='-l eng --psm 12', nice=0, output_type="dict")
        
        # get the words with a confidence of 25 or higher
        words = []
        for i, word in enumerate(results['text']):
            conf = int(results['conf'][i])
            if conf >= 40:
                words.append(word)
        
        # join the words together
        plate_text = "".join(words)
        plate_text = [char.upper() for char in plate_text if char.isalnum()]    #remove special characters
        plate_text = "".join(plate_text)    #join the characters together
        
        # filter out the states
        if self.filter is not None:
            for state in self.filter:
                if state.upper() in plate_text:
                    plate_text = plate_text.replace(state.upper(), '')  #remove the state from the plate text

        # return a max of 8 characters
        if len(plate_text) > 8:
            plate_text = plate_text[:8]

        elif len(plate_text) < 4:
            plate_text = "" #return an empty string if the plate is too small

        if plate_text.isnumeric():
            plate_text = "" #return an empty string if the plate is all numbers

        # return the plate text
        return plate_text
   
    """
    @Author: xdevilscloverx
    @Description: This function is the primary handler for the class. It is used to process the video and upload the results to a database
    """
    def process_video(self):
        # initialize the video reader
        if self.usbwebcam:    
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
        else:
            # initialize the Pi camera with the config size of 1280x720 and 30 fps
            #picam2 = Picamera2()
            #config = picam2.create_still_configuration(main={"size": (1280, 720)})
            #picam2.configure(config)

            # start the camera
            #picam2.start()
            pass
        # create a byte stream obj to store the image data
        stream = BytesIO()

        # wait for the camera to initialize
        time.sleep(3)

        # loop over the frames
        while True:
           
            if self.usbwebcam:    
                # read the frame
                ret, frame = cap.read()
                if not ret:
                    raise Exception('Video not found')
                frame = Image.fromarray(frame)
                frame = frame.convert('L')
            else:
                # read the frame, convert to grayscale
                #frame = picam2.capture_image("main")
                #frame = frame.convert('L')
                pass 
            new_frame = Image.new('RGB', frame.size)
            new_frame.putdata([(x, x, x) for x in frame.getdata()])
            frame = np.array(new_frame)
           
            # process the frame and
            detections = self.predict_plates(frame)

            # process the detections
            for key, item in detections.items():
                # get the coordinates
                x1, y1, x2, y2 = item

                # get the plate image
                plate_img = frame[y1:y2, x1:x2]

                crop = Image.fromarray(plate_img)
                crop.save("crop.jpeg")

                copy = cv2.imread("crop.jpeg")
                copy = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)
                copy = cv2.bilateralFilter(copy, 11, 17, 17)    #remove noise
                thresh = cv2.threshold(copy, 170, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
                close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
                result = 255 - close

                #cv2.imwrite("result.jpg", result)
                # read the plate
                plate_text = self.read_plate(result)

                # if the plate is not empty, print it
                if plate_text != "":
                    print(plate_text)
                    # create a byte stream obj to store the image data
                    crop.save(stream, format='JPEG')
                    # get the image data
                    image_data = stream.getvalue()

                    # reset the stream to the beginning
                    stream.seek(0)
                    stream.truncate()   #clear the stream

                    # add the plate text, image, and time to the buffer
                    self.buffer.update({plate_text: image_data})

            # upload the results to a database
            self.__upload_results()
   
    """
    @Author: xdevilscloverx
    @Description: This function uploads the results to a database
    """
    def __upload_results(self):
        #check if 10 seconds have passed
        if datetime.now() - self.time > timedelta(seconds=10):
            i = 0  #initialize the counter
            self.time = datetime.now()   #reset the timer
            if self.connected:
                print("Uploading results to database...")
                #loop over the buffer
                for i, plate in enumerate(self.buffer.keys()):
                    if plate not in self.time_buffer.keys():
                        with self.connection.cursor() as cursor:
                            #create the sql query
                            sql = "INSERT INTO `licenses` (`license_pl`, `plate_img`) VALUES (%s, %s)"
                            #execute the query
                            cursor.execute(sql, (plate, self.buffer[plate]))
                            #commit the changes
                            self.connection.commit()
                       
                        #upload the results to the database
                        print(f"{plate}: {self.buffer[plate]} uploaded to database!")
            else:
                print("Printing results...")
                #loop over the buffer
                for i, plate in enumerate(self.buffer.keys()):
                    if plate not in self.time_buffer.keys():
                        print(f"{plate}")  #print the plate

            print(f"{i} results captured + pushed!")

            #update the time buffer
            self.time_buffer.clear()              #clear the time buffer before updating
            self.time_buffer.update(self.buffer)  #update the time buffer

            #print(self.time_buffer.keys())               #print the time buffer -> show plate when empty!

            #clear the buffer
            self.buffer.clear()
        return None
   
    """
    @Author xdevilscloverx
    @Description: This function defines a bounding box for the plate
    """
    def predict_plates(self, frame, min_conf=0.6):

        # Load the Tensorflow Lite model into memory
        self.platemodel.allocate_tensors()

        # Get model details
        input_details = self.platemodel.get_input_details()
        output_details = self.platemodel.get_output_details()

        float_input = (input_details[0]['dtype'] == np.float32)

        input_mean = 127.5
        input_std = 127.5

        imH, imW, _ = frame.shape  # image size
        copy = frame.copy()
       
        # Resize image to model size
        copy = cv2.resize(copy, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
        input_data = np.expand_dims(copy, axis=0)

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

        detections = {}

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i, item in enumerate(scores):
            if ((scores[i] > min_conf) and (scores[i] <= 1.0)):

                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))

                #add the detection to the dictionary: these align with the original coordinates
                detections.update({i: (xmin, ymin, xmax, ymax)})
       
        return detections

#run test script
if __name__ == '__main__':

    # parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model_path", type=str, default=None,
                    help="Path to the input model")
    ap.add_argument("-f", "--filter_path", type=str, default=None,
                    help="Path to the filter csv file")
    ap.add_argument("-g", "--gpu", type=bool, default=False,
                    help="Use GPU? Default: False")
    ap.add_argument("-c", "--camera", type=bool, default=False,
                    help="Use the USB webcam? Default: False")
    ap.add_argument("-d", "--database", type=bool, default=False,
                    help="Connect to a database? Default: False")        
    args = vars(ap.parse_args())

    model_path = args['model_path']
    filter_file = args['filter_path']
    gpu = args['gpu']
    cam = args['camera']
    cam_setting = args['camera']
    database = args['database']

    # open the filter file
    try:
        filter_file = read_csv(filter_file)
        filter = tuple(filter_file['State'])
    except Exception as e:
        print(e, "Filter file not found! Using no filter...")
        filter = None

    if database:    
        #define the sql connection
        host = "195.179.237.153"
        user = "u376552790_projectlabdb"
        password = "Ltg5075@"
        db = "u376552790_projectlabdb"
    else:
        host = None
        user = None
        password = None
        db = None
   
    #initialize the reader
    detector = plate_reader(filter=filter, model_path=model_path, usbwebcam=cam_setting,
                            host=host, user=user, password=password, db=db)

    detector.process_video()