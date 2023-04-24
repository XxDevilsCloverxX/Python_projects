#import packages
from datetime import datetime, timedelta
import cv2
from pandas import read_csv
import pymysql
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter
import argparse
from PIL import Image
from io import BytesIO  #used to convert the image to a byte stream
import pytesseract
from threading import Thread
import importlib.util
import time
import sys
from glob import glob   #used to get the list of files in a directory
import os

"""
@Author xdevilscloverx
@Description: This class is used to scan for license plates
"""
class platescanner:
   
    """
    @Author xdevilscloverx
    @Description: This function initializes the class
    @Parameters:    filter - a list of strings to filter out of the plate
                    interpreter - the tflite interpreter object
                    usbwebcam - a boolean to determine if the camera is a usb webcam or a pi camera
                    sqlconnector - the sql connector object
    """
    def __init__(self, filter=None, interpreter=None, usbwebcam=False, sqlconnector=None):
        # initalize objects that the class will use
        self.filter = filter    #filter out the states
        self.model = interpreter    #tflite model
        self.connection = sqlconnector   #database connection
        self.buffer = {}            #buffer to store the results
        self.shift_buffer = {}      #buffer that stores previous results
        self.time = datetime.now()  #time to check if 10 seconds have passed
        self.stream = BytesIO()     #stream to store the image
        self.usbwebcam = usbwebcam  #determine if the camera is a usb webcam or a pi camera
        self.threads = []           #list of threads
        self.stop = False           #boolean to stop the threads
        #create the appropriate video stream
        if self.usbwebcam:
            #create a video capture object
            self.cam = cv2.VideoCapture(0)
            self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cam.set(cv2.CAP_PROP_FPS, 30)
            # start the camera
            ret, self.frame = self.cam.read()   #read the first frame
            if not ret:
                print("Error: Unable to read first frame")
                exit()
            self.frame = Image.fromarray(self.frame)   #convert the frame to a PIL image
        else:
            #create a video capture object using pi camera
            self.cam = Picamera2()
            config = self.cam.create_still_configuration(main={"size": (1280, 720)})
            self.cam.configure(config)
            # start the camera
            self.cam.start()
            self.frame = self.cam.capture_image()   #read the first frame
            if self.frame is None:
                print("Error: Unable to read first frame")
                exit()
        # self.frame is now a PIL image
        self.frame = self.frame.convert("L")    #convert the image to grayscale
        new_frame = Image.new('RGB', self.frame.size)
        new_frame.putdata([(x, x, x) for x in self.frame.getdata()])
        self.frame = np.array(new_frame)    #convert the image to an array
        
        if self.connection is not None:
            self.connected = True
            # upload plates from previous runs
            for file in glob(os.path.join(os.getcwd(), "outputs", "reconnect", "*.bin")):
                filename = file.split("\\")[-1]    #get the filename not extension
                try:
                    with self.connection.cursor() as cursor:
                        #create the sql query
                        sql = "INSERT INTO `licenses` (`license_pl`, `plate_img`) VALUES (%s, %s)"
                        #execute the query
                        cursor.execute(sql, (filename, file))
                        #commit the changes
                        self.connection.commit()
                    #upload the results to the database
                    print(f"'{filename}' uploaded to database!")
                except pymysql.err.OperationalError as e:
                    print(f"Error uploading '{filename}' to database: {e}")
                    
        else:
            self.connected = False

    """
    @Author xdevilscloverx
    @Description: This function reads a frame from the camera and returns it
    """
    def readframe(self):
        while not self.stop:
            # read the frame
            if self.usbwebcam:
                ret, frame = self.cam.read()
                if not ret:
                    print("Error: Unable to read frame")
                    exit()
                frame = Image.fromarray(frame)
            else:
                frame = self.cam.capture_image()
                if frame is None:
                    print("Error: Unable to read frame")
                    exit()
            # frame is now a PIL image
            frame = frame.convert("L")    #convert the image to grayscale
            new_frame = Image.new('RGB', frame.size)
            new_frame.putdata([(x, x, x) for x in frame.getdata()])
            self.frame = np.array(new_frame)    #convert the image to an array

    """
    @Author: xdevilscloverx
    @Description: This function handles the threads
    """
    def start(self):
        #create a thread to read the frame
        self.readthread = Thread(target=self.readframe, args=())
        self.handler = Thread(target=self.handle_frame, args=())
        self.uploader = Thread(target=self.upload_to_database, args=())
        self.threads.append(self.readthread)
        self.threads.append(self.handler)
        self.threads.append(self.uploader)
        self.stop = False   #set stop to false to start the threads
        #start the threads
        for thread in self.threads:
            thread.start()
        return self
    
    """
    Stop the threads
    """
    def stop_threads(self):
        self.stop = True
        for thread in self.threads:
            thread.join()
        self.threads.clear()

    """
    @Author xdevilscloverx
    @Description: This function takes a frame and returns a dictionary of the detected plates and their coordinates
    """
    def predict_plates(self, frame, min_conf=0.5, input_mean=127.5, input_std=127.5):

        # Get model details
        input_details = self.model.get_input_details()
        output_details = self.model.get_output_details()

        float_input = (input_details[0]['dtype'] == np.float32)

        imH, imW, _ = frame.shape  # image size
        copy = frame.copy()
       
        # Resize image to model size
        copy = cv2.resize(copy, (input_details[0]['shape'][2], input_details[0]['shape'][1]))
        input_data = np.expand_dims(copy, axis=0)

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if float_input:
            input_data = (np.float32(input_data) - input_mean) / input_std
       
        # Perform the actual detection by running the model with the image as input
        self.model.set_tensor(input_details[0]['index'],input_data)
        self.model.invoke()

        # Retrieve detection results
        boxes = self.model.get_tensor(output_details[1]['index'])[0] # Bounding box coordinates of detected objects
        classes = self.model.get_tensor(output_details[3]['index'])[0] # Class index of detected objects
        scores = self.model.get_tensor(output_details[0]['index'])[0] # Confidence of detected objects

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

    """
    @Author xdevilscloverx
    @Description: This function takes a frame and returns a string of the detected plate
    """
    def read_plate(self, plate_img):
        # read the plate
        results = pytesseract.image_to_data(plate_img, lang = "en", config='-l eng --psm 7', nice=0, output_type="dict")
        
        # get the words with a confidence of 15 or higher
        words = []
        for i, word in enumerate(results['text']):
            conf = int(results['conf'][i])
            if conf >= 15:
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
    @Author xdevilscloverx
    @Description: This function handles the frame and adds the plate to the buffer and database
    """
    def handle_frame(self):
        while not self.stop:
            # access the frame
            frame = self.frame

            # get the detections
            detections = self.predict_plates(frame) # grayscale image RGB color space

            # loop over the detections
            for i, detection in detections.items():
                # get the coordinates
                xmin, ymin, xmax, ymax = detection

                # get the original plate image
                plate_img = frame[ymin:ymax, xmin:xmax]

                crop = Image.fromarray(plate_img)
                crop.save(f"outputs/crop{i}.jpeg")

                copy = cv2.imread(f"outputs/crop{i}.jpeg")
                copy = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)
                copy = cv2.bilateralFilter(copy, 11, 17, 17)    #remove noise
                thresh = cv2.adaptiveThreshold(copy, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
                close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
                result = 255 - close

                cv2.imwrite(f"outputs/result{i}.jpeg", result)
                # read the plate text
                plate_text = self.read_plate(result)

                # check if text was detected
                if plate_text != "":
                    print(plate_text)
                    # create a byte stream obj to store the image data
                    crop.save(self.stream, format="JPEG")
                    
                    # get the image data
                    image_data = self.stream.getvalue()

                    # reset the stream to the beginning
                    self.stream.seek(0)
                    self.stream.truncate()   #clear the stream

                    # add the plate text, image, and time to the buffer
                    self.buffer.update({plate_text: image_data})

                    # save the frame with the plate drawn on it
                    cv2.imwrite(f"outputs/frame{i}.jpeg", copy)
    
    """
    @Author xdevilscloverx
    @Description: This function uploads the results to the database or prints the results to the console every 10 seconds
    """
    def upload_to_database(self):
        while not self.stop:
            i = -1  #initialize the default index for printing
            #check if 5 seconds have passed
            if datetime.now() - self.time > timedelta(seconds=5):
                self.time = datetime.now()   #reset the timer
                if self.connected:
                    print("Uploading results to database...")
                    #loop over the buffer
                    for i, plate in enumerate(self.buffer.keys()):
                        if plate not in self.shift_buffer.keys():
                            try:
                                with self.connection.cursor() as cursor:
                                    #create the sql query
                                    sql = "INSERT INTO `licenses` (`license_pl`, `plate_img`) VALUES (%s, %s)"
                                    #execute the query
                                    cursor.execute(sql, (plate, self.buffer[plate]))
                                    #commit the changes
                                    self.connection.commit()
                                #upload the results to the database
                                print(f"'{plate}' uploaded to database!")
                            except pymysql.err.OperationalError as e:
                                print(f"Error uploading '{plate}' to database: {e}")
                                print("Storing results locally... (will upload when connection is restored)")
                                with open(f"{plate}", 'wb') as f:
                                    f.write(self.buffer[plate])
                else:
                    print("Printing results...")
                    #loop over the buffer
                    for i, plate in enumerate(self.buffer.keys()):
                        if plate not in self.shift_buffer.keys():
                            print(f"{plate}")  #print the plate

                print(f"{i+1} results captured + pushed!")

                #update the time buffer
                self.shift_buffer.clear()              #clear the time buffer before updating
                self.shift_buffer.update(self.buffer)  #update the time buffer
                #print(self.time_buffer.keys())               #print the time buffer -> show plate when empty!
                
                #clear the buffer
                self.buffer.clear()

    """
    @Author xdevilscloverx
    @Description: This function stops the threads and closes the connection to the database and frees resources
    """
    def stop_and_exit(self):
        while True:
            try:
                self.stop = True #stop the threads
                for thread in self.threads:
                    thread.join()   #stop the threads
                if self.connected:
                    self.connection.close() #close the connection to the database
        
                    if self.usbwebcam:
                        self.cam.release()
                    else:
                        self.cam.stop()
            except KeyboardInterrupt:
                print("Keyboard interrupt detected!")
                print("Stopping threads...")
                self.stop = True
                continue
            break
        
        print("Scanner stopped!")
        sys.exit(0) #exit the program

#run script
if __name__ == '__main__':

    # parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--modelpath", type=str, required=True,
                    help="Path to the input model")
    ap.add_argument("-f", "--filterpath", type=str, default=None,
                    help="Path to the filter csv file")
    ap.add_argument("-c", "--camera", type=bool, default=False,
                    help="Use the USB webcam? Default: False")
    ap.add_argument("-d", "--database", type=bool, default=False,
                    help="Connect to a database? Default: False")
    args = vars(ap.parse_args())

    # define the arguments
    model_path = args['modelpath']
    filter_file = args['filterpath']
    cam_setting = args['camera']
    database = args['database']
    
    # open the filter file
    try:
        filter_file = read_csv(filter_file)
        filter = tuple(filter_file['State'])
    except Exception as e:
        print(e, "Filter file not found! Proceeding with no text filter.")
        filter = None

    # if database setting true, attempt to connect to the database below
    if database:
        try:    
            #define the sql connection
            conn = pymysql.connect(
            host = "195.179.237.153",   #ip address of the database
            user = "u376552790_projectlabdb",   #username of the database
            password = "Ltg5075@",  #password of the database
            db = "u376552790_projectlabdb"  #name of the database
            )
            print("Database connection successful!")
        except Exception as e:
            print(e, "Database connection failed!")
            conn = None
    else:
        conn = None
    
    # if camera setting is false, import Picamera
    if not cam_setting:
        from picamera2 import Picamera2

   # import the necessary packages for the model
    pkg = importlib.util.find_spec('tflite_runtime')
    if pkg:
        from tflite_runtime.interpreter import Interpreter
    else:
        from tensorflow.lite.python.interpreter import Interpreter

    # Load the Tensorflow Lite model into memory 
    interpreter = Interpreter(model_path=model_path)

    # Allocate memory for the model
    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    floating_model = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5

    # Check output layer name to determine if this model was created with TF2 or TF1,
    # because outputs are ordered differently for TF2 and TF1 models
    outname = output_details[0]['name']

    if ('StatefulPartitionedCall' in outname): # This is a TF2 model
        boxes_idx, classes_idx, scores_idx = 1, 3, 0
    else: # This is a TF1 model
        boxes_idx, classes_idx, scores_idx = 0, 1, 2

    # Initialize the scanner
    scanner = platescanner(filter=filter, interpreter=interpreter,sqlconnector=conn, usbwebcam=cam_setting)

    # Start the scanner
    scanner.start()

    while True:
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            scanner.stop_and_exit()
            break