#import libraries
import torch
import pandas as pd
from detecto.utils import read_image
from detecto.core import Dataset, DataLoader, Model
from detecto.visualize import show_labeled_image
import torchvision.models as models
from random import randint
#scanning libraries
import cv2
import pytesseract
import imutils

states = pd.read_csv('/home/xxdevilscloverxx/Documents/Vs_CodeSpace/python_projects/torch_lite/states.csv')
states = tuple(states['State'].to_list())
states = [state.upper() for state in states]
base = '/home/xxdevilscloverxx/Documents/models'

#create a model object
label = ['licence']
plates_model = Model(label)

def most_frequent(List):
    try:
        return max(set(List), key = List.count)
    except:
        return None

# Define a function to detect license plates in a single frame
def detect_license_plates(frame, model):
    # Resize the frame to a fixed height for consistency
    frame = imutils.resize(frame, height=720)
    
    # Apply the pre-trained model to the grayscale frame
    labels, boxes, scores = model.predict(frame)
    
    #get the predictions the model wasn't confident about
    filter = [index for index,val in enumerate(scores) if val > .5]
    boxes = boxes[filter]  #return tensors from the filter
    labels = [labels[index] for index in filter]
    if len(boxes) > 0:
        boxes = boxes[0] #get the largest plate
        labels = labels[0]
    else:
        return None
    # Iterate over the license plate detections and extract the text
    x1,y1,x2,y2 = boxes
    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)

    plate_img = frame[y1-10:y2+10, x1-10:x2+10]
    plate_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    plate_img = cv2.medianBlur(plate_img, 3)

    cv2.imshow("Cropped", plate_img)
    plate_text = pytesseract.image_to_string(plate_img, config='--psm 11', lang='eng').strip() 
    
    #clean the text
    plate_text = [char for char in plate_text if char.isalnum()] #remove special characters
    plate_text = "".join(plate_text) #join the characters
    plate_text = plate_text.upper() #convert to uppercase
    
    #check if the plate is from a state
    for state in states:
        if state in plate_text:
            plate_text = plate_text.replace(state, '')
    
    #check if the plate is valid
    if len(plate_text) < 5 or len(plate_text) > 7:
        return None
    
    #finally, decline all-numeric plates
    if plate_text.isnumeric():
        return None

    return plate_text

# Define a function to process a live video stream
def process_live_video(model):
    # Initialize the video capture object
    cap = cv2.VideoCapture(0)

    i = 0
    buffer = [30]
    # Loop over frames from the video stream
    while True:
        if i > 30:
            print(most_frequent(buffer))
            buffer.clear()
            i = 0
        i+=1
        
        # Read the next frame from the video stream
        ret, frame = cap.read()

        # If we couldn't read the next frame, break
        if not ret:
            raise ValueError("Frame could not be read, aborting!")

        # Detect license plates in the frame
        plate_text = detect_license_plates(frame, model)

        # Print the license plate texts to the console
        if plate_text != None:
            buffer.append(plate_text)

        # Display the frame (optional)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the video capture object and close any open windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # fill your architecture with the trained weights
    plates_model = Model.load(file=f"{base}/plates.pth", classes=['licence'])
    process_live_video(plates_model)