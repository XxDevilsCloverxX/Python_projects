#import libraries
import statistics as st
import pandas as pd
import easyocr as ocr
from detecto.core import Model
from random import randint
#scanning libraries
import cv2
import imutils

states = pd.read_csv('/home/xxdevilscloverxx/Documents/Vs_CodeSpace/python_projects/torch_lite/states.csv')
states = tuple(states['State'].to_list())
states = [state.upper() for state in states]
base = '/home/xxdevilscloverxx/Documents/models'

#create a model object
label = ['licence']
plates_model = Model(label)

# Define a function to detect license plates in a single frame
def detect_license_plates(frame, model):
    # Resize the frame to a fixed height for consistency
    frame = imutils.resize(frame, height=720)
    
    # Apply the pre-trained model to the frame
    labels, boxes, scores = model.predict(frame)
    
    #get the predictions the model wasn't confident about
    filter = [index for index,val in enumerate(scores) if val > .5]
    boxes = boxes[filter]  #return tensors from the filter
    labels = [labels[index] for index in filter]

    #zip the plates into a dictionary of label: index pairs    
    plates = {key: index for index, key in enumerate(labels)}

    #return the recovered plates
    return plates, boxes

# Define a function to process a dictionary of plates and boxes
def process_plates(frame, plates, boxes):
    #create a reader obj
    reader = ocr.Reader(['en'], gpu=True)
    #create a storage for the plate texts
    plate_texts = []
    # draw bounding boxes around detected plates
    if len(boxes) > 0:
        for key, index in plates.items():
            x1,y1,x2,y2 = boxes[index]
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            #draw bounding boxes for debugging
            cv2.rectangle(frame, (x1, y1), (x2, y2), (randint(0,255), randint(0,255), randint(0,255)), 2)
            cv2.putText(frame, key, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            #crop the plate
            plate_img = frame[y1-10:y2+10, x1-10:x2+10]
            plate_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            plate_img = cv2.medianBlur(plate_img, 3)
            #read the plate from image
            results = reader.readtext(plate_img, detail=0)

            #filter the results for the plate + remove useless strings and states
            results = [_ for _ in results if len(_) > 5 and len(_) < 9 and not _.isnumeric() and _.upper() not in states]

            for detection in results:
                #clean the text
                plate_text = [char for char in detection if char.isalnum()]
                plate_text = "".join(plate_text)
                plate_text = plate_text.upper()
                #append the text to the list
                plate_texts.append(plate_text)
                
    return plate_texts


# Define a function to process a live video stream
def process_live_video(model):
    # Initialize the video capture object
    cap = cv2.VideoCapture(0)
    i = 0
    buffer = []
    # Loop over frames from the video stream
    while True:
        if i > 30 and len(buffer) > 0:
            print(st.mode(buffer))
            buffer.clear()
            i = 0
        i+=1
        
        # Read the next frame from the video stream
        ret, frame = cap.read()

        # If we couldn't read the next frame, break
        if not ret:
            raise ValueError("Frame could not be read, aborting!")

        # Detect license plates in the frame
        plates, boxes = detect_license_plates(frame, model)

        # Process the detected plates
        plate_texts = process_plates(frame=frame, plates=plates, boxes=boxes)            

        # add the plate texts to the buffer
        buffer.extend(plate_texts)

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