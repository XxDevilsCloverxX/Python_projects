#training libraries
import torch
from detecto.utils import read_image
from detecto.core import Dataset, DataLoader, Model
from detecto.visualize import show_labeled_image
import torchvision.models as models
from random import randint
#scanning libraries
import cv2
import pytesseract

validate = True

#files used in this trainer
files = ('validate1.jpeg', 'validate2.jpeg',
          'validate3.jpeg', 'validate4.jpeg')
base = '/home/xxdevilscloverxx/Documents/Vs_CodeSpace/Python_Projects/torch_lite'
location = '/home/xxdevilscloverxx/Documents/Vs_CodeSpace/Python_Projects/torch_lite/images/annotations'

#create a model object
label = ['licence']
plates_model = Model(label)

#try to load a model if one exists
try:
    # fill your architecture with the trained weights
    plates_model = Model.load(file=f"{base}/plates.pth", classes=['licence'])
#construct a model otherwise
except:
    print(f"{base}/plates.pth was not found: Constructing model from {location}...")
    #display if GPU is in use
    print(torch.cuda.is_available())

    #get base location of image set
    data = Dataset(location)

    #show a random image of first 100 + label
    image, targets = data[randint(0, 100)]
    show_labeled_image(image, targets['boxes'], targets['labels'])

    #train the model
    plates_model.fit(data)

    # save the weights of the model to a .pth file
    plates_model.save(f"{base}/plates.pth")

#test the model with some images the dataset has never seen
finally:
    pass
    if (validate):

        for file in files:
            image = read_image(f"{base}/{file}")
            labels, boxes, scores = plates_model.predict(image)
            print(f"x1, y1, x2, y2 = {boxes}\n Scores:{scores}")
            #get the predictions the model wasn't confident about
            filter = [index for index,val in enumerate(scores) if val > .5]
            boxes = boxes[filter]
            print(f"Revised boxes: {boxes}")

            show_labeled_image(image, boxes, labels)
            #cv2.imwrite("temp.jpg", image[boxes[0][1]:boxes[0][3], boxes[0][0]:boxes[0][1]])
            #image = cv2.imread("temp.jpg")        
            #cv2.imshow("Cropped Tensor:", image)
            #cv2.waitKey(0)
            #collect and print the plate
            #print(pytesseract.image_to_string(image))      
    