#training libraries
import torch
from detecto.utils import read_image
from detecto.core import Dataset, DataLoader, Model
from detecto.visualize import show_labeled_image
from random import randint
# import the modules
import os
from os import listdir

#list a base directory
base = "/home/xxdevilscloverxx/Documents/Vs_CodeSpace/python_projects/sara"
#list a location of the images
training = f"{base}/images/train"
testing = f"{base}/images/test"

#create a model object using the existing labels
with open(f"{base}/labels.txt", "r") as f:
    labels = f.read().splitlines()
    f.close()

sara_model = Model(labels)

#try to load a model if one exists
try:
    # fill your architecture with the trained weights
    sara_model = Model.load(file=f"{base}/sara.pth", classes=labels)
    validate = True

#construct a model otherwise
except:
    #do not validate the model
    validate = False

    print(f"{base}/sara.pth was not found: Constructing model from {training}...")
    #display if GPU is in use
    print(f"GPU usage: {torch.cuda.is_available()}")

    #get base location of image set
    data = Dataset(training)

    #show a random image of first 100 + labels
    image, targets = data[randint(0, 100)]
    show_labeled_image(image, targets['boxes'], targets['labels'])

    #train the model
    sara_model.fit(data)

    # save the weights of the model to a .pth file
    sara_model.save(f"{base}/sara.pth")

#test the model with some images the dataset has never seen
finally:
    if (validate):

        for image in os.listdir(testing):
            if(image.endswith(".jpg") or image.endswith(".jpeg") or image.endswith(".png") or image.endswith(".JPEG") or image.endswith(".JPG") or image.endswith(".PNG")):
               
                image = read_image(f"{testing}/{image}")

                labels, boxes, scores = sara_model.predict(image)

                #get the predictions the model wasn't confident about
                filter = [index for index,val in enumerate(scores) if val > .5]
                boxes = boxes[filter]  #return tensors from the filter
                labels = [labels[index] for index in filter]
                scores = [scores[index] for index in filter]
                
                
                print(f"Revised Boxes: {boxes}")
                print(f"Revised Labels: {labels}")
                print(f"Scores: {scores}")

                show_labeled_image(image, boxes, labels)
