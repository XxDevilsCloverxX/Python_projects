#training libraries
import torch
from detecto.utils import read_image
from detecto.core import Dataset, DataLoader, Model
from detecto.visualize import show_labeled_image
from random import randint
# import the modules
import os
from os import listdir
import time

validate = True

#files used in this trainer
file_base = 'validate'

base = os.path.dirname(os.path.abspath(__file__))
location = os.path.join(base, 'images/train')

#create a model object
label = ['license']
plates_model = Model(label)

#try to load a model if one exists
try:
    # fill your architecture with the trained weights
    plates_model = Model.load(file=f"{base}/plates.pth", classes=label)
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
    if (validate):

        for image in os.listdir(base):
            if(image.startswith('validate') and (image.endswith(".jpg") or image.endswith(".jpeg") or image.endswith(".png"))):
               
                image = read_image(f"{base}/{image}")
                #rate the speed
                start = time.time()
                labels, boxes, scores = plates_model.predict(image)
                end = time.time()

                #get the predictions the model wasn't confident about
                filter = [index for index,val in enumerate(scores) if val > .5]
                boxes = boxes[filter]  #return tensors from the filter
                labels = [labels[index] for index in filter]
                print(f"Revised Boxes: {boxes}")
                print(f"Revised Labels: {labels}")
                print(f"Scores: {scores}")  #I want to print all scores regardless to see what was filtered out
                print(f"Time for prediction: {round(end - start, 3)}, 'seconds")

                show_labeled_image(image, boxes, labels)
