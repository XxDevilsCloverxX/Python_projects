#library for image preprocessing
from PIL import Image
import os
import random

#generate a path to the images folder from cwd
path = os.path.dirname(__file__) + "/images/"    #<-- absolute dir the script is in + images folder

#testing and traing data paths
testing_path = path + "/testing_data/"
training_path = path + "/training_data/"
i = 0

#loop through all the images in the folder and resize them
for filename in os.listdir(path):
    try:
        #read the image
        img = Image.open(path + filename)
        #resize the image
        img = img.resize(size=(1280, 720))  # (width, height) 16:9 ratio
        #gray scale the image
        img = img.convert('L')
        #save the image to the training or testing folder -> 80% training, 20% testing
        if random.randint(0, 10) > 8:
            img.save(testing_path + filename)
        else:
            img.save(training_path + filename)
        #close the image
        img.close()
        #delete the image from the images folder
        os.remove(path + filename)
    except Exception as e:
        #if there is an error, print the error
        print(f"{e}")
        #if there is an error, close the image and leave it
        img.close()
    finally:
        #increment the counter and print the progress
        i +=1
        if i % 20 == 0:
            print("Processing image: " + filename + "|" + str(i) + " images processed and saved.")

print("Done!")