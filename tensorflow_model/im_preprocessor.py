#library for image preprocessing
from PIL import Image
import os
from random import randint

diff_dir = False
rgb = True

#generate a path to the images folder from cwd
path = os.path.dirname(__file__) + "/images/"    #<-- absolute dir the script is in + images folder

#testing and traing data paths
testing_path = path + "/train/"
training_path = path + "/validate/"
i = 0

#loop through all the images in the folder and resize them
for i, filename in enumerate(os.listdir(path)):
    if filename.endswith(".xml"):
        continue
    try:
        #read the image
        img = Image.open(path + filename)
        #resize the image
        img = img.resize(size=(1280, 720))  # (width, height) 16:9 ratio
        
        if rgb:
            # convert the image to RGB format
            img = img.convert('RGB')
        else:
            # convert the image to grayscale
            img = img.convert('L')

        #overwrite the image
        img.save(path + filename)

        # move the image
        if diff_dir:
                #save the image to the training or testing folder -> 80% training, 20% testing
            if randint(0, 10) > 8:
                img.save(testing_path + filename)
            elif randint(0,10) <=8:
                img.save(training_path + filename)
            else:
                img.save(path + filename)   #<-- save the image to the images folder

            img.close() #<-- close the image
            os.remove(path + filename)  #<-- delete the image from the images folder if it was saved to the training or testing folder
            

    except Exception as e:
        #if there is an error, close the image and leave delete it
        img.close()
        os.remove(path + filename)
        print(f"{filename} deleted due to: {e}")
    finally:
        #increment the counter and print the progress
        if i % 20 == 0:
            print(f"Progress {i} images processed and saved.")

print("Done!")