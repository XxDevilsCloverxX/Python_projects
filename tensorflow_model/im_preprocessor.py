#library for image preprocessing
from PIL import Image
import os

diff_dir = False
rgb = False
ext = ".jpg"   #<-- file extension to save as

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
        with Image.open(path + filename) as img:
            #resize the image to 1280x720
            img = img.resize(size=(1280, 720))  # (width, height) 16:9 ratio
            
            if rgb:
                # convert the image to RGB format
                img = img.convert('RGB')
            else:
                # convert the image to grayscale
                img = img.convert('L')
                img = img.convert('RGB')    #convert to RGB to save as jpeg

            #overwrite the image
            img_name = os.path.join(path, os.path.splitext(filename)[0] + ext)
            img.save(img_name)

            if not filename.endswith(ext):
                #delete the original image
                os.remove(path + filename)

            #close the image
            img.close()


    except Exception as e:
        #if there is an error, close the image and leave delete it
        img.close()
        os.remove(path + filename)
        print(f"{filename} deleted due to: {e}")
    finally:
        #increment the counter and print the progress
        if i % 20 == 0:
            print(f"Progress {i} files processed and saved.")

print("Done!")