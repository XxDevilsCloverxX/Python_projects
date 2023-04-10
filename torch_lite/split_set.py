#import necessary packages
import os
import random
import shutil

# Set the path to the folder containing the images and their XML files
data_folder = os.path.join(os.path.dirname(__file__), "images")

# Set the percentage of images to use for testing
test_percent = 20

# Create the train and test subfolders
train_folder = os.path.join(data_folder, "train")
test_folder = os.path.join(data_folder, "test")
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Loop through each image file in the data folder
for filename in os.listdir(data_folder):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg") or filename.endswith(".JPG") or filename.endswith(".PNG") or filename.endswith(".JPEG"):
        
        # Get the corresponding XML file for this image
        xml_filename = os.path.splitext(filename)[0] + ".xml"
        xml_filepath = os.path.join(data_folder, xml_filename)
        
        # Determine whether to put this image in the train or test set
        if random.randint(0, 99) < test_percent:
            dest_folder = test_folder
        else:
            dest_folder = train_folder
        
        # Copy the image and XML file to the appropriate destination folder
        shutil.copy2(os.path.join(data_folder, filename), os.path.join(dest_folder, filename))
        shutil.copy2(xml_filepath, os.path.join(dest_folder, xml_filename))