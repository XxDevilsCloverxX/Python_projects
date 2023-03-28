import os
import shutil

# Set the source and destination directories
source_dir = path = os.path.dirname(__file__) + "/images/xmls"    #<-- absolute dir the script is in + images folder
dest_dir1 = os.path.dirname(__file__) + "/images/testing_data/"  #testing data is smaller than training data
dest_dir2 = os.path.dirname(__file__) + "/images/training_data/" #training data is larger than testing data

# Get a list of all the XML files in the source directory
xml_files = [f for f in os.listdir(source_dir) if f.endswith('.xml')]

# Loop through the XML files
for xml_file in xml_files:
    # Check if there is a corresponding PNG file in the destination directory
    png_file = xml_file.replace('.xml', '.png')
    if png_file in os.listdir(dest_dir1):
        # Move the XML file to destination folder 1
        shutil.move(os.path.join(source_dir, xml_file), dest_dir1)
    else:
        # Move the XML file to destination folder 2
        shutil.move(os.path.join(source_dir, xml_file), dest_dir2)