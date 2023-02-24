import cv2
import os

# Set the directory containing the images to convert
img_dir = '/home/xxdevilscloverxx/Documents/Vs_CodeSpace/python_projects/torch_lite/images/annotations'

# Loop through all files in the directory
for filename in os.listdir(img_dir):
    # Check if the file is an image (JPEG, PNG, etc.)
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
        # Read the image in RGB format
        img = cv2.imread(os.path.join(img_dir, filename))

        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Write the grayscale image to a new file
        new_filename = os.path.splitext(filename)[0] + os.path.splitext(filename)[1]
        cv2.imwrite(os.path.join(img_dir, new_filename), gray)