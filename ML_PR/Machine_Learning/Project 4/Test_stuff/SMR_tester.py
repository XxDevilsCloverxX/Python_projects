import tensorflow as tf
import os
import cv2
import argparse
from time import time
from ML_functions import *
from SMR import SoftMaxRegressor

def read_worm_image(file_path, img_size=(28, 28)):
    # Convert the image to grayscale
    gray_image = cv2.imread(file_path)
    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray_image, (img_size))
    
    # Apply Otsu's thresholding to the grayscale image
    _, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Convert to TensorFlow tensor
    img = tf.convert_to_tensor(otsu, dtype=tf.float32)
    # normalize
    img = tf.truediv(img, 255)

    # flatten the image
    img = tf.reshape(img, (1,-1))  # Flatten to a 1D tensor
    return img

def read_mnist_img(file_path, img_size=(28,28)):
    # Read TIFF image using OpenCV
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    # Resize image
    img = cv2.resize(img, img_size)
    # Convert image to TensorFlow tensor
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = tf.reshape(img, (1,-1)) / 255.
    return img

def process_images(image_dir, use_worm:bool=False, img_size=(28, 28)):
    # List image files in directory
    image_files = []
    for file in os.listdir(image_dir):
        if file.endswith('.png') or file.endswith('.tif'):
            image_files.append(os.path.join(image_dir, file))

    # Process each image one at a time
    for file_path in image_files:
        # Read image
        if use_worm:
            img_tensor = read_worm_image(file_path, img_size)
        else:
            img_tensor = read_mnist_img(file_path, img_size)
        yield img_tensor, file_path

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Image Batch Generator')
    parser.add_argument('-d', '--directory', type=str, required=True, help='Parent directory containing image data')
    args = parser.parse_args()
    # Determine if PNG or TIF images are present
    is_png = any(file.endswith('.png') for file in os.listdir(args.directory))
    is_tif = any(file.endswith('.tif') for file in os.listdir(args.directory))
    if is_png and is_tif:
        print('Images of two types detected, this cannot be inferenced properly please separate')
        return
    elif is_png:
        # load the worm weights
        smr = SoftMaxRegressor(init_weights='worm_weights.npy')
    else:
        # load the mnist weights
        smr = SoftMaxRegressor(init_weights='mnist_weights.npy')

    test_pred = []
    filenames = []
    time_x = time()
    for tensor, file_path in process_images(args.directory, use_worm=is_png):
        filenames.append(file_path)
        test_pred.extend(smr.predict(tensor))
    print(f'{time() - time_x:.3f} s to predict {len(filenames)} files')

    test_pred = np.array(test_pred)
    filenames = np.array(filenames)

    # Define a custom sorting key function
    def natural_sort_key(s):
        import re
        return [int(text) if text.isdigit() else text.lower() for text in re.split('(\d+)', s)]
    # Get the sorted indices based on natural sorting
    sorted_in = sorted(range(len(filenames)), key=lambda x: natural_sort_key(filenames[x]))

    test_pred = test_pred[sorted_in]
    filenames = filenames[sorted_in]

    write_out = 'worms.xlsx' if is_png else 'mnist.xlsx'
    outpath = write_predictions_to_excel(predictions=test_pred, filenames=filenames, output_file=write_out)
    print(f'Excel file written to {outpath}')

if __name__ == '__main__':
    main()
