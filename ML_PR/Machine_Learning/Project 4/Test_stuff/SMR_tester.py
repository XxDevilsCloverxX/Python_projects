import os
import argparse
from time import time
from ML_functions import *
from SMR import SoftMaxRegressor

def process_images(image_dir, img_size=(28, 28)):
    # List image files in directory
    image_files = []
    for file in os.listdir(image_dir):
        if file.endswith('.png') or file.endswith('.tif'):
            image_files.append(os.path.join(image_dir, file))

    # Process each image one at a time
    for file_path in image_files:
        # Read image
        img_tensor = read_test_image(file_path=file_path, img_size=img_size)
        yield img_tensor, file_path

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Image Batch Generator')
    parser.add_argument('-d', '--directory', type=str, required=True, help='Parent directory containing image data')
    args = parser.parse_args()
    # Determine if PNG or TIF images are present
    is_png = any(file.endswith('.png') for file in os.listdir(args.directory))
    is_tif = any(file.endswith('.tif') for file in os.listdir(args.directory))
    if is_png and is_tif or not (is_png or is_tif):
        raise FileNotFoundError('Varying image extensions detected, expected .png or .tif only')
    elif is_png:
        # load the worm weights
        smr = SoftMaxRegressor(init_weights='worm_weights.npy')
    else:
        # load the mnist weights
        smr = SoftMaxRegressor(init_weights='mnist_weights.npy')

    test_pred = []
    filenames = []
    time_x = time()
    for tensor, file_path in process_images(args.directory):
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
