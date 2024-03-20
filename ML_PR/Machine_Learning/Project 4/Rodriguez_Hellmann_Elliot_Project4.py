import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import math
import argparse
import os
import random

class LogisticRegressor:
    
    def __init__(self, weights:np.ndarray=None, alpha:float=0,
                kernel='rbf') -> None:
        # model parameters
        self.weights = weights
        self.alpha = alpha
        self.kernel = kernel

    def visualizeDataset(self, directory:str=None):
        """
        @purpose:
            visualizes a given dataset of images provided a relative filepath
        @params: 
            filepath - string to a relative filepath for a set of images
        """
        if directory is None or not os.path.isdir(directory):
            print("Invalid directory path.")
            return
        
        # Get list of image file names in the directory
        image_files = [file for file in os.listdir(directory) if file.endswith(('.jpg', '.jpeg', '.png', '.tif'))]
        num_images = len(image_files)

        # Randomly select 100 images
        selected_files = random.sample(image_files, min(100, num_images))

        # Calculate number of rows and columns for the grid
        num_rows = min(5, len(selected_files))
        num_cols = math.ceil(len(selected_files) / num_rows)

        # Create a new figure
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(num_cols * 10, num_rows * 10))
        fig.subplots_adjust(hspace=0.1, wspace=0.1)
        axs = axs.flatten()

        for i, image_file in enumerate(selected_files):
            # Read the image
            image_path = os.path.join(directory, image_file)
            img = cv2.imread(image_path)
            
            # Convert the image to grayscale
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding   
            _, thresholded_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Display the image on the corresponding subplot
            axs[i].imshow(thresholded_img, cmap='gray')
            axs[i].axis('off')
            
            # Extracting the last numbers from the filename to use as the title
            title = ''.join(filter(str.isdigit, os.path.splitext(image_file)[0]))
            # Set the title for the subplot
            axs[i].set_title(title)
        
        # Hide remaining empty subplots, if any
        for j in range(i + 1, num_rows * num_cols):
            axs[j].axis('off')
        
        plt.show()

    def mapFeatures(self):
        """
        """
        pass

    def rbfKernel(self, ):
        """
        """
        pass

    def fit(self, ):
        """
        """
        pass

    def predict(self, ):
        """
        """
        pass

    def lossFunction(self, ):
        """
        """
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Logistic Regression model object test programs')
    parser.add_argument('--directory', '-d', type=str, help='Directory containing images')

    args = parser.parse_args()

    # If directory is not provided, prompt the user to enter the directory path manually
    if not args.directory:
        args.directory = input("Enter the directory path containing images: ")

    model = LogisticRegressor()
    model.visualizeDataset(directory=args.directory)