import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import math
import argparse
import os
import random

class SoftMaxRegressor:
    
    def __init__(self, weights:np.ndarray=None, alpha:float=0,
                kernel='rbf') -> None:
        # model parameters
        self.weights = weights
        self.alpha = alpha
        self.kernel = kernel

    def img_preproc(self, img:np.ndarray) -> np.ndarray:
        """
        @purpose:
            Take an input image, apply preprocessing technique to it
        @return:
            Image with pre-processing accomplished
        """
        resize = cv2.resize(img, (30, 30))
        blur = cv2.GaussianBlur(resize, (3,3), sigmaX=0.8)

        # Convert the image to grayscale
        gray_img = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        norm_img = gray_img / 255
        return norm_img

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

            norm_img = self.img_preproc(img=img)

            # Display the image on the corresponding subplot
            axs[i].imshow(norm_img, cmap='gray')
            axs[i].axis('off')
            
            # Extracting the last numbers from the filename to use as the title
            title = ''.join(filter(str.isdigit, os.path.splitext(image_file)[0]))
            # Set the title for the subplot
            axs[i].set_title(title)
        
        # Hide remaining empty subplots, if any
        for j in range(i + 1, num_rows * num_cols):
            axs[j].axis('off')
        
        plt.show()

    def mapFeatures(self, input:np.ndarray):
        """
        @purpose:
            take an input vector & map its measurements to another feature space
        @param:
            input - input vector in input space
        @return:
            feature_vect - feature vector of input vector in mapped space
        """
        pass

    def rbfKernel(self, ):
        """
        """
        pass

    def gradient(self, predictions:np.ndarray, labels:np.ndarray, designMatrix:np.ndarray) -> np.ndarray:
        """
        @purpose:
            Compute the gradient of the iterattion of the cost function
        @return:
            gradient loss for each class
        """
        # Compute the error (difference between predictions and labels)
        error = predictions - labels  # N x K
        # Compute the gradient by multiplying the error by the design matrix
        grad = np.dot(designMatrix.T, error)    # (NxL)^T * N x K => L x K: Computes the loss for each measurement across all classes
        return grad            

    def fit(self, data:np.ndarray, labels:np.ndarray,
            epochs:int=10000, epsilon:float=1e-3, batch_size:int=1000, beta:float=0):
        """
        @purpose:
            Fit a model using gradient descent (mini batch)
        @param:
            data - data being fit -> feature space
            labels - target labels for the data - one-hot encoded
            epochs - max iterattions before grad descent is stopped
            epsilon - threshold to stop when delta ||Gradient|| less than this value
            batch_size - number of images to use in a batch for training
            beta - smoothing hyperparameter
        """
        assert beta >0, f'{beta} < 0 not permissible'
        assert batch_size >= 1, f'batch_size {batch_size} must be greater than or equal to 1'
        num_samples, num_features = data.shape
        num_classes = labels.shape[1]
        self.weights = np.zeros((num_features, num_classes))  # Initialize weights to zeros


    def predict(self, designMatrix:np.ndarray) -> np.ndarray:
        """
        @purpose:
            use the computed weights to make a prediction matrix, Y
        @param:
            designMatrix: Phi(x) -> mapped input vectors NxL
        @return:
            Y - > prediction matrix
        """
        z = designMatrix @ self.weights

        exp_z = np.exp(z)
        softmax_scores = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        return softmax_scores

    def lossFunction(self, labels:np.ndarray, predictions:np.ndarray)-> float:
        """
        @purpose:
            Compute the cost function error
        @params:
            labels      - one-hot labels of the data, NxK
            predictions - predictions of the data with Y matrix - mapped 0-1 NxK
        @return:
            Computed cross entropy Error - Soft-max regression
        """
        assert labels.shape == predictions.shape, f"Expected (NxK) matrices, got: {labels.shape}, {predictions.shape}"
        N = labels.shape[0]  # Number of samples
    
        # Compute cross-entropy loss using vectorized operations
        loss = -np.sum(labels * np.log(predictions)) / N
        
        return loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Logistic Regression model object test programs')
    parser.add_argument('--directory', '-d', type=str, help='Directory containing images')

    args = parser.parse_args()

    # If directory is not provided, prompt the user to enter the directory path manually
    if not args.directory:
        args.directory = input("Enter the directory path containing images: ")

    model = SoftMaxRegressor()
    model.visualizeDataset(directory=args.directory)