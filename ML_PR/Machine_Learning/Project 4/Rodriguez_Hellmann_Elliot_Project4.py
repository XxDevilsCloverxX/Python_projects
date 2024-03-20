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
        self.loss = []
        self.learn_rate = 1.0

    def imgPreProcess(self, img:np.ndarray) -> np.ndarray:
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
        norm_img = norm_img.flatten()
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

    def clearWeights(self):
        self.weights = np.zeros_like(self.weights) if self.weights is not None else None

    def generate_minibatch(self, directory:str, batch_size:int=1):
        """
            Generate a mini-batch of images along with labels from the given directory.
        Args:
            directory (str): Path to the directory containing images.
            batch_size (int): Number of images to include in each mini-batch.

        Yields:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the mini-batch of images and their labels.
        """
        subdirectories = sorted([os.path.join(directory, subdir) for subdir in os.listdir(directory) if
                            os.path.isdir(os.path.join(directory, subdir))])

        # for each subdirectory, get a list of image filenames
        subdir_images = []
        for subdir in subdirectories:
            image_files = [os.path.join(subdir, file) for file in os.listdir(subdir) if file.endswith(('.jpg', '.jpeg', '.png', '.tif'))]
            subdir_images.append(image_files)
        refill = [img.copy() for img in subdir_images]

        while True:
            # for each of the subdirectories of images, generate some images
            selected_images = []
            selected_labels = []
            for dir, imageset in enumerate(subdir_images):
                # if a list is exhausted, continue with other directories
                if len(imageset) == 0:
                    continue
                # shuffle the data
                random.shuffle(imageset)
                # select some of the data, and remove it from the imageset
                size = batch_size if batch_size < len(imageset) else len(imageset)
                select_indices = np.random.choice(len(imageset), size=size, replace=False)
                selected_images.append([imageset.pop(idx) for idx in sorted(select_indices, reverse=True)])

                # Create the labels for each of the images - one hot encoding
                Labels = np.array([np.eye(len(subdirectories))[dir] for _ in select_indices])
                selected_labels.append(Labels)
            
            # preprocess the images
            input_X = []
            for subdir in selected_images:
                for filename in subdir:
                    img = cv2.imread(filename)
                    norm_img = self.imgPreProcess(img=img)
                    input_X.append(norm_img)

            # images pre-processed, create a np array for the input matrix
            input_X = np.array(input_X)
            input_labels = np.vstack(selected_labels)
            # yield the batch, and the labels
            yield input_X, input_labels

            # if the subdir_images are empty, refill
            if all([len(images)==0 for images in subdir_images]):
                for dir, row in enumerate(refill):
                    subdir_images[dir] = row.copy()

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
        self.loss.clear()

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
    parser.add_argument('--batch_size', '-b', type=int, default=1000, help='Batch size for training')
    parser.add_argument('--epochs', '-e', type=int, default=10000, help='Num Epochs (max) for training')

    args = parser.parse_args()

    if not args.directory:
        args.directory = input("Enter the directory path containing images: ")

    model = SoftMaxRegressor()
    data_generator = model.generate_minibatch(args.directory, args.batch_size)

    for i in range(args.epochs):
        print(f'Batch {i}: ')
        batch_data, labels = next(data_generator)
        # Use batch_data and batch_labels for training
