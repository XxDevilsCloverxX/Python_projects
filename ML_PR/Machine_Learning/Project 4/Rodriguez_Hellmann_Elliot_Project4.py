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
            resize = cv2.resize(img, (30, 30))
            blur = cv2.GaussianBlur(resize, (3,3), sigmaX=0.8)

            # Convert the image to grayscale
            gray_img = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
            norm_img = gray_img / 255

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


        """
        """
        pass

    def clearWeights(self):
        self.weights = np.zeros_like(self.weights) if self.weights is not None else None

    def generate_minibatch(self, directory:str=None, batch_size:int=1):
        """
            Generate a mini-batch of images along with labels from the given directory.
        Args:
            directory (str): Path to the directory containing images.
            batch_size (int): Number of images to include in each mini-batch.

        Yields:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the mini-batch of images and their labels.
        """
        assert directory is not None, "A directory must be provided."

        subdirectories = sorted([os.path.join(directory, subdir) for subdir in os.listdir(directory) if
                            os.path.isdir(os.path.join(directory, subdir))])

        # for each subdirectory, get a list of image filenames
        subdir_images = []
        for subdir in subdirectories:
            image_files = [os.path.join(subdir, file) for file in os.listdir(subdir) if file.endswith(('.jpg', '.jpeg', '.png', '.tif'))]
            subdir_images.append(image_files)
        refill = [img.copy() for img in subdir_images]

        while True:
            # bool to track the state of refills - used for epochs
            exhaust = False
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
                size = batch_size//len(subdirectories) if batch_size//len(subdirectories) < len(imageset) else len(imageset)
                select_indices = np.random.choice(len(imageset), size=size, replace=False)
                selected_images.append([imageset.pop(idx) for idx in sorted(select_indices, reverse=True)])

                # Create the labels for each of the images - one hot encoding
                Labels = np.array([np.eye(len(subdirectories))[dir] for _ in select_indices])
                selected_labels.append(Labels)
            
            # if the subdir_images are empty, refill
            if all([len(images)==0 for images in subdir_images]):
                for dir, row in enumerate(refill):
                    subdir_images[dir] = row.copy()
                exhaust = True

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
            yield input_X, input_labels, exhaust

    def gradient(self, scores: np.ndarray, labels: np.ndarray, designMatrix: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the iteration of the cost function.

        Args:
            scores (np.ndarray): Predicted scores for each class.
            labels (np.ndarray): One-hot encoded target labels.
            designMatrix (np.ndarray): Design matrix or input features.

        Returns:
            np.ndarray: Gradient of the loss function with respect to the weights.
        """
        # Compute the gradient of the loss function
        gradient = np.mean(designMatrix.T @ (scores - labels), axis=0)
    
        # Calculate the norm of the gradient
        # gradient_norm = np.linalg.norm(gradient)
        
        # # If the gradient norm exceeds the maximum allowed norm, scale down the gradient
        # if gradient_norm > 5000:
        #     gradient *= 5000 / gradient_norm
        return gradient

    
    def fit(self, directory:str=None, batch_size:int=1,
            epochs:int=10000, epsilon:float=1e-3):
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
        # create a mini_batch generator
        generator = self.generate_minibatch(directory=directory, batch_size=batch_size)
        batch_x, batch_t, epoch_flag = next(generator)

        _, num_features = batch_x.shape
        num_classes = batch_t.shape[1]                        # one hot encoding shape
        self.weights = np.zeros((num_features, num_classes))  # Initialize weights to ones
        print(f'Init Weights: {self.weights.shape} * 0')
        self.loss.clear()                                     # calling this generator will clear the loss computed previously
        
        epoch_loss = []
        # upper limit on the epochs
        for epoch in range(epochs):
            print(f'Working Epoch: {epoch}/{epochs}')

            batch_losses = []            
            # train through the whole batch before next epoch
            while epoch_flag is not True:
                # calculate scores on the data
                probs = self.calcProbability(batch_x)
                # Compute gradient
                grad = self.gradient(scores=probs, labels=batch_t, designMatrix=batch_x)
                # Update weights using gradient descent with L2 regularization
                self.weights -= self.learn_rate * grad

                # Compute loss
                batch_loss = self.cross_entropy_cost(scores=probs, y_targets=batch_t)
                print(f'Batch Loss: {batch_loss}')
                batch_losses.append(batch_loss) # add the loss from this batch
                # update the data
                batch_x, batch_t, epoch_flag = next(generator)

            
            # reduce learning rate
            self.learn_rate *= 0.9
            # Calculate average loss for the epoch
            # avg_loss = total_loss / num_samples
            # self.loss.append(avg_loss)
            
            # Print loss for every 100 epochs
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {avg_loss}")

            # Check for convergence
            if len(self.loss) > 1 and abs(self.loss[-1] - self.loss[-2]) < epsilon:
                print(f"Minimum reached at epoch {epoch}. Stopping training.")
                break
            

    def calcProbability(self, designMatrix:np.ndarray) -> np.ndarray:
        """
        @purpose:
            use the computed weights to make a prediction matrix, Y
        @param:
            designMatrix: Phi(x) -> mapped input vectors NxL
        @return:
            Y - > prediction matrix of probabilites
        """
        probabilities = []
        for row in designMatrix:
            temp = row @ self.weights
            probabilities.append(self.softmax(temp))
        return np.array(probabilities)

    def softmax(self, input_vector):
        # Calculate the exponent of each element in the input vector
        copy = input_vector - np.max(input_vector)
        exponents = np.exp(copy)

        # divide the exponent of each value by the sum of the exponents
        sum_of_exponents = np.sum(exponents)
        probabilities = exponents / sum_of_exponents

        return probabilities

    def cross_entropy_cost(self, y_targets, scores):
        cost = 0
        samples, class_count = y_targets.shape
        # for every element
        for row in range(samples):
            print(y_targets * np.log(scores))
            cost -= np.sum(y_targets * np.log(scores))
        print(cost)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Logistic Regression model object test programs')
    parser.add_argument('--directory', '-d', type=str, help='Directory containing images')
    parser.add_argument('--batch_size', '-b', type=int, default=1000, help='Batch size for training')
    parser.add_argument('--epochs', '-e', type=int, default=10000, help='Num Epochs (max) for training')
    parser.add_argument('--lambda', '-l', type=float, default=0, help='Regularizer for training')
    args = parser.parse_args()

    if not args.directory:
        args.directory = input("Enter the directory path containing images: ")

    model = SoftMaxRegressor()
    
    model.fit(directory=args.directory, batch_size=args.batch_size, epochs=args.epochs,)