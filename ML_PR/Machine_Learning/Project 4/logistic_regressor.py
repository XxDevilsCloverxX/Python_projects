import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import argparse
import os

class LogisticRegressor:

    def __init__(self, regularizer:float,
                  initial_rate:float=1.0) -> None:
        """
        Constructor with training & testing parameters to initialize
        """
        assert regularizer >=0, f'{regularizer} is negative, expected positive'
        assert initial_rate >0, f'{initial_rate} not positive'
        self.weights = None
        self.regularizer = regularizer
        self.loss = []
        self.learn_rate = initial_rate

    def clearWeights(self):
        """
        A method to clear the current weight vector
        """
        self.weights= np.zeros_like(self.weights) if self.weights is not None else None

    def adjustLearningRate(self, newrate:float):
        """
        a method for manually adjusting learning rate (can be set higher, lower, etc)
        """
        assert newrate >0, f'{newrate} not > 0'
        self.learn_rate

    def clearLoss(self):
        """
        A method to clear the loss data
        """
        self.loss.clear()

    def predictScores(self, X):
        """
        A method to evaluate the probability of a singular sample or a matrix of samples X
        """
        if self.weights is None:
            raise ValueError("Model has not been trained yet. No weights available.")

        return self.sigmoid((X@self.weights))

    def sigmoid(self, x):
        """
        Sigmoid function to compute the sigmoid of input vector x
        """
        return 1 / (1 + np.exp(-x))

    def binaryCrossEntropyCost(self, y_true:np.ndarray, y_pred:np.ndarray):
        """
        a method to compute the cost function
        """
        m = len(y_true)
        epsilon = 1e-15  # small value to avoid log(0)
        cost = (-1 / m) * np.sum(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))
        regularization_term = (self.regularizer / (2 * m)) * np.sum(self.weights ** 2)
        return cost + regularization_term
    
    def costGradient(self, X, y_true, y_pred):
        """
        A function to compute the gradient of the binary cross-entropy cost function
        """
        m = len(y_true)
        gradient = (1 / m) * np.dot(X.T, (y_pred - y_true))
        regularization_term = (self.regularizer / m) * self.weights
        return gradient + regularization_term

    def train(self, X: np.ndarray, y: np.ndarray,
               verbose: bool = True):
        """
        A method to update the weights based on the samples X and corresponding labels y
        """
        # Initialize weights if not already initialized
        if self.weights is None:
            self.weights = np.zeros(X.shape[1])

        # Forward pass
        z_scores = self.predictScores(X)
        # convert scores to predictions
        y_pred = np.array([np.eye(2)[np.argmax(scores)] for scores in z_scores])

        # Compute cost
        cost = self.binaryCrossEntropyCost(y, y_pred)
        self.loss.append(cost)

        # Compute gradient
        gradient = self.costGradient(X, y, y_pred)

        # Update weights
        self.weights -= self.learn_rate * gradient

        if verbose:
            print("Weight update finished.")

def imgPreProcess(img:np.ndarray) -> np.ndarray:
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

def generate_minibatch(directory:str=None, batch_size:int=1):
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

    # This generator will last until all subdirectory images have been exhausted (i.e. epoch)
    while all(len(images)!= 0 for images in subdir_images):
        # print(f'{sum(len(images) for images in subdir_images)} images remaining in batch')
        # for each of the subdirectories of images, generate some images
        selected_images = []
        for imageset in subdir_images:
            # if a list is exhausted, continue with other directories
            if len(imageset) == 0:
                continue
            # shuffle the data
            np.random.shuffle(imageset)
            # select some of the data, and remove it from the imageset
            size = batch_size//len(subdirectories) if batch_size//len(subdirectories) < len(imageset) else len(imageset)
            select_indices = np.random.choice(len(imageset), size=size, replace=False)
            selected_images.append([imageset.pop(idx) for idx in sorted(select_indices, reverse=True)])

        # preprocess the images
        input_X = []
        input_t = []
        for dir, subdir in enumerate(selected_images):
            for filename in subdir:
                img = cv2.imread(filename)
                norm_img = imgPreProcess(img=img)
                input_X.append(norm_img)
                input_t.append([dir])
        # images pre-processed, create a np array for the input matrix
        input_X = np.array(input_X)
        input_labels=np.array(input_t)
        # shuffle the outputs
        indices = np.random.choice(input_X.shape[0], input_X.shape[0], replace=False)
        # yield the batch, and the labels
        yield input_X[indices], input_labels[indices]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Logistic Regression model object test programs')
    parser.add_argument('--directory', '-d', type=str, help='Directory containing images')
    parser.add_argument('--batch_size', '-b', type=int, default=1000, help='Batch size for training')
    parser.add_argument('--epochs', '-e', type=int, default=10000, help='Num Epochs (max) for training')
    parser.add_argument('--regularizer', '-r', type=float, default=0, help='Regularizer for training')
    args = parser.parse_args()

    if not args.directory:
        args.directory = input("Enter the directory path containing images: ")

    for epoch in range(args.epochs):
        training_data = generate_minibatch(directory=args.directory, batch_size=args.batch_size)
        print(f'Epoch {epoch+1}...')
        for batch_idx, dataset in enumerate(training_data):
            X, t = dataset

            # # Visualize the batch of images and their labels
            # plt.figure(figsize=(10, 10))
            # for i in range(X.shape[0]):
            #     plt.subplot(4, 4, i+1)
            #     plt.imshow(X[i].reshape((30, 30)), cmap='gray')
            #     plt.title(f'Label: {t[i]}')
            #     plt.axis('off')
            # plt.tight_layout()
            # plt.show()
