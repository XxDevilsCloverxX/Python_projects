import numpy as np
import seaborn as sns
import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from keras.datasets.mnist import load_data

class SoftMaxRegressor:

    def __init__(self, alpha = 0, classes=2, init_weights:str=None) -> None:
        self.weights = np.load(init_weights, allow_pickle=True) if init_weights is not None else None
        self.reg = alpha
        self.num_classes = classes

    def softmax(self, X:np.ndarray):
        """
        Numerically stable softmax function
        """
        x_max = np.amax(X, axis=1).reshape(-1,1)
        exp_x_shifted = np.exp(X - x_max)
        return exp_x_shifted / np.sum(exp_x_shifted, axis=1).reshape(-1,1)
    
    def reset_training(self):
        """
        clear the weights saved to reinitialize training
        """
        self.weights = None

    def predict(self, X:np.ndarray):
        if self.weights is None:
            print('Cannot Predict without weights, please call fit method')
            return
        Z = -X @ self.weights
        P = self.softmax(Z)
        return np.array(np.argmax(P, axis=1)).ravel()   # idk why, but numpy acts goofy without this @ confusion matrix

    def one_hot_encode(self, labels):
        """
        Encode a 1-D array of labels to one-hot encoded Y
        """
        one_hot_matrix = np.eye(self.num_classes)[labels]
        return one_hot_matrix

    def fit(self, X:np.ndarray, y:np.ndarray, rate=.01):
        # encode y into a one-hot matrix
        Y_onehot = self.one_hot_encode(y)
        # initialize weights -> L x K
        if self.weights is None:
            self.weights = np.zeros((X.shape[1], Y_onehot.shape[1]))

        # compute the gradient of the cross-entropy loss
        grad = self.gradient(X, Y_onehot)
        # update the weights
        self.weights -= rate * grad
        # compute the new loss
        loss = self.cross_entropy_loss(y_true=Y_onehot, X=X)
        # return the computed gradient norms + loss
        return np.linalg.norm(grad, axis=0), loss
    
    def gradient(self, X:np.ndarray, Y:np.ndarray):
        """
        Y - One hot encoded
        """
        # compute the scores
        Z = -X @ self.weights
        P = self.softmax(Z)
        N = X.shape[0]
        grad = 1/N * (X.T @ (Y - P)) + 2* self.reg * self.weights
        return grad
    
    def cross_entropy_loss(self, y_true:np.ndarray, X:np.ndarray, eps:float=1e-15):
        # compute the scores
        Z = -X @ self.weights
        y_pred = self.softmax(Z)
        # Clip predicted probabilities to prevent log(0) which is undefined
        y_pred = np.clip(y_pred, eps, 1 - eps)
        # Compute the cross-entropy loss
        loss = -np.sum(y_true * np.log(y_pred)) / len(y_true)
        return loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SMR Debug")
    parser.add_argument("-w","--weights", default=None, type=str, help="Path to the saved weights.")
    parser.add_argument("-s","--save", default='saved_weights', type=str, help="Path to save weights.")
    args = parser.parse_args()

    # open the mnist dataset
    (x_train, y_train), (x_test, y_test) = load_data()

    # image pre-proc happens here

    # flatten the images
    x_train = x_train.reshape(x_train.shape[0],-1)
    x_test  = x_test.reshape(x_test.shape[0],-1)

    x_train = x_train / 255
    x_test = x_test / 255

    # initialize the SoftMaxClassifier + regression
    smr = SoftMaxRegressor(alpha=0, classes=len(np.unique(y_train)), init_weights=args.weights)

    # Create a tf.data.Dataset to generate minibatches
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    # Define a batch size
    batch_size = 100

    # epoch limiter
    epochs = 10

    # train if not using old weights
    if args.weights is None:
        # Iterate over mini-batches
        epoch_loss = []
        epoch_grad_norms = []
        for i in range(epochs):
            # telemetry
            print(f'Epoch {i+1}...')
            # shuffle the order of the data to be presented
            train_miniset = train_dataset.shuffle(buffer_size=len(x_train)).batch(batch_size)

            batch_loss = []
            for batch in train_miniset:
                X_batch, y_batch = batch
                X_batch = X_batch.numpy()
                y_batch = y_batch.numpy()
                
                grad_norms, loss = smr.fit(X=X_batch, y=y_batch)
                batch_loss.append(loss)
            epoch_loss.append(np.mean(batch_loss))  # gets average loss for epoch
            epoch_grad_norms.append(grad_norms)
            if np.all(grad_norms < 1e-1):
                print(f'Convergence found @ epoch {i+1}')
        
        epoch_grad_norms = np.array(epoch_grad_norms)
        # save the weights
        np.save(args.save,smr.weights)

        x = np.arange(len(epoch_loss))
        plt.figure(figsize=(8,6))
        plt.plot(x, epoch_loss)
        for class_k in range(epoch_grad_norms.shape[1]):
            plt.plot(x, epoch_grad_norms[:, class_k])
        plt.show()

        # show training results:
        train_preds = []
        train_miniset = train_dataset.batch(batch_size)
        for batch in train_miniset:
            X_batch, y_batch = batch
            X_batch = X_batch.numpy()
            train_preds.extend(smr.predict(X_batch))
                
        cm = confusion_matrix(y_train, train_preds, labels=np.unique(y_train))

        # Plot the confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", cbar=False)
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")
        plt.show()

        # computing the accuracy with the confusion matrix
        accuracy = np.sum(np.diag(cm)) / np.sum(cm)
        print(f'Train Accuracy: {accuracy}')


    # compute testing results:
    test_preds = []
    test_miniset = test_dataset.batch(batch_size)
    for batch in test_miniset:
        X_batch, y_batch = batch
        X_batch = X_batch.numpy()
        test_preds.extend(smr.predict(X_batch))
            
    cm = confusion_matrix(y_test, test_preds, labels=np.unique(y_test))
    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", cbar=False)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()

    # computing the accuracy with the confusion matrix
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    print(f'Test Accuracy: {accuracy}')