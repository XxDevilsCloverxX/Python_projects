import numpy as np

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