import numpy as np
import warnings
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder
from scipy.special import softmax
from sklearn.datasets import load_iris

class SoftMaxRegressor:

    def __init__(self, rate:float=1, alpha:float=0) -> None:
        # params of the LR, weights will have bias included
        self.weights = None
        self.rate = rate
        self.reg = alpha
        self.encoder = OneHotEncoder()

    def softmax(self, x:np.ndarray):
        pass

    def clear_weights(self):
        """
        clear the weights saved to reinitialize training
        """
        self.weights = None

    def predict(self, X:np.ndarray):
        Z = -X @ self.weights
        P = softmax(Z, axis=1)
        return np.argmax(P, axis=1)

    def calcLoss(self, X:np.ndarray, Y:np.ndarray):
        """
        @params:
            X - Feature Matrix
            Y - True Labels One - Hot Encoded
        """
        if self.weights is None:
            warnings.warn('Attempting to calculate loss without a weight matrix... abandoning')
            return None
        Z = -X @ self.weights
        N = X.shape[0]
        loss = 1/N * (np.trace(X @ self.weights @ Y.T) + np.sum(np.log(np.sum(np.exp(Z), axis=1))))
        return loss

    def gradient(self, X:np.ndarray, Y:np.ndarray):
        """
        Y - One hot encoded
        """
        Z = -X @ self.weights
        P = softmax(Z, axis=1)
        N = X.shape[0]
        grad = 1/N * (X.T @ (Y - P)) + 2* self.reg * self.weights
        return grad
    
    def fit(self, X:np.ndarray, Y:np.ndarray, max_iter=1000):

        Y_onehot = self.encoder.fit_transform(Y.reshape(-1,1))
        
        # initialize the weights
        if self.weights is None:
            self.weights = np.zeros((X.shape[1], Y_onehot.shape[1]))
        
        grad_list = []
        weights_list = []
        loss_list= []
        steps = []
        for step in range(max_iter):
            steps.append(step)

            grad = self.gradient(X, Y_onehot)
            grad_list.append(grad)
            
            self.weights = self.weights - self.rate * grad
            weights_list.append(self.weights)
            
            loss = self.calcLoss(X, Y_onehot)
            loss_list.append(loss)

            self.rate *= .995

            # compute the norm of the gradients to see if all of them have reached some minimum
            grad_norms = np.linalg.norm(grad, axis=0)
            converge = grad_norms < 1e-1
            if np.all(converge):
                print(f'Early convergence @ step {step}')
                break

        df = pd.DataFrame({
                'epoch': steps,
                'loss': loss_list,
                'weights': weights_list,
                'gradients': grad_list
            })
        return df
    
    @staticmethod
    def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray):
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Greens')
        plt.title('Confusion Matrix of Classified Test Data')
        plt.show()  # Explicitly show the plot


if __name__ == '__main__':

    x = load_iris().data
    y = load_iris().target

    smr = SoftMaxRegressor(rate=0.1)
    training_eval = smr.fit(x, y)

    epochs = training_eval['epoch']
    loss = training_eval['loss']
    weights = np.linalg.norm(np.array(training_eval['weights'].to_list()), axis=1)
    grads = np.linalg.norm(np.array(training_eval['gradients'].to_list()), axis=1)

    plt.figure(figsize=(8,6))
    plt.subplot(3,1,1)
    plt.title('Loss vs Epochs')
    plt.plot(epochs, loss, label='Loss')
    plt.subplot(3,1,2)
    plt.title('||W|| vs Epochs')
    plt.plot(epochs, weights[:, 0], label='||W||: C=1')
    plt.plot(epochs, weights[:, 1], label='||W||: C=2')
    plt.plot(epochs, weights[:, 2], label='||W||: C=3')
    plt.subplot(3,1,3)
    plt.title('||G|| vs epochs')
    plt.plot(epochs, grads[:, 0], label='||G||: C=1')
    plt.plot(epochs, grads[:, 1], label='||G||: C=2')
    plt.plot(epochs, grads[:, 2], label='||G||: C=3')
    plt.legend()
    plt.show()

    pred = smr.predict(X=x)
    correct = np.sum(pred == y)
    print(f'{correct} / {y.shape[0]} = {100* correct / y.shape[0]}% ')

    smr.confusion_matrix(y, pred)