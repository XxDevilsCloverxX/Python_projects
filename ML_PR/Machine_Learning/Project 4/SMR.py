import numpy as np
import pandas as pd
import cv2 

from ML_functions import *
from sklearn.preprocessing import OneHotEncoder
from keras.datasets.mnist import load_data

class SoftMaxRegressor:

    def __init__(self, rate:float=1, alpha:float=0) -> None:
        # params of the LR, weights will have bias included
        self.weights = None
        self.rate = rate
        self.reg = alpha
        self.encoder = OneHotEncoder()

    def softmax(self, x:np.ndarray):
        x_max = np.amax(x, axis=1).reshape(-1,1)
        exp_x_shifted = np.exp(x - x_max)
        return exp_x_shifted / np.sum(exp_x_shifted, axis=1).reshape(-1,1)

    def reset_training(self):
        """
        clear the weights saved to reinitialize training
        """
        self.weights = None

    def predict(self, X:np.ndarray):
        Z = -X @ self.weights
        P = self.softmax(Z)
        return np.array(np.argmax(P, axis=1)).ravel()   # idk why, but numpy acts goofy without this @ confusion matrix

    def calcLoss(self, X:np.ndarray, Y:np.ndarray):
        if self.weights is None:
            print('Attempting to calculate loss without a weight matrix... abandoning')
            return None
    
        Z = -X @ self.weights
        exp_Z = np.exp(Z - np.max(Z, axis=1).reshape(-1, 1))  # Apply numerical stability trick
        N = X.shape[0]
        loss = 1/N * (np.trace(X @ self.weights @ Y.T) + np.sum(np.log(np.sum(exp_Z, axis=1))))
        return loss

    def gradient(self, X:np.ndarray, Y:np.ndarray):
        """
        Y - One hot encoded
        """
        Z = -X @ self.weights
        P = self.softmax(Z)
        N = X.shape[0]
        grad = 1/N * (X.T @ (Y - P)) + 2* self.reg * self.weights
        return grad
    
    def fit(self, X:np.ndarray, Y:np.ndarray, v_x:np.ndarray=None, v_y:np.ndarray=None, max_iter=1000) -> pd.DataFrame:

        Y_onehot = self.encoder.fit_transform(Y.reshape(-1,1))
        # initialize the weights
        if self.weights is None:
            self.weights = np.zeros((X.shape[1], Y_onehot.shape[1]))
            rate = self.rate

        grad_list = []
        weights_list = []
        train_loss= []
        val_loss =[]
        steps = []
        for step in range(max_iter):
            steps.append(step)

            grad = self.gradient(X, Y_onehot)
            grad_list.append(grad)
            
            self.weights = self.weights - rate * grad
            weights_list.append(self.weights)
            
            
            loss = self.calcLoss(X, Y_onehot)  # Compute the loss
            train_loss.append(loss)

            # compute validation loss
            if v_x is not None and v_y is not None:
                v_loss = self.calcLoss(v_x, self.encoder.transform(v_y.reshape(-1,1)).toarray())
                val_loss.append(v_loss)

            rate *= .995

            # compute the norm of the gradients to see if all of them have reached some minimum
            grad_norms = np.linalg.norm(grad, axis=0)
            converge = grad_norms < 1e-1
            if np.all(converge):
                print(f'Early convergence @ step {step}')
                break

        if v_x is None and v_y is None:
            val_loss = np.ones_like(train_loss)

        df = pd.DataFrame({
                'epoch': steps,
                'train_loss': train_loss,
                'validation_loss': val_loss,
                'weights': weights_list,
                'gradients': grad_list
            })
        
        return df

if __name__ == '__main__':

    (x_train, y_train), (x_test, y_test) = load_data()

    size= 10000
    test_s = 1000
    x_train = x_train.reshape(x_train.shape[0], -1)[:size, :]
    x_test = x_test.reshape(x_test.shape[0], -1)[:test_s, :]

    # Normalize the pixel values
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    smr = SoftMaxRegressor(rate=1)  # initialize the regressor with hyperparams

    test_acc = []

    print('Train set:', x_train.shape)
    print('Test set:', x_test.shape)

    training_eval = smr.fit(x_train, y_train[:size])
    trainerplot(training_eval)

    pred = smr.predict(X=x_test)
    correct = np.sum(pred == y_test[:test_s])
    test_acc_k = correct / y_test[:test_s].shape[0]
    test_acc.append(test_acc_k)
    print(f'{correct} / {y_test[:test_s].shape[0]} = {100* test_acc_k}% ')
    # smr.confusion_matrix(y_true=y_test, y_pred=pred)
    # clear the learned weights
    # smr.reset_training()

    #K-fold results :
    avg_test = np.mean(test_acc)
    print(f'Expected test accuracy: {avg_test}\n')

    # # train a final model over all the data:
    # smr.reset_training()
    # training_eval = smr.fit(x, y)
    # trainerplot(training_eval)
    
    # Evaluate training accuracy
    print(x_test.shape, smr.weights.shape)
    pred = smr.predict(X=x_test)
    correct = np.sum(pred == y_test[:test_s])
    test_acc_k = correct / y_test[:test_s].shape[0]
    print(f'{correct} / {y_test[:test_s].shape[0]} = {100* test_acc_k}% Training Accuracy')
    # smr.confusion_matrix(y_true=y, y_pred=pred)

    # Write predictions to an Excel file
    output_file = 'predictions.xlsx'
    output_file = write_predictions_to_excel(pred, y_test[:test_s], output_file)
    print(f'Predictions written to {output_file}')