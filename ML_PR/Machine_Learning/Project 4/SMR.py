import numpy as np
import warnings
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder
from scipy.special import softmax
from sklearn.datasets import load_iris, load_breast_cancer, load_digits
from sklearn.model_selection import KFold, train_test_split

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

    # def calcLoss(self, X:np.ndarray, Y:np.ndarray):
    #     """
    #     @params:
    #         X - Feature Matrix
    #         Y - True Labels One - Hot Encoded
    #     """
    #     if self.weights is None:
    #         warnings.warn('Attempting to calculate loss without a weight matrix... abandoning')
    #         return None
    #     Z = -X @ self.weights
    #     N = X.shape[0]
    #     loss = 1/N * (np.trace(X @ self.weights @ Y.T) + np.sum(np.log(np.sum(np.exp(Z), axis=1))))
    #     return loss

    def gradient(self, X:np.ndarray, Y:np.ndarray):
        """
        Y - One hot encoded
        """
        Z = -X @ self.weights
        P = self.softmax(Z)
        N = X.shape[0]
        grad = 1/N * (X.T @ (Y - P)) + 2* self.reg * self.weights
        return grad
    
    def fit(self, X:np.ndarray, Y:np.ndarray, max_iter=1000):

        Y_onehot = self.encoder.fit_transform(Y.reshape(-1,1))
        # initialize the weights
        if self.weights is None:
            self.weights = np.zeros((X.shape[1], Y_onehot.shape[1]))
            rate = self.rate

        grad_list = []
        weights_list = []
        loss_list= []
        steps = []
        for step in range(max_iter):
            steps.append(step)

            grad = self.gradient(X, Y_onehot)
            grad_list.append(grad)
            
            self.weights = self.weights - rate * grad
            weights_list.append(self.weights)
            
            loss = 0#self.calcLoss(X, Y_onehot)
            loss_list.append(loss)

            rate *= .995

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


def k_fold_cross_validation(X:np.ndarray, y:np.ndarray, k, shuffle=True):
    # Shuffle dataset once if required
    if shuffle:
        indices = np.random.choice(X.shape[0], size=X.shape[0], replace=False)
        shuffled_x = X[indices]
        shuffled_y = y[indices].reshape(-1,1)

    # join the dataset features and labels
    dataset = np.hstack((shuffled_x, shuffled_y))
    # Initialize KFold
    kf = KFold(n_splits=k)

    # Split dataset into train, validate, and test sets for each fold
    for train_idx, test_idx in kf.split(dataset):
        train_set = dataset[train_idx]
        test_set = dataset[test_idx]

        # Split the training set into train and validate sets
        train_set, validate_set = train_test_split(train_set, test_size=0.1, random_state=8)  # Adjust validation size as needed

        yield train_set, validate_set, test_set

def trainerplot(df: pd.DataFrame):
    epochs = df['epoch']
    loss = df['loss']

    # Plot loss vs epochs
    plt.figure(figsize=(8, 6))
    plt.subplot(2, 2, 1)
    plt.title('Loss vs Epochs')
    plt.plot(epochs, loss, label='Loss')

    # Get column names excluding 'epoch' and 'loss'
    columns_to_plot = [col for col in df.columns if col not in ['epoch', 'loss']]

    # Plot ||W|| and ||G|| vs epochs for each column
    for i, col in enumerate(columns_to_plot, start=2):
        plt.subplot(2, 2, i)
        plt.title(f'||{col}|| vs Epochs')
        values = np.linalg.norm(np.array(df[col].to_list()), axis=1)
        for j in range(values.shape[1]):
            plt.plot(epochs, values[:, j], label=f'||{col}||: C={j+1}')
        plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':

    x = load_digits().data
    y = load_digits().target

    smr = SoftMaxRegressor(rate=1)  # initialize the regressor with hyperparams

    # K-fold cross-validation
    k = 3
    folds = k_fold_cross_validation(x, y, k)

    test_acc = []
    for k, (train_set, val_set, test_set) in enumerate(folds):
        print(f'Fold {k+1}:')
        print('Train set:', train_set.shape[0])
        print('Validation set', val_set.shape[0])
        print('Test set:', test_set.shape[0])

        # separate into x, label pairs
        x_train = train_set[:, :-1]
        y_train = train_set[:, -1]
        
        x_val = val_set[:, :-1]
        y_val = val_set[:, -1]

        x_test = test_set[:, :-1]
        y_test = test_set[:, -1]

        # Flatten the labels
        y_train = y_train.flatten()
        y_val = y_val.flatten()
        y_test = y_test.flatten()

        training_eval = smr.fit(x_train, y_train)
        trainerplot(training_eval)

        pred = smr.predict(X=x_test)
        correct = np.sum(pred == y_test)
        test_acc_k = correct / y_test.shape[0]
        test_acc.append(test_acc_k)
        print(f'{correct} / {y_test.shape[0]} = {100* test_acc_k}% ')
        smr.confusion_matrix(y_true=y_test, y_pred=pred)
        # clear the learned weights
        smr.reset_training()

    #K-fold results :
    avg_test = np.mean(test_acc)
    print(f'Expected test accuracy: {avg_test}\n')

    # train a final model over all the data:
    smr.reset_training()
    training_eval = smr.fit(x, y)
    trainerplot(training_eval)
    
    # Evaluate training accuracy
    pred = smr.predict(X=x)
    correct = np.sum(pred == y)
    test_acc_k = correct / y.shape[0]
    print(f'{correct} / {y.shape[0]} = {100* test_acc_k}% Training Accuracy')
    smr.confusion_matrix(y_true=y, y_pred=pred)