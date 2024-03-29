import numpy as np
import tensorflow as tf

class SoftMaxRegressor:

    def __init__(self, alpha=0, classes=2, init_weights=None):
        if init_weights is not None:
            weights = np.load(init_weights, allow_pickle=True)
            self.weights = tf.convert_to_tensor(weights)
        else:
            self.weights = None

        self.reg = alpha
        self.num_classes = classes

    def softmax(self, X):
        """
        Numerically stable softmax function
        """
        x_max = tf.reduce_max(X, axis=1, keepdims=True)
        exp_x_shifted = tf.exp(X - x_max)
        return exp_x_shifted / tf.reduce_sum(exp_x_shifted, axis=1, keepdims=True)
    
    def reset_training(self):
        """
        clear the weights saved to reinitialize training
        """
        self.weights = None

    def predict(self, X):
        if self.weights is None:
            print('Cannot Predict without weights, please call fit method')
            return
        Z = -tf.matmul(X, self.weights)
        P = self.softmax(Z)
        return np.argmax(P, axis=1)

    def one_hot_encode(self, labels):
        """
        Encode a 1-D array of labels to one-hot encoded Y
        """
        one_hot_matrix = tf.one_hot(labels, depth=self.num_classes)
        return one_hot_matrix

    def fit(self, X, y, val_x=None, val_y=None, rate=0.01):
        # encode y into a one-hot matrix
        Y_onehot = self.one_hot_encode(y)
        
        # flag to perform validation error
        validate = False if val_x is None or val_y is None else True
        
        # initialize weights -> L x K
        if self.weights is None:
            self.weights = tf.Variable(tf.zeros((X.shape[1], Y_onehot.shape[1]), dtype=tf.float32))

        # compute the gradient of the cross-entropy loss
        grad = self.gradient(X, Y_onehot)
        
        # update the weights
        self.weights.assign_sub(rate * grad)
        
        # compute the loss
        loss = self.cross_entropy_loss(y_true=Y_onehot, X=X)
        
        if validate:
            v_y_onehot = self.one_hot_encode(val_y)
            val_loss = self.cross_entropy_loss(y_true=v_y_onehot, X=val_x)
        else:
            val_loss = 0
        # return the computed gradient norms + loss
        return tf.norm(grad, axis=0), loss, val_loss
    
    def gradient(self, X, Y):
        """
        Y - One hot encoded
        """
        # compute the scores
        Z = -tf.matmul(X, self.weights)
        P = self.softmax(Z)
        N = tf.cast(tf.shape(X)[0], dtype=tf.float32)
        grad = 1/N * tf.matmul(tf.transpose(X), (Y - P)) + 2 * self.reg * self.weights
        return grad
    
    def cross_entropy_loss(self, y_true, X, eps=1e-15):
        # compute the scores
        Z = -tf.matmul(X, self.weights)
        y_pred = self.softmax(Z)
        # Clip predicted probabilities to prevent log(0) which is undefined
        y_pred = tf.clip_by_value(y_pred, eps, 1 - eps)
        # Compute the cross-entropy loss
        loss = -tf.reduce_sum(y_true * tf.math.log(y_pred)) / tf.cast(tf.shape(y_true)[0], dtype=tf.float32)
        return loss