import numpy as np

class LogisticRegressor:

    def __init__(self, rate:float=1, alpha:float=0) -> None:
        # params of the LR, weights will have bias included
        self.weights = None
        self.bias = None
        self.rate = rate
        self.reg = alpha
        self.train_cost = []
        self.valid_cost = []
    
    def _positive_sigmoid(self, x):
        """
        Numerical stability sigmoid
        """
        return 1 / (1 + np.exp(-x))

    def _negative_sigmoid(self, x):
        """
        Numerical stability sigmoid
        """
        # Cache exp so you won't have to calculate it twice
        exp = np.exp(x)
        return exp / (exp + 1)

    def sigmoid(self, x:np.ndarray):
        """
        return scores given an input vector, X
        """
        positive = x >= 0
        # Boolean array inversion is faster than another comparison
        negative = ~positive

        # empty contains juke hence will be faster to allocate than zeros
        result = np.empty_like(x)
        result[positive] = self._positive_sigmoid(x[positive].astype(np.float64))
        result[negative] = self._negative_sigmoid(x[negative].astype(np.float64))

        return result

    def clear_weights(self):
        """
        clear the weights saved to reinitialize training
        """
        self.weights = None
        self.bias = None

    def clear_costs(self):
        self.train_cost.clear()
        self.valid_cost.clear()

    def validation_evaluation(self,V:np.ndarray, y_v:np.ndarray):
        # compute the cost over the validation set
        pass

    def fit(self, X:np.ndarray, y:np.ndarray):

        n_samples, d_features = X.shape
        
        # initial weights
        if self.weights is None:
            self.weights = np.zeros(d_features)
        # initial bias
        if self.bias is None:
            self.bias = 0

        # linear predictions
        pred = X @ self.weights + self.bias
        # map to sigmoid
        scores = self.sigmoid(pred)

        # compute the gradient
        dw = (1/n_samples) * (X.T @ (scores - y))
        db = (1/n_samples) * np.sum(scores - y)

        # update the weights
        self.weights = self.weights - self.rate * dw
        self.bias = self.bias - self.rate * db
        self.rate *= .995

    def predict(self, X:np.ndarray):
        # linear predictions
        pred = X @ self.weights + self.bias
        scores = self.sigmoid(pred)

        labels = [0 if score < 0.5 else 1 for score in scores]
        return labels