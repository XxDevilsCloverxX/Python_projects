import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load the iris dataset
iris = load_iris()

# Store the feature matrix (X) and response vector (y)
X = iris.data
y = iris.target

# Splitting X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

class NaiveBayes:
    def fit(self, X_train, y_train):
        self.classes = np.unique(y_train)
        self.parameters = []
        for i, c in enumerate(self.classes):
            X_c = X_train[y_train == c]
            self.parameters.append([])
            for feature in X_c.T:
                mean, std = np.mean(feature), np.std(feature)
                self.parameters[i].append({'mean': mean, 'std': std})
        self.y_train = y_train
                
    def _calculate_likelihood(self, mean, std, x):
        exponent = np.exp(-((x - mean) ** 2) / (2 * std ** 2))
        return (1 / (np.sqrt(2 * np.pi) * std)) * exponent
        
    def _calculate_prior(self, c):
        return np.mean(self.y_train == c)
        
    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            posteriors = []
            for i, c in enumerate(self.classes):
                prior = self._calculate_prior(c)
                likelihood = 1
                for feature, params in zip(x, self.parameters[i]):
                    likelihood *= self._calculate_likelihood(params['mean'], params['std'], feature)
                posterior = prior * likelihood
                posteriors.append(posterior)
            y_pred.append(self.classes[np.argmax(posteriors)])
        return np.array(y_pred)

# Train and test NumPy-based Naive Bayes model
model_numpy = NaiveBayes()
model_numpy.fit(X_train, y_train)
y_pred_numpy = model_numpy.predict(X_test)

# Check accuracy
accuracy_numpy = metrics.accuracy_score(y_test, y_pred_numpy)
print("NumPy-based Naive Bayes model accuracy (in %):", accuracy_numpy * 100)
