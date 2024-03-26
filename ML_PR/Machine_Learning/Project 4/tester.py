from logistic_regressor import LogisticRegressor
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

bc = datasets.load_breast_cancer()
X,y = bc.data, bc.target

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1234)

print(X_train.shape, y_train.shape)

clf = LogisticRegressor(rate=1)

for _ in range(1000):
    clf.fit(X_train, y_train)

pred = clf.predict(X_test)

def accuracy(y_pred, y_test):
    return np.sum(y_pred == y_test) / len(y_pred)

acc = accuracy(pred, y_test)
print(acc)

clf.confusion_matrix(y_true=y_test, y_pred=pred)