import numpy as np
from keras.datasets.mnist import load_data
from sklearn.metrics import confusion_matrix

from SMR import SoftMaxRegressor
from ML_functions import *

def generate_batches(data, labels, batch_size):
    num_samples = len(data)
    num_batches = num_samples // batch_size
    for i in range(num_batches):
        start_index = i * batch_size
        end_index = (i + 1) * batch_size
        yield data[start_index:end_index], labels[start_index:end_index]

def main():
    smr = SoftMaxRegressor()
    (x_train, y_train), (x_test, y_test) = load_data()
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    batch_size = 64
    epochs = 10
    rate = 1
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        for batch_x, batch_y in generate_batches(x_train, y_train, batch_size):
            smr.fit(X=batch_x, Y=batch_y, rate=rate)

        rate *= 0.995

    test_predictions = smr.predict(x_test)

    cm = confusion_matrix(y_true=y_test, y_pred=test_predictions, labels=np.unique(y_test))
    print(cm)

if __name__ == "__main__":
    main()
