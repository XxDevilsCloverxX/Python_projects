import tensorflow as tf
from keras.datasets.mnist import load_data
from SMR import SoftMaxRegressor

def main():
    # load the dataset
    (x_train, y_train), (x_test, y_test) = load_data()
    # Convert NumPy arrays to TensorFlow Dataset objects
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    # flatten the images and normalize them
    train_dataset = train_dataset.map(lambda x, y: (tf.cast(tf.reshape(x, [-1]), tf.float32) / 255., y))
    test_dataset = test_dataset.map(lambda x, y: (tf.cast(tf.reshape(x, [-1]), tf.float32) / 255., y))

    # Optionally shuffle and batch the datasets
    BATCH_SIZE = 64
    train_batches = train_dataset.shuffle(len(x_train)).batch(BATCH_SIZE)
    test_batches = test_dataset.batch(BATCH_SIZE)

    # initialize the trainer
    smr = SoftMaxRegressor(alpha=0, classes=len(tf.unique(y_train)), init_weights=None)

    # Example usage: iterate through batches of training data for set epochs
    gradient_norms = []
    epoch_loss = []
    for i in range(10):
        batch_loss = []
        for batch in train_batches:
            x_batch, y_batch = batch
            grad_norms, loss = smr.fit(X=x_batch, y=y_batch)
            batch_loss.append(loss)
        epoch_loss.append(tf.reduce_mean(batch_loss))


if __name__ == '__main__':
    main()