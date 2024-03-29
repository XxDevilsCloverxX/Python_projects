import argparse
import tensorflow as tf
from keras.datasets.mnist import load_data
from sklearn.metrics import confusion_matrix
from SMR import SoftMaxRegressor
from numpy import save
from time import time
from ML_functions import *

def main():
    parser = argparse.ArgumentParser(description="SMR Debug")
    parser.add_argument("-w","--weights", default=None, type=str, help="Path to the saved weights.")
    parser.add_argument("-s","--save", default='mnist_weights', type=str, help="Path to save weights.")
    parser.add_argument("-e","--epochs", default='20', type=int, help="Numnber of epochs for training.")
    args = parser.parse_args()
    
    # load the dataset
    (x_train, y_train), (x_test, y_test) = load_data()

    # Convert NumPy arrays to TensorFlow Dataset objects
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    # split test into test + validation data
    val_dataset = test_dataset.shard(2, 0)
    test_dataset = test_dataset.shard(2, 1)

    # flatten the images and normalize them
    train_dataset = train_dataset.map(tf_cv2_func)
    test_dataset = test_dataset.map(tf_cv2_func)
    val_dataset = val_dataset.map(tf_cv2_func)

    # Optionally shuffle and batch the datasets
    BATCH_SIZE = 64
    train_batches = train_dataset.shuffle(train_dataset.cardinality().numpy()).batch(BATCH_SIZE)
    test_batches = test_dataset.batch(BATCH_SIZE)
    # Convert validation dataset to a single batch
    val_x, val_y = next(iter(val_dataset.batch(len(val_dataset))))    

    # Get unique labels and count them
    unique_labels, _ = tf.unique(y_train)
    num_unique_labels = tf.size(unique_labels)

    train_start = time()
    # train on the data
    if args.weights is None:
        # initialize the trainer
        smr = SoftMaxRegressor(alpha=0, classes=num_unique_labels, momentum=0.9, init_weights=args.weights)

        # Example usage: iterate through batches of training data for set epochs
        gradient_norms = []
        epoch_loss = []
        epoch_val_loss =[]
        epoch_rate = 1
        for i in range(args.epochs):
            print(f'Epoch {i+1}')
            batch_loss = []
            batch_val_loss = []
            for batch in train_batches:
                x_batch, y_batch = batch
                grad_norms, loss, val_loss = smr.fit(X=x_batch, y=y_batch,
                                                      val_x=val_x, val_y=val_y, rate=epoch_rate)
                batch_val_loss.append(val_loss)
                batch_loss.append(loss)

            epoch_val_loss.append(tf.reduce_mean(batch_val_loss))
            epoch_loss.append(tf.reduce_mean(batch_loss))
            gradient_norms.append(grad_norms)
            epoch_rate *=0.8

            #if our validation loss is no longer decreasing, exit
            if i > 10 and (tf.math.reduce_std(epoch_val_loss[-10:]) < 0.1 or tf.argmin(epoch_val_loss[-10:]) < 4):
                print(f'Converged on epoch {i+1}')
                break

        train_time = time() - train_start
        print(f'Train time: {train_time:0.3} sec.')
        # convert training metrics into tensors
        grad_norms = tf.convert_to_tensor(gradient_norms)
        epoch_loss = tf.convert_to_tensor(epoch_loss)
        epoch_val_loss = tf.convert_to_tensor(epoch_val_loss)

        # plot metrics
        eval_train(epoch_loss=epoch_loss, gradient_norms=grad_norms,
                    epoch_val_loss=epoch_val_loss)
        
        # save the weight matrix
        args.save = args.save+'.npy' if not args.save.endswith('.npy') else args.save
        save(args.save, smr.weights)
        # predict training data
        train_pred = []
        true_labels = []
        for batch in train_batches:
            x_batch, y_batch = batch
            true_labels.extend(y_batch)
            train_pred.extend(smr.predict(X=x_batch))
        train_acc = conf_matrix_eval(true_labels, train_pred)
        print(f'Training Accuracy: {train_acc}')
    
    else:
        # load the classifier
        smr = SoftMaxRegressor(init_weights=args.weights)

    # perform test error calculation
    test_pred = []
    true_labels = []
    test_time_start = time()
    for batch in test_batches:
        x_batch, y_batch = batch
        true_labels.extend(y_batch)
        test_pred.extend(smr.predict(X=x_batch))
    test_time= time() - test_time_start
    acc = conf_matrix_eval(true_labels, test_pred)

    cm = confusion_matrix(true_labels, test_pred)
    show_confusion_matrix(cm)
    print(f'Testing accuracy {acc:.3f}')

    with open('MnistLogger.txt', 'w') as logger:
        data = []
        if args.weights is None:
            data.append(f'Train_time: {train_time:.3f}\n')
            data.append(f'Train accuracy: {train_acc:.3f}\n')
        data.append(f'Test time: {test_time:.3f}\n')
        data.append(f'Testing Accuracy: {acc}\n')
        logger.writelines(data)

if __name__ == '__main__':
    main()