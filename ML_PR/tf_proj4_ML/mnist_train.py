import argparse
import tensorflow as tf
from keras.datasets.mnist import load_data
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from SMR import SoftMaxRegressor
from numpy import save
from time import time

def eval_train(epoch_loss, gradient_norms, epoch_val_loss):
    plt.figure(figsize=(8, 6))

    # Plot gradient norms
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(gradient_norms) + 1), gradient_norms, marker='o')
    plt.title('Gradient Norms')
    plt.xlabel('Epochs')
    plt.ylabel('Norm')
    plt.grid(True)

    # Plot epoch loss
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(epoch_loss) + 1), epoch_loss, marker='o', color='orange', label='Train Loss')
    plt.plot(range(1, len(epoch_val_loss) + 1), epoch_val_loss, marker='o', color='blue', label='Validation loss')
    plt.title('Epoch Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

def conf_matrix_eval(y_true, y_pred):
    # make the confusion matrix for train data & compute accuracy
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    correct = tf.reduce_sum(tf.linalg.diag_part(cm))
    accuracy = correct / tf.reduce_sum(cm)
    return accuracy

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
    train_dataset = train_dataset.map(lambda x, y: (tf.cast(tf.reshape(x, [-1]), tf.float32) / 255., y))
    test_dataset = test_dataset.map(lambda x, y: (tf.cast(tf.reshape(x, [-1]), tf.float32) / 255., y))
    val_dataset = val_dataset.map(lambda x, y: (tf.cast(tf.reshape(x, [-1]), tf.float32) / 255., y))

    # Optionally shuffle and batch the datasets
    BATCH_SIZE = 128
    train_batches = train_dataset.shuffle(len(x_train)).batch(BATCH_SIZE)
    test_batches = test_dataset.batch(BATCH_SIZE)
    # Convert validation dataset to a single batch
    val_x, val_y = next(iter(val_dataset.batch(len(val_dataset))))    

    # Get unique labels and count them
    unique_labels, _ = tf.unique(y_train)
    num_unique_labels = tf.size(unique_labels)
    # initialize the trainer
    smr = SoftMaxRegressor(alpha=0, classes=num_unique_labels, init_weights=args.weights)

    train_start = time()
    # train on the data
    if args.weights is None:
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
            epoch_rate *=0.995

            #if our validation loss is no longer decreasing, exit
            if i > 10 and (tf.math.reduce_std(epoch_val_loss[-8:]) < 0.25 or tf.argmin(epoch_val_loss[-8:] < 4)):
                print(f'Converged on epoch {i}')
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
        acc = conf_matrix_eval(true_labels, train_pred)
        print(f'Training Accuracy: {acc}')
    
    # perform test error calculation
    test_pred = []
    true_labels = []
    for batch in test_batches:
        x_batch, y_batch = batch
        true_labels.extend(y_batch)
        test_pred.extend(smr.predict(X=x_batch))
    acc = conf_matrix_eval(true_labels, test_pred)
    print(f'Testing Accuracy: {acc}')

if __name__ == '__main__':
    main()