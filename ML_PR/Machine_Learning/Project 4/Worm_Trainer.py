import argparse
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from SMR import SoftMaxRegressor
from keras.utils import image_dataset_from_directory, split_dataset
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

def process_image(x, y):
    # Apply edge detection using a Sobel filter
    edge_detector = tf.image.sobel_edges(tf.expand_dims(x, axis=0))
    img = tf.image.adjust_gamma(edge_detector, 4/9)
    img = tf.norm(edge_detector, axis=-1)
    img = tf.cast(tf.reshape(img, [-1]), tf.float32) / 255.
    
    return img, y

def main():
    parser = argparse.ArgumentParser(description="SMR Debug")
    parser.add_argument("-d","--directory", default=None, type=str, help="Parent directory to worm images.")
    parser.add_argument("-w","--weights", default=None, type=str, help="Path to the saved weights.")
    parser.add_argument("-s","--save", default='worm_weights', type=str, help="Path to save weights.")
    parser.add_argument("-e","--epochs", default='100', type=int, help="Numnber of epochs for training.")
    args = parser.parse_args()
    
    # load the dataset
    train_dataset, test_dataset = image_dataset_from_directory(args.directory,
                                                               color_mode='grayscale',
                                                               image_size=(28,28),
                                                               validation_split=0.3,
                                                               seed=69,
                                                               subset='both',
                                                               batch_size=None)
    
    test_dataset, validation_dataset = split_dataset(test_dataset, left_size=0.5)
    # flatten the images and normalize them
    train_dataset = train_dataset.map(process_image)
    test_dataset = test_dataset.map(process_image)
    validation_dataset = validation_dataset.map(process_image)

    # Optionally shuffle and batch the datasets
    BATCH_SIZE = 64
    train_batches = train_dataset.shuffle(train_dataset.cardinality().numpy()).batch(BATCH_SIZE)
    test_batches = test_dataset.batch(BATCH_SIZE)
    # Convert validation dataset to a single batch
    val_x, val_y = next(iter(validation_dataset.batch(len(validation_dataset))))    

    # Get unique labels and count them
    unique_labels, _ = tf.unique(val_y)
    num_unique_labels = tf.size(unique_labels)
    # initialize the trainer
    smr = SoftMaxRegressor(alpha=1/4, classes=num_unique_labels, momentum=0.9, init_weights=args.weights)

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
            epoch_rate *=0.8

            #if our validation loss is no longer decreasing, exit
            if i > 10 and (tf.math.reduce_std(epoch_val_loss[-10:]) < 0.25 or tf.argmin(epoch_val_loss[-10:]) < 4):
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

    cm = confusion_matrix(true_labels, test_pred)

if __name__ == '__main__':
    main()