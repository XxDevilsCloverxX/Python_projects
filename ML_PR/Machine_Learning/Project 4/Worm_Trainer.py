import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from keras.utils import image_dataset_from_directory

from SMR import SoftMaxRegressor
from ML_functions import * 

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="SMR Debug")
    parser.add_argument("-w","--weights", default=None, type=str, help="Path to the saved weights.")
    parser.add_argument("-s","--save", default='worm_weights', type=str, help="Path to save weights.")
    parser.add_argument("-o","--output", default='predictions_sheets_WORM.xlsx', type=str, help="Path to save excel file.")
    parser.add_argument('-d', '--directory', required=True, type=str, help='Parent Directory to the images')

    args = parser.parse_args()

    # Open the worms dataset with train, validation split
    dataset_train, dataset_test = image_dataset_from_directory(args.directory,
                                                 image_size=(28, 28),
                                                 color_mode='grayscale',
                                                 validation_split=0.20,
                                                 seed=69,
                                                 subset="both")

    # Initialize the SoftMaxRegressorTF
    smr = SoftMaxRegressor(alpha=0, classes=len(dataset_train.class_names), init_weights=args.weights)

    # epoch limiter
    epochs = 10

    # Train if not using old weights
    if args.weights is None:
        # Iterate over mini-batches
        epoch_loss = []
        epoch_grad_norms = []
        for i in range(epochs):
            # telemetry
            print(f'Epoch {i+1}...')

            batch_loss = []
            for batch in dataset_train:
                X_batch, y_batch = batch
                
                X_batch = X_batch.numpy().reshape(X_batch.shape[0], -1)   # reshape and normalize
                y_batch = y_batch.numpy()
                
                grad_norms, loss = smr.fit(X=X_batch, y=y_batch)
                batch_loss.append(loss)

            epoch_loss.append(np.mean(batch_loss))  # gets average loss for epoch
            epoch_grad_norms.append(grad_norms)
            if np.all(grad_norms < 1e-1):
                print(f'Convergence found @ epoch {i+1}')
                break
        
        epoch_grad_norms = np.array(epoch_grad_norms)
        # Save the weights
        checkpoint = tf.train.Checkpoint(weights=smr.weights)
        checkpoint.save(args.save)

        x = np.arange(len(epoch_loss))
        plt.figure(figsize=(8, 6))
        plt.plot(x, epoch_loss)
        for class_k in range(epoch_grad_norms.shape[1]):
            plt.plot(x, epoch_grad_norms[:, class_k])
        plt.show()

        # Show training results:
        train_preds = []
        y_train = []
        for batch in dataset_train:
            X_batch, y_batch = batch
            X_batch = X_batch.numpy().reshape(X_batch.shape[0], -1) / 255
            train_preds.extend(smr.predict(X_batch))
            y_train.extend(y_batch)
                
        cm_train = confusion_matrix(y_train, train_preds, labels=np.unique(y_train))

        # Plot the confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_train, annot=True, fmt="d", cmap="Greens", cbar=False)
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Training Confusion Matrix")
        plt.show()

        # computing the accuracy with the confusion matrix
        accuracy = np.sum(np.diag(cm_train)) / np.sum(cm_train)
        print(f'Train Accuracy: {accuracy}')

    # Compute testing results:
    test_preds = []
    true_labels = []
    for batch in dataset_test:
        X_batch, y_batch = batch
        X_batch = X_batch.numpy().reshape(X_batch.shape[0], -1) / 255
        test_preds.extend(smr.predict(X_batch))
        true_labels.extend(y_batch)
    test_preds = np.array(test_preds)
    true_labels = np.array(true_labels)

    cm_test = confusion_matrix(true_labels, test_preds, labels=np.unique(true_labels))

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_test, annot=True, fmt="d", cmap="Greens", cbar=False)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Testing Confusion Matrix")
    plt.show()

    # computing the accuracy with the confusion matrix
    accuracy = np.sum(np.diag(cm_test)) / np.sum(cm_test)
    print(f'Test Accuracy: {accuracy}')

    # Write predictions to an excel file
    if not args.output.endswith('.xlsx'):
        args.output += '.xlsx'

    path = write_predictions_to_excel(y_true=true_labels, dataset_test=dataset_test, predictions=test_preds, output_file=args.output)
    print(f'Excel file with 2 sheets saved to: {path}')
