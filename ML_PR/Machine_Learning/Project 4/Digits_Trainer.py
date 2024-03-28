import argparse
from SMR import SoftMaxRegressor
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from keras.datasets.mnist import load_data
from ML_functions import * 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SMR Debug")
    parser.add_argument("-w","--weights", default=None, type=str, help="Path to the saved weights.")
    parser.add_argument("-s","--save", default='MNIST_weights', type=str, help="Path to save weights.")
    parser.add_argument("-o","--output", default='predictions_sheets_MNIST.xlsx', type=str, help="Path to save excel file.")
    args = parser.parse_args()

    # open the mnist dataset
    (x_train, y_train), (x_test, y_test) = load_data()

    # image pre-proc happens here

    # flatten the images
    x_train = x_train.reshape(x_train.shape[0],-1)
    x_test  = x_test.reshape(x_test.shape[0],-1)

    x_train = x_train / 255
    x_test = x_test / 255

    # initialize the SoftMaxClassifier + regression
    smr = SoftMaxRegressor(alpha=0, classes=len(np.unique(y_train)), init_weights=args.weights)

    # Create a tf.data.Dataset to generate minibatches
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    # Define a batch size
    batch_size = 100

    # epoch limiter
    epochs = 1000

    # train if not using old weights
    if args.weights is None:
        # Iterate over mini-batches
        epoch_loss = []
        epoch_grad_norms = []
        for i in range(epochs):
            # telemetry
            print(f'Epoch {i+1}...')
            # shuffle the order of the data to be presented
            train_miniset = train_dataset.shuffle(buffer_size=len(x_train)).batch(batch_size)

            batch_loss = []
            for batch in train_miniset:
                X_batch, y_batch = batch
                
                grad_norms, loss = smr.fit(X=X_batch, y=y_batch)
                batch_loss.append(loss)
            epoch_loss.append(np.mean(batch_loss))  # gets average loss for epoch
            epoch_grad_norms.append(grad_norms)
            if np.all(grad_norms < 1e-1):
                print(f'Convergence found @ epoch {i+1}')
                break
        
        epoch_grad_norms = np.array(epoch_grad_norms)
        # save the weights
        np.save(args.save,smr.weights)

        x = np.arange(len(epoch_loss))
        plt.figure(figsize=(8,6))
        plt.plot(x, epoch_loss)
        for class_k in range(epoch_grad_norms.shape[1]):
            plt.plot(x, epoch_grad_norms[:, class_k])
        plt.show()

        # show training results:
        train_preds = []
        train_miniset = train_dataset.batch(batch_size)
        for batch in train_miniset:
            X_batch, y_batch = batch
            train_preds.extend(smr.predict(X_batch))
                
        cm = confusion_matrix(y_train, train_preds, labels=np.unique(y_train))

        # Plot the confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", cbar=False)
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")
        plt.show()

        # computing the accuracy with the confusion matrix
        accuracy = np.sum(np.diag(cm)) / np.sum(cm)
        print(f'Train Accuracy: {accuracy}')


    # compute testing results:
    test_preds = []
    test_miniset = test_dataset.batch(batch_size)
    for batch in test_miniset:
        X_batch, y_batch = batch
        X_batch = X_batch.numpy()
        test_preds.extend(smr.predict(X_batch))
    test_preds = np.array(test_preds)

    cm = confusion_matrix(y_test, test_preds, labels=np.unique(y_test))
    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", cbar=False)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()

    # computing the accuracy with the confusion matrix
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    print(f'Test Accuracy: {accuracy}')

    # write predictions to an excel file
    if not args.output.endswith('.xlsx'):
        args.output += '.xlsx'

    path = write_predictions_to_excel(y_true=y_test,predictions=test_preds, output_file=args.output)
    print(f'Excel file with 2 sheets saved to: {path}')