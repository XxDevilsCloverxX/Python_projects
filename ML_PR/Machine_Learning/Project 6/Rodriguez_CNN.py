import argparse
import tensorflow as tf
from keras.utils import image_dataset_from_directory, split_dataset
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="SMR Debug")
    parser.add_argument("-d","--directory", default=None, type=str, help="Parent directory to worm images.")
    parser.add_argument("-e","--epochs", default='10', type=int, help="Number of epochs for training.")
    args = parser.parse_args()
    
    # load the dataset
    train_dataset, test_dataset = image_dataset_from_directory(args.directory,
                                                               color_mode='grayscale',
                                                               image_size=(30, 30),
                                                               validation_split=0.3,
                                                               seed=69,
                                                               subset='both',
                                                               batch_size=None)
    
    test_dataset, validation_dataset = split_dataset(test_dataset, left_size=0.5, shuffle=True)
    
    # Optionally shuffle and batch the datasets
    BATCH_SIZE = 64
    train_batches = train_dataset.shuffle(train_dataset.cardinality().numpy()).batch(BATCH_SIZE)
    test_batches = test_dataset.batch(BATCH_SIZE)
    # Convert validation dataset to a single batch
    val_x, val_y = next(iter(validation_dataset.batch(len(validation_dataset))))

    # Design a CNN model for binary classification
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(30, 30, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer with 1 neuron and 'sigmoid' activation for binary classification
    ])

    # Compile the model with binary cross-entropy loss
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(train_batches, epochs=args.epochs, validation_data=(val_x, val_y))

    # Plot loss over each epoch during training
    plt.plot(history.history['loss'], label='training loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Evaluate the model on test dataset
    test_loss, test_accuracy = model.evaluate(test_batches)
    print(f'Testing Accuracy: {test_accuracy * 100:.2f}%')

    # Get true labels and predictions
    true_labels = []
    test_pred = []
    for batch in test_batches:
        x_batch, y_batch = batch
        true_labels.extend(y_batch)
        test_pred.extend(model.predict(x_batch).flatten() > 0.5)  # Threshold predictions at 0.5

    # Calculate percentage correct
    percentage_correct = accuracy_score(true_labels, test_pred) * 100
    print(f'Percentage Correct: {percentage_correct:.2f}%')

    # Generate confusion matrix
    cm = confusion_matrix(true_labels, test_pred)
    print("Confusion Matrix:")
    print(cm)

    # Plot confusion matrix with annotations
    plt.imshow(cm, cmap='Greens', interpolation='nearest')
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='black')

    plt.show()

    # Save the trained model in native Keras format
    model.save('worm_cnn_model.keras')

if __name__ == '__main__':
    main()
