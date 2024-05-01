import argparse
import tensorflow as tf
from keras.utils import image_dataset_from_directory, split_dataset
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import cv2
import numpy as np

def preprocess_bin(image, label):
    img = np.array(image)
    # Convert image to uint8
    img = img.astype(np.uint8)

    # Apply Otsu's thresholding to the grayscale image
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Convert to TensorFlow tensor
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    # normalize
    img = tf.truediv(img, 255)

    # reshape the image to (30, 30, 1)
    img = tf.reshape(img, [30, 30, 1])

    return img, label

def preprocess_edge(image, label):
    img = image.numpy()
    # Convert image to uint8
    img = img.astype(np.uint8)

    # Apply Canny edge detection
    img = cv2.Canny(img, 100, 200)

    # Convert to TensorFlow tensor
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    # normalize
    img = tf.truediv(img, 255)

    # reshape the image to (30, 30, 1)
    img = tf.reshape(img, [30, 30, 1])

    return img, label

def train_model(train_batches, val_x, val_y, test_batches, epochs, model_name):
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
    history = model.fit(train_batches, epochs=epochs, validation_data=(val_x, val_y))

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
    print(f'{model_name} Testing Accuracy: {test_accuracy * 100:.2f}%')

    # Get true labels and predictions
    true_labels = []
    test_pred = []
    for batch in test_batches:
        x_batch, y_batch = batch
        true_labels.extend(y_batch)
        test_pred.extend(model.predict(x_batch).flatten() > 0.5)  # Threshold predictions at 0.5

    # Calculate percentage correct
    percentage_correct = accuracy_score(true_labels, test_pred) * 100
    print(f'{model_name} Percentage Correct: {percentage_correct:.2f}%')

    # Generate confusion matrix
    cm = confusion_matrix(true_labels, test_pred)
    print(f"{model_name} Confusion Matrix:")
    print(cm)

    # Plot confusion matrix with annotations
    plt.imshow(cm, cmap='Greens', interpolation='nearest')
    plt.title(f'{model_name} Confusion Matrix')
    plt.colorbar()
    plt.xticks(ticks=[0, 1], labels=['0', '1'])
    plt.yticks(ticks=[0, 1], labels=['0', '1'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='black')

    plt.show()

    # Save the trained model in native Keras format
    model.save(f'{model_name}.keras')

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

    # Train with raw dataset
    train_model(train_batches, val_x, val_y, test_batches, args.epochs, "Raw Data")

    # Preprocess dataset using binarization
    train_dataset_bin = train_dataset.map(preprocess_bin)
    test_dataset_bin = test_dataset.map(preprocess_bin)
    validation_dataset_bin = validation_dataset.map(preprocess_bin)

    train_batches_bin = train_dataset_bin.shuffle(train_dataset_bin.cardinality().numpy()).batch(BATCH_SIZE)
    test_batches_bin = test_dataset_bin.batch(BATCH_SIZE)

    train_model(train_batches_bin, val_x, val_y, test_batches_bin, args.epochs, "Binarized Data")

    # Preprocess dataset using edge detection
    train_dataset_edge = train_dataset.map(preprocess_edge)
    test_dataset_edge = test_dataset.map(preprocess_edge)
    validation_dataset_edge = validation_dataset.map(preprocess_edge)

    train_batches_edge = train_dataset_edge.shuffle(train_dataset_edge.cardinality().numpy()).batch(BATCH_SIZE)
    test_batches_edge = test_dataset_edge.batch(BATCH_SIZE)

    train_model(train_batches_edge, val_x, val_y, test_batches_edge, args.epochs, "Edge Detected Data")

if __name__ == '__main__':
    main()
