import argparse
import tensorflow as tf
from time import time
from keras.utils import image_dataset_from_directory, split_dataset
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import numpy as np

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

def preprocess_original(image, label):
    # Convert image to float32
    img = tf.image.convert_image_dtype(image, tf.float32)
    # Normalize the pixel values to [0, 1]
    img /= 255.0
    return img, label

def preprocess_sobel(image, label):
    # Expand the dimensions to include a batch dimension
    img = tf.expand_dims(image, axis=0)
    
    # Convert image to float32
    img = tf.image.convert_image_dtype(img, tf.float32)
    # Normalize the pixel values to [0, 1]
    img /= 255.0

    # Apply Sobel edge detection
    img = tf.image.sobel_edges(img)  # Sobel edge detection

    # Remove the batch dimension after Sobel edge detection
    img = tf.squeeze(img, axis=0)

    # Calculate the magnitude of gradient
    img = tf.norm(img, axis=-1)

    # Normalize
    img = img / tf.reduce_max(img)

    return img, label

def preprocess_contrast(image, label, factor=1.0):
    # Convert image to float32
    img = tf.image.convert_image_dtype(image, tf.float32)
    # Normalize the pixel values to [0, 1]
    img /= 255.0

    # Adjust contrast
    img = tf.image.adjust_contrast(img, contrast_factor=factor)

    return img, label



def main():
    parser = argparse.ArgumentParser(description="SMR Debug")
    parser.add_argument("-d","--directory", default=None, type=str, help="Parent directory to worm images.")
    parser.add_argument("-e","--epochs", default='10', type=int, help="Number of epochs for training.")
    args = parser.parse_args()
    
    # Load the dataset
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
    
    # Preprocess validation data
    val_x, val_y = next(iter(validation_dataset.batch(len(validation_dataset))))

    start = time()
    # Train with raw dataset
    train_model(train_batches, val_x, val_y, test_batches, args.epochs, "raw")

    # Preprocess dataset using Sobel edge detection
    train_dataset_sobel = train_dataset.map(preprocess_sobel)
    test_dataset_sobel = test_dataset.map(preprocess_sobel)
    validation_dataset_sobel = validation_dataset.map(preprocess_sobel)
    val_x, val_y = next(iter(validation_dataset_sobel.batch(len(validation_dataset))))

    train_batches_sobel = train_dataset_sobel.shuffle(train_dataset_sobel.cardinality().numpy()).batch(BATCH_SIZE)
    test_batches_sobel = test_dataset_sobel.batch(BATCH_SIZE)

    # Train with Sobel edge detection
    train_model(train_batches_sobel, val_x, val_y, test_batches_sobel, args.epochs, "edge")

    # Preprocess dataset using gamma correction
    train_dataset_gamma = train_dataset.map(lambda x, y: preprocess_contrast(x, y, factor=2))  # You can adjust gamma value as needed
    test_dataset_gamma = test_dataset.map(lambda x, y: preprocess_contrast(x, y, factor=2))
    validation_dataset_gamma = validation_dataset.map(lambda x, y: preprocess_contrast(x, y, factor=2))
    val_x, val_y = next(iter(validation_dataset_gamma.batch(len(validation_dataset))))


    train_batches_gamma = train_dataset_gamma.shuffle(train_dataset_gamma.cardinality().numpy()).batch(BATCH_SIZE)
    test_batches_gamma = test_dataset_gamma.batch(BATCH_SIZE)

    # Train with gamma corrected data
    train_model(train_batches_gamma, val_x, val_y, test_batches_gamma, args.epochs, "contrast")
    end = time()

    print(f'Elapsed train time: {(end-start)/60:.3f} min')
    # Load trained models
    model_original = load_model("raw.keras")
    model_sobel = load_model("edge.keras")
    model_contrast = load_model("contrast.keras")
    
    test_x_raw, test_y = next(iter(test_dataset.batch(len(validation_dataset))))
    test_x_sobel, _ = next(iter(test_dataset_sobel.batch(len(validation_dataset))))
    test_x_contrast, _ = next(iter(test_dataset_gamma.batch(len(validation_dataset))))
    pred_original = model_original.predict(test_x_raw)
    pred_sobel = model_sobel.predict(test_x_sobel)
    pred_contrast = model_contrast.predict(test_x_contrast)
    
    predictions = tf.stack([pred_original, pred_sobel, pred_contrast])
    ensemble_pred = tf.reduce_mean(predictions, axis=0)
    ensemble_pred = tf.round(ensemble_pred)

    cm = confusion_matrix(test_y, ensemble_pred)
    print(f"Test Accuracy: {100 * np.sum(np.diag(cm)) / np.sum(cm):.2f}%")
    print(cm)

if __name__ == '__main__':
    main()
