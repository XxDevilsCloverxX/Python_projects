import os
import cv2
import numpy as np
import pandas as pd
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

def read_test_image(file_path, img_size=(28, 28)):
    # Convert the image to grayscale
    gray_image = cv2.imread(file_path)
    gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(gray_image, (img_size))
    
    # Apply Otsu's thresholding to the grayscale image
    _, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Convert to TensorFlow tensor
    img = tf.convert_to_tensor(otsu, dtype=tf.float32)
    # normalize
    img = tf.truediv(img, 255)

    # flatten the image
    img = tf.reshape(img, (1,-1))  # Flatten to a 1D tensor
    return img


def show_confusion_matrix(cm: np.ndarray) -> None:
    """
    Display a confusion matrix
    """
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Greens')
    plt.title('Confusion Matrix of Classified Test Data')
    plt.show()  # Explicitly show the plot

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

def cv2_func(image, label):
    img = image.numpy()
    # Convert image to uint8
    img = img.astype(np.uint8)

    # Apply Otsu's thresholding to the grayscale image
    _, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Convert to TensorFlow tensor
    img = tf.convert_to_tensor(otsu, dtype=tf.float32)
    # normalize
    img = tf.truediv(img, 255)

    # flatten the image
    img = tf.reshape(img, [-1])  # Flatten to a 1D tensor
    return img, label

def tf_cv2_func(image, label):
    [image, label] = tf.py_function(cv2_func, [image, label], [tf.float32, tf.uint8])
    return image, label

def write_predictions_to_excel(predictions: np.ndarray, filenames: list, output_file: str) -> str:
    """
    This function takes a list of predictions and true labels, and creates a confusion matrix.
    From here, it will write to excel sheets:
        Sheet 1: Filename | Predicted Label | Label | Predicted Count
    """
    predictions = predictions.astype(int)
    
    # Create DataFrames
    df = pd.DataFrame({'Filename': filenames, 'Predicted Label': predictions})
    
    # Calculate and append the count of each label
    pred_counts = np.bincount(predictions)
    label_df = pd.DataFrame({'Label': range(len(pred_counts)), 'Predicted Count': pred_counts})
    
    # Write the DataFrame to an Excel file using openpyxl
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Write DataFrame to the sheet
        df.to_excel(writer, index=False, sheet_name='Predictions')
        
        # Get the workbook and the sheet
        workbook = writer.book
        sheet = workbook['Predictions']
        
        # Write the label DataFrame to the same sheet but starting from column E
        label_df.to_excel(writer, index=False, sheet_name='Predictions', startcol=4)
        
        # Set the column widths for better visibility
        sheet.column_dimensions['A'].width = 15
        sheet.column_dimensions['B'].width = 15
        sheet.column_dimensions['C'].width = 15
        sheet.column_dimensions['D'].width = 15
        sheet.column_dimensions['E'].width = 15
        sheet.column_dimensions['F'].width = 15
    
    # Get the absolute path of the file
    output_file = os.path.abspath(output_file)

    # Return the file path
    return output_file

if __name__ == '__main__':
    pass