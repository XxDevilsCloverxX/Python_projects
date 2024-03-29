import os
import cv2
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def show_confusion_matrix(cm: np.ndarray) -> None:
    """
    Display a confusion matrix
    """
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Greens')
    plt.title('Confusion Matrix of Classified Test Data')
    plt.show()  # Explicitly show the plot

# def imgPreProc(img_Filename:str) -> np.ndarray:
#     """
#     Preprocess images
#     """
#     img = cv2.imread(img_Filename)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     resize = cv2.resize(gray, (30, 30))

#     # Normalize pixel values to range [0, 1]
#     normalized_img = resize / 255.0
#     # flatten the image input
#     flattened = normalized_img.flatten()
#     return flattened

def write_predictions_to_excel(predictions: np.ndarray, y_true: np.ndarray, dataset_test, output_file: str) -> str:
    """
    This function takes a list of predictions and true labels, and creates a confusion matrix.
    From here, it will write to excel sheets:
        Sheet 1: Filename | Predicted Label
        Sheet 2: Label | True Count | Predicted Count | Correct Count
    """
    # Get filenames from the dataset
    filenames = [os.path.basename(image_path.numpy().decode("utf-8")) for image_path, _ in dataset_test]

    # Compute the confusion matrix
    cm = confusion_matrix(y_true=y_true, y_pred=predictions)
    correct = np.diag(cm)

    # Convert labels array to integers
    labels = y_true.astype(int)
    predictions = predictions.astype(int)

    # Create a DataFrame to store predictions and labels
    df = pd.DataFrame({'image_filename': filenames, 'predicted_label': predictions})

    # Write the DataFrame to an Excel file using openpyxl
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Predictions')

        # Calculate and append the count of each label
        label_counts = np.bincount(labels)
        pred_counts = np.bincount(predictions)
        label_df = pd.DataFrame({'Label': range(len(label_counts)), 'True Count': label_counts, 'Predicted Count': pred_counts, 'Correct Predictions': correct})

        label_df.to_excel(writer, index=False, sheet_name='Label Counts')

    # Get the absolute path of the file
    output_file = os.path.abspath(output_file)

    # Return the file path
    return output_file

if __name__ == '__main__':
    pass