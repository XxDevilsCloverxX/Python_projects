import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

def show_confusion_matrix(cm: np.ndarray) -> None:
    """
    Display a confusion matrix
    """
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Greens')
    plt.title('Confusion Matrix of Classified Test Data')
    plt.show()  # Explicitly show the plot


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