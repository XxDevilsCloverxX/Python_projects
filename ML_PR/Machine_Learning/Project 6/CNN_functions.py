import os
import numpy as np
import pandas as pd
import tensorflow as tf

def preprocess_original(image):
    # Convert image to float32
    img = tf.image.convert_image_dtype(image, tf.float32)
    # Normalize the pixel values to [0, 1]
    img /= 255.0
    return img

def preprocess_sobel(image):
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

    return img

def preprocess_contrast(image, factor=1.0):
    # Convert image to float32
    img = tf.image.convert_image_dtype(image, tf.float32)
    # Normalize the pixel values to [0, 1]
    img /= 255.0

    # Adjust contrast
    img = tf.image.adjust_contrast(img, contrast_factor=factor)

    return img

def write_predictions_to_excel(predictions: tf.Tensor, filenames: list, output_file: str) -> str:
    """
    This function takes a list of predictions and true labels, and creates a confusion matrix.
    From here, it will write to excel sheets:
        Sheet 1: Filename | Predicted Label | Label | Predicted Count
    """
    predictions = tf.cast(predictions, tf.int8)
    
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