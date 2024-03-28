import os
import cv2
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import KFold, train_test_split

def show_confusion_matrix(cm: np.ndarray) -> None:
    """
    Display a confusion matrix
    """
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Greens')
    plt.title('Confusion Matrix of Classified Test Data')
    plt.show()  # Explicitly show the plot

def imgPreProc(img_Filename:str) -> np.ndarray:
    """
    Preprocess images
    """
    img = cv2.imread(img_Filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resize = cv2.resize(gray, (30, 30))

    # Normalize pixel values to range [0, 1]
    normalized_img = resize / 255.0
    # flatten the image input
    flattened = normalized_img.flatten()
    return flattened

def write_predictions_to_excel(predictions: np.ndarray, labels: np.ndarray, output_file: str) -> str:
    """
    This function takes a list of predictions and true labels, and creates a confusion matrix
    From here, it will write to excel sheets:
        Sheet 1: Filename / Index | Predicted Label 
        Sheet 2: Label | True Count | Predicted Count | Correct Count
    """
    # Convert labels array to integers
    labels = labels.astype(int)
    predictions = predictions.astype(int)
    # Create a DataFrame to store predictions and labels
    df = pd.DataFrame({'image_filename': range(len(predictions)), 'label': predictions})

    # Write the DataFrame to an Excel file
    writer = pd.ExcelWriter(output_file, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Predictions')

    # Calculate and append the count of each label
    label_counts = np.bincount(labels)
    pred_counts = np.bincount(predictions)
    label_df = pd.DataFrame({'Label': range(len(label_counts)), 'True Count': label_counts, 'Predicted Count': pred_counts})

    label_df.to_excel(writer, index=False, sheet_name='Label Counts')

    # Save the Excel file
    writer._save()

    # Get the absolute path of the file
    output_file = os.path.abspath(output_file)

    # Return the file path
    return output_file

def trainerplot(df: pd.DataFrame) -> None:
    """
    @param:
        df - dataframe that is returned from the SMR training fit process
    """
    epochs = df['epoch']
    train_loss = df['train_loss']
    validation_loss = df['validation_loss']
   
    # Plot loss vs epochs
    plt.figure(figsize=(8, 6))
    plt.subplot(2, 2, 1)
    plt.title('Loss vs Epochs')
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, validation_loss, label='Validation Loss')
    plt.legend()
    # Get column names excluding 'epoch' and 'loss'
    columns_to_plot = [col for col in df.columns if col not in ['epoch', 'train_loss', 'validation_loss']]

    # Plot ||W|| and ||G|| vs epochs for each column
    for i, col in enumerate(columns_to_plot, start=2):
        plt.subplot(2, 2, i)
        plt.title(f'||{col}|| vs Epochs')
        values = np.linalg.norm(np.array(df[col].to_list()), axis=1)
        for j in range(values.shape[1]):
            plt.plot(epochs, values[:, j], label=f'||{col}||: C={j+1}')
        plt.legend()

    plt.tight_layout()
    plt.show()

def k_fold_cross_validation(X:np.ndarray, y:np.ndarray, k:int, shuffle=True):
    """
    Separate the dataset into train, test, validate groups
    """
    # Shuffle dataset once if required
    if shuffle:
        indices = np.random.choice(X.shape[0], size=X.shape[0], replace=False)
        shuffled_x = X[indices]
        shuffled_y = y[indices].reshape(-1,1)

    # join the dataset features and labels
    dataset = np.hstack((shuffled_x, shuffled_y))
    # Initialize KFold
    kf = KFold(n_splits=k)

    # Split dataset into train, validate, and test sets for each fold
    for train_idx, test_idx in kf.split(dataset):
        train_set = dataset[train_idx]
        test_set = dataset[test_idx]

        # Split the training set into train and validate sets
        train_set, validate_set = train_test_split(train_set, test_size=0.1, random_state=8)  # Adjust validation size as needed

        yield train_set, validate_set, test_set

def ImageYield(directory:str, classLabel:int):
    """
    @purpose:
        Yield image filenames and class labels from a specified directory
    @params:    
        directory: directory to yield images from
        size: how many images to yield from the directory until exhaustion
        classLabel: which class this directory belongs to
    @return:
        filenames: filenames of images to be returned
        class_labels: labels to be assigned to this imageset
    """
    assert directory is not None, "A directory must be provided."

    # get a list of image filepaths
    image_files = sorted([os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(('.jpg', '.jpeg', '.png', '.tif'))])
    class_labels = np.ones(len(image_files)) * classLabel
    return image_files, class_labels

def DataSetYield(directory:str=None):
    """
        Get a list of all filenames + labels in subdirectories from a parent Directory
    Args:
        directory (str): Path to the directory containing images.
    Return:
        lists of filename, label pairs for the data
    """
    assert directory is not None, "A Parent directory must be provided."

    # Sorted subdirectories for 0, 1, 2, etc
    subdirectories = sorted([os.path.join(directory, subdir) for subdir in os.listdir(directory) if
                        os.path.isdir(os.path.join(directory, subdir))])

    # for each subdirectory, get a list of image filenames
    dataset = []
    for k, subdir in enumerate(subdirectories):
        x, t = ImageYield(directory=subdir, classLabel=k)
        for filename, label in zip(x, t):
            dataset.append({'filename': filename,
                                  'label': label})
    return dataset

def miniBatchGenerator(subset:pd.DataFrame, size:int):
    """
    @purpose:
        Generate a mini-batch of images from the given set
    @params:
        subset: training, validation, or test set to make a mini-batch from
        size: size of the batch to generate, 1-N
    @return:
        batch_x: np.array of preprocessed images
        batch_y: np.array of image labels
    """
    mini_set = subset.to_numpy()
    files = mini_set[:, 0]
    batch_y = mini_set[:, -1]   # get all the rows, but the last column for the label
    cur = 0

    # keep yielding full batches
    while cur + size < len(files):
        # create a new array to yield
        batch_x = []
        # open the images between cur & cur + size
        current_batch = files[cur:cur+size]
        for file in current_batch:
            img = imgPreProc(file)
            batch_x.append(img)
        yield np.array(batch_x), batch_y[cur:cur+size]
        cur += size
    
    # yield a partial batch
    batch_x = []
    final_batch = files[cur:]
    for file in final_batch:
        img = imgPreProc(file)
        batch_x.append(img)
    yield np.array(batch_x), batch_y[cur:]

if __name__ == '__main__':
    pass