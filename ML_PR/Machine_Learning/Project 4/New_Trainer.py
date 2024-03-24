import numpy as np
import argparse
import os
import pandas as pd
import cv2
from logistic_regressor import LogisticRegressor
from sklearn.model_selection import KFold, train_test_split

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

def k_fold_cross_validation(dataset, k, shuffle=True):
    # Shuffle dataset if required
    if shuffle:
        dataset = dataset.sample(frac=1).reset_index(drop=True)
    
    # Initialize KFold
    kf = KFold(n_splits=k)

    # Split dataset into train, validate, and test sets for each fold
    for train_idx, test_idx in kf.split(dataset):
        train_set = dataset.iloc[train_idx]
        test_set = dataset.iloc[test_idx]

        # Split the training set into train and validate sets
        train_set, validate_set = train_test_split(train_set, test_size=0.2, random_state=8)  # You can adjust test_size as needed

        yield train_set, validate_set, test_set

def imgPreProc(img_Filename:str):
    """
    Preprocess images for predictions and training
    """
    img = cv2.imread(img_Filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resize = cv2.resize(gray, (30, 30))

    # Normalize pixel values to range [0, 1]
    normalized_img = resize / 255.0
    # flatten the image input
    flattened = normalized_img.flatten()
    return flattened

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
    parser = argparse.ArgumentParser(description='Logistic Regression model object test programs')
    parser.add_argument('--directory', '-d', type=str, help='Directory containing images')
    parser.add_argument('--batch_size', '-b', type=int, default=1000, help='Batch size for training')
    parser.add_argument('--epochs', '-e', type=int, default=10000, help='Num Epochs (max) for training')
    parser.add_argument('--regularizer', '-r', type=float, default=0, help='Regularizer for training')
    args = parser.parse_args()

    if not args.directory:
        args.directory = input("Enter the parent directory path containing images: ")

    # get a dataset of all the image filenames, and labels
    Dataset = pd.DataFrame(DataSetYield(directory=args.directory))

    # create a logistic regressor object
    clf = LogisticRegressor()

    # K-fold cross-validation
    k = 4               # Number of folds
    test_errors = []    # Test errors to avg
    for i, (train_set, validation_set, test_set) in enumerate(k_fold_cross_validation(Dataset, k=k)):
        print(f'Fold {i+1}:')
        print('Train set:', train_set.shape)
        print('Validation set', validation_set.shape)
        print('Test set:', test_set.shape)
        clf.clear_weights()
        clf.clear_costs()

        for j in range(args.epochs):
            print(f'Working Epoch: {j+1}')
            for k, (batch_x, batch_y) in enumerate(miniBatchGenerator(train_set, 1000)):
                clf.fit(X=batch_x, y=batch_y)   # update the weights
        
        # compute the test error for each test batch
        test_errors = []
        for test_x, test_y in miniBatchGenerator(test_set, 1000):
            k_pred = clf.predict(test_x)
            acc = np.mean(test_y == k_pred)  # append accuracy
            test_errors.append(acc)
        
        acc = np.mean(test_errors)
        print(f'Test acc for fold {i+1}: {acc}')