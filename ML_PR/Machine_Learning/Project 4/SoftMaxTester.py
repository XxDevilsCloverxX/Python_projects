import argparse
from keras.preprocessing import image_dataset_from_directory
from SMR import SoftMaxRegressor

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Image Batch Generator')
    parser.add_argument('-d', '--directory', type=str, required=True, help='Parent directory containing image data')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for the generator')
    parser.add_argument('-w', '--weights', type=str, required=True, help='Path to the weights')
    args = parser.parse_args()

    # load the dataset
    test_dataset = image_dataset_from_directory(args.directory,
                                                color_mode='grayscale',
                                                image_size=(28,28))
    # create a softmax clf
    smr = SoftMaxRegressor(init_weights=args.weights)
    test_predictions = []
    for batch_data in test_dataset:
        # Perform training/testing with the batch data
        X_batch = batch_data  # Placeholder for actual training/testing code
        # Perform softmax regression
        predictions = smr.predict(X_batch)
        test_predictions.extend(predictions)
    
    print(test_predictions)