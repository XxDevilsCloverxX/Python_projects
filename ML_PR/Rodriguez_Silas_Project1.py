"""
@Name: Silas Rodriguez
@Date: 01/24/2024
@Dependencies: Numpy, Matplotlib, argparse
@purpose: Demonstrate Linear Regression using Closed-Form Solution
& Gradient Descent Solution of a weight vector over carbig dataset
"""
import numpy as np
import matplotlib.pyplot as plt
import argparse

"""
@purpose: Standardize a vector
@return: standardized vector of the input
"""
def standardize(vector:np.ndarray):
    # Min-max normalization
    min_vals = vector.min(axis=0)
    max_vals = vector.max(axis=0)
    mean_vals = vector.mean(axis=0)
    feature_dev = vector.std(axis=0)

    vector_normalized = (vector - min_vals) / (max_vals - min_vals)
    standardized = (vector_normalized - mean_vals) / feature_dev
    return standardized

"""
@purpose:
    load data organized from a csv file into design matrix X and
    target vector, v
@return:
    predictors - Design Matrix X
    targets    - target vector v
    labels     - labels provided for X's columns
"""
def load_data(csv_file: str):
    # Get the column labels
    try:
        with open(csv_file, 'r') as file:
            header = file.readline().strip()
            labels = tuple(header.split(','))
    except Exception as e:
        print(f'File Not Found: {e}')
        exit()
    # Load data from CSV file, skipping the labels
    data = np.genfromtxt(csv_file, delimiter=',')

    # Extract predictors (1:(D-1) columns) and targets (Dth column)
    predictors = data[1:, :-1]
    targets = data[1:, -1]

    # Find rows with non-NaN targets
    valid_rows = ~np.isnan(targets)

    # Filter predictors and targets based on valid rows
    predictors = predictors[valid_rows, :]
    targets = targets[valid_rows]
    
    # Get standard vectors
    stand_targets = standardize(targets)
    stand_predictors = standardize(predictors)

    # Append 1s to the predictors rows in the last col
    stand_predictors = np.c_[stand_predictors,
                        np.ones(stand_predictors.shape[0])]
    predictors = np.c_[predictors,
                       np.ones(predictors.shape[0])]
    
    return labels, predictors, stand_predictors, targets, stand_targets

"""
@purpose:
    perform gradient descent given max itterations desired and D
@return:
    weight_vector - D+1 vector of weights computed using gradient descent
"""
def gradient_descent(max_itterations:int, X:np.ndarray, t:np.ndarray):
    wk = np.zeros(X.shape[1])   # initial guess
    rho_k = 1e-4                # initial learning rate
    threshold = 1e-3            # condition where change in cost is very small

    for k in range(max_itterations):

        # compute the predictions
        predictions_k = np.dot(X, wk)
        # compute the gradient of the cost function : least squares squared, respect to w
        gradient = 2*np.dot(X.T, (predictions_k - t)) / X.shape[0] # 1/N

        if np.linalg.norm(gradient) < threshold:
            print(f"Gradient small @ step {k+1}")
            break
        # update the weights using gradient and current learn rate
        wk -= rho_k * gradient
        rho_k *=.95
        print(wk)
    return wk

"""
Driver for the program control
"""
def main(csv_file:str, itterations:int):
    # Load data from CSV file
    labels, predictors, X_stand, targets, t_stand = load_data(csv_file)
    # linear regression: closed-form solution w = p_inv(X)*t
    weights_closed_form = np.dot(np.linalg.pinv(predictors), targets)

    # linear regression: gradient descent algorithm
    weights_gradient_descent = gradient_descent(itterations,
                                                X_stand, t_stand)
    print(weights_gradient_descent)
    print(weights_closed_form)
    # plot the data - taking the first column of the predictors (appended 1s removed):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
    plt.scatter(predictors[:, 0], targets, marker='x', color='red')
    plt.plot(predictors[:, 0], np.dot(predictors, weights_closed_form),
            color='blue', label='Closed-Form')
    plt.xlabel(labels[0])
    plt.ylabel(labels[-1])
    plt.title('Closed-Form Solution')
    plt.xticks(np.arange(1500, 6000, 500))
    plt.yticks(np.arange(40, 260, 20))
    plt.legend()

    # Second subplot
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
    plt.scatter(predictors[:, 0], targets, marker='x', color='red')
    plt.plot(predictors[:, 0], np.dot(predictors, weights_gradient_descent),
            color='green', label='Gradient-Descent')
    plt.xlabel(labels[0])
    plt.ylabel(labels[-1])
    plt.title('Gradient Descent')
    plt.xticks(np.arange(1500, 6000, 500))
    plt.yticks(np.arange(40, 260, 20))
    plt.legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()

    plt.show()
    
    
if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(
        prog='Rodriguez_Silas_Project1.py',
        description='Linear Regression on dataset',
        epilog='Dep: Numpy, matplotlib, argparse')

    # Add the CSV + max itterations arguments
    parser.add_argument('-f','--csv',
                         type=str,
                         help='Relative path to CSV file',
                         required=True)
    parser.add_argument('-l', '--limit',
                        type=int,
                        metavar='MAX',
                        help='Maximum itterations to use for gradient descent (default 1000)',
                        default=1000)
    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with the provided CSV file
    main(args.csv, args.limit)