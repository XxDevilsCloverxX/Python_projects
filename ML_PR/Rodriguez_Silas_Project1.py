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

    # Append 1s to the predictors rows in the last col
    predictors = np.c_[predictors, np.ones(predictors.shape[0])]

    return labels, predictors, targets

"""
@purpose:
    perform gradient descent given max itterations desired and D
@return:
    weight_vector - D+1 vector of weights computed using gradient descent
"""
def gradient_descent(max_itterations:int, X:np.ndarray, t:np.ndarray):
    wk = np.zeros(X.shape[1])   # initial guess
    rho_k = .1         # initial learning rate
    threshold = 1e-2  # condition where change in cost is very small
    assert max_itterations >= 10, "Req: Max_itterations >= 10"
    
    for k in range(max_itterations):

        # compute the predictions
        predictions_k = np.dot(X, wk)

        # compute the gradient of the cost function : least squares squared, respect to w
        gradient = 2*np.dot(X.T, (predictions_k - t))
        print(f'Gradient: {gradient}')
        if np.linalg.norm(gradient) < threshold:
            break
        # update the weights using gradient and current learn rate
        wk -= rho_k * gradient
        print(f'Weights: {wk}')
        # every itteration, reduce learning rate by 1/5
        rho_k*=.2
    return wk

def main(csv_file:str, itterations:int):
    # Load data from CSV file
    labels, predictors, targets = load_data(csv_file)
    # linear regression: closed-form solution w = p_inv(X)*t
    weights_closed_form = np.dot(np.linalg.pinv(predictors), targets)

    # linear regression: gradient descent algorithm
    weights_gradient_descent = gradient_descent(itterations,
                                                predictors, targets)
    print(weights_gradient_descent)
    print(weights_closed_form)
    return
    # plot the data - taking the first column of the predictors (appended 1s removed):
    plt.figure()
    plt.scatter(predictors[:,0], targets, marker='x', color='red')
    plt.plot(predictors[:,0], np.dot(predictors, weights_closed_form),
            color='blue', label='Closed-Form')
    # Format the plots
    plt.xlabel(labels[0])   # Assuming first label is predictor variable desired
    plt.ylabel(labels[-1])  # Assumes targets label was last column
    plt.title('Matlab\'s "carbig" dataset')
    plt.xticks(np.arange(1500,6000, 500))
    plt.yticks(np.arange(40,260,20))
    plt.legend()
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
                        help='Maximum itterations to use for gradient descent (default 100)',
                        default=100)
    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with the provided CSV file
    main(args.csv, args.limit)