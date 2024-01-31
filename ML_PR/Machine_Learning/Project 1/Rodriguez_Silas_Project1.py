"""
@Name: Silas Rodriguez
@Date: 01/24/2024
@Dependencies: pandas, numpy, Matplotlib, argparse
@purpose: Demonstrate Linear Regression using Closed-Form Solution
& Gradient Descent Solution of a weight vector over carbig dataset
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import argparse

def closed_form(X, t):
    """
    @purpose:
        compute closed form solution (model) for X & t using P inv
    @param:
        X - design matrix
        t - target vector
    @return:
        weights - weight vector with solution of Xw that optimizes L2^2 norm
    """
    weights = np.linalg.pinv(X).dot(t)
    return weights

def gradientDescent(X: np.ndarray, t: np.ndarray,
                     iterations: int, tolerance=1e-3):
    """
    @purpose: 
        - compute the gradient descent solution for w
    @paramters:
        - X : design matrix
        - t : target vector
        - tolerance: termination critera for gradient <= tolerance, exit
    """
    m,n = X.shape
    np.random.seed(8)   # for consistency
    wk = np.random.randn(n, 1)  # weight vector initial guess
    rho_k = 1
    for k in range(iterations):
        # compute predictions
        predictions = X.dot(wk)
        gradient = 2/m * X.T.dot(predictions - t)
        wk -= rho_k * gradient
        # degrade rho_k
        if k % 10 == 0:
            rho_k*=.8
        # exit if the weight vector sees no significant change
        if np.linalg.norm(gradient) <= tolerance:
            break
    print(f"Converged in {k+1} itterations")
    return wk

def main(csv:str, limit:int):
    """
    @purpose:
        Driver function for control flow of the program
    """
    # Open the data and normalize it, keeping a copy of original data
    df = pd.read_csv(csv)
    df = df.interpolate()   #fill nan values with a linear interpolation of the data
    
    X = df.iloc[:, :-1].values # get all the cols except the last
    t = df.iloc[:, -1:].values  # last column

    # standardize the input matrix
    scaler = StandardScaler()
    X_standard = scaler.fit_transform(X)

    # Add bias terms
    X_b = np.c_[np.ones(X_standard.shape[0]), X_standard]
    # compute the closed form solution for w
    w_closed = closed_form(X=X_b, t=t)
    print(f'Closed-Form soln: {w_closed}')

    # compute the gradient descent aproximation for w
    w_grad = gradientDescent(X=X_b, t=t, iterations=limit)
    print(f'Gradient-D soln: {w_grad}')

    print(f'Difference in soln: {w_grad - w_closed}')

    # Plot the data
    plt.figure(figsize=(12, 5))
    # # Plotting the closed-form solution
    plt.subplot(1, 2, 1)
    plt.scatter(X, t, marker='x', color='red')
    plt.plot(X, X_b.dot(w_closed), color='blue', label='Closed-Form Solution')
    plt.title('Matlab\'s "carbig" dataset')
    plt.xlabel('Weight')
    plt.ylabel('Horsepower')
    plt.xlim([1500, 5500])
    plt.ylim([40, 240])
    plt.xticks(np.arange(1500, 5501, 500))
    plt.yticks(np.arange(40, 241, 20))
    plt.legend()
    
    # Plotting the gradient descent data
    plt.subplot(1, 2, 2)
    plt.scatter(X, t, marker='x', color='red')
    plt.plot(X, X_b.dot(w_grad), color='green', label='Gradient Descent Solution')
    plt.title('Matlab\'s "carbig" dataset')
    plt.xlabel('Weight')
    plt.ylabel('Horsepower')
    plt.xlim([1500, 5500])
    plt.ylim([40, 240])
    plt.xticks(np.arange(1500, 5501, 500))
    plt.yticks(np.arange(40, 241, 20))
    plt.legend()
    # Show the plots
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(
        prog='Rodriguez_Silas_Project1.py',
        description='Linear Regression on dataset',
        epilog='Dep: Numpy, matplotlib, pandas, sklearn')

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