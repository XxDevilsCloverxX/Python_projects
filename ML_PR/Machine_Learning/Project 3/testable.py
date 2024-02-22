"""
@Engineer: Silas Rodriguez
@Class: ECE 5370 - Machine Learning
@Date: 2/16/2024
"""

import numpy as np
import matplotlib.pyplot as plt
from math import ceil

def kernel_RBF(X: np.ndarray, k:int=1, s:float=0.1):
    """
    Purpose:
        Generate radial basis function values based on Gaussian radial basis functions.
    Parameters:
    - X: Data points to generate basis functions for
    - s: std dev of the Gaussian radial basis functions
    - k: number of RBFs to fit to the data - Hyperparameter
    Returns:
    - phi: Radial basis function values
    """
    # get a singular mean over all the input data: - Expected value of X
    mean = np.mean(X)
    
    # Calculate the step size between RBF centers based on the standard deviation
    step_size = s * np.sqrt(2 * np.log(2))  # Full Width Half Max - property of Normal distributions

    # Generate RBF centers around the 'population' mean - assuming what we know about the data is that low variance, we can expect means around this mean
    centers = np.linspace(mean - (k-1) * step_size, mean + (k-1) * step_size, num=k)

    # calculate all the exponents - should return N x K exponenets for the data
    exponents = -((X-centers)**2 / (2*s**2))

    # e raise to the exponents powers
    phi = np.exp(exponents)
    # prepend the ones for Phi (will lead to bias fit)
    phi = np.c_[np.ones(phi.shape[0]), phi] 
    return phi

def fit_curve(Phi:np.ndarray, t:np.ndarray, alpha:float=0):
    """
    @purpose:
        fit a curve using least-squares regression and return weights
    @param:
        Phi - Feature matrix fit to some basis functions
        t - target vector
        alpha - regularization parameter
    @return:
        Weights- parameters of the model
    """
    lin_kern = Phi.T.dot(Phi)
    reg_mat = np.eye(lin_kern.shape[0])
    weights = np.linalg.inv(lin_kern + alpha * reg_mat).dot(Phi.T).dot(t)
    return weights

def k_fold_cross_validation(X:np.ndarray, targets:np.ndarray, lambdas:np.ndarray, k:int=5):
    """
    @purpose:
        Perform k-fold cross-validation to estimate the optimal value for the regularization parameter (lambda).

    @parameters:
        - X (numpy.ndarray): Feature matrix.
        - targets (numpy.ndarray): Target vector.
        - k (int): Number of folds for cross-validation.
        - lambdas (numpy.ndarray): Array of regularization parameter values to be tested.

    @returns:
        - best_lambda (float): The optimal regularization parameter that minimizes the average error across folds.
    """
    # Calculate number of samples
    N = X.shape[0]

    # Shuffle indices
    indices = np.random.permutation(N)

    # Use shuffled indices to reorder data
    X_shuffled = X[indices]
    targets_shuffled = targets[indices]

    # Split shuffled data into k folds
    fold_size = N // k
    folds_X = [X_shuffled[i*fold_size:(i+1)*fold_size, :] for i in range(k)]
    folds_targets = [targets_shuffled[i*fold_size:(i+1)*fold_size] for i in range(k)]

    # Perform k-fold cross-validation
    errors = []
    for alpha in lambdas:
        fold_errors = []
        for i in range(k):
            # Use the i-th fold for testing, others for training
            X_test, targets_test = folds_X[i], folds_targets[i]
            X_train = np.concatenate([folds_X[j] for j in range(k) if j != i], axis=0)
            targets_train = np.concatenate([folds_targets[j] for j in range(k) if j != i], axis=0)

            # Generate Phi for training set
            phi_train = kernel_RBF(X_train, k=5)

            # Fit the curve on the training set
            weights = fit_curve(Phi=phi_train, t=targets_train, alpha=alpha)

            # Generate Phi for the test set
            phi_test = kernel_RBF(X_test, k=5)

            # Calculate the error on the test set
            error = np.linalg.norm(phi_test.dot(weights) - targets_test)**2 / len(targets_test)
            fold_errors.append(error)

        # Average the errors across folds for the current lambda
        errors.append(np.mean(fold_errors))

    # Find the lambda that minimizes the average error
    best_lambda = lambdas[np.argmin(errors)]

    return best_lambda

def generate_Dataset(N:int, std=0.3):
    """
    @purpose:
        generate data matrix X from a normal distribution
        that is N data points x 1 features
    @param:
        N- num data points
        std - std dev to sample noise
    @return:
        X - Nx1 feature matrix
        T - Nx1 matrix of targets
    """
    X = np.random.uniform(low=0,high=1, size=(N,1))
    noise = np.random.normal(loc=0, scale=std, size=(N,1))
    t = np.sin(2 * np.pi * X)   # apply the sin function along the rows
    t += noise                  # add some noise to the targets
    return X, t

def compute_cost(predictions:np.ndarray, targets:np.ndarray):
    return np.linalg.norm(predictions - targets) ** 2

def main(L:int = 100):
    # Seed the program for reproducibility - Experimental
    np.random.seed(52)
    
    # generate dataset and matching targets
    X, targets = generate_Dataset(N=25)
    
    # generate Phi for our dataset: Maps X (Nx1) -> Phi (N x K+1)
    phi = kernel_RBF(X=X, k=5)

    # create permissible values for lambda
    lambdas = np.linspace(start=-2.5, stop=1.75, num=10)  # get a 100 evenly spaced values of lambda between -2.5 & 1.75
    lambdas = np.exp(lambdas)   # So we can recreate the ln(lambda) axis between -2.5 to 1.5

    # # Perform k-fold cross-validation - hyper param lambda
    # best_lambda = k_fold_cross_validation(X=X, targets=targets, k=5, lambdas=lambdas)
    # print("Best Lambda:", best_lambda)

    computed_weights = []
    for alpha in lambdas:
        # compute the weights with the regularization term
        weights_k = fit_curve(Phi=phi, t=targets, alpha=alpha)
        computed_weights.append(weights_k)

    # Plot fits for different lambdas
    fig_row, fig_col = 1, 2
    num_figs = ceil(len(lambdas) / (fig_col * fig_row))

    # Generate Phi for the test set
    X_test, targ_test = generate_Dataset(N=1000)
    Phi_test = kernel_RBF(X=X_test, k=5)
    test_pts = np.linspace(X.min(), X.max(), num=1000).reshape(-1,1)
    test_rbf = kernel_RBF(test_pts, k=5)

    for i in range(num_figs):
        # create a figure
        plt.figure(figsize=(8,6))
        # get the weights being used in this iteration
        start = i*fig_row*fig_col
        end = (start+fig_col*fig_row)
        plotting_weights = computed_weights[start:end]
        for fig_num, plot_weight in enumerate(plotting_weights):    
            # Calculate the predicted values using the computed weights
            predictions = test_rbf.dot(plot_weight)
            # identify which subplot this belongs to:
            plt.subplot(fig_row,fig_col,fig_num+1)
            plt.title(f'N= {1000}, Weights[{start+fig_num}], lambda = {lambdas[start+fig_num]:.4f}')
            # Plot the fit curve along with the test data
            plt.scatter(X_test, targ_test, label="Test Data", color="black")
            plt.plot(test_pts, predictions, color="red", label=f'Cost: {compute_cost(predictions=predictions, targets=targ_test):.3f}')
            plt.tight_layout()
            plt.legend()
    # show the subplots
    plt.show()


if __name__ == '__main__':
    main(L=100)