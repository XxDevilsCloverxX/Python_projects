"""
@Engineer: Silas Rodriguez
@Class: ECE 5370 - Machine Learning
@Date: 2/16/2024
"""
import numpy as np
import matplotlib.pyplot as plt

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
    # get the global mean
    mean = np.mean(X)

    # Calculate the step size for symmetrically placing the Gaussians
    step_size = s * np.sqrt(2 * np.log(2))  # Full Width Half Max - property of Normal distributions

    # Set Gaussians around the data using means around the global mean
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
    
    # Setup loop variables + hyperparams
    L = 100
    num_rbfs = 5    # increasing this puts more equi-distant means around the true mean - leads to overfitting
    Datasets = {}    # will collect information about each dataset, (X, t)
    weights_dataset = []    # this will hold onto the weights for each dataset (L x lambdas x rbfs + 1 x 1)

    # create permissible values for lambda
    lambdas = np.linspace(start=-2.5, stop=1.75, num=10)  # get a num evenly spaced values of lambda between -2.5 & 1.75
    lambdas = np.exp(lambdas)   # So we can recreate the ln(lambda) axis between -2.5 to 1.5

    for set in range(L):
        # generate dataset and matching targets
        X, targets = generate_Dataset(N=25)

        # generate Phi for our dataset: Maps X (Nx1) -> Phi (N x K+1)
        phi = kernel_RBF(X=X, k=num_rbfs)

        computed_weights = []
        for alpha in lambdas:
            # compute the weights with the regularization term
            weights_k = fit_curve(Phi=phi, t=targets, alpha=alpha)
            computed_weights.append(weights_k)

        # add all the computed weights to this dataset's weights
        Datasets.update({set: (X, targets)})
        weights_dataset.append(computed_weights)

    # convert the gathered weights to a np array
    weights_dataset = np.array(weights_dataset)
    
    # get the average weights across the 100 datasets axis = 0 means over L
    avg_weights = np.mean(weights_dataset, axis=0)  # (lambdas x rbfs + 1 x 1)

    # for each set of weights, compute the bias by applying the weights to the points, and subtracting my generated points
    for key, dataset in Datasets.items():
        X, t = dataset
        for weights in avg_weights:
            # apply the weights to the all the dataset
            pass

if __name__ == '__main__':
    main(L=100)