import numpy as np
import matplotlib.pyplot as plt


def generateVectors(N: int, duplicates: int = 0):
    X = np.random.uniform(low=0, high=1, size=(N, 1))
    epsilon = np.random.normal(loc=0, scale=0.3, size=(N, 1))
    t = np.sin(2 * np.pi * X) + epsilon

    # Introduce duplicates in x values
    if duplicates!=0:
        X_dupe = np.random.random_integers(low=0, high=X.shape[0], size=duplicates)
        X_dupe = X[X_dupe]  # resample the rows of X
        epsilon = np.random.normal(loc=0, scale=0.3, size=(duplicates, 1))
        t_dupe = np.sin(2 * np.pi * X_dupe) + epsilon
        X = np.vstack((X, X_dupe))
        t = np.vstack((t, t_dupe))
    return X, t

def generateRadialBasis(X: np.ndarray, centers: np.ndarray, var: float):
    """
    Generate radial basis function values based on Gaussian radial basis functions.

    Parameters:
    - X: Data points to generate basis functions for
    - centers: Means of the distribution to be fitted to the normal distributions
    - var: Variance of the Gaussian radial basis functions

    Returns:
    - phi: Radial basis function values
    """
    exponents = -0.5 * ((X - centers) / var)**2
    phi = np.exp(exponents)
    phi = np.c_[np.ones(phi.shape[0]), phi]
    return phi

def generateBasisFunction(degree: int, X: np.ndarray):
    phi = np.power(X, np.arange(degree + 1))
    return phi

def linearRegression(phi: np.ndarray, t: np.ndarray):
    weights = np.linalg.pinv(phi).dot(t)
    return weights

def main():
    np.random.seed(52)
    # Generate the training set
    X_train, t_train = generateVectors(N=10)

    # Generate a degree set
    degrees = np.arange(10)

    for degree in degrees:
        phi_train = generateBasisFunction(degree=degree, X=X_train)
        weights_k = linearRegression(phi=phi_train, t=t_train)

        # Create a densely sampled X for plotting the polynomial curve
        X_plot = np.linspace(X_train.min(), X_train.max(), 1000).reshape(-1, 1)
        phi_plot = generateBasisFunction(degree=degree, X=X_plot)

        predictions = phi_plot.dot(weights_k)

        plt.figure(figsize=(10, 8))
        plt.title(f'Fitting {degree} degree polynomial to data')
        plt.scatter(X_train, t_train, label='Training Data')
        plt.plot(X_plot, predictions, label=f'{degree} degree Polynomial Fit', c='red')
        plt.legend()
        plt.show()


    # Generate the new training set
    X_train, t_train = generateVectors(N=100, duplicates=50)

    # Generate a variance for the Gaussian radial basis functions
    var = 1

    # Calculate unique X values and corresponding means for radial basis functions
    unique_X, unique_indices = np.unique(X_train, return_inverse=True)
    centers = np.array([np.mean(t_train[unique_indices == i]) for i in range(len(unique_X))])

    print(f'X_train datapoints: {X_train.shape[0]}, Unique means: {centers.shape[0]}')

    # Generate radial basis functions for training set
    phi_train = generateRadialBasis(X=X_train, centers=centers, var=var)
    weights_k = linearRegression(phi=phi_train, t=t_train)

    # Create a densely sampled X for plotting the radial basis function curve
    X_plot = np.linspace(X_train.min(), X_train.max(), 1000).reshape(-1, 1)
    phi_plot = generateRadialBasis(X=X_plot, centers=centers, var=var)

    predictions = phi_plot.dot(weights_k)

    # Plot the results
    plt.figure(figsize=(10, 8))
    plt.title(f'Fitting Gaussian Radial Basis Function to data and {X_train.shape[0] - centers.shape[0]} conflicting X-t pairs')
    plt.scatter(X_train, t_train, label='Training Data', c='black')
    plt.plot(X_plot, predictions, label='Gaussian Radial Basis Function Fit', c='red')
    plt.scatter(unique_X, centers, marker='x', color='red', label='Centers')
    plt.ylim(t_train.min(), t_train.max())
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
