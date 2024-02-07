"""
@Engineer: Silas Rodriguez
@Date: February 3, 2024
@Course: Machine Learning ECE 5370
"""
import numpy as np
import matplotlib.pyplot as plt

def generateVectors(N:int):
    """
    @purpose:
        - Generate sets of data from uniform or gaussian distributions
    @param:
        N - how many data points to generate
    @return:
        data- data generated from uniform distribution
        targets- values assosciated with data + some noise
    """
    X = np.random.uniform(low=0,high=1, size=(N,1))
    epsilon = np.random.normal(loc=0,scale=0.3, size=(N,1))
    t = np.sin(2 * np.pi * X) + epsilon
    return X, t

def generateBasisFunction(degree:int, X:np.ndarray):
    """
    @purpose: 
        generate a vector phi(x) using basis functions
    @param:
        X - dataset needing to be fit to a mth degree polynomial
        degree - the largest power to fit X to
    @return:
        phi - basis function vector phi(X)
    """
    phi = np.power(X, np.arange(degree + 1))
    return phi

def linearRegression(phi:np.ndarray, t:np.ndarray):
    """
    @purpose:
        fit Phi using linear regression with t
    @param:
        phi - phi(X_train) of basis functions
        t - target vector corresponding to phi
    @return:
        weights - weights fitting phi_train to t
    """
    weights = np.linalg.pinv(phi).dot(t)
    return weights

def calculateError(phi:np.ndarray, weights:np.ndarray, t:np.ndarray):
    """
    @purpose:
        Calculate the least squares error of phi from t using w
    @param:
        phi - basis function vector phi(X)
        weights - weights that dot with phi to t
        t - target vector corresponding to phi
    @return:
        error - reported least squares error of phi with w against t
    """
    # calculate the prediction, its difference from target, then square it
    predictions = np.power(phi.dot(weights) - t, 2)
    # compute the error
    error = np.sum(predictions)
    return error

def plotErrors(degrees:np.ndarray, errors_train:list, errors_test:list, N:int):
    """
    @purpose: 
        plot the errors corresponding to a model of certain degree
    @param:
        degrees - vector of degrees corresponding to errors
        errors_train - errors corresponding to training sets
        errors_test - errors corresponding to testing sets
        N - training dataset size
    """
        # Plot the E_RMS with degrees
    plt.figure()
    plt.plot(degrees, errors_train, marker='o', color='blue', label='Training')
    plt.plot(degrees, errors_test, marker='o', color='red', label='Testing')
    plt.legend()
    plt.title(f'M vs Training & Testing Error N={N}')
    plt.xlabel('M')
    plt.ylabel('E_RMS')
    plt.xlim(0,10)
    plt.ylim(0, 1)
    plt.show()

def main():
    """
    @purpose:
        driver function / control flow
    52 - from experimenting, 12 alternative
    """

    np.random.seed(52)
    # Generate the training set
    X_train, t_train = generateVectors(N=10)
    # Generate the test set
    X_test, t_test = generateVectors(N=100)

    # Generate a degree set
    degrees = np.arange(10)

    # track RMS errors
    E_RMS_train = []
    E_RMS_test = []
    for degree in degrees:
        phi_train = generateBasisFunction(degree=degree, X = X_train)
        phi_test = generateBasisFunction(degree=degree, X = X_test)
        weights_k = linearRegression(phi=phi_train, t=t_train)
        error_train = calculateError(phi= phi_train, weights=weights_k, t=t_train)
        error_test  = calculateError(phi= phi_test, weights=weights_k,t=t_test)
        E_RMS_train.append(np.sqrt(error_train/phi_train.shape[0]))
        E_RMS_test.append(np.sqrt(error_test/phi_test.shape[0]))
    
    # plot the data
    plotErrors(degrees=degrees, errors_test=E_RMS_test, errors_train=E_RMS_train, N=phi_train.shape[0])

    # Repeat the experiment for 100 data points in X_train
    X_train_100, t_train_100 = generateVectors(N=100)
    X_test_100, t_test_100 = generateVectors(N=100)
    
    # track RMS errors
    E_RMS_train = []
    E_RMS_test = []
    for degree in degrees:
        phi_train = generateBasisFunction(degree=degree, X = X_train_100)
        phi_test = generateBasisFunction(degree=degree, X = X_test_100)
        weights_k = linearRegression(phi=phi_train, t=t_train_100)
        error_train = calculateError(phi= phi_train, weights=weights_k, t=t_train_100)
        error_test  = calculateError(phi= phi_test, weights=weights_k,t=t_test_100)
        E_RMS_train.append(np.sqrt(error_train/phi_train.shape[0]))
        E_RMS_test.append(np.sqrt(error_test/phi_test.shape[0]))
    
    # plot the data
    plotErrors(degrees=degrees, errors_test=E_RMS_test, errors_train=E_RMS_train, N=phi_train.shape[0])

if __name__ == "__main__":
    main()