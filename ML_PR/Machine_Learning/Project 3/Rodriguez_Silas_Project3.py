"""
@Engineer: Silas Rodriguez
@Class: ECE 5370 - Machine Learning
@Date: 2/16/2024
"""

import numpy as np
import matplotlib.pyplot as plt

def basis_functions(X:np.ndarray):
    """
    @purpose: 
        Fit guassian basis functions to the dataset, where s=0.1,
        and mu seleected from different parts of the data.
    @param:
        X - input feature matrix
    @return:
        phi - feature matrix with basis functions applied to X
    """
    phi = 0
    return phi

def calcF_bar(model:np.ndarray, data:np.ndarray):
    L = data.shape[1]   # extract the columns
    fbar = 0
    return fbar

def calc_bias_squared():
    pass

def calc_variance():
    pass



def generate_Data(N:int, L:int, std=0.3):
    """
    @purpose:
        generate data matrix X from a normal distribution
        that is N data points x L features
    @param:
        N- num data points
        L- num of features per data point
        var - variance to sample noise
    @return:
        X - NxL feature matrix
        T - NxL matrix of targets
    """
    X = np.random.uniform(low=0,high=1, size=(N,L))
    noise = np.random.normal(loc=0, scale=std, size=(N,1))
    t = np.sin(2 * np.pi * X)   # apply the sin function along the rows
    t += noise                  # add some noise to the targets
    return X, t

def main():
    # Seed the program for reproducibility - Experimental
    np.random.seed(52)
    X, targets = generate_Data(N=25, L=100)
    print(X.shape, targets.shape)

if __name__ == '__main__':
    main()