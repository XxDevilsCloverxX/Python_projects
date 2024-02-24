"""
@Engineer: Silas Rodriguez
@Course: ECE 5363 - Pattern Recognition
@Date: Feb 18, 2024
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

def generate_linearly_separable_data(N:int):
    """
    @purpose:
        Generate linearly separable data easily with labels
    @param:
        N - Number of data points per class
    @return:
        X - data points
        y - labels
    """
    class1_x = np.random.normal(loc=25, size=(N,2))
    class2_x = np.random.normal(loc=30, size=(N,2))

    class1_t = np.zeros((N, 1))
    class2_t = np.ones((N, 1))

    dataset = np.vstack((np.hstack((class1_x, class1_t)), np.hstack((class2_x, class2_t))))
    np.random.shuffle(dataset)

    X = dataset[:, :-1]
    y = dataset[:, -1].reshape(-1, 1)
    return X, y

def SVM_Evaluator(weights: np.ndarray, X: np.ndarray, t: np.ndarray):
    """
    @purpose:
        - Evaluate the SVM classifier boundary
    @params:
        - weights - weight vector with bias prepended
        - X - feature matrix with 1s prepended
        - t - target labels
    @return:
        - misclasses - number of misclasses
        - accuracy  - % accuracy of the SVM
    """
    predictions = X.dot(weights)
    classes = np.where(predictions > 0, 0, 1)
    misclasses = np.sum(classes!=t)
    accuracy = 1 - misclasses / t.shape[0]
    return misclasses, accuracy*100

def main():

    # open the dataset
    dataset = pd.read_excel('Proj2DataSet.xlsx')
    data_np = dataset.to_numpy()

    X = data_np[:, :-1]
    t = data_np[:, -1].reshape(-1,1)

    # visualize the generated data
    plt.figure(figsize=(8,6))
    plt.scatter(X[:,0], X[:,1], c=t, cmap='coolwarm')
    plt.title('Data Plotted x1 vs x2')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.colorbar(label='Classes')
    plt.show()
    #########################################################################
    # Dual Form solution for the dataset
    dual_X = np.c_[X, np.ones(X.shape[0])]
    # Generate q'
    q = -np.ones(dual_X.shape[0]).reshape(-1,1)
    # Extract y: -1 for class 2 and 1 for class 1
    y = t.copy()
    # Generate the kernel of x: linear - X dot X^T
    K = dual_X.dot(dual_X.T)
    # Generate P: yy^T(K)
    P = y.dot(y.T) @ K
    # Generate A: Ax = 0 -- lambda (y) = 0 - (1xN) * (Nx1)
    A = y.T
    # Generate b : 0
    b = 0
    # Generate G: Gx <= 0 -lambda <=0 : -I
    G = np.vstack((-np.eye(y.shape[0]), np.eye(y.shape[0])))
    # Generate h
    C_hyperparam = 0.1
    h = np.hstack((np.zeros(y.shape[0]), C_hyperparam*np.ones(y.shape[0]))).reshape(-1,1)
    # convert all the objects to matrix
    P_cvxopt = matrix(P, tc='d')
    q_cvxopt = matrix(q, tc='d')
    G_cvxopt = matrix(G, tc='d')
    h_cvxopt = matrix(h, tc='d')
    A_cvxopt = matrix(A, tc='d')
    b_cvxopt = matrix(b, tc='d')

    # Solve the quadratic programming problem
    sol = solvers.qp(P=P_cvxopt, q=q_cvxopt, G=G_cvxopt, h=h_cvxopt, A=A_cvxopt, b=b_cvxopt)

    # Extract the optimal solution
    # lambdas = np.array(sol['x'])

    # Identify support vectors
    # support_vectors = dual_X[lambdas.flatten() > 1e-6]


if __name__ == '__main__':
    main()
