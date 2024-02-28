"""
@Engineer: Silas Rodriguez
@Course: ECE 5363 - Pattern Recognition
@Date: Feb 18, 2024
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from sklearn.svm import SVC

from time import time

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
    predictions = X @ weights
    classes = np.sign(predictions)
    misclasses = np.sum(classes!=t)
    index_misclasses = np.nonzero(classes!=t)
    accuracy = 1 - misclasses / t.shape[0]
    return misclasses, index_misclasses[0], accuracy*100

def SoftMargin_SVM(X:np.ndarray, t:np.ndarray, C:float, req_reclass:bool=False, plot_en:bool=True):
    """
    @purpose:
        - Calcaulte the Soft margin SVM using the dual form solution
    @params:
        - X - input matrix
        - t - target labels
        - C - slack variable
    @return:
        - weights: weights and bias from the SVM solution
        - time_end - time_start : elapsed time to compute the SVM solution
    """
    # Dual Form solution for the dataset - 1/2 x' P x + qx st Gx <= h & Ax = b
    time_start = time()
    # copy the data
    dual_X= X.copy()
    # extract some information from the data
    N, L = dual_X.shape
    y = np.where(t==0, 1, -1).astype('float64') if req_reclass else t.copy()

    # get the H that is used to compute P
    H = dual_X * y
    P = H @ H.T
    # generate A
    A = y.T
    # generate q
    q = -np.ones(N).astype('float64')
    # # generate G
    G = np.vstack((-np.eye(N).astype('float64'), np.eye(N).astype('float64')))
    # generate h
    h = np.hstack((np.zeros(N).astype('float64'), C * np.ones(N).astype('float64')))
    # generate b
    b = np.zeros(1).astype('float64')

    # # convert all the objects to matrix
    P_cvxopt = matrix(P, tc='d')
    q_cvxopt = matrix(q, tc='d')
    G_cvxopt = matrix(G, tc='d')
    h_cvxopt = matrix(h, tc='d')
    A_cvxopt = matrix(A, tc='d')
    b_cvxopt = matrix(b, tc='d')

    # # Solve the quadratic programming problem
    solvers.options['show_progress'] = False
    sol = solvers.qp(P=P_cvxopt, q=q_cvxopt, G=G_cvxopt, h=h_cvxopt, A=A_cvxopt, b=b_cvxopt)

    # Extract the optimal solution
    lambdas = np.array(sol['x'])

    # w parameter in vectorized form
    weights = ((y * lambdas).T @ dual_X).reshape(-1,1)

    # Selecting the set of indices S corresponding to non zero parameters
    S = (lambdas > 1e-4).flatten()

    # Computing b and append to weights
    b = y[S] - dual_X[S] @ weights
    weights = np.vstack((weights, b.mean()))
    time_end = time()

    # identify the support vectors
    support_vectors = dual_X[S]

    # plot the decision boundary and margins
    x_line = np.linspace(X[:, 0].min(), X[:, 0].max(), 100) # create a 100 evenly spaced points over x1
    y_line = -(x_line * weights[0] + weights[-1]) / weights[1]
    d = 1 / np.linalg.norm(weights[:-1])    # perpendicular distance from y_line to 1 decision boundary, no offset in this calc
    
    # applying trig, we can get the other two margins through the SV
    slope = weights[0] / weights[1]           # getting the slope of the decision boundary in terms of x2
    vert_offset = np.sqrt(1 + slope**2) * d   # proof here: https://www.geeksforgeeks.org/distance-between-two-lines/ x2 =y x1 =x for slope
    margin1 = y_line + vert_offset
    margin2 = y_line - vert_offset

    # Test classifications:
    dual_X = np.c_[X, np.ones(X.shape[0])]
    misclasses, index_misclasses, accuracy = SVM_Evaluator(weights=weights, X=dual_X, t=y)
    
    # Display results
    if plot_en:
        print("Optimal solution w:")
        print(weights)
        print("Lambdas (dual):")
        print(lambdas[S])
        print(f'Margin width: 2/||w|| = {2 * d}')
        print(f'{misclasses} misclasses with {accuracy}% accuracy!')

    # Plotting misclassified points with black squares
    misclassified_points = X[index_misclasses]  # Extract misclassified points

    if plot_en:
        plt.figure(figsize=(8,6))
        plt.title(f'Data with d(x) + Margins: C={C}')
        plt.scatter(support_vectors[:, 0], support_vectors[:, 1], facecolors='none', edgecolors='k', s=150, label=f'Support Vectors: {np.sum(S)}')    # highlight SVs
        plt.scatter(misclassified_points[:, 0], misclassified_points[:, 1], marker='s', edgecolors='black', facecolors='none', s=150, label=f'Misclassified Vectors: {misclasses}')
        plt.scatter(X[:,0], X[:,1], c=t, cmap='coolwarm')
        plt.plot(x_line, y_line, c='black', label='d(x) = 0')
        plt.plot(x_line, margin1, c='red', label='d(x) = 1')
        plt.plot(x_line, margin2, c='blue', label='d(x) = -1')

        # Shade the entire region below margin2 in blue
        plt.fill_between(x_line, y_line, min(margin2), color='blue', alpha=0.2, label='Positive Region (C0)')
        # Shade the entire region above margin1 in red
        plt.fill_between(x_line, y_line, max(margin1), color='red', alpha=0.2, label='Negative Region (C1)')
        # shade the margin with gray
        plt.fill_between(x_line, margin1, margin2, color='black', alpha=0.2, label='Maximized Margin')

        plt.legend()
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.colorbar(label='Classes')
        plt.show()

    return weights, time_end-time_start

def SkLearn_SVM(X:np.ndarray, t:np.ndarray, C:float, req_reclass:bool=False, plot_en:bool=True):
    """
    @purpose:
        - Calcaulte the Soft margin SVM using the sklearn solution
    @params:
        - X - input matrix
        - t - target labels
        - C - slack variable
    @return:
        - weights: weights and bias from the SVM solution
        - time_end - time_start : elapsed time to compute the SVM solution
    """
    time_start = time()
    sk_x = X.copy()
    sk_y = t.copy() if not req_reclass else np.where(t==0, 1, -1).astype('float64')
    clf = SVC(C = C, kernel = 'linear')
    clf.fit(sk_x, sk_y.ravel()) 
    time_end = time()

    # get the support vectors
    support_vectors = sk_x[clf.support_]
    # get the weights
    weights = clf.coef_
    intercept = clf.intercept_
    weights = np.append(weights, intercept).reshape(-1, 1)

    # plot the decision boundary and margins
    x_line = np.linspace(X[:, 0].min(), X[:, 0].max(), 100) # create a 100 evenly spaced points over x1
    y_line = -(x_line * weights[0] + weights[-1]) / weights[1]
    d = 1 / np.linalg.norm(weights[:-1])    # perpendicular distance from y_line to 1 decision boundary, no offset in this calc
    
    # applying trig, we can get the other two margins through the SV
    slope = weights[0] / weights[1]           # getting the slope of the decision boundary in terms of x2
    vert_offset = np.sqrt(1 + slope**2) * d   # proof here: https://www.geeksforgeeks.org/distance-between-two-lines/ x2 =y x1 =x for slope
    margin1 = y_line + vert_offset
    margin2 = y_line - vert_offset

    # Test classifications:
    sk_x = np.c_[X, np.ones(X.shape[0])]
    misclasses, index_misclasses, accuracy = SVM_Evaluator(weights=weights, X=sk_x, t=sk_y)

    if plot_en:
        print('w = ',clf.coef_)
        print('b = ',clf.intercept_)
        print('Indices of support vectors = ', clf.support_)
        print('Support vectors = ', clf.support_vectors_)
        print('Number of support vectors for each class = ', clf.n_support_)
        print('Coefficients of the support vector in the decision function = ', np.abs(clf.dual_coef_))
        print(f'Margin width: 2/||w|| = {2 * d}')
        print(f'{misclasses} misclasses with {accuracy}% accuracy!')

    # Plotting misclassified points with black squares
    misclassified_points = X[index_misclasses]  # Extract misclassified points

    if plot_en:
        plt.figure(figsize=(8,6))
        plt.title(f'Data with d(x) + Margins - SkLearn C={C}')
        plt.scatter(support_vectors[:, 0], support_vectors[:, 1], facecolors='none', edgecolors='k', s=150, label=f'Support Vectors: {len(support_vectors)}')    # highlight SVs
        plt.scatter(misclassified_points[:, 0], misclassified_points[:, 1], marker='s', edgecolors='black', facecolors='none', s=150, label=f'Misclassified Vectors: {misclasses}')
        plt.scatter(X[:,0], X[:,1], c=t, cmap='coolwarm')
        plt.plot(x_line, y_line, c='black', label='d(x) = 0')
        plt.plot(x_line, margin1, c='red', label='d(x) = 1')
        plt.plot(x_line, margin2, c='blue', label='d(x) = -1')

        # Shade the entire region below margin2 in blue
        plt.fill_between(x_line, y_line, min(margin2), color='blue', alpha=0.2, label='Positive Region (C0)')
        # Shade the entire region above margin1 in red
        plt.fill_between(x_line, y_line, max(margin1), color='red', alpha=0.2, label='Negative Region (C1)')
        # shade the margin with gray
        plt.fill_between(x_line, margin1, margin2, color='black', alpha=0.2, label='Margin')

        plt.legend()
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.colorbar(label='Classes')
        plt.show()


    return weights, time_end-time_start

def generate_linearly_nonseparable_data(N:int):
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
    class2_x = np.random.normal(loc=28.5, size=(N,2))

    class1_t = np.zeros((N, 1))
    class2_t = np.ones((N, 1))

    dataset = np.vstack((np.hstack((class1_x, class1_t)), np.hstack((class2_x, class2_t))))
    np.random.shuffle(dataset)

    X = dataset[:, :-1]
    y = dataset[:, -1].reshape(-1, 1)
    return X, y

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
    # soft margin SVM with C = 0.1
    SoftMargin_SVM(X = X, t=t, C=0.1)
    #########################################################################
    # soft margin SVM with C = 100
    SoftMargin_SVM(X = X, t=t, C=100)
    
    #########################################################################
    # sklearn solution for C = 0.1
    SkLearn_SVM(X = X, t= t, C=0.1)
    
    #########################################################################
    # sklearn solution for C = 100
    SkLearn_SVM(X = X, t= t, C=100)
    
    #########################################################################
    # Time differences
    times_dual  = []
    times_sklrn = []
    x_plot = [i for i in range(50, 501, 50)]

    np.random.seed(0)
    for samples in x_plot:
        print(f'Num Samples working : {samples}')
        X, t = generate_linearly_nonseparable_data(N=samples)
        _, time_dual = SoftMargin_SVM(X=X, t=t, C=10, req_reclass=True, plot_en=False)
        _, time_sklr = SkLearn_SVM(X=X, t=t, C=10, req_reclass=True, plot_en=False)

        times_dual.append(time_dual)
        times_sklrn.append(time_sklr)
    
    print('Plotting...')
    plt.figure(figsize=(8,6))
    plt.plot(x_plot, times_dual, c='red', label='CVXOPT time')
    plt.plot(x_plot, times_sklrn, c='blue', label='LIBCVM time')
    plt.xlabel('N Samples')
    plt.ylabel('Elapsed Time (s)')
    plt.title('LIBSVM vs CVXOPT Computational Time for Weights vs # Samples')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
