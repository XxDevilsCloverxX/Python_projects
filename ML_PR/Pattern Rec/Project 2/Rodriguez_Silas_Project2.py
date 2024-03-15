"""
@Engineer: Silas Rodriguez
@Course: ECE 5363 - Pattern Recognition
@Date: Feb 28, 2024
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from sklearn.svm import SVC

import timeit

def SVM_Evaluator(predictions:np.ndarray, t: np.ndarray):
    """
    @purpose:
        - Evaluate the SVM classifier boundary
    @params:
        - predictions - predicted labels
        - t - target labels
    @return:
        - misclasses - number of misclasses
        - accuracy  - % accuracy of the SVM
    """
    misclasses = np.sum(predictions!=t)
    index_misclasses = np.nonzero(predictions!=t)
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
    # copy the data
    dual_X= X.copy()
    # extract some information from the data
    N, L = dual_X.shape
    y = np.where(t==0, 1, -1).astype('float64') if req_reclass else t.copy()

    # get the H that is used to compute P
    K = dual_X @ dual_X.T
    P = np.outer(y, y) * K
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

    # Support vectors have non zero lagrange multipliers
    sv = (lambdas > 1e-3).flatten()
    ind = np.arange(len(lambdas))[sv]
    a_sv = lambdas[sv].ravel()
    X_sv = X[sv].ravel()
    y_sv = y[sv].ravel()

    weights = np.sum(y[sv] * lambdas[sv] * dual_X[sv], axis=0)

    # Intercept - average over indices where 0 < alpha_i < C
    b = 0
    alpha_diff_C_count = 0
    is_eq_to_C = lambda a: C is not None and a > C - 1e-3

    for n in range(len(a_sv)):
        if is_eq_to_C(a_sv[n]):
            continue

        alpha_diff_C_count += 1
        b += y_sv[n]
        b -= np.sum(a_sv * y_sv * K[ind[n],sv])

    b = b / alpha_diff_C_count if alpha_diff_C_count > 0 else 0

    weights = np.append(weights, b)
    # identify the support vectors
    support_vectors = dual_X[sv]

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
    predictions = np.sign(dual_X @ weights)
    misclasses, index_misclasses, accuracy = SVM_Evaluator(predictions=predictions, t=y)
    
    # Display results
    if plot_en:
        print("Optimal solution w:")
        print(weights)
        print("Lambdas (dual):")
        print(lambdas[sv])
        print(f'Margin width: 2/||w|| = {2 * d}')
        print(f'{misclasses} misclasses with {accuracy}% accuracy!')

    # Plotting misclassified points with black squares
    misclassified_points = X[index_misclasses]  # Extract misclassified points

    if plot_en:
        plt.figure(figsize=(8,6))
        plt.title(f'CVXOPT d(x) + Margins: C={C}')
        plt.scatter(support_vectors[:, 0], support_vectors[:, 1], facecolors='none', edgecolors='k', s=150, label=f'Support Vectors: {np.sum(sv)}')    # highlight SVs
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
        # plt.show()

    return weights

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
    sk_x = X.copy()
    sk_y = t.copy() if not req_reclass else np.where(t==0, 1, -1).astype('float64')
    clf = SVC(C = C, kernel = 'linear')
    clf.fit(sk_x, sk_y.ravel()) 

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
    predictions = np.sign(sk_x @ weights)
    misclasses, index_misclasses, accuracy = SVM_Evaluator(predictions=predictions, t=sk_y)

    if plot_en:
        print('w = ',clf.coef_)
        print('b = ',clf.intercept_)
        print('Indices of support vectors = ', clf.support_)
        print('Support vectors = ', clf.support_vectors_)
        print('Number of support vectors for each class = ', clf.n_support_)
        print('Coefficients of the support vector in the decision function = ', np.abs(clf.dual_coef_))
        print(f'Margin width: 2/||w|| = {2 * d:.2f}')
        print(f'{misclasses} misclasses with {accuracy:.2f}% accuracy!')

    # Plotting misclassified points with black squares
    misclassified_points = X[index_misclasses]  # Extract misclassified points

    if plot_en:
        plt.figure(figsize=(8,6))
        plt.title(f'SkLearn d(x) + Margins C={C}')
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
        # plt.show()

    return weights

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

    class1_t = np.ones((N, 1))
    class2_t = -1*np.ones((N, 1))

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

    #########################################################################
    # soft margin SVM with C = 0.1
    SoftMargin_SVM(X = X, t=t, C=0.1)
    print()
    #########################################################################
    # soft margin SVM with C = 100
    SoftMargin_SVM(X = X, t=t, C=100)
    print()
    #########################################################################
    # sklearn solution for C = 0.1
    SkLearn_SVM(X = X, t= t, C=0.1)
    print()
    #########################################################################
    # sklearn solution for C = 100
    SkLearn_SVM(X = X, t= t, C=100)
    print()    
    # plt.show()
    return
    #########################################################################
    # Create a logarithmically spaced array
    n_values = np.logspace(np.log2(16), np.log2(1024), num=25, base=2.0, dtype=int)
    n_values = np.unique(n_values)  # drop any duplicate values
    times_cvxopt = []
    times_sklrn  = []

    print('Entering timing loop... This will take a while')
    for num_samples in n_values:
        print(f'Num Samples working : {2*num_samples}') # My function generates N * 2 for the dataset (evenly filled classes)
        # Generate some data for this itteration
        X, t = generate_linearly_nonseparable_data(N=num_samples)

        # Timing SoftMargin_SVM function
        time_cvx = timeit.repeat(lambda: SoftMargin_SVM(X=X, t=t, C=1, plot_en=False), number=1, repeat=10)
        min_time_cvx = min(time_cvx)

        # Timing SkLearn_SVM function
        time_sklr = timeit.repeat(lambda: SkLearn_SVM(X=X, t=t, C=1, plot_en=False), number=1, repeat=1000)
        min_time_sklr = min(time_sklr)

        times_cvxopt.append(min_time_cvx)
        times_sklrn.append(min_time_sklr)

    times_cvxopt = np.array(times_cvxopt)
    times_sklrn = np.array(times_sklrn)

    plt.figure(figsize=(14, 6))
    plt.subplot(1,2,1)
    plt.plot(n_values*2, times_cvxopt, label='CVXOPT Average Execution Time (s)', color='red')
    plt.plot(n_values*2, times_sklrn, label='LIBSVM Average Execution Time (s)', color='blue')
    plt.ylabel("Average Execution Time")
    plt.xlabel("Number of Samples, N")
    plt.title('Execution Time vs N (LIBSVM vs CVXOPT)- Same Scale')
    plt.legend()
    plt.grid(True)

    plt.subplot(1,2,2)
    plt.plot(n_values*2, times_cvxopt, label='CVXOPT Average Execution Time (s)', color='red')
    plt.plot(n_values*2, times_sklrn*1000, label='LIBSVM Average Execution Time (ms)', color='blue')
    plt.ylabel("Average Execution Time")
    plt.xlabel("Number of Samples, N")
    plt.title('Execution Time vs N (LIBSVM vs CVXOPT)- Scaled')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()