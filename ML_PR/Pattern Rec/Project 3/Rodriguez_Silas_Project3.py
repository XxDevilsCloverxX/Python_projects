"""
@Engineer: Silas Rodriguez
@Course: ECE 5363 - Pattern Recognition
@Date: March 15th, 2024
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from sklearn.svm import SVC

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

def get_rbf_kernel(sigma:float):
    """
    returns the RBF kernel map for 2 vectors
    """
    return lambda x1,x2: np.exp(-np.linalg.norm(x1-x2)**2/(2*sigma**2))

def SoftMargin_SVM(X:np.ndarray, t:np.ndarray, C:float, sigma:float=1.75, plot_en:bool=True):
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
    y = t.copy()

    # get the H that is used to compute P
    rbf_kern = get_rbf_kernel(sigma=sigma)
    K = np.zeros((N,N))
    for i in range(len(dual_X)):
        for j in range(len(dual_X)):
            K[i,j] = rbf_kern(dual_X[i], dual_X[j])

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

    # Solve the quadratic programming problem
    solvers.options['show_progress'] = False
    sol = solvers.qp(P=P_cvxopt, q=q_cvxopt, G=G_cvxopt, h=h_cvxopt, A=A_cvxopt, b=b_cvxopt)

    # Extract the optimal solution
    lambdas = np.array(sol['x'])

    # Support vectors have non zero lagrange multipliers
    sv = (lambdas > 1e-3).flatten()
    ind = np.arange(len(lambdas))[sv]
    a_sv = lambdas[sv].ravel()
    X_sv = dual_X[sv]
    y_sv = y[sv].ravel()

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

    ## This predicts the training data
    predictions = np.zeros(N)
    for n, sample in enumerate(dual_X):
        prediction = 0
        for alpha, label, sv in zip(a_sv, y_sv, X_sv):
            prediction += alpha * label * rbf_kern(sv, sample)
        prediction += b
        predictions[n] = np.sign(prediction)
    predictions=predictions.reshape(-1,1)
    misclasses, index_misclasses, accuracy = SVM_Evaluator(predictions=predictions, t=y)
    print(f"{misclasses} misclassifications with {accuracy:.2f} % training accuracy!")

    if plot_en:
        # Plot decision surfaces and support vectors
        plt.figure(figsize=(8, 6))
        plt.scatter(dual_X[:, 0], dual_X[:, 1], c=t, cmap='coolwarm', s=20, edgecolors='k')
        plt.colorbar(label='Classes', ticks=[-1, 0, 1])  # Add colorbar
        plt.scatter(X_sv[:, 0], X_sv[:, 1], facecolors='none', edgecolors='k', s=150,
                    label=f'Support Vectors: {len(X_sv)}')  # highlight SVs
        # Box misclassified points
        misclassified_points = dual_X[index_misclasses]
        plt.scatter(misclassified_points[:, 0], misclassified_points[:, 1], marker='s',
                     edgecolors='black', facecolors='none', s=150, label=f'Misclassified Vectors: {misclasses}')
        
        # Plot decision surfaces
        x_min, x_max = dual_X[:, 0].min() - 1, dual_X[:, 0].max() + 1
        y_min, y_max = dual_X[:, 1].min() - 1, dual_X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

        Z = np.zeros_like(xx)
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                sample = np.array([xx[i, j], yy[i, j]])
                prediction = b
                for alpha, label, sv in zip(a_sv, y_sv, X_sv):
                    prediction += alpha * label * rbf_kern(sv, sample)
                Z[i, j] = (prediction)

        # Plot decision surfaces filled with class colors
        plt.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.4)

        # Plot +1 and -1 hyperplanes
        plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=[':', '-', '--'])

        plt.legend()
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title(f'Soft Margin SVM Decision Surfaces with C={C}, sigma={sigma}')
        plt.show()

    return X_sv, a_sv, b, y_sv

def SkLearn_SVM(X:np.ndarray, t:np.ndarray, C:float, gamma:float=8/49, plot_en:bool=True):
    """
    @purpose:
        - Calculate the Soft margin SVM using the sklearn solution
    @params:
        - X - input matrix
        - t - target labels
        - C - slack variable
        - gamma - 1/2(sigma)**2 : default for sigma = 1.75
    @return:
        - weights: weights and bias from the SVM solution
        - time_end - time_start : elapsed time to compute the SVM solution
    """
    sk_x = X.copy()
    sk_y = t.copy()
    clf = SVC(C=C, kernel='rbf', gamma=gamma)
    clf.fit(sk_x, sk_y.ravel()) 

    # get the support vectors
    support_vectors = sk_x[clf.support_]
    dual_coefs = clf.dual_coef_
    n_support = clf.n_support_
    intercept = clf.intercept_

    predictions = clf.predict(sk_x).reshape(-1, 1)
    misclasses, index_misclasses, accuracy = SVM_Evaluator(predictions=predictions, t=sk_y)
    
    if plot_en:
        print('b = ', clf.intercept_)
        print('Indices of support vectors = ', clf.support_)
        print('Support vectors = ', clf.support_vectors_)
        print('Number of support vectors for each class = ', clf.n_support_)
        print('Coefficients of the support vector in the decision function = ', np.abs(clf.dual_coef_))
        print(f'{misclasses} misclasses with {accuracy:.2f}% accuracy!')

        # Plot decision surfaces and support vectors
        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], X[:, 1], c=t, cmap='coolwarm', s=20, edgecolors='k')
        plt.colorbar(label='Classes', ticks=[-1,0,1])  # Add colorbar
        plt.scatter(support_vectors[:, 0], support_vectors[:, 1], facecolors='none', edgecolors='k', s=150, label=f'Support Vectors: {len(support_vectors)}')    # highlight SVs

        # Plot decision surfaces
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')

        # Plot +1 and -1 hyperplanes
        plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=[':', '-', '--'])

        # Box misclassified points
        misclassified_points = X[index_misclasses]
        plt.scatter(misclassified_points[:, 0], misclassified_points[:, 1], marker='s', edgecolors='black', facecolors='none', s=150, label=f'Misclassified Vectors: {misclasses}')

        plt.legend()
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title(f'SkLearn SVM Decision Surfaces with C={C}, sigma={1.75}')
        plt.show()

    return support_vectors, dual_coefs, intercept, n_support


def main():

    # open the dataset
    dataset = pd.read_excel('Proj2DataSet.xlsx')
    data_np = dataset.to_numpy()

    X = data_np[:, :-1]
    t = data_np[:, -1].reshape(-1,1)

    # # visualize the generated data
    # plt.figure(figsize=(8,6))
    # plt.scatter(X[:,0], X[:,1], c=t, cmap='coolwarm')
    # plt.title('Data Plotted x1 vs x2')
    # plt.xlabel('x1')
    # plt.ylabel('x2')
    # plt.colorbar(label='Classes')

    #########################################################################
    # soft margin SVM with C = 10
    SoftMargin_SVM(X = X, t=t, C=10)
    print()
    #########################################################################
    # # soft margin SVM with C = 100
    SkLearn_SVM(X = X, t=t, C=10)
    print()
    # #########################################################################
    # # sklearn solution for C = 0.1
    # SkLearn_SVM(X = X, t= t, C=0.1)
    # print()
    # #########################################################################
    # # sklearn solution for C = 100
    # SkLearn_SVM(X = X, t= t, C=100)
    # print()    
    # plt.show()

if __name__ == '__main__':
    main()
