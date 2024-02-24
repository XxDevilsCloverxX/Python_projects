"""
@Engineer: Silas Rodriguez
@Course: ECE 5363 - Pattern Recognition
@Date: Feb 18, 2024
"""
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from sklearn.svm import SVC

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
    # Seed the program for reproducibility
    np.random.seed(8)
    # Generate some data
    X, t = generate_linearly_nonseparable_data(N=25)

    # visualize the generated data
    plt.figure(figsize=(5,4))
    plt.scatter(X[:,0], X[:,1], c=t, cmap='coolwarm')
    plt.title('Data Plotted x1 vs x2')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.colorbar(label='Classes')
    plt.show()

    #################################################################################################
    # Dual Form solution for the dataset - 1/2 x' P x + qx st Gx <= h & Ax = b

    # copy the data
    dual_X= X.copy()
    # initialize a regularization parameter
    C = 10
    # extract some information from the data
    N, L = dual_X.shape
    y = np.where(t==0, 1, -1).astype('float64')

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
    sol = solvers.qp(P=P_cvxopt, q=q_cvxopt, G=G_cvxopt, h=h_cvxopt, A=A_cvxopt, b=b_cvxopt)

    # Extract the optimal solution
    lambdas = np.array(sol['x'])

    # w parameter in vectorized form
    weights = ((y * lambdas).T @ dual_X).reshape(-1,1)

    # Selecting the set of indices S corresponding to non zero parameters
    S = (lambdas > 1e-4).flatten()

    # Computing b and append to weights
    b = y[S] - np.dot(dual_X[S], weights)
    weights = np.vstack((weights, b.min()))

    # identify the support vectors
    support_vectors = dual_X[S]
    
    # Display results
    print("Optimal solution w (dual):")
    print(weights)
    print("Lambdas (dual):")
    print(lambdas[S])

    # plot the decision boundary and margins
    x_line = np.linspace(X[:, 0].min(), X[:, 0].max(), 100) # create a 100 evenly spaced points over x1
    y_line = -(x_line * weights[0] + weights[-1]) / weights[1]
    d = 1 / np.linalg.norm(weights[:-1])    # perpendicular distance from y_line to 1 decision boundary, no offset in this calc
    
    # applying trig, we can get the other two margins through the SV
    slope = weights[0] / weights[1]           # getting the slope of the decision boundary in terms of x2
    vert_offset = np.sqrt(1 + slope**2) * d   # proof here: https://www.geeksforgeeks.org/distance-between-two-lines/ x2 =y x1 =x for slope
    margin1 = y_line + vert_offset
    margin2 = y_line - vert_offset
    print(f'Margin width: 2/||w|| = {2 * d}')

    plt.figure(figsize=(8,6))
    plt.title('Data with d(x) + Margins - Dual')
    plt.scatter(support_vectors[:, 0], support_vectors[:, 1], facecolors='none', edgecolors='k', s=150, label='Support Vectors')    # highlight SVs
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

    # Test classifications:
    dual_X = np.c_[X, np.ones(X.shape[0])]
    print(dual_X.shape, weights.shape, t.shape)
    misclasses, accuracy = SVM_Evaluator(weights=weights, X=dual_X, t=t)
    print(f'{misclasses} misclasses with {accuracy}% accuracy!')

#######################################################################################################3
    #sklearn section
    sk_x = X.copy()

    clf = SVC(C = 10, kernel = 'linear')
    clf.fit(sk_x, y.ravel()) 

    print('w = ',clf.coef_)
    print('b = ',clf.intercept_)
    print('Indices of support vectors = ', clf.support_)
    print('Support vectors = ', clf.support_vectors_)
    print('Number of support vectors for each class = ', clf.n_support_)
    print('Coefficients of the support vector in the decision function = ', np.abs(clf.dual_coef_))

    # get the support vectors
    support_vectors = sk_x[clf.support_]
    # get the weights
    weights = clf.coef_
    intercept = clf.intercept_
    weights = np.append(weights, intercept).reshape(-1, 1)

    plt.figure(figsize=(8,6))
    plt.title('Data with d(x) + Margins - SkLearn')
    plt.scatter(support_vectors[:, 0], support_vectors[:, 1], facecolors='none', edgecolors='k', s=150, label='Support Vectors')    # highlight SVs
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

    # Test classifications:
    sk_x = np.c_[X, np.ones(X.shape[0])]
    print(sk_x.shape, weights.shape, t.shape)
    misclasses, accuracy = SVM_Evaluator(weights=weights, X=sk_x, t=t)
    print(f'{misclasses} misclasses with {accuracy}% accuracy!')

if __name__ == '__main__':
    main()
