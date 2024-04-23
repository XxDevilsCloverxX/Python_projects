# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# define functions

# compute a sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    return sigmoid(x) * (1-sigmoid(x))

# tanh units
def tanh(x):
    return np.tanh(x)

# derivative of tanh
def d_tanh(x):
    return 1 - np.tanh(x)**2

# MSE cost
def MSE(t, y):
    return 1/2 * np.mean((y-t)**2)

# Binary Cross entropy loss
def Binary_Cross_Entropy_loss(t, y):
    return -np.mean(t*np.log(y)+(1-t)*np.log(1-y))

# Model Arch Functions

# initialize model arch
def init_params(num_in, num_h, num_out):
    W1 = np.random.randn(num_in, num_h)
    b1 = np.zeros((num_h, 1))
    W2 = np.random.randn(num_h, num_out)
    b2 = np.zeros((num_out, 1))
    return W1, b1, W2, b2
    
# forward propogate a batch of data
def Fprop(X, W1, b1, W2, b2, clf=True):
    Z0 = X.T # D x N
    A1 = W1.T @ Z0 + b1 # (D x num_hidden).T x (D x N) = num_hidden x N
    # activation func
    Z1 = tanh(A1)
    A2 = W2.T @ Z1 + b2 # (num_h x num_out).T x (num_h x N) = num_out x N
    # activation func
    Z2 = sigmoid(A2) if clf else A2 # num_out x N
    return A1, Z1, A2, Z2

def train(X,Y, num_in,num_h,num_out, loss, clf:bool, epochs=1000, rate=0.1):
    W1, b1, W2, b2 = init_params(num_in,num_h,num_out)
    costs = []

    plt.ion()  # Turn on interactive mode
    plt.figure()  # Create a new figure

    for i in range(epochs):
        # Forward pass the data
        A1, Z1, A2, Z2 = Fprop(X, W1, b1, W2, b2, clf)
        # Measure the cost of the prediction
        cost = loss(Y, Z2)
        costs.append(cost)

        # # classification 100%
        # if np.all(np.round(Z2)==y):
        #     print("100% Acc")
        #     break

        # Compute gradients for output layer
        dZ2 = Z2 - Y
        dA2 = (dZ2 * d_sigmoid(A2)) if clf else dZ2
        dW2 = (1/Y.size) * (dA2 @ Z1.T).T
        db2 = (1/Y.size) * np.sum(dA2, axis=1, keepdims=True)
        # Compute gradients for hidden layer
        dZ1 = (dA2.T @ W2.T).T
        dA1 = dZ1 * d_tanh(A1)
        dW1 = (1/Y.size) * (dA1 @ X).T
        db1 = (1/Y.size) * np.sum(dA1, axis=1, keepdims=True)
    
        # update the weights
        W2 -= rate*dW2
        W1 -= rate*dW1
        b1 -= rate*db1
        b2 -= rate*db2

        # Plot the cost in 'real-time'
        if i % 10 == 0:
            plt.clf()  # Clear previous plot
            plt.plot(costs)
            plt.xlabel('Epoch')
            plt.ylabel(f'Avg/ Cost {loss.__name__}')
            plt.title('Avg Cost vs Epoch')
            plt.pause(0.001)  # Pause to update plot
            
    plt.ioff()  # Turn off interactive mode
        
    return W1, b1, W2, b2, costs

#######################################################################
if __name__ == '__main__':
    np.random.seed(666)
    # create the XOR data
    X = np.array((
        (-1,-1),
        (1,1),
        (1,-1),
        (-1,1)
    ))
    y = np.array((0,0,1,1))
    
    # train the network
    W1, b1, W2, b2, costs = train(X,y,2,2,1, loss=Binary_Cross_Entropy_loss, clf=True)

    # Generate grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                        np.arange(y_min, y_max, 0.1))

    # Flatten grid points
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Forward pass for each point on the grid
    _, _, _, Z2 = Fprop(grid_points, W1, b1, W2, b2, clf=True)
    _, _, _, ZTrain = Fprop(X, W1, b1, W2, b2, clf=True)

    # Reshape predictions to match the grid and round to get binary decisions
    Z = np.round(Z2.reshape(xx.shape))

    # Plot decision surface with 'coolwarm' colormap
    fig = plt.figure(figsize=(12, 6))

    # 3D surface plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(xx, yy, Z, alpha=0.5, cmap='coolwarm')
    ax1.scatter(X[:, 0], X[:, 1], np.round(ZTrain), color='black')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.set_zlabel('Output')
    ax1.set_title('3D Decision Surface')

    # 2D contour plot
    ax2 = fig.add_subplot(122)
    ax2.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
    ax2.scatter(X[:, 0][y==0], X[:, 1][y==0], marker='o', edgecolors='black')
    ax2.scatter(X[:, 0][y==1], X[:, 1][y==1], marker='s', edgecolors='black')
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    ax2.set_title('2D Decision Contour')

    plt.tight_layout()

    # solve the regression problem
    X = pd.read_excel('Proj5Dataset.xlsx').to_numpy()
    np.random.shuffle(X)
    y = X[:, -1]
    X = X[:,:-1]

    x_test = np.linspace(np.min(X), np.max(X)).reshape(-1,1)

    W1, b1, W2, b2, costs3 = train(X,y,1,3,1, loss=MSE, clf=False)
    _,_,_,ztest3 = Fprop(x_test, W1, b1, W2, b2, clf=False)

    W1, b1, W2, b2, costs20 = train(X,y,1,20,1, loss=MSE, clf=False)
    _,_,_,ztest20 = Fprop(x_test, W1, b1, W2, b2, clf=False)

    plt.figure()
    plt.scatter(X, y, label='Samples', marker='s', edgecolors='black')
    plt.plot(x_test, ztest3.T, label=f'3 units - Cost = {costs3[-1]:.3f}')
    plt.plot(x_test, ztest20.T, label=f'20 units - Cost = {costs20[-1]:.3f}')

    plt.legend()
    plt.show()