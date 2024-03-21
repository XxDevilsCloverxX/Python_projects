import numpy as np
import matplotlib.pyplot as plt

def gaussian_phi(input_x: np.ndarray, k=int, sigma=0.3):
    center_indices = np.random.choice(input_x.shape[0], k, replace=False)
    centers = input_x[center_indices]

    # Compute exponents
    distances_squared = np.sum((input_x[:, None] - centers) ** 2, axis=1)
    exponents = -distances_squared / (2 * sigma ** 2)
    
    # Compute Gaussian basis functions
    phi = np.exp(exponents)
    
    # Prepend a column of ones for bias term
    phi = np.concatenate((np.ones((phi.shape[0], 1)), phi), axis=1)
    
    return phi

def cross_entropy_gradient(scores: np.ndarray, labels: np.ndarray, designMatrix: np.ndarray) -> np.ndarray:
    """
    Compute the gradient of the iteration of the cost function.

    Args:
        scores (np.ndarray): Predicted scores for each class.
        labels (np.ndarray): One-hot encoded target labels.
        designMatrix (np.ndarray): Design matrix or input features.

    Returns:
        np.ndarray: Gradient of the loss function with respect to the weights.
    """
    # Compute difference NxK
    differ = scores - labels
    print('differ: ', differ)
    N = scores.shape[0]

    # Multiply by design matrix to get the gradient with respect to weights : (NxL)^T = L x N dot N x K => L x K gradient
    gradient = (designMatrix.T @ differ)
    return gradient

def softmax(input_vector):

    # Subtract the maximum value from each element to prevent overflow
    max_value = np.max(input_vector)
    centered_input = input_vector - max_value

    # Calculate the exponent of each element in the input vector
    exponents = np.exp(centered_input)

    # Sum of exponents along the columns
    sum_of_exponents = np.sum(exponents)

    # divide the exponent of each value by the sum of the exponents
    probabilities = exponents / sum_of_exponents

    return probabilities

def cross_entropy_cost(y_targets, scores):
    """
    y_targets & scores are N x K
    """
    avgcost = -np.mean(np.sum(y_targets * np.log(scores), axis=1))
    return avgcost
#############################################################
samples = 100
classes = 2
img_dim = 30
epochs = 500
rho = 1 
alpha = 5
imgs = []
for n in range(samples):
    img = np.random.randint(low=0,high=255, size=(img_dim,img_dim))
    img = img.flatten()
    img = img / 255
    imgs.append(img)

imgs = np.array(imgs)
targets = np.array([np.eye(classes)[np.random.randint(0,2)] for i in range(samples)])

phi = gaussian_phi(imgs, imgs.shape[0])

# initial weights
weights = np.random.uniform(low=0, high=1, size=(phi.shape[1], classes))

costs = []
for i in range(epochs):
    print(f'Epoch {i}...')    
    # apply softmax
    dot = phi@weights
    probs = []
    for row in dot:
        probs.append(softmax(row))
    probs = np.array(probs)
 
    # compute gradient:
    grad = cross_entropy_gradient(scores=probs, labels=targets, designMatrix=phi)
    
    weights = weights - (rho * grad + alpha * weights)
    rho *=0.9

    avg_cost = cross_entropy_cost(y_targets=targets, scores=probs)
    costs.append(avg_cost)
    if i >= 10 and np.abs(costs[-1] - costs[-2]) < 1e-3:
        print('Loss function convergence')
        print(weights)
        break
    
# compute training predictions
# apply softmax
dot = phi@weights
probs = []
for row in dot:
    probs.append(softmax(row))
probs = np.array(probs)
pred_labels = [np.eye(targets.shape[1])[np.argmax(row)] for row in probs]

misclassifications =  np.count_nonzero((pred_labels != targets).flatten())
print(f"{misclassifications} train misclassifications with accuracy {100*(1 - misclassifications/targets.shape[0])}%")

x = np.arange(i+1)
plt.plot(x, costs)
plt.title('Training Error')
plt.ylabel('Cost')
plt.xlabel('Epoch')
plt.show()

test_imgs = []
for n in range(samples):
    img = np.random.randint(low=0,high=255, size=(img_dim,img_dim))
    img = img.flatten()
    img = img / 255
    test_imgs.append(img)

test_imgs = np.array(test_imgs)
test_targets = np.array([np.eye(classes)[np.random.randint(0,2)] for i in range(samples)])
test_phi = gaussian_phi(test_imgs, test_imgs.shape[0])

dot = test_phi@weights
probs = []
for row in dot:
    probs.append(softmax(row))
probs = np.array(probs)
pred_labels = [np.eye(test_targets.shape[1])[np.argmax(row)] for row in probs]

misclassifications =  np.count_nonzero((pred_labels != test_targets).flatten())
print(f"{misclassifications} test misclassifications with accuracy {100*(1 - misclassifications/test_targets.shape[0])}%")