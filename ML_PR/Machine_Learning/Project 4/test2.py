import numpy as np
from time import sleep

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

def guassian_phi(input_x:np.ndarray, k=int, sigma=0.3):
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
    # Compute the gradient of the loss function
    gradient = np.mean(designMatrix.T @ (scores - labels), axis=0)

    # Calculate the norm of the gradient
    # gradient_norm = np.linalg.norm(gradient)
    
    # # If the gradient norm exceeds the maximum allowed norm, scale down the gradient
    # if gradient_norm > 5000:
    #     gradient *= 5000 / gradient_norm
    return gradient 

def cross_entropy_cost(y_targets, scores):
    cost = 0
    samples, class_count = y_targets.shape
    # for every element
    for row in range(samples):
        cost -= np.sum(y_targets * np.log(scores))
    return cost
#############################################################
samples = 100
classes = 2
img_dim = 30

imgs = []
for n in range(samples):
    img = np.random.randint(low=0,high=255, size=(img_dim,img_dim))
    img = img.flatten()
    img = img / 255
    imgs.append(img)

imgs = np.array(imgs)
targets = np.array([np.eye(classes)[np.random.randint(0,2)] for i in range(samples)])

weights = np.zeros((1+img_dim**2, classes))
print(f'Initial weights {weights} {weights.shape}')
rho  = 1
costs = []
epochs = 10000
for i in range(epochs):
    print(f"Epoch {i}")
    indices = np.random.choice(imgs.shape[0], size=imgs.shape[0], replace=False)
    batch_x = imgs[indices]
    batch_phi = guassian_phi(batch_x, len(batch_x))
    batch_t = targets[indices]

    labels = []
    scores = []
    for img in batch_phi:
        pred = img@weights
        probs = softmax(pred)
        scores.append(probs.flatten())
        labels.append(np.eye(weights.shape[1])[np.argmax(probs)])

    scores = np.array(scores)
    labels = np.array(labels)
    print(f"Scores: {scores}, {scores.shape}")
    print(f"labels: {labels}, {labels.shape}")
    exit()
    print(np.count_nonzero((labels!=targets).flatten()), "Misclassed")

    # compute the cost for this inference
    # print(scores)
    cost = cross_entropy_cost(y_targets=targets, scores=scores)
    costs.append(cost)

    grad = cross_entropy_gradient(scores=scores, labels=targets, designMatrix=batch_phi)
    print(grad.shape)

    weights = weights + rho * grad
    print(f'Updated Weights: {weights}')
    #reduce learning rate
    rho *= 0.9
    sleep(1)
