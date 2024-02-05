import numpy as np
import matplotlib.pyplot as plt

# Generate some random data for two classes with 2 features (introducing skewness)
np.random.seed(42)

# Generate 30 points for each class
num_points_per_class = 30
X_class0 = np.random.rand(num_points_per_class, 2) * 0.5  # Class 0 points closer together
X_class1 = 1 + np.random.rand(num_points_per_class, 2) * 0.5  # Class 1 points closer together

X = np.vstack((X_class0, X_class1))
y = np.array([0] * num_points_per_class + [1] * num_points_per_class).reshape(-1, 1)  # Use 0 and 1 for binary classification

# Add a bias term to the input features
X_b = np.c_[np.ones(X.shape[0]), X]  # Add bias term
print(X_b)
# Compute the least squares solution for binary classification
theta = np.linalg.pinv(X_b).dot(y)

# Plot the original data and the decision boundary
plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap=plt.cm.Paired, label='Original Data')

# Plot decision boundary
plt.plot(X[:, 0], -(theta[0] + X[:, 0] * theta[1]) / theta[2], color='black')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
