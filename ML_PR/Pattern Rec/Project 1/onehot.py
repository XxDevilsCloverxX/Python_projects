import numpy as np
import matplotlib.pyplot as plt

# Set a seed for reproducibility
np.random.seed(42)

# Number of samples per class
num_samples = 100

# Class 1: Linearly separable
class1_mean = [2, 2]
class1_cov = [[1, 0], [0, 1]]
class1_data = np.random.multivariate_normal(class1_mean, class1_cov, num_samples)
class1_labels = np.ones(num_samples)

# Class 2: Randomly generated
class2_mean = [6, 6]
class2_cov = [[1, 0.5], [0.5, 1]]
class2_data = np.random.multivariate_normal(class2_mean, class2_cov, num_samples)
class2_labels = 2 * np.ones(num_samples)

# Class 3: Randomly generated
class3_mean = [8, 2]
class3_cov = [[1, -0.5], [-0.5, 1]]
class3_data = np.random.multivariate_normal(class3_mean, class3_cov, num_samples)
class3_labels = 3 * np.ones(num_samples)

# Combine Class 2 and Class 3
combined_class2_3_data = np.vstack((class2_data, class3_data))
combined_class2_3_labels = np.hstack((class2_labels, class3_labels))

# Change labels for Class 2 and Class 3 to -1
combined_class2_3_labels[combined_class2_3_labels != 1] = -1

# Plot the generated data
plt.scatter(class1_data[:, 0], class1_data[:, 1], label='Class 1', marker='o')
plt.scatter(class2_data[:, 0], class2_data[:, 1], label='Class 2', marker='x')
plt.scatter(class3_data[:, 0], class3_data[:, 1], label='Class 3', marker='^')
plt.title('Synthetic Separable Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# Plot the combined Class 2 & 3 data
plt.scatter(class1_data[:, 0], class1_data[:, 1], label='Class 1', marker='o')
plt.scatter(combined_class2_3_data[:, 0], combined_class2_3_data[:, 1], label='Class 2 & 3', marker='x')
plt.title('Combined Class 2 & 3 Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# Use Least Squares to find a separating line
X = np.vstack((class1_data, combined_class2_3_data))
y = np.hstack((class1_labels, combined_class2_3_labels))

# Add bias term to features
X_bias = np.c_[np.ones(X.shape[0]), X]
print(X_bias)
# Use least squares to find weights
weights = np.linalg.pinv(X_bias).dot(y)
print(weights)
# Plot the separating line
# x_line = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
# y_line = -(weights[0] + weights[1] * x_line) / weights[2]

# Plot the data and the separating line
plt.scatter(class1_data[:, 0], class1_data[:, 1], label='Class 1', marker='o')
plt.scatter(combined_class2_3_data[:, 0], combined_class2_3_data[:, 1], label='Class 2 & 3', marker='x')
plt.plot(X_bias[:, 1], -(weights[0] + weights[1] * X_bias[:, 1]) / weights[2], color='red', label='Least-Squares')
plt.title('Synthetic Linearly Separable Data with Corrected Separating Line')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()
