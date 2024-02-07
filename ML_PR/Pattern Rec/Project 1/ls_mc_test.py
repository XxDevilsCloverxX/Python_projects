import numpy as np
import matplotlib.pyplot as plt

# generate some data
np.random.seed(0)
num_features = 2
num_samples = 150
# Generate data with points close to themselves but separated linearly
class1 = 0.75 * np.random.rand(num_samples//3, num_features)  # Scale down the values
class1 += 1 * np.ones(num_features)  # Adjust the mean

class2 = 0.75 * np.random.rand(num_samples//3, num_features)
class2 += 2 * np.ones(num_features)  # Adjust the mean

class3 = 0.75 * np.random.rand(num_samples//3, num_features)
class3 += 3 * np.ones(num_features)  # Adjust the mean

# Prepend a column of ones to each class
class1 = np.c_[np.ones((class1.shape[0], 1)), class1]
class2 = np.c_[np.ones((class2.shape[0], 1)), class2]
class3 = np.c_[np.ones((class3.shape[0], 1)), class3]

# Combine the classes
X = np.vstack((class1, class2, class3))
print(X)
print(X.shape)

# generate some labels
t1 = np.ones((num_samples//3, 1))
t2 = 2*np.ones((num_samples//3, 1))
t3 = 3*np.ones((num_samples//3, 1))
t = np.vstack((t1, t2, t3))
print(t, t.shape)

# One-hot encoding
num_classes = 3
T_one_hot = np.eye(num_classes)[t.flatten().astype(int) - 1]    # indexes the identity matrix by row (from the label) to get the encoded t(n)
print(T_one_hot)
print(T_one_hot.shape)

# Compute W - equivalent methods when full col rank
# W = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(T_one_hot)
W = np.linalg.pinv(X).dot(T_one_hot)
print(W, W.shape)   # verify this is (l + 1) x (M)

# classify the dataset using the W dot X (both Transpose + see notes)
prediction_matrix = W.T.dot(X.T)
print(prediction_matrix, prediction_matrix.shape)
# recreate the predictions as one hot encoding
P_one_hot = np.eye(num_classes)[np.argmax(prediction_matrix, axis=0)]
print(P_one_hot, P_one_hot.shape)

# Find misclassified points
misclassified_indices = np.where(~np.all(T_one_hot == P_one_hot, axis=1))[0]

# Print misclassified indices and count
print("Misclassified Indices:", misclassified_indices)
print("Number of Misclassified Points:", len(misclassified_indices))

# Extract misclassified points
misclassified_points = X[misclassified_indices]

# Print the misclassified points
print("Misclassified Points:\n", misclassified_points)
print(f"Training accuracy: {1 - len(misclassified_points)/num_samples}%")

# Plot decision boundaries
if num_features == 2:
    # Create the decision matrix for plottable LS multiclass
    x_line = np.linspace(X[:, 1].min(), X[:, 1].max(), num_samples)  # Assuming Feature 1 for x-axis

    # Calculate decision boundaries - points along the hyperplane in terms of feature 2
    d1 = -(W[0, 0] + W[1, 0]*x_line) / W[2,0]
    d2 = -(W[0, 1] + W[1, 1]*x_line) / W[2,1]
    d3 = -(W[0, 2] + W[1, 2]*x_line) / W[2,2]


    # Plot the decision boundaries
    plt.scatter(class1[:, 1], class1[:, 2], marker='x', color='red', label='Class 1')
    plt.scatter(class2[:, 1], class2[:, 2], marker='o', color='green', label='Class 2')
    plt.scatter(class3[:, 1], class3[:, 2], marker='^', color='blue', label='Class 3')
    
    plt.plot(x_line, d1, color='black', label='Decision Boundary 1')
    plt.plot(x_line, d2, color='orange', label='Decision Boundary 2')
    plt.plot(x_line, d3, color='purple', label='Decision Boundary 3')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()
elif num_features == 1:
    pass
else:
    print(f'{num_features} >= 2. Cannot Plot without 3D visualizer tools.')