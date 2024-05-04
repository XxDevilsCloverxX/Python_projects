import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import MDS

# Load a sample dataset (e.g., the digits dataset)
data = load_digits()
X, y = data.data, data.target

print('Original Dimension of X = ', X.shape)
# Create an MDS model with the desired number of dimensions
# Number of dimensions for visualization
n_components = 2
mds = MDS(n_components=n_components)

# Fit the MDS model to your data
X_reduced = mds.fit_transform(X)
print('Dimension of X after MDS = ', X_reduced.shape)

# Visualize the reduced data
plt.figure(figsize=(8, 6))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap=plt.cm.get_cmap('jet', 10))
plt.colorbar(label='Digit Label', ticks=range(10))
plt.title("MDS Visualization of Digits Dataset")
plt.xlabel("MDS Dimension 1")
plt.ylabel("MDS Dimension 2")
plt.show()
