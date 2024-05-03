import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import MDS
from sklearn.decomposition import PCA

# Load the Swiss roll dataset
X, _ = datasets.make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)

# Plot the Swiss roll dataset in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=X[:, 2], cmap='rainbow')
ax.set_title('Swiss Roll Dataset in 3D')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()


# Perform Multidimensional Scaling (MDS) to project the data to 2D
mds = MDS(n_components=2, n_jobs=-1)
X_2d_mds = mds.fit_transform(X)

# Plot the transformed data in 2D using MDS
plt.figure(figsize=(8, 6))
plt.subplot(1,2,1)
plt.scatter(X_2d_mds[:, 0], X_2d_mds[:, 1], c=X[:, 2], cmap='rainbow')
plt.title('MDS: Swiss Roll Dataset projected to 2D')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.colorbar(label='Z')
plt.grid(True)

# Plot the rotated data in 2D using MDS
plt.subplot(1,2,2)
plt.scatter(X_2d_mds.T[1, :], X_2d_mds.T[0, :], c=X[:, 2], cmap='rainbow')
plt.title('MDS Rotated: Swiss Roll Dataset projected to 2D')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.colorbar(label='Z')
plt.grid(True)
plt.show()

# Perform Principal Component Analysis (PCA) to project the data to 2D
pca = PCA(n_components=2)
X_2d_pca = pca.fit_transform(X)

# Plot the transformed data in 2D using PCA
plt.figure(figsize=(8, 6))
plt.scatter(X_2d_pca[:, 0], X_2d_pca[:, 1], c=X[:, 2], cmap='rainbow')
plt.title('PCA: Swiss Roll Dataset projected to 2D')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.colorbar(label='Z')
plt.grid(True)
plt.show()
