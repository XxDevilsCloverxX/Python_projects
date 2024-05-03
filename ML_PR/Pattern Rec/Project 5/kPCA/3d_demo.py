import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits, load_wine
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler

# Load Digits dataset
digits = load_digits()
X = digits.data
y = digits.target

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply kernel PCA with RBF kernel
kpca = KernelPCA(n_components=3, kernel='rbf')  # Changed number of components to 3
X_kpca = kpca.fit_transform(X_scaled)

# Plot the reduced data in a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_kpca[:, 0], X_kpca[:, 1], X_kpca[:, 2], c=y, cmap='rainbow', s=15)
plt.colorbar(scatter, ticks=np.arange(10), label='Digit Label')
ax.set_title('Digits Dataset - kPCA with RBF Kernel (3D)')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
plt.show()

# Load Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply kernel PCA with RBF kernel
kpca = KernelPCA(n_components=3, kernel='rbf')  # Changed number of components to 3
X_kpca = kpca.fit_transform(X_scaled)

# Plot the reduced data in a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_kpca[:, 0], X_kpca[:, 1], X_kpca[:, 2], c=y, cmap='rainbow', s=50)
plt.colorbar(scatter, ticks=np.arange(3), label='Wine Class')
ax.set_title('Wine Dataset - kPCA with RBF Kernel (3D)')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
plt.show()

