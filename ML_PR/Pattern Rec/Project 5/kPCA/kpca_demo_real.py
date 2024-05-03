import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits, load_wine
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap

# Load Digits dataset
digits = load_digits()
X = digits.data
y = digits.target

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply kernel PCA with RBF kernel
kpca = KernelPCA(n_components=2, kernel='rbf')
X_kpca = kpca.fit_transform(X_scaled)

# Define a custom colormap (rainbow)
colors = plt.cm.rainbow(np.linspace(0, 1, len(np.unique(y))))
custom_cmap = ListedColormap(colors)

# Plot the reduced data in a scatter plot
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=y, cmap=custom_cmap, s=15)
plt.colorbar(scatter, ticks=np.arange(10), label='Digit Label')
plt.title('Digits Dataset - kPCA with RBF Kernel')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Load Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply kernel PCA with RBF kernel
kpca = KernelPCA(n_components=2, kernel='rbf')
X_kpca = kpca.fit_transform(X_scaled)

# Plot the reduced data in a scatter plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=y, cmap='rainbow', s=50)
plt.colorbar(scatter, ticks=np.arange(3), label='Wine Class')
plt.title('Wine Dataset - kPCA with RBF Kernel')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
