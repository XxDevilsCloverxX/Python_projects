from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA

# Generating and splitting the dataset
X, y = make_circles(n_samples=1000, factor=0.3, noise=0.05, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

# Visualizing the dataset
_, (train_ax, test_ax) = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(8, 4))

train_ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
train_ax.set_ylabel("Feature #1")
train_ax.set_xlabel("Feature #0")
train_ax.set_title("Training data")

test_ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
test_ax.set_xlabel("Feature #0")
test_ax.set_title("Testing data")
plt.show()

# PCA and KernelPCA transformation
pca = PCA(n_components=2)

#  gamma = 1/2(sigma)**2 : Relation of gamma to the variance {Gamma up = sigma down}
kernel_pca = KernelPCA(
    n_components=None, kernel="rbf", gamma=5, fit_inverse_transform=True, alpha=0.1
)
kernel_pca_cos = KernelPCA(
    n_components=None, kernel="cosine", fit_inverse_transform=True, alpha=0.1
)

X_test_pca = pca.fit(X_train).transform(X_test)
X_test_kernel_pca = kernel_pca.fit(X_train).transform(X_test)
X_test_kernel_pca_cos = kernel_pca_cos.fit(X_train).transform(X_test)

# Plotting
fig, ((orig_data_ax, pca_proj_ax), (kernel_pca_proj_ax, kernel_pca_proj_ax_cos)) = plt.subplots(
    nrows=2, ncols=2, figsize=(10, 10)
)

# Plot original data
orig_data_ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
orig_data_ax.set_ylabel("Feature #1")
orig_data_ax.set_xlabel("Feature #0")
orig_data_ax.set_title("Testing data")

# Plot PCA projection
pca_proj_ax.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test)
pca_proj_ax.set_ylabel("Principal component #1")
pca_proj_ax.set_xlabel("Principal component #0")
pca_proj_ax.set_title("Projection of testing data\n using PCA")

# Plot KernelPCA with RBF projection
kernel_pca_proj_ax.scatter(X_test_kernel_pca[:, 0], X_test_kernel_pca[:, 1], c=y_test)
kernel_pca_proj_ax.set_ylabel("Principal component #1")
kernel_pca_proj_ax.set_xlabel("Principal component #0")
kernel_pca_proj_ax.set_title("Projection of testing data\n using KernelPCA (RBF)")

# Plot KernelPCA with cosine projection
kernel_pca_proj_ax_cos.scatter(X_test_kernel_pca_cos[:, 0], X_test_kernel_pca_cos[:, 1], c=y_test)
kernel_pca_proj_ax_cos.set_ylabel("Principal component #1")
kernel_pca_proj_ax_cos.set_xlabel("Principal component #0")
kernel_pca_proj_ax_cos.set_title("Projection of testing data\n using KernelPCA (Cosine)")

plt.tight_layout()
plt.show()

# Reconstruction
X_reconstructed_pca = pca.inverse_transform(pca.transform(X_test))
X_reconstructed_kernel_pca = kernel_pca.inverse_transform(kernel_pca.transform(X_test))
X_reconstructed_kernel_pca_cos = kernel_pca_cos.inverse_transform(kernel_pca_cos.transform(X_test))

fig, ((orig_data_ax, pca_back_proj_ax), (kernel_pca_back_proj_ax, kernel_pca_back_proj_ax_cos)) = plt.subplots(
    nrows=2, ncols=2, sharex=True, sharey=True, figsize=(10, 10)
)

# Plot original data
orig_data_ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test)
orig_data_ax.set_ylabel("Feature #1")
orig_data_ax.set_xlabel("Feature #0")
orig_data_ax.set_title("Original test data")

# Plot PCA reconstruction
pca_back_proj_ax.scatter(X_reconstructed_pca[:, 0], X_reconstructed_pca[:, 1], c=y_test)
pca_back_proj_ax.set_xlabel("Feature #0")
pca_back_proj_ax.set_title("Reconstruction via PCA")

# Plot KernelPCA (RBF) reconstruction
kernel_pca_back_proj_ax.scatter(X_reconstructed_kernel_pca[:, 0], X_reconstructed_kernel_pca[:, 1], c=y_test)
kernel_pca_back_proj_ax.set_xlabel("Feature #0")
kernel_pca_back_proj_ax.set_title("Reconstruction via KernelPCA (RBF)")

# Plot KernelPCA (Cosine) reconstruction
kernel_pca_back_proj_ax_cos.scatter(X_reconstructed_kernel_pca_cos[:, 0], X_reconstructed_kernel_pca_cos[:, 1], c=y_test)
kernel_pca_back_proj_ax_cos.set_xlabel("Feature #0")
_ = kernel_pca_back_proj_ax_cos.set_title("Reconstruction via KernelPCA (Cosine)")

plt.tight_layout()
plt.show()
