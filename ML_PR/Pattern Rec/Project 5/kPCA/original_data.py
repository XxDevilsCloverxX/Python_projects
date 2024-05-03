from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine, load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Wine dataset
data = load_wine()
X = data.data

scaler = StandardScaler()
x_scaled = scaler.fit_transform(X)

# Initialize PCA
pca = PCA()

# Fit PCA
X_pca = pca.fit_transform(x_scaled)

# Obtain eigenvalues
eigenratios = pca.explained_variance_ratio_
print(eigenratios)

# Plot scree plot
sns.barplot(eigenratios)
plt.show()

# Load Digits dataset
digits_data = load_digits()

# Extract images and labels
images = digits_data.images
labels = digits_data.target

# Display a few sample images
num_images_to_display = 5
fig, axes = plt.subplots(1, num_images_to_display, figsize=(12, 4))

for i in range(num_images_to_display):
    axes[i].imshow(images[i], cmap='gray')
    axes[i].set_title(f"Label: {labels[i]}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()
