import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA

# Generate angles from 0 to 2*pi
angles = np.linspace(0, 2*np.pi, 30)

def transform_show(n_comp, angles):
    np.random.seed(0)
    # Generate x and y coordinates for the circle
    x = np.cos(angles)
    y = np.sin(angles)

    # generate noisy datapoints
    x_noisy = x + np.random.normal(0, 0.2, x.size)
    y_noisy = y + np.random.normal(0, 0.2, y.size)
    
    # Combine x and y coordinates into a single array for noisy data
    data = np.column_stack((x_noisy, y_noisy))

    # configure transformations
    pca = PCA(n_components=n_comp)
    kpca_rbf = KernelPCA(n_components=n_comp, kernel='rbf')
    kpca_cos = KernelPCA(n_components=n_comp, kernel='cosine', )

    # transform the data
    transformed_pca = pca.fit_transform(data)
    transformed_kpca_rbf = kpca_rbf.fit_transform(data)
    transformed_kpca_cos = kpca_cos.fit_transform(data)

    plt.subplot(2,2,1)
    # Plot eigenvectors
    eigenvectors = pca.components_.T
    for i, eigenvector in enumerate(eigenvectors):
        plt.quiver(pca.mean_[0], pca.mean_[1], eigenvector[0], eigenvector[1], scale=3, color='red')
        plt.annotate(f'PCA {i+1}', (pca.mean_[0] + eigenvector[0], pca.mean_[1] + eigenvector[1]), color='red', fontsize=10, ha='center')


    plt.plot(x, y, color='blue', label='Intrinsic Manifold (1-D) r=1')  
    for i, txt in enumerate(range(len(data))):
        plt.annotate(txt, (data[i,0], data[i,1]), textcoords="offset points", xytext=(0,10), ha='center')
    plt.scatter(data[:,0], data[:, 1], c=np.arange(data.shape[0]), cmap='rainbow')
    plt.legend()
    plt.title('Input data')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.axis('equal')

    plt.subplot(2,2,2)
    plt.plot(x, y, color='blue', label='Intrinsic Manifold (1-D) r=1')  
    for i, txt in enumerate(range(len(transformed_pca))):
        plt.annotate(txt, (transformed_pca[i,0], transformed_pca[i,1]), textcoords="offset points", xytext=(0,10), ha='center')
        # Plot quiver for x-axis
    plt.quiver(0, 0, 1, 0, scale=5, color='red')
    # Plot quiver for y-axis
    plt.quiver(0, 0, 0, 1, scale=5, color='red')
    plt.scatter(transformed_pca[:, 0], transformed_pca[:, 1], c=np.arange(data.shape[0]), cmap='rainbow')
    plt.title('kPCA Plot (linear)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True)
    plt.axis('equal')

    plt.subplot(2,2,3)
    plt.plot(x, y, color='blue', label='Intrinsic Manifold (1-D) r=1')
    for i, txt in enumerate(range(len(transformed_kpca_rbf))):
        plt.annotate(txt, (transformed_kpca_rbf[i,0], transformed_kpca_rbf[i,1]), textcoords="offset points", xytext=(0,10), ha='center')
    plt.scatter(transformed_kpca_rbf[:, 0], transformed_kpca_rbf[:, 1], c=np.arange(data.shape[0]), cmap='rainbow')
    plt.title('kPCA Plot (rbf)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True)
    plt.axis('equal')

    plt.subplot(2,2,4)
    plt.plot(x, y, color='blue', label='Intrinsic Manifold (1-D) r=1')
    for i, txt in enumerate(range(len(transformed_kpca_cos))):
        plt.annotate(txt, (transformed_kpca_cos[i,0], transformed_kpca_cos[i,1]), textcoords="offset points", xytext=(0,10), ha='center')
    plt.scatter(transformed_kpca_cos[:, 0], transformed_kpca_cos[:, 1], c=np.arange(data.shape[0]), cmap='rainbow')
    plt.title('kPCA Plot (cos)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True)
    plt.axis('equal')

    plt.tight_layout()
    plt.show()

transform_show(2, angles)