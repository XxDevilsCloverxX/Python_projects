import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.datasets.mnist import load_data
import matplotlib.pyplot as plt
import numpy as np

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = load_data()

mnist1 = tf.convert_to_tensor(x_train[0])
mnist2 = tf.convert_to_tensor(x_test[0])

# Convert MNIST images to grayscale
mnist1 = tf.expand_dims(mnist1, axis=-1)
mnist2 = tf.expand_dims(mnist2, axis=-1)

# Convert the MNIST images to float32 and normalize
mnist1_normalized = tf.cast(mnist1, tf.float32) / 255.0
mnist2_normalized = tf.cast(mnist2, tf.float32) / 255.0

# Load the images from file
img_path1 = r"C:\Users\silas\Documents\vs-codespace\Python_Projects\ML_PR\Machine_Learning\Project 4\Celegans_ModelGen\1\image_17.png"
img_path0 = r"C:\Users\silas\Documents\vs-codespace\Python_Projects\ML_PR\Machine_Learning\Project 4\Celegans_ModelGen\0\image_17.png"

img1 = load_img(img_path1, color_mode='grayscale')
img0 = load_img(img_path0, color_mode='grayscale')

img1 = tf.image.adjust_gamma(img1, gamma=4/9)
img0 = tf.image.adjust_gamma(img0, gamma=4/9)
img1 = tf.image.adjust_gamma(img1, gamma=4/9)
img0 = tf.image.adjust_gamma(img0, gamma=4/9)

# Convert the images to TensorFlow tensors
img_tensor0 = tf.keras.preprocessing.image.img_to_array(img0)
img_tensor1 = tf.keras.preprocessing.image.img_to_array(img1)

# Normalize the pixel values to range [0, 1]
img_tensor0_normalized = img_tensor0 / 255.0
img_tensor1_normalized = img_tensor1 / 255.0

# Apply edge detection using a Sobel filter
edge_detector0 = tf.image.sobel_edges(tf.expand_dims(img_tensor0_normalized, axis=0))
edge_detector1 = tf.image.sobel_edges(tf.expand_dims(img_tensor1_normalized, axis=0))

edges0 = tf.norm(edge_detector0, axis=-1)
edges1 = tf.norm(edge_detector1, axis=-1)

# Compute the heatmap of pixel intensities
heatmap0 = np.mean(img_tensor0_normalized, axis=-1)
heatmap1 = np.mean(img_tensor1_normalized, axis=-1)

# Display the images, edges, and heatmaps for file images
plt.figure(figsize=(10, 10))

# Image 0 (Loaded from file)
plt.subplot(3, 3, 1)
plt.imshow(img0, cmap='gray')
plt.title('Image 0 (File)')

plt.subplot(3, 3, 2)
plt.imshow(edges0[0], cmap='gray')
plt.title('Edges 0 (File)')

plt.subplot(3, 3, 3)
plt.imshow(heatmap0, cmap='hot')
plt.title('Heatmap of Intensities 0 (File)')

# Image 1 (Loaded from file)
plt.subplot(3, 3, 4)
plt.imshow(img1, cmap='gray')
plt.title('Image 1 (File)')

plt.subplot(3, 3, 5)
plt.imshow(edges1[0], cmap='gray')
plt.title('Edges 1 (File)')

plt.subplot(3, 3, 6)
plt.imshow(heatmap1, cmap='hot')
plt.title('Heatmap of Intensities 1 (File)')

plt.tight_layout()
plt.show()

# Compute edges and heatmaps for MNIST images
edge_detector_mnist1 = tf.image.sobel_edges(tf.expand_dims(mnist1_normalized, axis=0))
edge_detector_mnist2 = tf.image.sobel_edges(tf.expand_dims(mnist2_normalized, axis=0))

edges_mnist1 = tf.norm(edge_detector_mnist1, axis=-1)
edges_mnist2 = tf.norm(edge_detector_mnist2, axis=-1)

heatmap_mnist1 = np.mean(mnist1_normalized.numpy(), axis=-1)
heatmap_mnist2 = np.mean(mnist2_normalized.numpy(), axis=-1)

# Display the MNIST images, edges, and heatmaps
plt.figure(figsize=(10, 10))

# MNIST Image 1
plt.subplot(3, 3, 1)
plt.imshow(mnist1[:, :, 0], cmap='gray')
plt.title('MNIST Image 1')

plt.subplot(3, 3, 2)
plt.imshow(edges_mnist1[0], cmap='gray')
plt.title('Edges MNIST 1')

plt.subplot(3, 3, 3)
plt.imshow(heatmap_mnist1, cmap='hot')
plt.title('Heatmap of Intensities MNIST 1')

# MNIST Image 2
plt.subplot(3, 3, 4)
plt.imshow(mnist2[:, :, 0], cmap='gray')
plt.title('MNIST Image 2')

plt.subplot(3, 3, 5)
plt.imshow(edges_mnist2[0], cmap='gray')
plt.title('Edges MNIST 2')

plt.subplot(3, 3, 6)
plt.imshow(heatmap_mnist2, cmap='hot')
plt.title('Heatmap of Intensities MNIST 2')

plt.tight_layout()
plt.show()
