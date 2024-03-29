import argparse
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

def process_image(image_path):
    # Read the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read the image at {image_path}.")
        return

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Otsu's thresholding to the grayscale image
    _, otsu = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Convert to TensorFlow tensor
    img = tf.convert_to_tensor(otsu, dtype=tf.float32)
    # Apply gamma adjustment
    img = tf.image.adjust_gamma(img, 4/9)
    img = tf.truediv(img, 255)
    
    return img

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Image Processing')
    parser.add_argument('-i', '--image', type=str, required=True, help='Path to the input image')
    args = parser.parse_args()

    # Process the image
    processed_image = process_image(args.image)

    # Display the processed image
    plt.imshow(processed_image, cmap='gray')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()
