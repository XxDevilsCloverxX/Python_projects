import os
import argparse
from PIL import Image

def rotate_images(directory):
    # Iterate over all files and subdirectories in the specified directory
    for root, _, files in os.walk(directory):
        for file in files:
            # Check if the file is an image (you might want to extend this check for other image formats)
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_path = os.path.join(root, file)
                image = Image.open(image_path)

                # Create rotated copies
                for i in range(1, 4):
                    rotated_image = image.rotate(90 * i)  # Rotate the image by 90 degrees * i
                    rotated_image_path = os.path.join(root, f"{os.path.splitext(file)[0]}_rot{i}{os.path.splitext(file)[1]}")
                    rotated_image.save(rotated_image_path)
                    print(f"Rotated image saved: {rotated_image_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create rotated copies of images in specified directory.")
    parser.add_argument("-d","--directory", type=str, help="Path to the directory containing images.")
    args = parser.parse_args()

    rotate_images(args.directory)
