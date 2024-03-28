import idx2numpy
import numpy as np

def load_mnist_images(file_path):
    return idx2numpy.convert_from_file(file_path)

def load_mnist_labels(file_path):
    return idx2numpy.convert_from_file(file_path)

def main():
    with open('train-images-idx3-ubyte', 'rb') as file:
        for line in file.readlines():
            print(line)

if __name__ == "__main__":
    main()
