import tkinter as tk
from tkinter import filedialog, scrolledtext, Radiobutton
import os
from time import time
from ML_functions import *
from SMR import SoftMaxRegressor

def process_images(image_dir, img_size=(28, 28)):
    # List image files in directory
    image_files = []
    for file in os.listdir(image_dir):
        if file.endswith('.png') or file.endswith('.tif'):
            image_files.append(os.path.join(image_dir, file))

    # Process each image one at a time
    for file_path in image_files:
        # Read image
        img_tensor = read_test_image(file_path=file_path, img_size=img_size)
        yield img_tensor, file_path

class ImageBatchGeneratorGUI:
    def __init__(self, master):
        self.master = master
        master.title("Worm & Digits Evaluator")

        # Weights selection
        self.weights_var = tk.StringVar(value="Worm")
        self.radio_worm = Radiobutton(
            master, text="Worm Weights", variable=self.weights_var, value="Worm"
        )
        self.radio_digits = Radiobutton(
            master, text="Digits Weights", variable=self.weights_var, value="Digits"
        )
        self.radio_worm.grid(row=0, column=0, sticky="w")
        self.radio_digits.grid(row=1, column=0, sticky="w")

        # Directory selection
        self.dir_button = tk.Button(
            master, text="Select Directory", command=self.select_directory
        )
        self.dir_button.grid(row=2, column=0, sticky="ew")

        # Text window
        self.text_window = scrolledtext.ScrolledText(master, width=40, height=10)
        self.text_window.grid(row=0, column=1, rowspan=4, padx=10, pady=5)

        # Start
        self.start_button = tk.Button(master, text="Start", command=self.start_process)
        self.start_button.grid(row=3, column=0, sticky="ew")

    def select_directory(self):
        self.directory = filedialog.askdirectory()
        if self.directory:
            self.text_window.insert(tk.END, f"Selected Directory: {self.directory}\n")

    def start_process(self):
        self.text_window.insert(tk.END, "Predicting, please wait...\n")
        self.master.update()

        # Construct the path to the weights file.
        script_dir = os.path.dirname(os.path.abspath(__file__))
        worm_weights_path = os.path.join(script_dir, "worm_weights.npy")
        mnist_weights_path = os.path.join(script_dir, "mnist_weights.npy")
        weights_path = (
            worm_weights_path
            if self.weights_var.get() == "Worm"
            else mnist_weights_path
        )
        smr = SoftMaxRegressor(init_weights=weights_path)
        test_predictions = []
        filenames = []

        # Check if directory has been selected
        if not self.directory:
            self.text_window.insert(tk.END, "Error: Please select a directory first!\n")
            return

        # Determine if we should use worm logic based on user selection
        use_worm = self.weights_var.get() == "Worm"

        excel_file = (
            "predictions_sheets_WORM.xlsx"
            if use_worm
            else "predictions_sheets_MNIST.xlsx"
        )

        start_time = time()

        for img_tensor, file_path in process_images(self.directory):
            predictions = smr.predict(img_tensor)
            test_predictions.extend(predictions)
            filenames.append(os.path.basename(file_path))

        elapsed_time = time() - start_time

        test_predictions = np.array(test_predictions)
        filenames = np.array(filenames)

        # Define a custom sorting key function
        def natural_sort_key(s):
            import re

            return [
                int(text) if text.isdigit() else text.lower()
                for text in re.split("(\d+)", s)
            ]

        # Get the sorted indices based on natural sorting
        sorted_in = sorted(
            range(len(filenames)), key=lambda x: natural_sort_key(filenames[x])
        )

        test_predictions = test_predictions[sorted_in]
        filenames = filenames[sorted_in]

        outpath = write_predictions_to_excel(
            predictions=test_predictions, filenames=filenames, output_file=excel_file
        )

        self.text_window.insert(tk.END, f"Finished. Written outputs to {outpath}\n")
        self.text_window.insert(tk.END, f"Processed in {elapsed_time:.3f} seconds.\n")
        # self.text_window.insert(tk.END, f"Predictions: {test_predictions}\n")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageBatchGeneratorGUI(root)
    root.mainloop()