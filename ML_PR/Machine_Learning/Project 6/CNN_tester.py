import os
import tkinter as tk
from tkinter import filedialog, scrolledtext
import tensorflow as tf
from time import time
from tensorflow.keras.models import load_model # type: ignore
from keras.utils import image_dataset_from_directory
from CNN_functions import *

class ImageBatchGeneratorGUI:
    def __init__(self, master):
        self.master = master
        master.title("Worm Image Evaluator")

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
        edge_path = os.path.join(script_dir, "edge.keras")
        normal_path = os.path.join(script_dir, "raw.keras")
        contrast_path = os.path.join(script_dir, "contrast.keras")

        # Check if directory has been selected
        if not self.directory:
            self.text_window.insert(tk.END, "Error: Please select a directory first!\n")
            return

        # load the models
        try:
            normal_model = load_model(normal_path)
            edge_model = load_model(edge_path)
            contrast_model = load_model(contrast_path)
        except Exception as e:
            print(f'{e}')
            self.text_window.insert(tk.END, "Error: Model NOT found!\n")
            return

        # Create image dataset from directory
        image_size = (30, 30)
        batch_size = 64
        dataset = image_dataset_from_directory(
            self.directory,
            labels="inferred",
            label_mode=None,
            batch_size=None,
            image_size=image_size,
            color_mode="grayscale"  # Convert images to grayscale
        )

        # Initialize batch variables
        filenames = dataset.file_paths

        # raw input predictions
        start = time()
        normal_dataset = dataset.map(preprocess_original)
        normal_dataset = normal_dataset.batch(batch_size)
        normal_predictions = normal_model.predict(normal_dataset)        

        # edge detect predictions
        edge_dataset = dataset.map(preprocess_sobel)
        edge_dataset = edge_dataset.batch(batch_size)
        edge_predictions = edge_model.predict(edge_dataset)
        
        # contrast predictions
        contrast_dataset = dataset.map(lambda x: preprocess_contrast(x,factor=2))
        contrast_dataset = contrast_dataset.batch(batch_size)
        contrast_predictions = contrast_model.predict(contrast_dataset)

        elapsed_time = time() - start

        # Perform ensemble prediction
        all_pred = tf.stack((normal_predictions, edge_predictions, contrast_predictions))
        ensemble_pred = tf.round(tf.reduce_mean(all_pred, axis=0))
        ensemble_pred = tf.squeeze(ensemble_pred)   # remove dim 1 from shapes
        outpath = write_predictions_to_excel(
            predictions=ensemble_pred, filenames=filenames, output_file="Worms_Pred_CNN.xlsx"
        )

        self.text_window.insert(tk.END, f"Finished. Written outputs to {outpath}\n")
        self.text_window.insert(tk.END, f"Processed in {elapsed_time/60:.3f} mins.\n")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageBatchGeneratorGUI(root)
    root.mainloop()
