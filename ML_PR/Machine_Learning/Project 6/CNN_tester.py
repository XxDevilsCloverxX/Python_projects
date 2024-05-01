import os
import tkinter as tk
from tkinter import filedialog, scrolledtext
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.utils import image_dataset_from_directory

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
            batch_size=batch_size,
            image_size=image_size,
            color_mode="grayscale"  # Convert images to grayscale
        )

        # Initialize batch variables
        normal_predictions = []
        edge_predictions = []
        contrast_predictions = []
        filenames = []

        # Predict on the dataset in batches
        for batch in dataset:
            batch_images = batch[0]
            batch_filenames = batch[1]
            print(batch_images)
            # print(batch_images.shape)
            continue
            # Predict on the batch for each model
            normal_pred = normal_model.predict(batch_images)
            edge_pred = edge_model.predict(batch_images)
            contrast_pred = contrast_model.predict(batch_images)

            # Extend predictions into separate lists
            normal_predictions.extend(normal_pred)
            edge_predictions.extend(edge_pred)
            contrast_predictions.extend(contrast_pred)
            filenames.extend(batch_filenames)

        # Perform ensemble prediction
        all_pred = tf.stack((normal_predictions, edge_predictions, contrast_predictions))
        ensemble_pred = tf.round(tf.reduce_mean(all_pred, axis=0))

        # Write predictions to Excel file or do further processing
        self.text_window.insert(tk.END, "Finished prediction.\n")
        # You can perform further processing or save predictions here


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageBatchGeneratorGUI(root)
    root.mainloop()
