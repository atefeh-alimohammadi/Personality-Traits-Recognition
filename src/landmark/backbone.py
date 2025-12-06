#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import pickle

class Landmark(Dataset):
    def __init__(self, root_dir, labels_pickle, input_size):
        """
        Initialize the dataset loader for landmarks only.
        :param root_dir: Path to the root directory containing train/test/validation directories.
        :param labels_pickle: Path to the pickle file containing labels (not used here but kept for compatibility).
        :param input_size: Size to which images will be resized (e.g., 224 for 224x224).
        """
        self.root_dir = root_dir
        self.labels_pickle = labels_pickle  # This won't be used anymore
        self.input_size = input_size
        self.annotations = self._load_labels()  # This won't be used anymore, kept for consistency with original code

    def _load_labels(self):
        """
        Load labels from the pickle file. Here, we are not using them, but we still need to load them
        to keep the structure similar to the original code.
        :return: Dictionary mapping video names to labels.
        """
        # If you still want to load pickle but not use it, you can just return an empty dict
        with open(self.labels_pickle, 'rb') as f:
            labels_dict = pickle.load(f, encoding='latin1')
        return labels_dict

    def load_dataset(self):
        """
        Load the dataset by iterating through the directory structure.
        :return: A list of (landmark_tensor) tuples.
        """
        dataset = []

        # First loop: Iterate over main directories (e.g., train, test, validation)
        for main_dir in os.listdir(self.root_dir):
            main_dir_path = os.path.join(self.root_dir, main_dir)
            if not os.path.isdir(main_dir_path):
                continue  # Skip if it's not a directory

            # Second loop: Iterate over video directories inside each main directory
            for video_dir in os.listdir(main_dir_path):
                video_dir_path = os.path.join(main_dir_path, video_dir)
                if not os.path.isdir(video_dir_path):
                    continue  # Skip if it's not a directory

                frames = []
                landmarks = []

                # Third loop: Collect frames and landmarks inside each video directory
                for file_name in os.listdir(video_dir_path):
                    file_path = os.path.join(video_dir_path, file_name)
                    if file_name.startswith("face_landmark_") and file_name.endswith(".jpg"):
                        landmarks.append(file_path)

                # Sort landmarks by their frame number
                landmarks.sort(key=lambda x: int(os.path.basename(x).split("_")[2].split(".")[0]))

                # Ensure we have landmarks
                if not landmarks:
                    raise ValueError(f"No landmarks found in directory {video_dir_path}")

                # Process each landmark and convert to tensor
                for landmark_path in landmarks:
                    landmark_tensor = self._load_image_as_tensor(landmark_path)
                    dataset.append((landmark_tensor))  # Only append landmark tensor

        return dataset

    def _load_image_as_tensor(self, image_path):
        """
        Load and preprocess an image (landmark), then convert it to a PyTorch tensor.
        :param image_path: Path to the image.
        :return: A PyTorch tensor representing the image.
        """
        img = Image.open(image_path).convert('RGB')  # Open image and convert to RGB
        img = img.resize((self.input_size, self.input_size))  # Resize to required input size
        img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
        img_tensor = torch.tensor(img_array, dtype=torch.float32).permute(2, 0, 1)  # Convert to tensor and rearrange dimensions
        return img_tensor


# In[ ]:


# get_ipython().system('jupyter nbconvert --to script backbone.ipynb')

