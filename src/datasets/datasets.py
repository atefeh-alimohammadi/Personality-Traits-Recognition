




import os
import pickle
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset



class FrameDataset(Dataset):
    def __init__(self, root_dir, labels_pickle, input_size, interval=10):
        """
        Initialize the dataset for loading frames, landmarks, and labels.
        :param root_dir: Path to the root directory containing train/test/validation directories.
        :param labels_pickle: Path to the pickle file containing labels.
        :param input_size: Size to which images will be resized (e.g., 224 for 224x224).
        :param interval: Interval for selecting frames and landmarks (e.g., 10 for selecting every 10th frame).
        """
        self.root_dir = root_dir
        self.input_size = input_size
        self.interval = interval  # Interval for selecting frames and landmarks
        self.annotations = self._load_labels(labels_pickle)
        self.data = self._load_data()
    
    def _load_labels(self, labels_pickle):
        """
        Load labels from the pickle file.
        :return: Dictionary mapping video names to labels.
        """
        with open(labels_pickle, 'rb') as f:
            labels_dict = pickle.load(f, encoding='latin1')
        return labels_dict
    

    def _load_data(self):
        """
        Load all video frames, corresponding landmarks, and labels.
        :return: A list of (frame_path, landmark_path, label_tensor) tuples.
        """
        data = []
    
        # Iterate over main directories (train/test/validation)
        for main_dir in os.listdir(self.root_dir):
            main_dir_path = os.path.join(self.root_dir, main_dir)
            if not os.path.isdir(main_dir_path):
                continue  # Skip non-directory files
    
            # Iterate over video directories inside each main directory
            for video_dir in os.listdir(main_dir_path):
                video_dir_path = os.path.join(main_dir_path, video_dir)
                if not os.path.isdir(video_dir_path):
                    continue  # Skip non-directory files
    
                frames = []
                landmarks = []
    
                # Collect all frames and landmarks in the video directory
                for file_name in os.listdir(video_dir_path):
                    file_path = os.path.join(video_dir_path, file_name)
                    if file_name.startswith("face_") and not file_name.startswith("face_landmark_") and file_name.endswith(".jpg"):
                        frames.append(file_path)
                    elif file_name.startswith("face_landmark_") and file_name.endswith(".jpg"):
                        landmarks.append(file_path)
    
                # Sort frames and landmarks by their frame number
                frames.sort(key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0]))
                landmarks.sort(key=lambda x: int(os.path.basename(x).split("_")[2].split(".")[0]))
    
                # Ensure the number of frames matches the number of landmarks
                if len(frames) != len(landmarks):
                    raise ValueError(f"Mismatch between frames and landmarks in directory {video_dir_path}")
    
                # Select every 10th frame and landmark
                frames = frames[::10]
                landmarks = landmarks[::10]
    
                # Extract video name for label lookup
                video_name = os.path.basename(video_dir)
    
                # Prepare label for this video
                label = []
                for trait in self.annotations.keys():
                    if trait != "interview":
                        trait_dict = self.annotations[trait]
                        file_name = video_name + ".mp4"  # Assuming video files are named like this
                        if file_name in trait_dict:
                            label.append(trait_dict[file_name])
                        else:
                            label.append(-1.0)  # Default value if label is not found
    
                # Convert label to tensor
                label_tensor = torch.tensor(label, dtype=torch.float32)
            
                # Add each frame, corresponding landmark, and label to the dataset
                for frame_path, landmark_path in zip(frames, landmarks):
                    data.append((frame_path, landmark_path, label_tensor))
    
        return data
        
    def _load_image_as_tensor(self, image_path):
        """
        Load and preprocess an image, then convert it to a PyTorch tensor.
        :param image_path: Path to the image.
        :return: A PyTorch tensor representing the image.
        """
        img = Image.open(image_path).convert('RGB')  # Open image and convert to RGB
        img = img.resize((self.input_size, self.input_size))  # Resize to required input size
        img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
        img_tensor = torch.tensor(img_array, dtype=torch.float32).permute(2, 0, 1)  # Convert to tensor and rearrange dimensions
        return img_tensor
   
    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset.
        :param idx: Index of the sample to retrieve.
        :return: A tuple of (frame_tensor, landmark_tensor, label_tensor).
        """
        frame_path, landmark_path, label_tensor = self.data[idx]
        frame_tensor = self._load_image_as_tensor(frame_path)
        landmark_tensor = self._load_image_as_tensor(landmark_path)
        return frame_tensor, landmark_tensor, label_tensor
     


