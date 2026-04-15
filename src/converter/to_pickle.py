import os
import pickle
import numpy as np
from src.config import PICKLE_DIR


class PickleConverter:
    def __init__(self):
        os.makedirs(PICKLE_DIR, exist_ok=True)

    def save(self, keypoints: np.ndarray, video_name: str, label: int = None, output_subpath=""):
        """
        keypoints shape: (T, 86, 3)
        """
        output_dir = os.path.join(PICKLE_DIR, output_subpath)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{video_name}.pkl")

        data = {
            "video_name": video_name,
            "data": keypoints.astype("float32"),
            "label": label
        }

        with open(output_path, "wb") as f:
            pickle.dump(data, f)

        print(f"Pickle saved to {output_path}")