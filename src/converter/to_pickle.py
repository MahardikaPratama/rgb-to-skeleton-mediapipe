import os
import pickle
import numpy as np
from src.config import PICKLE_DIR


class PickleConverter:
    def __init__(self):
        os.makedirs(PICKLE_DIR, exist_ok=True)

    def save(self, keypoints: np.ndarray, video_name: str, label: int = None):
        """
        keypoints shape: (T, 86, 3)
        """

        output_path = os.path.join(PICKLE_DIR, f"{video_name}.pkl")

        data = {
            "video_name": video_name,
            "data": keypoints.astype("float32"),
            "label": label
        }

        with open(output_path, "wb") as f:
            pickle.dump(data, f)

        print(f"Pickle saved to {output_path}")