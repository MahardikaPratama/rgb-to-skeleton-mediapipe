import os
import json
import numpy as np
from src.config import JSON_DIR, JSON_INDENT


class JSONConverter:
    def __init__(self):
        os.makedirs(JSON_DIR, exist_ok=True)

    def save(self, keypoints: np.ndarray, video_name: str, output_subpath=""):
        """
        keypoints shape: (T, 86, 3)
        """
        output_dir = os.path.join(JSON_DIR, output_subpath)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{video_name}.json")

        data = {
            "video_name": video_name,
            "num_frames": int(keypoints.shape[0]),
            "num_keypoints": int(keypoints.shape[1]),
            "dimensions": int(keypoints.shape[2]),
            "keypoints": keypoints.tolist()
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=JSON_INDENT)

        print(f"JSON saved to {output_path}")