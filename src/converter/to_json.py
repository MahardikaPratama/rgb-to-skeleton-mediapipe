"""
JSON Converter Module

This module provides functionality to export extracted skeleton keypoints
into a structured JSON format.
"""

import os
import json
import numpy as np
from src.config import JSON_DIR, JSON_INDENT


class JSONConverter:
    """
    Handles the conversion and saving of keypoints to JSON format.
    """

    def __init__(self):
        """
        Initializes the JSONConverter and ensures the base JSON output directory exists.
        """
        os.makedirs(JSON_DIR, exist_ok=True)

    def save(self, keypoints: np.ndarray, video_id: str, output_subpath: str = "") -> None:
        """
        Saves the extracted keypoints into a JSON file.

        The resulting JSON contains metadata about the frames, keypoints, and dimensions,
        along with the raw keypoints array converted to nested lists.

        Args:
            keypoints (np.ndarray): The extracted keypoints array of shape (T, 86, C).
            video_id (str): The standardized ID representing the video (e.g., P01_S001_R01).
            output_subpath (str, optional): A relative path to append to the base JSON_DIR 
                                            to maintain directory structures. Defaults to "".
        """
        output_dir = os.path.join(JSON_DIR, output_subpath)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{video_id}.json")

        data = {
            "video_id": video_id,
            "num_frames": int(keypoints.shape[0]),
            "num_keypoints": int(keypoints.shape[1]),
            "dimensions": int(keypoints.shape[2]),
            "keypoints": keypoints.tolist()
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=JSON_INDENT)

        print(f"JSON saved to {output_path}")