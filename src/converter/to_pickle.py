"""
Pickle Converter Module

This module provides functionality to export skeleton keypoints to a
Pickle (.pkl) file format. It integrates all video samples into a
single aggregated dataset file, which is highly optimized for model training.
"""

import os
import pickle
from typing import Tuple
import numpy as np
from src.config import PICKLE_DIR


class PickleConverter:
    """
    Handles the serialization and accumulation of keypoints to a Pickle dataset.
    """

    def __init__(self):
        """
        Initializes the PickleConverter and ensures the Pickle output directory exists.
        """
        os.makedirs(PICKLE_DIR, exist_ok=True)

    @staticmethod
    def _dataset_filename() -> str:
        """
        Returns the hardcoded dataset filename based on reference conventions.

        Returns:
            str: The target filename for the combined dataset.
        """
        return "pose_data_isharah2000_hands_lips_body_phase2_SI.pkl"

    def get_output_path(self, output_subpath: str = "", filename: str = None) -> str:
        """
        Constructs the absolute output path for the dataset file.
        Note: We ignore output_subpath to accumulate all data into a single Pickle file.

        Args:
            output_subpath (str, optional): Ignored.
            filename (str, optional): Custom filename for the dataset Pickle file.

        Returns:
            str: The absolute path pointing to the dataset Pickle file.
        """
        os.makedirs(PICKLE_DIR, exist_ok=True)
        target_name = filename or self._dataset_filename()
        return os.path.join(PICKLE_DIR, target_name)

    def save(
        self,
        keypoints: np.ndarray,
        video_id: str,
        label: int = None,
        output_subpath: str = "",
        filename: str = None,
    ) -> Tuple[str, str]:
        """
        Saves or updates the aggregated dataset with the extracted keypoints.

        The structure of the Pickle dictionary maintains the format:
        {
            "P1_S01_R1": {"keypoints": np.ndarray(T, 86, 2)},
            "P1_S01_R2": {"keypoints": np.ndarray(T, 86, 2)},
            ...
        }

        Args:
            keypoints (np.ndarray): The keypoints array of shape (T, 86, C). 
                                    Only the (x, y) coordinates are saved.
            video_id (str): The filename-based key representing the video (e.g., P1_S01_R1).
            label (int, optional): An optional integer class label. Defaults to None.
            output_subpath (str, optional): Optional sub-path. Defaults to "".
            filename (str, optional): Custom filename for the dataset Pickle file.

        Returns:
            Tuple[str, str]: A tuple containing the `sample_id` used as the dictionary key, 
                             and the absolute path of the Pickle file updated.
                             
        Raises:
            ValueError: If the keypoints shape is invalid or if the existing Pickle file is corrupted.
        """
        output_path = self.get_output_path(output_subpath, filename=filename)

        if keypoints.ndim != 3 or keypoints.shape[1] != 86:
            raise ValueError(f"Expected keypoints shape (T, 86, C), got {keypoints.shape}")

        # Match reference structure shape: (T, 86, 2)
        keypoints_xy = keypoints[:, :, :2].astype("float64")

        if os.path.exists(output_path):
            with open(output_path, "rb") as f:
                data = pickle.load(f)
            if not isinstance(data, dict):
                raise ValueError(f"Existing pickle root must be dict, got {type(data)}")
        else:
            data = {}

        # The pipeline passes the renamed video stem directly (e.g. P1_S01_R1)
        sample_id = video_id

        data[sample_id] = {
            "keypoints": keypoints_xy
        }

        tmp_path = output_path + ".tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump(data, f)
        os.replace(tmp_path, output_path)

        print(f"Pickle sample saved: {sample_id} -> {output_path}")
        return sample_id, output_path