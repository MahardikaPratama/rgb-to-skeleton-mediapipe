"""
Excel Converter Module

This module provides functionality to export skeleton keypoints to Excel format.
It supports grouping multiple video outputs into a single Excel file per subject,
where each sheet represents a unique video.
"""

import os
import pandas as pd
import numpy as np
from src.config import EXCEL_DIR

class ExcelConverter:
    """
    Handles the conversion and saving of keypoints to Excel format.
    """

    def __init__(self):
        """
        Initializes the ExcelConverter and ensures the base Excel output directory exists.
        """
        os.makedirs(EXCEL_DIR, exist_ok=True)

    def save(self, keypoints: np.ndarray, video_id: str, output_subpath: str = "") -> None:
        """
        Saves the extracted keypoints into an Excel file.

        The structure aggregates all data for a single subject into one file (e.g., P01.xlsx),
        where each sheet contains the data for one specific video (e.g., P01_S001_R01).
        
        The resulting tabular format contains columns:
        [kode_video, frame, keypoint_id, x, y]

        Args:
            keypoints (np.ndarray): The extracted keypoints array of shape (T, 86, C).
            video_id (str): The standardized ID representing the video (e.g., P01_S001_R01).
            output_subpath (str, optional): A relative path to append to the base EXCEL_DIR 
                                            to maintain directory structures. Defaults to "".
        """
        # Ensure we only take (x,y)
        keypoints_xy = keypoints[:, :, :2].astype("float64")
        T, num_kp, dims = keypoints_xy.shape

        # Flatten structure: frame, keypoint_id, x, y
        # Using numpy advanced indexing to create columns quickly
        frames = np.repeat(np.arange(T), num_kp)
        kp_ids = np.tile(np.arange(num_kp), T)
        xs = keypoints_xy[:, :, 0].flatten()
        ys = keypoints_xy[:, :, 1].flatten()

        df = pd.DataFrame({
            'kode_video': video_id,
            'frame': frames,
            'keypoint_id': kp_ids,
            'x': xs,
            'y': ys
        })

        # Determine subject ID from video_id (e.g., P01 from P01_S001_R01)
        parts = video_id.split('_')
        subject_id = parts[0] if len(parts) >= 1 and parts[0].startswith('P') else "UNKNOWN"

        output_dir = os.path.join(EXCEL_DIR, output_subpath)
        os.makedirs(output_dir, exist_ok=True)
        excel_path = os.path.join(output_dir, f"{subject_id}.xlsx")

        # Append to existing excel or create new
        if os.path.exists(excel_path):
            with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                df.to_excel(writer, sheet_name=video_id, index=False)
        else:
            with pd.ExcelWriter(excel_path, engine='openpyxl', mode='w') as writer:
                df.to_excel(writer, sheet_name=video_id, index=False)

        print(f"Excel sheet saved: {video_id} -> {excel_path}")
