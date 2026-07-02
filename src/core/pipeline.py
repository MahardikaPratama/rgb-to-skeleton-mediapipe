"""
Core Pipeline Module

This module orchestrates the extraction of keypoints from video files
and dispatches the results to various converters.
"""

import os
import time
from datetime import timedelta
from pathlib import Path
import numpy as np
import cv2

import pandas as pd
from src.config import PICKLE_DIR, PROJECT_ROOT
from src.extractor.holistic_86 import Holistic86Extractor
from src.converter.to_pickle import PickleConverter
from src.converter.to_excel import ExcelConverter

SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


class SkeletonPipeline:
    """
    Main orchestration class for the RGB-to-Skeleton workflow.
    """
    
    def __init__(self, save_pickle=True, save_excel=True, pickle_filename: str = None):
        """
        Initializes the pipeline with desired output targets.

        Args:
            save_pickle (bool): Whether to export the dataset to a Pickle file.
            save_excel (bool): Whether to export to Excel file formats.
            pickle_filename (str, optional): Custom filename for the aggregated Pickle output.
        """
        self.extractor     = Holistic86Extractor()
        self.pickle_conv   = PickleConverter()
        self.excel_conv    = ExcelConverter()

        self.save_pickle   = save_pickle
        self.save_excel    = save_excel
        self.last_pickle_sample_id = None
        self.last_pickle_path = None
        self.pickle_filename = pickle_filename or "pose_bisindo"
        self.start_time = time.time()

        # Load split mapping for train_dev / test separation
        self.split_mapping = self._load_split_mapping()

        os.makedirs(PICKLE_DIR, exist_ok=True)

    def _elapsed(self) -> str:
        """Returns the elapsed time since pipeline initialization formatted as HH:MM:SS."""
        elapsed = int(time.time() - self.start_time)
        return str(timedelta(seconds=elapsed))

    def _load_split_mapping(self):
        """
        Loads the video ID to split mapping from the data/results directory.
        """
        split_dir = os.path.join(PROJECT_ROOT, "data", "results")
        mapping = {}
        
        # Combine train and dev into 'train_dev', keep 'test' separate
        splits = {
            'train.csv': 'train_dev',
            'dev.csv': 'train_dev',
            'test.csv': 'test'
        }
        
        for csv_file, target_split in splits.items():
            csv_path = os.path.join(split_dir, csv_file)
            if os.path.exists(csv_path):
                try:
                    # Use sep='|' as configured in the splitting notebook
                    df = pd.read_csv(csv_path, sep='|')
                    if 'id' in df.columns:
                        for vid in df['id']:
                            mapping[str(vid).strip()] = target_split
                except Exception as e:
                    print(f"[{self._elapsed()}] [WARN] Failed to read split file {csv_file}: {e}")
            else:
                print(f"[{self._elapsed()}] [WARN] Split file not found: {csv_path}")
        
        return mapping

    def process_video(self, video_path: str, label: int = None, output_subpath: str = "") -> np.ndarray:
        """
        Extracts skeleton keypoints from a single video file and saves to configured targets.

        Args:
            video_path (str): The path to the input RGB video.
            label (int, optional): An optional integer class label.
            output_subpath (str, optional): Directory subpath used to mirror input folder structures.

        Returns:
            np.ndarray: The extracted keypoints of shape (T, 86, 3).
        
        Raises:
            FileNotFoundError: If the video path does not exist.
        """
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        path_obj = Path(video_path)
        video_id = path_obj.stem
        
        print(f"\n[{self._elapsed()}] [INFO] Processing: {path_obj.name} -> {video_id} (Subpath: {output_subpath or '.'})")

        keypoints = self.extractor.extract_video(video_path)
        
        # Ensure only 2D coordinates (x, y) are kept across all formats
        if keypoints.ndim == 3 and keypoints.shape[2] >= 2:
            keypoints = keypoints[:, :, :2].astype("float64")
            
        print(f"[{self._elapsed()}]        Frames extracted: {keypoints.shape[0]}")

        if self.save_pickle:
            # Determine target split (train_dev or test)
            split_suffix = self.split_mapping.get(video_id)
            if split_suffix:
                target_filename = f"{self.pickle_filename}_{split_suffix}.pkl"
            else:
                target_filename = f"{self.pickle_filename}.pkl"

            sample_id, pickle_path = self.pickle_conv.save(
                keypoints,
                video_id,
                label,
                output_subpath,
                filename=target_filename,
            )
            self.last_pickle_sample_id = sample_id
            self.last_pickle_path = pickle_path

        if self.save_excel:
            self.excel_conv.save(keypoints, video_id, output_subpath)

        print(f"[{self._elapsed()}] [DONE] {video_id}\n")
        return keypoints

    def process_folder(self, folder_path: str, label: int = None) -> None:
        """
        Recursively processes all supported video files within a directory.

        Args:
            folder_path (str): The root directory containing input videos.
            label (int, optional): A class label to be attached to all processed videos.
            
        Raises:
            NotADirectoryError: If the provided folder_path is not a directory.
        """
        if not os.path.isdir(folder_path):
            raise NotADirectoryError(f"Not a directory: {folder_path}")

        all_files = Path(folder_path).rglob("*")
        video_files = sorted([f for f in all_files if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS])

        if len(video_files) == 0:
            print(f"[{self._elapsed()}] [WARN] No supported video files found in: {folder_path}")
            return

        print(f"[{self._elapsed()}] [INFO] Found {len(video_files)} video(s) recursively in: {folder_path}")

        for i, filepath in enumerate(video_files, 1):
            video_path = str(filepath)
            
            rel_path = filepath.relative_to(Path(folder_path))
            output_subpath = str(rel_path.parent)
            if output_subpath == ".":
                output_subpath = ""

            print(f"[{self._elapsed()}] [{i}/{len(video_files)}] {rel_path}")
            try:
                self.process_video(video_path, label=label, output_subpath=output_subpath)
            except Exception as e:
                print(f"[{self._elapsed()}] [ERROR] Failed on {filepath.name}: {e}")

        print(f"\n[{self._elapsed()}] [INFO] Batch complete. {len(video_files)} video(s) processed.")
