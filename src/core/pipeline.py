"""
Core Pipeline Module

This module orchestrates the extraction of keypoints from video files
and dispatches the results to various converters and visualizers.
"""

import os
from pathlib import Path
import numpy as np
import cv2

from src.config import SKELETON_DIR
from src.core.metadata import parse_video_id
from src.extractor.holistic_86 import Holistic86Extractor
from src.converter.to_json import JSONConverter
from src.converter.to_pickle import PickleConverter
from src.converter.to_excel import ExcelConverter
from src.visualizer.preview_generator import PreviewGenerator

SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


class SkeletonPipeline:
    """
    Main orchestration class for the RGB-to-Skeleton workflow.
    """
    
    def __init__(self, save_npy=True, save_json=True, save_pickle=True, save_excel=True,
                 generate_preview=True, generate_overlay=True,
                 generate_skeleton_only=True, pickle_filename: str = None):
        """
        Initializes the pipeline with desired output targets.

        Args:
            save_npy (bool): Whether to export raw keypoints as .npy files.
            save_json (bool): Whether to export structured data as .json files.
            save_pickle (bool): Whether to export the dataset to a Pickle file.
            save_excel (bool): Whether to export to Excel file formats.
            generate_preview (bool): Master switch for generating any video previews.
            generate_overlay (bool): Whether to generate skeletons overlaid on original RGB.
            generate_skeleton_only (bool): Whether to generate standalone skeleton videos.
            pickle_filename (str, optional): Custom filename for the aggregated Pickle output.
        """
        self.extractor     = Holistic86Extractor()
        self.json_conv     = JSONConverter()
        self.pickle_conv   = PickleConverter()
        self.excel_conv    = ExcelConverter()
        self.preview_gen   = PreviewGenerator()

        self.save_npy      = save_npy
        self.save_json     = save_json
        self.save_pickle   = save_pickle
        self.save_excel    = save_excel
        self.gen_preview   = generate_preview
        self.gen_overlay   = generate_overlay
        self.gen_skel_only = generate_skeleton_only
        self.last_pickle_sample_id = None
        self.last_pickle_path = None
        self.pickle_filename = pickle_filename

        os.makedirs(SKELETON_DIR, exist_ok=True)

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
        video_id = parse_video_id(path_obj)
        
        print(f"\n[INFO] Processing: {path_obj.name} -> {video_id} (Subpath: {output_subpath or '.'})")

        keypoints = self.extractor.extract_video(video_path)
        
        # Ensure only 2D coordinates (x, y) are kept across all formats
        if keypoints.ndim == 3 and keypoints.shape[2] >= 2:
            keypoints = keypoints[:, :, :2].astype("float64")
            
        print(f"       Frames extracted: {keypoints.shape[0]}")

        if self.save_npy:
            output_dir = os.path.join(SKELETON_DIR, output_subpath)
            os.makedirs(output_dir, exist_ok=True)
            npy_path = os.path.join(output_dir, f"{video_id}.npy")
            np.save(npy_path, keypoints)
            print(f"       Saved .npy  → {npy_path}")

        if self.save_json:
            self.json_conv.save(keypoints, video_id, output_subpath)

        if self.save_pickle:
            sample_id, pickle_path = self.pickle_conv.save(
                keypoints,
                video_id,
                label,
                output_subpath,
                filename=self.pickle_filename,
            )
            self.last_pickle_sample_id = sample_id
            self.last_pickle_path = pickle_path

        if self.save_excel:
            self.excel_conv.save(keypoints, video_id, output_subpath)

        if self.gen_preview:
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                resolution = (w, h)
                cap.release()
            else:
                resolution = None

            if self.gen_skel_only:
                self.preview_gen.generate_skeleton_only(keypoints, video_id, resolution, output_subpath)

            if self.gen_overlay:
                self.preview_gen.generate_overlay(keypoints, video_path, video_id, resolution, output_subpath)

        print(f"[DONE] {video_id}\n")
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
            print(f"[WARN] No supported video files found in: {folder_path}")
            return

        print(f"[INFO] Found {len(video_files)} video(s) recursively in: {folder_path}")

        for i, filepath in enumerate(video_files, 1):
            video_path = str(filepath)
            
            rel_path = filepath.relative_to(Path(folder_path))
            output_subpath = str(rel_path.parent)
            if output_subpath == ".":
                output_subpath = ""

            print(f"[{i}/{len(video_files)}] {rel_path}")
            try:
                self.process_video(video_path, label=label, output_subpath=output_subpath)
            except Exception as e:
                print(f"[ERROR] Failed on {filepath.name}: {e}")

        print(f"\n[INFO] Batch complete. {len(video_files)} video(s) processed.")
