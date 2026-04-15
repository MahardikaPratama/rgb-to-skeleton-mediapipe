"""
Pipeline: RGB Video → Skeleton (.npy) + Preview

Usage
-----
Single video:
    python main.py --input data/raw/marah.mp4

Folder of videos:
    python main.py --input data/raw/

Optional flags:
    --no-preview        Skip generating preview videos
    --no-overlay        Skip skeleton-overlay preview
    --no-skeleton-only  Skip skeleton-only preview
    --no-json           Skip JSON export
    --no-pickle         Skip Pickle export
    --no-npy            Skip NumPy export
"""

import os
import argparse
import numpy as np
import cv2

from src.config import (
    SKELETON_DIR,
    SAVE_NUMPY,
    SAVE_JSON,
    SAVE_PICKLE,
)

from src.extractor.holistic_86 import Holistic86Extractor
from src.converter.to_json import JSONConverter
from src.converter.to_pickle import PickleConverter
from src.visualizer.preview_generator import PreviewGenerator

SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


class SkeletonPipeline:
    def __init__(self, save_npy=True, save_json=True, save_pickle=True,
                 generate_preview=True, generate_overlay=True,
                 generate_skeleton_only=True):

        self.extractor     = Holistic86Extractor()
        self.json_conv     = JSONConverter()
        self.pickle_conv   = PickleConverter()
        self.preview_gen   = PreviewGenerator()

        self.save_npy      = save_npy
        self.save_json     = save_json
        self.save_pickle   = save_pickle
        self.gen_preview   = generate_preview
        self.gen_overlay   = generate_overlay
        self.gen_skel_only = generate_skeleton_only

        os.makedirs(SKELETON_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Single video
    # ------------------------------------------------------------------
    def process_video(self, video_path, label=None, output_subpath=""):
        """
        Extract skeleton from a single video file.

        Parameters
        ----------
        video_path : str
            Path to the input RGB video.
        label : int or None
            Class label (stored in pickle if provided).
        output_subpath : str
            Subpath to prepend to output directories for mirroring structure.
        """

        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"\n[INFO] Processing: {video_name} -> {output_subpath or '.'}")

        # 1. Extract raw keypoints
        keypoints = self.extractor.extract_video(video_path)
        # shape: (T, 86, 3)
        print(f"       Frames extracted: {keypoints.shape[0]}")

        # 2. Save NumPy
        if self.save_npy:
            # Create the subdirectory if it doesn't exist
            output_dir = os.path.join(SKELETON_DIR, output_subpath)
            os.makedirs(output_dir, exist_ok=True)
            npy_path = os.path.join(output_dir, f"{video_name}.npy")
            np.save(npy_path, keypoints)
            print(f"       Saved .npy  → {npy_path}")

        # 3. Save JSON
        if self.save_json:
            self.json_conv.save(keypoints, video_name, output_subpath)

        # 4. Save Pickle
        if self.save_pickle:
            self.pickle_conv.save(keypoints, video_name, label, output_subpath)

        # 5. Preview
        if self.gen_preview:
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)   # respect rotation metadata
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                resolution = (w, h)
                cap.release()
            else:
                resolution = None

            if self.gen_skel_only:
                self.preview_gen.generate_skeleton_only(keypoints, video_name, resolution, output_subpath)

            if self.gen_overlay:
                self.preview_gen.generate_overlay(keypoints, video_path, video_name, resolution, output_subpath)

        print(f"[DONE] {video_name}\n")
        return keypoints

    # ------------------------------------------------------------------
    # Folder of videos
    # ------------------------------------------------------------------
    def process_folder(self, folder_path, label=None):
        """
        Process all supported video files inside a folder.

        Parameters
        ----------
        folder_path : str
            Path to the folder containing input videos.
        label : int or None
            Class label applied to all videos (optional).
        """

        if not os.path.isdir(folder_path):
            raise NotADirectoryError(f"Not a directory: {folder_path}")

        video_files = sorted([
            f for f in os.listdir(folder_path)
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
        ])

        if len(video_files) == 0:
            print(f"[WARN] No supported video files found in: {folder_path}")
            return

        print(f"[INFO] Found {len(video_files)} video(s) in: {folder_path}")

        for i, filename in enumerate(video_files, 1):
            video_path = os.path.join(folder_path, filename)
            print(f"[{i}/{len(video_files)}] {filename}")
            try:
                self.process_video(video_path, label=label)
            except Exception as e:
                print(f"[ERROR] Failed on {filename}: {e}")

        print(f"\n[INFO] Batch complete. {len(video_files)} video(s) processed.")


# ======================================================================
# CLI Entry Point
# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="RGB → Skeleton pipeline using MediaPipe Holistic (86 keypoints)"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to a single video file or a folder of videos"
    )
    parser.add_argument(
        "--label", "-l",
        type=int,
        default=None,
        help="Class label to attach in pickle output (optional)"
    )
    parser.add_argument("--no-npy",           action="store_true", help="Skip .npy export")
    parser.add_argument("--no-json",          action="store_true", help="Skip .json export")
    parser.add_argument("--no-pickle",        action="store_true", help="Skip .pkl export")
    parser.add_argument("--no-preview",       action="store_true", help="Skip all preview generation")
    parser.add_argument("--no-overlay",       action="store_true", help="Skip overlay preview only")
    parser.add_argument("--no-skeleton-only", action="store_true", help="Skip skeleton-only preview")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    pipeline = SkeletonPipeline(
        save_npy            = not args.no_npy,
        save_json           = not args.no_json,
        save_pickle         = not args.no_pickle,
        generate_preview    = not args.no_preview,
        generate_overlay    = not args.no_overlay,
        generate_skeleton_only = not args.no_skeleton_only,
    )

    input_path = args.input

    if os.path.isdir(input_path):
        pipeline.process_folder(input_path, label=args.label)
    elif os.path.isfile(input_path):
        pipeline.process_video(input_path, label=args.label)
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")