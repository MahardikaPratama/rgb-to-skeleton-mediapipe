import os
import numpy as np
import cv2

from src.config import (
    SKELETON_DIR,
    SAVE_NUMPY,
    SAVE_JSON,
    SAVE_PICKLE
)

from src.extractor.holistic_86 import Holistic86Extractor
from src.processor.keypoint_selector import KeypointSelector
from src.processor.normalizer import SkeletonNormalizer
from src.processor.temporal_handler import TemporalHandler
from src.converter.to_json import JSONConverter
from src.converter.to_pickle import PickleConverter
from src.visualizer.preview_generator import PreviewGenerator


class SkeletonPipeline:
    def __init__(self):

        self.extractor = Holistic86Extractor()
        self.selector = KeypointSelector()
        self.normalizer = SkeletonNormalizer()
        self.temporal = TemporalHandler()

        self.json_converter = JSONConverter()
        self.pickle_converter = PickleConverter()
        self.preview = PreviewGenerator()

        os.makedirs(SKELETON_DIR, exist_ok=True)

    def process_video(self, video_path, label=None):

        video_name = os.path.splitext(os.path.basename(video_path))[0]

        print(f"Processing {video_name}...")

        raw_keypoints = self.extractor.extract_video(video_path)

        self.selector.validate(raw_keypoints)

        # Copy untuk training
        keypoints = self.normalizer.normalize_video(raw_keypoints)
        keypoints = self.temporal.process(keypoints)

        # 5. Save NumPy
        if SAVE_NUMPY:
            npy_path = os.path.join(SKELETON_DIR, f"{video_name}.npy")
            np.save(npy_path, keypoints)
            print(f"Numpy saved to {npy_path}")

        # 6. Save JSON
        if SAVE_JSON:
            self.json_converter.save(keypoints, video_name)

        # 7. Save Pickle
        if SAVE_PICKLE:
            self.pickle_converter.save(keypoints, video_name, label)

        # 8. Generate preview (use raw_keypoints for correct visualization)
        # Try to get original video resolution so previews match source
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            resolution = (width, height)
            cap.release()
        else:
            # fallback to default configured resolution
            resolution = None

        # Use raw_keypoints for preview (not normalized)
        self.preview.generate_skeleton_only(raw_keypoints, video_name, resolution)
        self.preview.generate_overlay(raw_keypoints, video_path, video_name, resolution)

        print("Done.\n")