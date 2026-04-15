import cv2
import numpy as np
import os

from src.config import (
    PREVIEW_FPS,
    PREVIEW_RESOLUTION,
    PREVIEW_SKELETON_DIR,
    PREVIEW_OVERLAY_DIR
)

from src.visualizer.draw_skeleton import SkeletonDrawer


class PreviewGenerator:
    def __init__(self):
        # Don't create a single drawer with fixed resolution.
        # Drawer will be created per-preview using the actual resolution.
        self.default_resolution = PREVIEW_RESOLUTION

    def generate_skeleton_only(self, keypoints, output_name, resolution=None):
        """
        keypoints: (T, 86, 3)
        """

        os.makedirs(PREVIEW_SKELETON_DIR, exist_ok=True)

        output_path = os.path.join(
            PREVIEW_SKELETON_DIR,
            f"{output_name}_skeleton.mp4"
        )

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        if resolution is None:
            resolution = self.default_resolution

        # Ensure integers
        res = (int(resolution[0]), int(resolution[1]))

        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            PREVIEW_FPS,
            res
        )

        drawer = SkeletonDrawer(res)

        for frame_kp in keypoints:
            frame = drawer.draw_frame(frame_kp, background=None)
            writer.write(frame)

        writer.release()
        print(f"Skeleton-only preview saved to {output_path}")

    def generate_overlay(self, keypoints, original_video_path, output_name, resolution=None):
        """
        keypoints: (T, 86, 3)
        """

        os.makedirs(PREVIEW_OVERLAY_DIR, exist_ok=True)

        output_path = os.path.join(
            PREVIEW_OVERLAY_DIR,
            f"{output_name}_overlay.mp4"
        )

        cap = cv2.VideoCapture(original_video_path)

        # Respect rotation metadata so output matches original orientation
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)

        # Determine resolution AFTER enabling auto-rotation
        # (rotated videos swap width/height)
        if resolution is None:
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            resolution = (width, height)

        res = (int(resolution[0]), int(resolution[1]))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            PREVIEW_FPS,
            res
        )

        drawer = SkeletonDrawer(res)

        t = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or t >= len(keypoints):
                break

            overlay_frame = drawer.draw_frame(
                keypoints[t],
                background=frame
            )

            writer.write(overlay_frame)
            t += 1

        cap.release()
        writer.release()

        print(f"Overlay preview saved to {output_path}")