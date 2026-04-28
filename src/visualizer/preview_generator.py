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

    @staticmethod
    def _create_writer(output_path, fps, resolution):
        # Try a small set of common codecs to avoid OpenH264/runtime issues.
        # Put mp4v first to avoid libopenh264 missing warnings on Windows.
        candidates = ["mp4v", "avc1", "XVID"]
        for codec in candidates:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(output_path, fourcc, fps, resolution)
            if writer.isOpened():
                if codec != "avc1":
                    print(f"[WARN] Using fallback codec '{codec}' for preview output")
                return writer
            writer.release()
        raise RuntimeError("Failed to initialize VideoWriter with any supported codec")

    def generate_skeleton_only(self, keypoints, output_name, resolution=None, output_subpath=""):
        """
        keypoints: (T, 86, 3)
        """
        output_dir = os.path.join(PREVIEW_SKELETON_DIR, output_subpath)
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(
            output_dir,
            f"{output_name}_skeleton.mp4"
        )

        # Force skeleton-only previews to a fixed 256x256 resolution.
        # This ensures consistency across videos with different resolutions
        # and avoids estimation / visual mismatch caused by varying sizes.
        resolution = (256, 256)

        # Ensure integers
        res = (int(resolution[0]), int(resolution[1]))

        writer = self._create_writer(output_path, PREVIEW_FPS, res)

        drawer = SkeletonDrawer(res)

        for frame_kp in keypoints:
            frame = drawer.draw_frame(frame_kp, background=None)
            writer.write(frame)

        writer.release()
        print(f"Skeleton-only preview saved to {output_path}")

    def generate_overlay(self, keypoints, original_video_path, output_name, resolution=None, output_subpath=""):
        """
        keypoints: (T, 86, 3)
        """
        output_dir = os.path.join(PREVIEW_OVERLAY_DIR, output_subpath)
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(
            output_dir,
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

        writer = self._create_writer(output_path, PREVIEW_FPS, res)

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