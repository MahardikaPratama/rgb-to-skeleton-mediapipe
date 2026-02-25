import numpy as np
from src.config import NORMALIZATION_REFERENCE


class SkeletonNormalizer:
    def __init__(self):
        pass

    def _get_reference_point(self, frame):
        """
        Get reference point for normalization based on config.
        frame shape: (86, 3)
        """

        # Index shoulder based on order in Holistic86Extractor:
        # Pose: index 1 = left shoulder
        # Pose: index 2 = right shoulder

        left_shoulder = frame[1]
        right_shoulder = frame[2]

        if NORMALIZATION_REFERENCE == "shoulder_center":
            return (left_shoulder + right_shoulder) / 2

        elif NORMALIZATION_REFERENCE == "nose":
            return frame[0]

        elif NORMALIZATION_REFERENCE == "hip_center":
            left_hip = frame[7]
            right_hip = frame[8]
            return (left_hip + right_hip) / 2

        else:
            raise ValueError("Normalization reference not recognized.")

    def normalize_frame(self, frame):
        """
        Normalize a single frame (86,3)
        """
        reference = self._get_reference_point(frame)

        # Centering
        frame_centered = frame - reference

        # Scaling berdasarkan jarak bahu
        left_shoulder = frame[1]
        right_shoulder = frame[2]

        scale = np.linalg.norm(left_shoulder - right_shoulder)

        if scale > 1e-6:
            frame_centered = frame_centered / scale

        return frame_centered

    def normalize_video(self, keypoints):
        """
        Normalize entire video.
        Input: (T, 86, 3)
        """
        normalized = []

        for frame in keypoints:
            normalized.append(self.normalize_frame(frame))

        return np.stack(normalized)