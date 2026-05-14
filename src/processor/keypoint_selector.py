from typing import List
import numpy as np


class KeypointSelector:
    """Selects and formats landmarks from MediaPipe-like outputs.

    Responsibilities:
      - Accept a MediaPipe landmarks object (or None) and a list of indices
      - Return a list of coordinate lists with dimensions matching `use_3d`
    """

    def __init__(self, use_3d: bool = False):
        self.use_3d = use_3d
        self.dims = 3 if use_3d else 2

    def select_landmarks(self, landmarks, selection: List[int]) -> List[List[float]]:
        """Extract selected landmarks into coordinate lists.

        Args:
            landmarks: MediaPipe landmarks object or None
            selection: list of indices to extract

        Returns:
            List of lists: each inner list is [x, y] or [x, y, z]
        """
        if landmarks is None:
            return [[0.0] * self.dims for _ in selection]

        output: List[List[float]] = []
        for idx in selection:
            lm = landmarks.landmark[idx]
            coords = [float(lm.x), float(lm.y)]
            if self.use_3d:
                coords.append(float(getattr(lm, "z", 0.0)))
            output.append(coords)
        return output

    def validate(self, arr: np.ndarray, expected_kpts: int, expected_dims: int) -> bool:
        """Basic validation helper for downstream tests.

        Raises ValueError on mismatch.
        """
        if not isinstance(arr, np.ndarray):
            raise ValueError("Keypoints must be a numpy.ndarray")
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array for single-frame keypoints, got ndim={arr.ndim}")
        if arr.shape[0] != expected_kpts:
            raise ValueError(f"Expected {expected_kpts} keypoints, got {arr.shape[0]}")
        if arr.shape[1] != expected_dims:
            raise ValueError(f"Expected {expected_dims} dims, got {arr.shape[1]}")
        return True


__all__ = ["KeypointSelector"]
"""
KeypointSelector

Validates shape of extracted keypoint arrays.

Expected shape: (T, 86, C) where:
    T = number of frames
    86 = total keypoints (isharah layout)
    C = coordinate dimensions (2 or 3)

Keypoint layout:
    Index  0 – 20  : Left Hand  (GL)
    Index 21 – 41  : Right Hand (GR)
    Index 42 – 60  : Mouth      (GM)
    Index 61 – 85  : Pose       (GP)
"""

import numpy as np
from src.config import TOTAL_KEYPOINTS


class KeypointSelector:
    def __init__(self):
        self.total_keypoints = TOTAL_KEYPOINTS

    def validate(self, keypoints: np.ndarray):
        """
        Validate that extracted keypoints have the expected shape.

        Parameters
        ----------
        keypoints : np.ndarray, shape (T, 86, C)

        Returns
        -------
        bool : True if valid

        Raises
        ------
        ValueError
        """
        if keypoints.ndim != 3:
            raise ValueError(
                f"Keypoints must be a 3D array (T, N, C), got shape {keypoints.shape}"
            )
        if keypoints.shape[1] != self.total_keypoints:
            raise ValueError(
                f"Expected {self.total_keypoints} keypoints per frame, "
                f"got {keypoints.shape[1]}"
            )
        return True