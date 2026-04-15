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