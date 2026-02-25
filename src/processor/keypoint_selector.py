import numpy as np
from src.config import TOTAL_KEYPOINTS


class KeypointSelector:
    def __init__(self):
        self.total_keypoints = TOTAL_KEYPOINTS

    def validate(self, keypoints: np.ndarray):
        """
        Validation for input keypoints.
        Expected shape: (T, 86, C)
        """
        if keypoints.ndim != 3:
            raise ValueError("Input must be a 3D tensor (T, N, C)")
        if keypoints.shape[1] != self.total_keypoints:
            raise ValueError(
                f"Number of keypoints must be {self.total_keypoints}, "
                f"but found {keypoints.shape[1]}"
            )

        return True

    def reorder(self, keypoints: np.ndarray):
        """
        If in the future you want to change the order of nodes.
        Currently, it just returns the original.
        """
        return keypoints