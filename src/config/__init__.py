"""
Configuration Package

This package consolidates directory paths, system settings, and keypoint layout
definitions used by the RGB-to-skeleton pipeline.
"""

from .paths import (
    PROJECT_ROOT,
    DATA_DIR,
    RAW_VIDEO_DIR,
    PICKLE_DIR,
    EXCEL_DIR,
)

from .settings import (
    MEDIAPIPE_CONFIG,
    TOTAL_KEYPOINTS,
    LEFT_HAND_RANGE,
    RIGHT_HAND_RANGE,
    MOUTH_RANGE,
    POSE_RANGE,
    USE_3D_COORDINATES,
    SAVE_PICKLE,
    SAVE_EXCEL,
)

# New selection lists for processors
from .keypoint_layout import (
    LEFT_HAND_SELECTION,
    RIGHT_HAND_SELECTION,
    MOUTH_SELECTION,
    POSE_SELECTION,
    KEYPOINT_SELECTIONS,
)

__all__ = [
    # paths
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_VIDEO_DIR",
    "PICKLE_DIR",
    "EXCEL_DIR",
    # settings
    "MEDIAPIPE_CONFIG",
    "TOTAL_KEYPOINTS",
    "USE_3D_COORDINATES",
    "SAVE_PICKLE",
    "SAVE_EXCEL",
    # ranges (backwards compat)
    "LEFT_HAND_RANGE",
    "RIGHT_HAND_RANGE",
    "MOUTH_RANGE",
    "POSE_RANGE",
    # selection lists
    "LEFT_HAND_SELECTION",
    "RIGHT_HAND_SELECTION",
    "MOUTH_SELECTION",
    "POSE_SELECTION",
    "KEYPOINT_SELECTIONS",
]
