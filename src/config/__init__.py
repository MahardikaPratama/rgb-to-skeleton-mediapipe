"""
Configuration Package

This package consolidates directory paths, system settings, and metadata mappings.
All constants are re-exported here for backwards compatibility and ease of import.
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

from .mappings import (
    PERSON_MAP,
    SENTENCE_MAP,
    SENTENCE_FOLDERS,
)
