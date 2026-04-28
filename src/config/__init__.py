"""
Configuration Package

This package consolidates directory paths, system settings, and metadata mappings.
All constants are re-exported here for backwards compatibility and ease of import.
"""

from .paths import (
    PROJECT_ROOT,
    DATA_DIR,
    RAW_VIDEO_DIR,
    SKELETON_DIR,
    JSON_DIR,
    PICKLE_DIR,
    EXCEL_DIR,
    PREVIEW_DIR,
    PREVIEW_SKELETON_DIR,
    PREVIEW_OVERLAY_DIR,
)

from .settings import (
    MEDIAPIPE_CONFIG,
    TOTAL_KEYPOINTS,
    LEFT_HAND_RANGE,
    RIGHT_HAND_RANGE,
    MOUTH_RANGE,
    POSE_RANGE,
    USE_3D_COORDINATES,
    SAVE_NUMPY,
    SAVE_JSON,
    SAVE_PICKLE,
    SAVE_EXCEL,
    JSON_INDENT,
    PREVIEW_FPS,
    PREVIEW_RESOLUTION,
    DRAW_CONNECTIONS,
    DRAW_JOINTS,
    JOINT_RADIUS,
    LINE_THICKNESS,
    COLOR_LEFT_HAND,
    COLOR_RIGHT_HAND,
    COLOR_MOUTH,
    COLOR_POSE,
)

from .mappings import (
    PERSON_MAP,
    SENTENCE_MAP,
    SENTENCE_FOLDERS,
)
