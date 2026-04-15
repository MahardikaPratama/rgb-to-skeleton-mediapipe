"""
Global Configuration File for RGB-to-Skeleton MediaPipe Pipeline

Author: Mahardika

Description:
    This file contains all global configuration parameters used in the
    RGB-to-Skeleton extraction pipeline based on MediaPipe Holistic.

    The configuration includes:
    1. Project directory structure
    2. MediaPipe detection settings
    3. Keypoint selection strategy (86 selected keypoints)
    4. Data format configuration
    5. Output saving options
    6. Visualization and preview settings

Keypoint Layout (86 total, matches isharah format):
    Index  0 – 20  : Left Hand  (GL) — 21 keypoints
    Index 21 – 41  : Right Hand (GR) — 21 keypoints
    Index 42 – 60  : Mouth      (GM) — 19 keypoints
    Index 61 – 85  : Pose       (GP) — 25 keypoints

    left_hand_idx  = np.arange(0,  21)
    right_hand_idx = np.arange(21, 42)
    mouth_idx      = np.arange(42, 61)
    pose_idx       = np.arange(61, 86)
"""

import os


# ==========================================================
# 1. PROJECT ROOT AND DIRECTORY STRUCTURE
# ==========================================================

# Absolute path to the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Main data directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_VIDEO_DIR = os.path.join(DATA_DIR, "raw")          # Input RGB videos
SKELETON_DIR = os.path.join(DATA_DIR, "skeleton")     # Extracted skeleton data (.npy)
JSON_DIR = os.path.join(DATA_DIR, "json")             # JSON export files
PICKLE_DIR = os.path.join(DATA_DIR, "pickle")         # Pickle serialized files

# Preview and visualization directories
PREVIEW_DIR = os.path.join(DATA_DIR, "preview")
PREVIEW_SKELETON_DIR = os.path.join(PREVIEW_DIR, "skeleton_only")
PREVIEW_OVERLAY_DIR = os.path.join(PREVIEW_DIR, "skeleton_rgb")


# ==========================================================
# 2. MEDIAPIPE CONFIGURATION
# ==========================================================

MEDIAPIPE_CONFIG = {
    "static_image_mode": False,        # False for video tracking mode
    "model_complexity": 2,             # 0=light, 1=balanced, 2=heavy/accurate
    "smooth_landmarks": True,          # Apply temporal smoothing
    "enable_segmentation": False,      # Disable segmentation mask
    "refine_face_landmarks": True,     # Enable refined face mesh detection
    "min_detection_confidence": 0.5,   # Minimum confidence for detection
    "min_tracking_confidence": 0.5,    # Minimum confidence for tracking
}


# ==========================================================
# 3. KEYPOINT LAYOUT (Isharah Format)
# ==========================================================
# Total = 21 (LH) + 21 (RH) + 19 (Mouth) + 25 (Pose) = 86
#
# The specific MediaPipe landmark indices that are selected
# for each region are defined in src/extractor/holistic_86.py.
#
# Isharah index mapping:
#   left_hand_idx  = np.arange(0,  21)   # GL
#   right_hand_idx = np.arange(21, 42)   # GR
#   mouth_idx      = np.arange(42, 61)   # GM
#   pose_idx       = np.arange(61, 86)   # GP

TOTAL_KEYPOINTS  = 86
LEFT_HAND_RANGE  = (0,  21)
RIGHT_HAND_RANGE = (21, 42)
MOUTH_RANGE      = (42, 61)
POSE_RANGE       = (61, 86)


# ==========================================================
# 4. DATA FORMAT CONFIGURATION
# ==========================================================

USE_3D_COORDINATES = True  # If False, only (x, y) are stored


# ==========================================================
# 5. OUTPUT CONFIGURATION
# ==========================================================

SAVE_NUMPY  = True    # Save as .npy
SAVE_JSON   = True    # Save as .json
SAVE_PICKLE = True    # Save as .pkl

JSON_INDENT = 2       # JSON formatting indentation


# ==========================================================
# 6. PREVIEW / VISUALIZATION CONFIGURATION
# ==========================================================

PREVIEW_FPS = 25
PREVIEW_RESOLUTION = (640, 480)

DRAW_CONNECTIONS = True   # Draw skeleton edges
DRAW_JOINTS      = True   # Draw joint circles

JOINT_RADIUS    = 2       # Pixel radius (kecil untuk akurasi analisis)
LINE_THICKNESS  = 1       # Garis tipis agar tidak menutupi titik

# Color per region — BGR format
COLOR_LEFT_HAND  = (0,   255,  0)    # GL — green
COLOR_RIGHT_HAND = (0,   100, 255)   # GR — orange-red
COLOR_MOUTH      = (0,   220, 220)   # GM — yellow
COLOR_POSE       = (0,   0,   200)   # GP — red