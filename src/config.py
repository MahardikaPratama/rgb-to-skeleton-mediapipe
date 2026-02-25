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
7. Graph configuration (for future GCN integration)

This file is designed to ensure modularity, reproducibility,
and easy experimentation in skeleton-based research.
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
SKELETON_DIR = os.path.join(DATA_DIR, "skeleton")     # Extracted skeleton data
JSON_DIR = os.path.join(DATA_DIR, "json")             # JSON export files
PICKLE_DIR = os.path.join(DATA_DIR, "pickle")         # Pickle serialized files

# Preview and visualization directories
PREVIEW_DIR = os.path.join(DATA_DIR, "preview")
PREVIEW_SKELETON_DIR = os.path.join(PREVIEW_DIR, "skeleton_only")
PREVIEW_OVERLAY_DIR = os.path.join(PREVIEW_DIR, "skeleton_rgb")


# ==========================================================
# 2. MEDIAPIPE CONFIGURATION
# ==========================================================

"""
MediaPipe Holistic configuration parameters.

These parameters control detection stability, tracking behavior,
and landmark refinement quality.
"""

MEDIAPIPE_CONFIG = {
    "static_image_mode": False,        # False for video tracking mode
    "model_complexity": 1,             # 0=light, 1=balanced, 2=heavy
    "smooth_landmarks": True,          # Apply temporal smoothing
    "enable_segmentation": False,      # Disable segmentation mask
    "refine_face_landmarks": True,     # Enable refined face mesh detection
    "min_detection_confidence": 0.5,   # Minimum confidence for detection
    "min_tracking_confidence": 0.5,    # Minimum confidence for tracking
}


# ==========================================================
# 3. KEYPOINT SELECTION (TOTAL = 86)
# ==========================================================

"""
Keypoint selection strategy.

From MediaPipe Holistic outputs:
- 33 Pose landmarks
- 21 Left hand landmarks
- 21 Right hand landmarks
- 468 Face landmarks

This configuration selects a subset (86 keypoints total)
to reduce noise and computational complexity while preserving
expressive features relevant to sign language analysis.
"""


# -----------------------------
# POSE (12 Upper Body Keypoints)
# -----------------------------
# Selected indices focus on upper-body motion and head orientation.

POSE_SELECTED = [
    0,      # Nose
    11, 12, # Left and Right Shoulder
    13, 14, # Left and Right Elbow
    15, 16, # Left and Right Wrist
    23, 24, # Left and Right Hip
    7, 8,   # Left and Right Ear
    9       # Left Eye (additional orientation reference)
]


# -----------------------------
# HANDS (21 per hand)
# -----------------------------
# Use full 21 hand keypoints for detailed finger articulation.

HAND_SELECTED = list(range(21))


# -----------------------------
# FACE (32 Expressive Keypoints)
# -----------------------------
# Focused on regions important for expressive features:
# eyebrows, eyes, nose, and mouth.


# Face contour (4 points)
FACE_CONTOUR = [
    10,     # Forehead top
    152,    # Chin bottom
    234,    # Left jaw
    454     # Right jaw
]

# Eyebrows and forehead region (important for frowning expressions)
EYEBROWS_AND_FOREHEAD = [
    70, 63, 105,   # Left eyebrow region
    300, 293, 334  # Right eyebrow region
]

# Eyes (upper and lower eyelids)
EYES = [
    # Left eye
    33, 159, 158,      # Upper eyelid
    153, 145, 133,     # Lower eyelid

    # Right eye
    362, 386, 385,     # Upper eyelid
    380, 374, 263      # Lower eyelid
]

# Nose region (minimal but expressive)
NOSE = [
    1, 2, 5, 6
]

# Mouth outer contour (clockwise order)
MOUTH = [
    61,    # Left mouth corner
    40,    # Upper-left lip
    0,     # Upper lip center
    270,   # Upper-right lip
    291,   # Right mouth corner
    17     # Lower lip center
]

# Final face selection
FACE_SELECTED = (
    FACE_CONTOUR
    + EYEBROWS_AND_FOREHEAD
    + EYES
    + NOSE
    + MOUTH
)

# Total keypoints:
# 12 Pose + 42 Hands + 32 Face = 86
TOTAL_KEYPOINTS = 86


# ==========================================================
# 4. DATA FORMAT CONFIGURATION
# ==========================================================

"""
Controls how skeleton data is stored and normalized.
"""

USE_3D_COORDINATES = True  # If False, only (x, y) are stored
NORMALIZE_KEYPOINTS = True # Apply coordinate normalization

# Reference point used for normalization
# Options: "shoulder_center", "nose", "hip_center"
NORMALIZATION_REFERENCE = "shoulder_center"


# ==========================================================
# 5. OUTPUT CONFIGURATION
# ==========================================================

"""
Controls output file formats.
"""

SAVE_NUMPY = True     # Save as .npy
SAVE_JSON = True      # Save as .json
SAVE_PICKLE = True    # Save as .pkl

JSON_INDENT = 2       # JSON formatting indentation


# ==========================================================
# 6. PREVIEW / VISUALIZATION CONFIGURATION
# ==========================================================

"""
Controls visualization settings for preview videos.
"""

PREVIEW_FPS = 25
PREVIEW_RESOLUTION = (640, 480)

DRAW_CONNECTIONS = True   # Draw skeleton edges
DRAW_JOINTS = True        # Draw joint circles

JOINT_RADIUS = 4          # Pixel radius of joint circle
LINE_THICKNESS = 2        # Thickness of skeleton lines


# ==========================================================
# 7. GRAPH STRUCTURE CONFIGURATION (FOR GCN)
# ==========================================================

"""
Graph mode configuration for future Graph Convolutional Network (GCN) integration.

If enabled, adjacency matrix and graph structure will be generated
based on TOTAL_KEYPOINTS.
"""

ENABLE_GRAPH_MODE = False

GRAPH_CONFIG = {
    "num_nodes": TOTAL_KEYPOINTS,
    "self_loops": True,  # Include self-connections in adjacency matrix
}