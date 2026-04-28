"""
Settings Configurations

This module contains application-wide settings such as MediaPipe parameters,
keypoint constants, output switches, and visualization configurations.
"""

# ==========================================================
# 1. MEDIAPIPE CONFIGURATION
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
# 2. KEYPOINT LAYOUT (Isharah Format)
# ==========================================================
# Total = 21 (LH) + 21 (RH) + 19 (Mouth) + 25 (Pose) = 86

TOTAL_KEYPOINTS  = 86
LEFT_HAND_RANGE  = (0,  21)
RIGHT_HAND_RANGE = (21, 42)
MOUTH_RANGE      = (42, 61)
POSE_RANGE       = (61, 86)


# ==========================================================
# 3. DATA FORMAT CONFIGURATION
# ==========================================================

USE_3D_COORDINATES = True  # If False, only (x, y) are stored


# ==========================================================
# 4. OUTPUT CONFIGURATION
# ==========================================================

SAVE_NUMPY  = True    # Save as .npy
SAVE_JSON   = True    # Save as .json
SAVE_PICKLE = True    # Save as .pkl
SAVE_EXCEL  = True    # Save as .xlsx

JSON_INDENT = 2       # JSON formatting indentation


# ==========================================================
# 5. PREVIEW / VISUALIZATION CONFIGURATION
# ==========================================================

PREVIEW_FPS = 25
PREVIEW_RESOLUTION = (640, 480)

DRAW_CONNECTIONS = True   # Draw skeleton edges
DRAW_JOINTS      = True   # Draw joint circles

JOINT_RADIUS    = 2       # Pixel radius
LINE_THICKNESS  = 1       # Garis tipis agar tidak menutupi titik

# Color per region — BGR format
COLOR_LEFT_HAND  = (0,   255,  0)    # GL — green
COLOR_RIGHT_HAND = (0,   100, 255)   # GR — orange-red
COLOR_MOUTH      = (0,   220, 220)   # GM — yellow
COLOR_POSE       = (0,   0,   200)   # GP — red
