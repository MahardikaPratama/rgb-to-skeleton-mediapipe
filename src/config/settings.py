"""
Settings Configurations

This module contains application-wide settings such as MediaPipe parameters,
keypoint constants, and output switches.
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

USE_3D_COORDINATES = False  # If False, only (x, y) are stored


# ==========================================================
# 4. OUTPUT CONFIGURATION
# ==========================================
SAVE_PICKLE = True    # Save as .pkl
SAVE_EXCEL  = True    # Save as .xlsx
