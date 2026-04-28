"""
Path Configurations

This module defines all the absolute paths used across the project,
ensuring consistent directory structuring and file output locations.
"""

import os

# ==========================================================
# 1. PROJECT ROOT AND DIRECTORY STRUCTURE
# ==========================================================

# Absolute path to the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Main data directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_VIDEO_DIR = os.path.join(DATA_DIR, "raw")          # Input RGB videos
SKELETON_DIR = os.path.join(DATA_DIR, "skeleton")     # Extracted skeleton data (.npy)
JSON_DIR = os.path.join(DATA_DIR, "json")             # JSON export files
PICKLE_DIR = os.path.join(DATA_DIR, "pickle")         # Pickle serialized files
EXCEL_DIR = os.path.join(DATA_DIR, "excel")           # Excel export files

# Preview and visualization directories
PREVIEW_DIR = os.path.join(DATA_DIR, "preview")
PREVIEW_SKELETON_DIR = os.path.join(PREVIEW_DIR, "skeleton_only")
PREVIEW_OVERLAY_DIR = os.path.join(PREVIEW_DIR, "skeleton_rgb")
