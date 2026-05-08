"""
Command Line Interface (CLI) Module

This module is responsible for parsing input arguments from the terminal.
"""

import argparse

def parse_args():
    """
    Parses command line arguments for the RGB-to-Skeleton extraction pipeline.

    Returns:
        argparse.Namespace: An object containing all parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="RGB → Skeleton pipeline using MediaPipe Holistic (86 keypoints)"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to a single video file or a folder of videos"
    )
    parser.add_argument(
        "--pickle-name",
        default="pose_bisindo",
        help="Base filename for the pickle outputs (default: pose_bisindo)"
    )
    return parser.parse_args()
