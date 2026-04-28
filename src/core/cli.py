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
        "--label", "-l",
        type=int,
        default=None,
        help="Class label to attach in pickle output (optional)"
    )
    parser.add_argument("--no-npy",           action="store_true", help="Skip .npy export")
    parser.add_argument("--no-json",          action="store_true", help="Skip .json export")
    parser.add_argument("--no-pickle",        action="store_true", help="Skip .pkl export")
    parser.add_argument("--no-excel",         action="store_true", help="Skip .xlsx export")
    parser.add_argument("--no-preview",       action="store_true", help="Skip all preview generation")
    parser.add_argument("--no-overlay",       action="store_true", help="Skip overlay preview only")
    parser.add_argument("--no-skeleton-only", action="store_true", help="Skip skeleton-only preview")
    return parser.parse_args()
