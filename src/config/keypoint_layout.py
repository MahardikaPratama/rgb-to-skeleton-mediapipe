"""
Keypoint layout and selection configuration.

Centralizes all keypoint selection indices used by the extractors and
processors. This module intentionally imports range constants from
`src.config.settings` so existing callers that reference ranges continue
to work while providing explicit selection lists for processors.
"""
from typing import Dict, List, Tuple

from .settings import (
    LEFT_HAND_RANGE,
    RIGHT_HAND_RANGE,
    MOUTH_RANGE,
    POSE_RANGE,
    TOTAL_KEYPOINTS,
)

# Convert ranges to explicit selections when needed by processors
LEFT_HAND_SELECTION: List[int] = list(range(LEFT_HAND_RANGE[0], LEFT_HAND_RANGE[1]))
RIGHT_HAND_SELECTION: List[int] = list(range(RIGHT_HAND_RANGE[0], RIGHT_HAND_RANGE[1]))

# Mouth selection uses specific 468-face-mesh indices targeting lips
# These indices were chosen for the Isharah specification and are stable
# for reproducibility across experiments.
MOUTH_SELECTION: List[int] = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,  # outer lip
    78, 191, 80, 81, 82, 13, 312, 311, 308,      # inner lip
]

POSE_SELECTION: List[int] = list(range(POSE_RANGE[0], POSE_RANGE[1]))

KEYPOINT_RANGES: Dict[str, Tuple[int, int]] = {
    "left_hand": LEFT_HAND_RANGE,
    "right_hand": RIGHT_HAND_RANGE,
    "mouth": MOUTH_RANGE,
    "pose": POSE_RANGE,
}

KEYPOINT_SELECTIONS: Dict[str, List[int]] = {
    "left_hand": LEFT_HAND_SELECTION,
    "right_hand": RIGHT_HAND_SELECTION,
    "mouth": MOUTH_SELECTION,
    "pose": POSE_SELECTION,
}

__all__ = [
    "LEFT_HAND_SELECTION",
    "RIGHT_HAND_SELECTION",
    "MOUTH_SELECTION",
    "POSE_SELECTION",
    "KEYPOINT_SELECTIONS",
    "KEYPOINT_RANGES",
    "TOTAL_KEYPOINTS",
]
