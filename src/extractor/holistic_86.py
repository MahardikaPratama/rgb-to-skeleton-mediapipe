"""
Holistic86Extractor

Extracts 86 keypoints per frame from an RGB video using MediaPipe Holistic.

Keypoint layout (matches isharah format):
    Index  0 – 20  : Left Hand  (GL) — 21 keypoints  → hand landmarks 0–20
    Index 21 – 41  : Right Hand (GR) — 21 keypoints  → hand landmarks 0–20
    Index 42 – 60  : Mouth      (GM) — 19 keypoints  → face landmarks 0–18
    Index 61 – 85  : Pose       (GP) — 25 keypoints  → pose landmarks 0–24

All regions are selected sequentially (landmark 0 to N-1).
Counts are derived from the range constants in src/config.py.
"""


import cv2
import mediapipe as mp
import numpy as np

from src.config import MEDIAPIPE_CONFIG, TOTAL_KEYPOINTS, USE_3D_COORDINATES
from src.config.keypoint_layout import (
    LEFT_HAND_SELECTION,
    RIGHT_HAND_SELECTION,
    MOUTH_SELECTION,
    POSE_SELECTION,
)
from src.processor.keypoint_selector import KeypointSelector
from src.utils.exceptions import ExtractionException, ValidationException
from src.utils.logger import get_logger


logger = get_logger(__name__)


class Holistic86Extractor:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.model = self.mp_holistic.Holistic(**MEDIAPIPE_CONFIG)
        self._dims = 3 if USE_3D_COORDINATES else 2
        self.selector = KeypointSelector(use_3d=USE_3D_COORDINATES)

    def extract_frame(self, frame):
        """
        Extract 86 keypoints from a single BGR video frame.

        Output shape: (86, 3) or (86, 2) depending on USE_3D_COORDINATES

        Layout:
            [  0– 20] Left Hand  (GL) — face landmarks  0–20
            [ 21– 41] Right Hand (GR) — hand landmarks  0–20
            [ 42– 60] Mouth      (GM) — face landmarks  0–18
            [ 61– 85] Pose       (GP) — pose landmarks  0–24
        """

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model.process(rgb)

        keypoints = []

        keypoints.extend(self.selector.select_landmarks(results.left_hand_landmarks, LEFT_HAND_SELECTION))
        keypoints.extend(self.selector.select_landmarks(results.right_hand_landmarks, RIGHT_HAND_SELECTION))
        keypoints.extend(self.selector.select_landmarks(results.face_landmarks, MOUTH_SELECTION))
        keypoints.extend(self.selector.select_landmarks(results.pose_landmarks, POSE_SELECTION))

        keypoints = np.array(keypoints, dtype=float)

        if keypoints.shape[0] != TOTAL_KEYPOINTS:
            raise ValidationException(f"Expected {TOTAL_KEYPOINTS} keypoints, got {keypoints.shape[0]}")

        return keypoints

    def extract_video(self, video_path):
        """
        Extract all frames from a video file.

        Returns
        -------
        np.ndarray, shape (T, 86, 3)
            T = number of frames
        """

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ExtractionException(f"Cannot open video: {video_path}")

        # Respect rotation metadata when possible
        try:
            cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
        except Exception:
            logger.debug("Video capture does not support ORIENTATION_AUTO on this platform")

        all_frames = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                all_frames.append(self.extract_frame(frame))

            if len(all_frames) == 0:
                raise ExtractionException(f"Video has no readable frames: {video_path}")

            return np.stack(all_frames)

        finally:
            cap.release()