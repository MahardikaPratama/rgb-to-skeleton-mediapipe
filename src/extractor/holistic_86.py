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

from src.config import (
    MEDIAPIPE_CONFIG,
    TOTAL_KEYPOINTS,
    USE_3D_COORDINATES,
    LEFT_HAND_RANGE,
    RIGHT_HAND_RANGE,
    MOUTH_RANGE,
    POSE_RANGE,
)

# Number of landmarks to take per region (derived from config ranges)
_N_LEFT_HAND  = LEFT_HAND_RANGE[1]  - LEFT_HAND_RANGE[0]   # 21
_N_RIGHT_HAND = RIGHT_HAND_RANGE[1] - RIGHT_HAND_RANGE[0]  # 21
_N_POSE       = POSE_RANGE[1]       - POSE_RANGE[0]         # 25

# Mouth: specific landmark indices from MediaPipe 468-point face mesh.
# Landmarks 0-18 (sequential) are nose/eye area — NOT the lip region.
# These specific indices select the outer and inner lip contour.
_MOUTH_SELECTED = [
    # Outer lip (10 points, clockwise)
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    # Inner lip (9 points)
    78, 191, 80, 81, 82, 13, 312, 311, 308,
]  # total: 19


class Holistic86Extractor:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.model = self.mp_holistic.Holistic(**MEDIAPIPE_CONFIG)
        self._dims = 3 if USE_3D_COORDINATES else 2

    def _landmarks_to_array(self, landmarks, selection):
        """
        Convert mediapipe landmarks to numpy array.

        Parameters
        ----------
        landmarks : mediapipe landmark object or None
        selection : int or list[int]
            - int  : take the first N landmarks sequentially (0 to N-1)
            - list : take landmarks at these specific indices

        Returns
        -------
        list of [x, y, z] or [x, y]
        """
        indices = range(selection) if isinstance(selection, int) else selection
        dims    = self._dims
        empty   = [0.0] * dims

        if landmarks is None:
            return [list(empty) for _ in indices]

        output = []
        for idx in indices:
            lm = landmarks.landmark[idx]
            if dims == 3:
                output.append([lm.x, lm.y, lm.z])
            else:
                output.append([lm.x, lm.y])
        return output

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

        # ---- LEFT HAND (GL) — output indices 0–20 ----
        keypoints.extend(self._landmarks_to_array(
            results.left_hand_landmarks, _N_LEFT_HAND
        ))

        # ---- RIGHT HAND (GR) — output indices 21–41 ----
        keypoints.extend(self._landmarks_to_array(
            results.right_hand_landmarks, _N_RIGHT_HAND
        ))

        # ---- MOUTH (GM) — output indices 42–60 ----
        # Uses specific lip landmark indices (not sequential) because
        # face mesh landmark order 0-18 maps to nose/eye area, not lips.
        keypoints.extend(self._landmarks_to_array(
            results.face_landmarks, _MOUTH_SELECTED
        ))

        # ---- POSE (GP) — output indices 61–85 ----
        keypoints.extend(self._landmarks_to_array(
            results.pose_landmarks, _N_POSE
        ))

        keypoints = np.array(keypoints)

        assert keypoints.shape[0] == TOTAL_KEYPOINTS, (
            f"Expected {TOTAL_KEYPOINTS} keypoints, got {keypoints.shape[0]}"
        )

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
            raise ValueError(f"Cannot open video: {video_path}")

        # Respect rotation metadata (e.g. videos shot on phone with portrait orientation)
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)

        all_frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            all_frames.append(self.extract_frame(frame))

        cap.release()

        if len(all_frames) == 0:
            raise ValueError(f"Video has no readable frames: {video_path}")

        return np.stack(all_frames)