import cv2
import mediapipe as mp
import numpy as np

from src.config import (
    MEDIAPIPE_CONFIG,
    POSE_SELECTED,
    HAND_SELECTED,
    FACE_SELECTED,
    TOTAL_KEYPOINTS,
    USE_3D_COORDINATES,
)


class Holistic86Extractor:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.model = self.mp_holistic.Holistic(**MEDIAPIPE_CONFIG)

    def _landmarks_to_array(self, landmarks, selected_indices):
        """
        Convert mediapipe landmarks to numpy array
        """
        output = []

        if landmarks is None:
            # Jika tidak terdeteksi, isi nol
            for _ in selected_indices:
                if USE_3D_COORDINATES:
                    output.append([0.0, 0.0, 0.0])
                else:
                    output.append([0.0, 0.0])
            return output

        for idx in selected_indices:
            lm = landmarks.landmark[idx]
            if USE_3D_COORDINATES:
                output.append([lm.x, lm.y, lm.z])
            else:
                output.append([lm.x, lm.y])

        return output

    def extract_frame(self, frame):
        """
        Extract 86 keypoints from single frame
        Output shape: (86, 3)
        """

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model.process(rgb)

        keypoints = []

        # ---- POSE ----
        pose_kp = self._landmarks_to_array(
            results.pose_landmarks,
            POSE_SELECTED
        )
        keypoints.extend(pose_kp)

        # ---- LEFT HAND ----
        left_hand_kp = self._landmarks_to_array(
            results.left_hand_landmarks,
            HAND_SELECTED
        )
        keypoints.extend(left_hand_kp)

        # ---- RIGHT HAND ----
        right_hand_kp = self._landmarks_to_array(
            results.right_hand_landmarks,
            HAND_SELECTED
        )
        keypoints.extend(right_hand_kp)

        # ---- FACE ----
        face_kp = self._landmarks_to_array(
            results.face_landmarks,
            FACE_SELECTED
        )
        keypoints.extend(face_kp)

        keypoints = np.array(keypoints)

        # Safety check
        assert keypoints.shape[0] == TOTAL_KEYPOINTS, \
            f"Expected {TOTAL_KEYPOINTS}, got {keypoints.shape[0]}"

        return keypoints

    def extract_video(self, video_path):
        """
        Extract full video into tensor shape:
        (num_frames, 86, 3)
        """

        cap = cv2.VideoCapture(video_path)
        all_frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_kp = self.extract_frame(frame)
            all_frames.append(frame_kp)

        cap.release()

        if len(all_frames) == 0:
            raise ValueError("Video contains no frames or cannot be read.")

        return np.stack(all_frames)