"""
Skeleton Drawer Module

This module provides visualization utilities for rendering
86-keypoint skeleton data extracted from MediaPipe Holistic.

The skeleton structure consists of:
- 12 upper-body pose keypoints
- 21 left-hand keypoints
- 21 right-hand keypoints
- 32 selected facial expression keypoints

The class supports:
- Joint visualization
- Skeleton edge connections
- Facial expression rendering
- Optional RGB background overlay

Designed for preview, debugging, and qualitative analysis.
"""

import cv2
import numpy as np
from src.config import (
    DRAW_CONNECTIONS,
    DRAW_JOINTS,
    JOINT_RADIUS,
    LINE_THICKNESS
)


class SkeletonDrawer:
    """
    Utility class for drawing 86-keypoint skeleton frames.

    Parameters
    ----------
    resolution : tuple(int, int)
        Output canvas resolution as (width, height).
    """

    def __init__(self, resolution=(640, 480)):
        self.width, self.height = resolution

        # --------------------------------------------------
        # Keypoint index ranges (Total = 86 keypoints)
        # --------------------------------------------------
        self.pose_range = (0, 12)          # 12 pose points
        self.left_hand_range = (12, 33)    # 21 left-hand points
        self.right_hand_range = (33, 54)   # 21 right-hand points
        self.face_range = (54, 86)         # 32 facial expression points

        # --------------------------------------------------
        # Pose connections (based on POSE_SELECTED order)
        # --------------------------------------------------
        self.pose_connections = [
            (1, 2),                # Shoulder line
            (1, 3), (3, 5),        # Left arm
            (2, 4), (4, 6),        # Right arm
            (1, 7), (2, 8),        # Shoulders to hips
            (7, 8),                # Hip line
            (0, 1), (0, 2),        # Nose to shoulders
            (0, 9), (0, 10),       # Nose to ears
        ]

        # --------------------------------------------------
        # Hand skeletal structure (21 keypoints)
        # --------------------------------------------------
        self.hand_connections = [
            # Thumb
            (0, 1), (1, 2), (2, 3), (3, 4),

            # Index finger
            (0, 5), (5, 6), (6, 7), (7, 8),

            # Middle finger
            (0, 9), (9, 10), (10, 11), (11, 12),

            # Ring finger
            (0, 13), (13, 14), (14, 15), (15, 16),

            # Pinky
            (0, 17), (17, 18), (18, 19), (19, 20),

            # Palm structure
            (5, 9), (9, 13), (13, 17)
        ]

        # --------------------------------------------------
        # Face structure (relative indices within 32-face subset)
        # --------------------------------------------------
        self.face_offset = 54

        self.face_contour = [0, 1, 2, 3]

        self.left_eyebrow = [4, 5, 6]
        self.right_eyebrow = [7, 8, 9]

        self.left_eye_upper = [10, 11, 12]
        self.left_eye_lower = [15, 14, 13]

        self.right_eye_upper = [16, 17, 18]
        self.right_eye_lower = [21, 20, 19]

        self.nose = [22, 23, 24, 25]
        self.mouth = [26, 27, 28, 29, 30, 31]

    # ======================================================
    # Utility Methods
    # ======================================================

    def _is_valid_point(self, point):
        """
        Check whether a keypoint contains valid coordinates.

        A point is considered valid if it is not near (0, 0).
        This helps avoid drawing invalid or missing landmarks.

        Parameters
        ----------
        point : array-like
            (x, y) or (x, y, z) normalized coordinates.

        Returns
        -------
        bool
        """
        return abs(point[0]) > 1e-6 or abs(point[1]) > 1e-6

    def _denormalize(self, point):
        """
        Convert normalized MediaPipe coordinates (0–1 range)
        into pixel coordinates based on canvas resolution.

        Parameters
        ----------
        point : array-like
            Normalized (x, y) coordinates.

        Returns
        -------
        tuple(int, int)
            Pixel coordinates (x, y).
        """
        x = int(point[0] * self.width)
        y = int(point[1] * self.height)
        return x, y

    def _draw_line_connection(self, canvas, keypoints, indices, color):
        """
        Draw consecutive line connections between a list of indices.

        This function draws open chains (not closed loops).

        Parameters
        ----------
        canvas : ndarray
        keypoints : list or ndarray
        indices : list[int]
        color : tuple(B, G, R)
        """
        for i in range(len(indices) - 1):
            start = indices[i]
            end = indices[i + 1]

            if (
                start < len(keypoints)
                and end < len(keypoints)
                and self._is_valid_point(keypoints[start])
                and self._is_valid_point(keypoints[end])
            ):
                x1, y1 = self._denormalize(keypoints[start])
                x2, y2 = self._denormalize(keypoints[end])
                cv2.line(canvas, (x1, y1), (x2, y2), color, LINE_THICKNESS)

    # ======================================================
    # Main Drawing Function
    # ======================================================

    def draw_frame(self, keypoints, background=None):
        """
        Render a full skeleton frame.

        Parameters
        ----------
        keypoints : list or ndarray
            Array of 86 keypoints in normalized format.
        background : ndarray or None
            Optional RGB image background.

        Returns
        -------
        ndarray
            Rendered image with skeleton overlay.
        """

        # Create blank white canvas if no background is provided
        if background is None:
            canvas = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        else:
            canvas = cv2.resize(background, (self.width, self.height))

        # --------------------------------------------------
        # Draw joints
        # --------------------------------------------------
        if DRAW_JOINTS:
            for i, point in enumerate(keypoints):

                if not self._is_valid_point(point):
                    continue

                # Color coding by body region
                if i < 12:
                    color = (0, 0, 255)        # Pose (Red)
                elif i < 54:
                    color = (255, 0, 0)        # Hands (Blue)
                else:
                    color = (0, 165, 255)      # Face (Orange)

                x, y = self._denormalize(point)
                cv2.circle(canvas, (x, y), JOINT_RADIUS, color, -1)

        # --------------------------------------------------
        # Draw skeleton connections
        # --------------------------------------------------
        if DRAW_CONNECTIONS:

            # ---- Pose edges ----
            for start, end in self.pose_connections:
                if (
                    self._is_valid_point(keypoints[start])
                    and self._is_valid_point(keypoints[end])
                ):
                    x1, y1 = self._denormalize(keypoints[start])
                    x2, y2 = self._denormalize(keypoints[end])
                    cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 0), LINE_THICKNESS)

            # ---- Left Hand ----
            left_hand_points = keypoints[self.left_hand_range[0]:self.left_hand_range[1]]

            for start, end in self.hand_connections:
                if (
                    start < len(left_hand_points)
                    and end < len(left_hand_points)
                    and self._is_valid_point(left_hand_points[start])
                    and self._is_valid_point(left_hand_points[end])
                ):
                    x1, y1 = self._denormalize(left_hand_points[start])
                    x2, y2 = self._denormalize(left_hand_points[end])
                    cv2.line(canvas, (x1, y1), (x2, y2), (255, 255, 0), LINE_THICKNESS)

            # ---- Right Hand ----
            right_hand_points = keypoints[self.right_hand_range[0]:self.right_hand_range[1]]

            for start, end in self.hand_connections:
                if (
                    start < len(right_hand_points)
                    and end < len(right_hand_points)
                    and self._is_valid_point(right_hand_points[start])
                    and self._is_valid_point(right_hand_points[end])
                ):
                    x1, y1 = self._denormalize(right_hand_points[start])
                    x2, y2 = self._denormalize(right_hand_points[end])
                    cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 255), LINE_THICKNESS)

            # ---- Face Region ----
            face_points = keypoints[self.face_offset:self.face_offset + 32]

            # Eyebrows
            self._draw_line_connection(canvas, face_points, self.left_eyebrow, (255, 100, 0))
            self._draw_line_connection(canvas, face_points, self.right_eyebrow, (255, 100, 0))

            # Eyes
            for upper, lower in [
                (self.left_eye_upper, self.left_eye_lower),
                (self.right_eye_upper, self.right_eye_lower),
            ]:
                self._draw_line_connection(canvas, face_points, upper, (0, 200, 200))
                self._draw_line_connection(canvas, face_points, lower, (0, 200, 200))

            # Nose
            self._draw_line_connection(canvas, face_points, self.nose, (100, 100, 100))

            # Mouth (closed loop)
            mouth_valid = [
                i for i in self.mouth
                if i < len(face_points)
                and self._is_valid_point(face_points[i])
            ]

            for i in range(len(mouth_valid)):
                a = mouth_valid[i]
                b = mouth_valid[(i + 1) % len(mouth_valid)]
                x1, y1 = self._denormalize(face_points[a])
                x2, y2 = self._denormalize(face_points[b])
                cv2.line(canvas, (x1, y1), (x2, y2), (0, 0, 200), LINE_THICKNESS)

        return canvas