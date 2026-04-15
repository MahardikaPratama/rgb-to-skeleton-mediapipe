"""
Skeleton Drawing Module

Renders 86-keypoint skeletons on a canvas or over an RGB frame.
Simple style: small dots + connecting lines, no legend.

Color scheme:
    GL (left hand)  — Green
    GR (right hand) — Orange-red
    GM (mouth)      — Yellow
    GP (pose)       — Red
"""

import cv2
import numpy as np

from src.config import (
    DRAW_CONNECTIONS,
    DRAW_JOINTS,
    JOINT_RADIUS,
    LINE_THICKNESS,
    LEFT_HAND_RANGE,
    RIGHT_HAND_RANGE,
    MOUTH_RANGE,
    POSE_RANGE,
    COLOR_LEFT_HAND,
    COLOR_RIGHT_HAND,
    COLOR_MOUTH,
    COLOR_POSE,
)


class SkeletonDrawer:
    """
    Renders 86-keypoint skeleton frames.

    Parameters
    ----------
    resolution : tuple(int, int)
        Output canvas resolution as (width, height).
    """

    def __init__(self, resolution=(640, 480)):
        self.width, self.height = resolution

        self.left_hand_range  = LEFT_HAND_RANGE
        self.right_hand_range = RIGHT_HAND_RANGE
        self.mouth_range      = MOUTH_RANGE
        self.pose_range       = POSE_RANGE

        # Hand connections (21 keypoints, local indices)
        self.hand_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17),
        ]

        # Pose connections (25 keypoints, local indices 0-24)
        self.pose_connections = [
            (0, 11), (0, 12),
            (11, 12),
            (11, 13), (13, 15),
            (12, 14), (14, 16),
            (15, 17), (15, 19), (15, 21),
            (16, 18), (16, 20), (16, 22),
            (11, 23), (12, 24), (23, 24),
        ]

        # Mouth rings (sequential, local indices)
        self.mouth_outer = list(range(10))
        self.mouth_inner = list(range(10, 19))

    # --------------------------------------------------
    # Utilities
    # --------------------------------------------------

    def _is_valid(self, point):
        return abs(float(point[0])) > 1e-6 or abs(float(point[1])) > 1e-6

    def _px(self, point):
        x = int(float(point[0]) * self.width)
        y = int(float(point[1]) * self.height)
        return x, y

    def _draw_joints(self, canvas, points, color):
        for pt in points:
            if self._is_valid(pt):
                # Thin outline circle — prevents adjacent dots from merging into blobs
                cv2.circle(canvas, self._px(pt), JOINT_RADIUS, color, 1)

    def _draw_edges(self, canvas, points, connections, color):
        for s, e in connections:
            if s < len(points) and e < len(points):
                if self._is_valid(points[s]) and self._is_valid(points[e]):
                    cv2.line(canvas, self._px(points[s]), self._px(points[e]),
                             color, LINE_THICKNESS)

    def _draw_ring(self, canvas, points, indices, color, closed=True):
        valid = [i for i in indices if i < len(points) and self._is_valid(points[i])]
        n = len(valid)
        if n < 2:
            return
        steps = n if closed else n - 1
        for k in range(steps):
            a, b = valid[k], valid[(k + 1) % n]
            cv2.line(canvas, self._px(points[a]), self._px(points[b]),
                     color, LINE_THICKNESS)

    # --------------------------------------------------
    # Main Render
    # --------------------------------------------------

    def draw_frame(self, keypoints, background=None):
        """
        Render a full skeleton frame.

        Parameters
        ----------
        keypoints : np.ndarray, shape (86, 3) or (86, 2)
        background : ndarray or None
            Optional BGR image. If None, dark canvas is used.

        Returns
        -------
        ndarray : rendered BGR image
        """

        if background is None:
            canvas = np.full((self.height, self.width, 3), 30, dtype=np.uint8)
        else:
            canvas = cv2.resize(background, (self.width, self.height))

        lh = keypoints[self.left_hand_range[0]:self.left_hand_range[1]]
        rh = keypoints[self.right_hand_range[0]:self.right_hand_range[1]]
        mo = keypoints[self.mouth_range[0]:self.mouth_range[1]]
        po = keypoints[self.pose_range[0]:self.pose_range[1]]

        if DRAW_CONNECTIONS:
            self._draw_edges(canvas, po, self.pose_connections,  COLOR_POSE)
            self._draw_edges(canvas, lh, self.hand_connections,  COLOR_LEFT_HAND)
            self._draw_edges(canvas, rh, self.hand_connections,  COLOR_RIGHT_HAND)
            self._draw_ring(canvas, mo, self.mouth_outer, COLOR_MOUTH, closed=True)
            self._draw_ring(canvas, mo, self.mouth_inner, COLOR_MOUTH, closed=True)

        if DRAW_JOINTS:
            self._draw_joints(canvas, po, COLOR_POSE)
            self._draw_joints(canvas, lh, COLOR_LEFT_HAND)
            self._draw_joints(canvas, rh, COLOR_RIGHT_HAND)
            self._draw_joints(canvas, mo, COLOR_MOUTH)

        return canvas