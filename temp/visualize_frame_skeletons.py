#!/usr/bin/env python3
"""Render MediaPipe Holistic skeleton variants for a single image frame.

This script reads one RGB frame, runs MediaPipe Holistic once, and writes five
JPG outputs:
- full skeleton
- left hand only
- right hand only
- pose only
- mouth only
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import mediapipe as mp
import numpy as np

from src.config.keypoint_layout import MOUTH_SELECTION


@dataclass(frozen=True)
class RenderVariant:
    """Defines which landmark groups should be rendered for one output."""

    suffix: str
    draw_left_hand: bool = False
    draw_right_hand: bool = False
    draw_pose: bool = False
    draw_mouth: bool = False


MOUTH_CONNECTIONS: tuple[tuple[int, int], ...] = (
    *tuple((idx, idx + 1) for idx in range(0, 9)),
    (9, 0),
    *tuple((idx, idx + 1) for idx in range(10, 18)),
    (18, 10),
)

RENDER_VARIANTS: tuple[RenderVariant, ...] = (
    RenderVariant("full", True, True, True, True),
    RenderVariant("left_hand_only", True, False, False, False),
    RenderVariant("right_hand_only", False, True, False, False),
    RenderVariant("pose_only", False, False, True, False),
    RenderVariant("mouth_only", False, False, False, True),
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description="Render full / partial skeleton JPGs from a single image frame"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/temp/frame-169.jpg"),
        help="Path to the input frame image (default: data/temp/frame-169.jpg)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where JPG outputs will be saved (default: <input parent>/skeleton_outputs)",
    )
    return parser.parse_args()


def load_image(image_path: Path) -> cv2.Mat:
    """Load a BGR image from disk."""

    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    return image


def normalized_to_pixel_coordinates(landmark, image_width: int, image_height: int) -> tuple[int, int]:
    """Convert a normalized landmark to pixel coordinates."""

    return int(landmark.x * image_width), int(landmark.y * image_height)


def draw_landmark_group(
    image,
    landmarks,
    indices: Sequence[int],
    connections: Sequence[tuple[int, int]] | None,
    landmark_color: tuple[int, int, int],
    connection_color: tuple[int, int, int],
    circle_radius: int,
    thickness: int,
) -> None:
    """Draw a selected landmark group onto the provided image."""

    if landmarks is None:
        return

    image_height, image_width = image.shape[:2]
    points: list[tuple[int, int]] = []
    for index in indices:
        landmark = landmarks.landmark[index]
        points.append(normalized_to_pixel_coordinates(landmark, image_width, image_height))

    if connections is not None:
        for start, end in connections:
            if start >= len(points) or end >= len(points):
                continue
            cv2.line(image, points[start], points[end], connection_color, thickness)

    for point in points:
        cv2.circle(image, point, circle_radius, landmark_color, -1)


def create_holistic_model():
    """Create the legacy MediaPipe Holistic model used by this project."""

    solutions = getattr(mp, "solutions", None)
    if solutions is None or not hasattr(solutions, "holistic"):
        raise RuntimeError(
            "This script requires the legacy MediaPipe Holistic API (mediapipe==0.10.14)."
        )

    return solutions.holistic.Holistic(
        static_image_mode=True,
        model_complexity=2,
        smooth_landmarks=False,
        enable_segmentation=False,
        refine_face_landmarks=True,
        min_detection_confidence=0.5,
    )


def render_variant(image, results, variant: RenderVariant):
    """Render a single output variant from MediaPipe results."""

    canvas = np.full_like(image, 255)

    if variant.draw_pose:
        draw_landmark_group(
            canvas,
            results.pose_landmarks,
            range(25),
            mp.solutions.pose.POSE_CONNECTIONS,
            (0, 200, 255),
            (0, 165, 255),
            circle_radius=3,
            thickness=2,
        )

    if variant.draw_left_hand:
        draw_landmark_group(
            canvas,
            results.left_hand_landmarks,
            range(21),
            mp.solutions.hands.HAND_CONNECTIONS,
            (0, 255, 0),
            (0, 200, 0),
            circle_radius=3,
            thickness=2,
        )

    if variant.draw_right_hand:
        draw_landmark_group(
            canvas,
            results.right_hand_landmarks,
            range(21),
            mp.solutions.hands.HAND_CONNECTIONS,
            (255, 0, 0),
            (200, 0, 0),
            circle_radius=3,
            thickness=2,
        )

    if variant.draw_mouth:
        draw_landmark_group(
            canvas,
            results.face_landmarks,
            MOUTH_SELECTION,
            MOUTH_CONNECTIONS,
            (255, 255, 0),
            (255, 215, 0),
            circle_radius=2,
            thickness=2,
        )

    return canvas


def save_outputs(image_path: Path, output_dir: Path) -> list[Path]:
    """Render all variants for one image and save them to disk."""

    output_dir.mkdir(parents=True, exist_ok=True)
    image = load_image(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    holistic = create_holistic_model()

    try:
        results = holistic.process(rgb_image)
    finally:
        holistic.close()

    saved_paths: list[Path] = []
    for variant in RENDER_VARIANTS:
        rendered = render_variant(image, results, variant)
        output_path = output_dir / f"{image_path.stem}_{variant.suffix}.jpg"
        if not cv2.imwrite(str(output_path), rendered):
            raise IOError(f"Failed to write output image: {output_path}")
        saved_paths.append(output_path)

    return saved_paths


def main() -> None:
    """Command line entry point."""

    args = parse_args()
    image_path = args.input
    output_dir = args.output_dir or image_path.parent / "skeleton_outputs"

    saved_paths = save_outputs(image_path, output_dir)

    print(f"[DONE] Saved {len(saved_paths)} image(s) to: {output_dir}")
    for path in saved_paths:
        print(f" - {path}")


if __name__ == "__main__":
    main()