"""
Converts the raw BISINDO skeleton pickle (Mediapipe-extracted: 0-1 normalized
coordinates, multi-underscore video_id naming) into the exact format and
filename expected by datasets/skeleton_feeder.py, WITHOUT modifying any
baseline code.

Two adjustments are applied to the data itself:

  1. video_id renaming: collapse to exactly one underscore
     ("P1_S01_R1" -> "P1_S01R1", "P6_S01_MJ" -> "P6_S01MJ"), matching the
     `signer, sentence_id = video_id.split('_')` parsing used by
     preprocess/mslr2025/mslr_process.py.

  2. coordinate rescaling: multiply every (x, y) by --scale-factor so the
     existing `coord / norm_div - 1` formula in skeleton_feeder.py's
     simple_normalize() (norm_div = (10240 - 1) / 2 = 5119.5) lands on a
     well-used slice of [-1, 1] instead of collapsing near -1. Default
     scale factor mirrors norm_div itself, so the combined effect is just
     `raw_bisindo - 1`.

Usage:
    python preprocess/bisindo/convert_pickle.py
    python preprocess/bisindo/convert_pickle.py --scale-factor 6000
"""
import argparse
import pickle
from pathlib import Path

import numpy as np

DEFAULT_SCALE_FACTOR = 5119.5  # = (10240 - 1) / 2, mirrors skeleton_feeder.py's norm_div


def fix_video_id(video_id: str) -> str:
    """Collapse a multi-underscore video_id into exactly one underscore.

    Keeps the first segment (signer) as-is; concatenates the remaining
    segments without a separator, since they are fixed-width/fixed-prefix
    tokens (S01, R1, MJ, MN) and remain unambiguous when concatenated.
    """
    parts = video_id.split("_")
    return parts[0] + "_" + "".join(parts[1:])


def rescale_keypoints(keypoints: np.ndarray, scale_factor: float) -> np.ndarray:
    """Multiply every (x, y) coordinate by scale_factor.

    Using the same factor on both axes preserves aspect ratio / relative
    topology between keypoints.
    """
    return keypoints * scale_factor


def convert_pickle(input_path: Path, output_path: Path, scale_factor: float) -> dict:
    """Load input_path, rename keys, rescale coordinates, save to output_path.

    Returns a summary dict (sample counts, coordinate ranges before/after)
    for sanity-checking the conversion.
    """
    with open(input_path, "rb") as f:
        data = pickle.load(f)

    converted = {}
    for video_id, entry in data.items():
        new_id = fix_video_id(video_id)
        if new_id in converted:
            raise ValueError(f"Collision after renaming: {video_id!r} -> {new_id!r} already exists")
        converted[new_id] = {"keypoints": rescale_keypoints(entry["keypoints"], scale_factor)}

    with open(output_path, "wb") as f:
        pickle.dump(converted, f)

    sample_keys = list(data.keys())[:50]
    raw_stack = np.concatenate([data[k]["keypoints"] for k in sample_keys], axis=0)
    new_stack = np.concatenate([converted[fix_video_id(k)]["keypoints"] for k in sample_keys], axis=0)

    return {
        "n_samples_in": len(data),
        "n_samples_out": len(converted),
        "raw_range": (float(raw_stack.min()), float(raw_stack.max())),
        "scaled_range": (float(new_stack.min()), float(new_stack.max())),
        "scale_factor": scale_factor,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="datasets/pose_bisindo.pkl")
    parser.add_argument(
        "--output",
        default="datasets/pose_data_isharah1000_hands_lips_body_May12.pkl",
        help="Must match the filename skeleton_feeder.py expects (train/dev path).",
    )
    parser.add_argument("--scale-factor", type=float, default=DEFAULT_SCALE_FACTOR)
    args = parser.parse_args()

    summary = convert_pickle(Path(args.input), Path(args.output), args.scale_factor)

    print(f"Input:  {args.input}  ({summary['n_samples_in']} samples)")
    print(f"Output: {args.output}  ({summary['n_samples_out']} samples)")
    print(f"Scale factor: {summary['scale_factor']}")
    print(f"Coordinate range before: {summary['raw_range']}")
    print(f"Coordinate range after:  {summary['scaled_range']}")


if __name__ == "__main__":
    main()
