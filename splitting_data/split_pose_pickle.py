"""
Split pose_bisindo.pkl into train_dev and test pickle files.

The script reads the split CSV files in splitting_data/results and filters the
input pickle dictionary by sample id.
"""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path
from typing import Dict, Set, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PICKLE = PROJECT_ROOT / "data" / "pickle" / "pose_bisindo.pkl"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "splitting_data" / "results"
DEFAULT_TRAIN_DEV_OUTPUT = PROJECT_ROOT / "data" / "pickle" / "pose_bisindo_train_dev.pkl"
DEFAULT_TEST_OUTPUT = PROJECT_ROOT / "data" / "pickle" / "pose_bisindo_test.pkl"


def load_ids(csv_path: Path) -> Set[str]:
    """Load sample ids from a pipe-separated split CSV."""
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="|")
        if not reader.fieldnames or "id" not in reader.fieldnames:
            raise ValueError(f"Missing 'id' column in {csv_path}")

        return {str(row["id"]).strip() for row in reader if row.get("id")}


def load_split_ids(results_dir: Path) -> Tuple[Set[str], Set[str]]:
    """Return the ids for train_dev and test splits."""
    train_dev_ids = set()
    for split_name in ("train.csv", "dev.csv"):
        split_path = results_dir / split_name
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")
        train_dev_ids.update(load_ids(split_path))

    test_path = results_dir / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"Split file not found: {test_path}")

    test_ids = load_ids(test_path)
    return train_dev_ids, test_ids


def split_pickle_data(data: Dict[str, object], train_dev_ids: Set[str], test_ids: Set[str]):
    """Split the pickle dictionary into train_dev and test dictionaries."""
    train_dev_data = {}
    test_data = {}
    missing_ids = []

    for sample_id, sample_value in data.items():
        if sample_id in train_dev_ids:
            train_dev_data[sample_id] = sample_value
        elif sample_id in test_ids:
            test_data[sample_id] = sample_value
        else:
            missing_ids.append(sample_id)

    return train_dev_data, test_data, missing_ids


def save_pickle(data: Dict[str, object], output_path: Path) -> None:
    """Persist a dictionary to disk using pickle."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(data, handle)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split pose_bisindo.pkl into pose_bisindo_train_dev.pkl and pose_bisindo_test.pkl"
    )
    parser.add_argument(
        "--input-pickle",
        type=Path,
        default=DEFAULT_INPUT_PICKLE,
        help=f"Path to pose_bisindo.pkl (default: {DEFAULT_INPUT_PICKLE})",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"Directory containing train.csv, dev.csv, and test.csv (default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--train-dev-output",
        type=Path,
        default=DEFAULT_TRAIN_DEV_OUTPUT,
        help=f"Output path for train_dev pickle (default: {DEFAULT_TRAIN_DEV_OUTPUT})",
    )
    parser.add_argument(
        "--test-output",
        type=Path,
        default=DEFAULT_TEST_OUTPUT,
        help=f"Output path for test pickle (default: {DEFAULT_TEST_OUTPUT})",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.input_pickle.exists():
        raise FileNotFoundError(f"Input pickle not found: {args.input_pickle}")

    with args.input_pickle.open("rb") as handle:
        data = pickle.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Expected pickle root to be a dict, got {type(data)}")

    train_dev_ids, test_ids = load_split_ids(args.results_dir)
    overlap_ids = train_dev_ids & test_ids
    if overlap_ids:
        raise ValueError(f"Found ids in both train_dev and test splits: {sorted(overlap_ids)[:10]}")

    train_dev_data, test_data, missing_ids = split_pickle_data(data, train_dev_ids, test_ids)

    save_pickle(train_dev_data, args.train_dev_output)
    save_pickle(test_data, args.test_output)

    print(f"Loaded samples: {len(data)}")
    print(f"Saved train_dev samples: {len(train_dev_data)} -> {args.train_dev_output}")
    print(f"Saved test samples: {len(test_data)} -> {args.test_output}")

    if missing_ids:
        preview = ", ".join(missing_ids[:10])
        suffix = "..." if len(missing_ids) > 10 else ""
        print(f"[WARN] {len(missing_ids)} ids were not found in any split csv: {preview}{suffix}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())