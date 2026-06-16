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
from typing import Dict, List, Optional, Set, Tuple

import sys
# Make `src/` importable when running this script directly (script entrypoint).
# Inserted early so subsequent `from src...` imports work.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Add project root so `import src.*` resolves to PROJECT_ROOT/src
sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.logger import get_logger
from src.utils import exceptions


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PICKLE = PROJECT_ROOT / "data" / "pickle" / "pose_bisindo.pkl"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "splitting_data" / "results"
DEFAULT_TRAIN_DEV_OUTPUT = PROJECT_ROOT / "data" / "pickle" / "pose_bisindo_train_dev_sd.pkl"
DEFAULT_TEST_OUTPUT = PROJECT_ROOT / "data" / "pickle" / "pose_bisindo_test_sd.pkl"
DEFAULT_TEST_SI_MAJ_OUTPUT = PROJECT_ROOT / "data" / "pickle" / "pose_bisindo_test_si-maj.pkl"
DEFAULT_TEST_SI_MIN_OUTPUT = PROJECT_ROOT / "data" / "pickle" / "pose_bisindo_test_si-min.pkl"

logger = get_logger(__name__)


def load_ids(csv_path: Path) -> Set[str]:
    """Load sample ids from a pipe-separated split CSV.

    Args:
        csv_path: Path to the split CSV (pipe-delimited) containing an `id` column.

    Returns:
        A set of sample id strings.

    Raises:
        exceptions.ValidationException: If the CSV does not contain an `id` column.
    """
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="|")
        if not reader.fieldnames or "id" not in reader.fieldnames:
            raise exceptions.ValidationException(f"Missing 'id' column in {csv_path}")

        return {str(row["id"]).strip() for row in reader if row.get("id")}


def load_sd_split_ids(results_dir: Path) -> Tuple[Set[str], Set[str]]:
    """Return the ids for train_dev and test splits."""
    sd_dir = results_dir / "SD"
    train_dev_ids = set()
    for split_name in ("train.csv", "dev.csv"):
        split_path = sd_dir / split_name
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")
        train_dev_ids.update(load_ids(split_path))

    test_path = sd_dir / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"Split file not found: {test_path}")

    test_ids = load_ids(test_path)
    return train_dev_ids, test_ids


def load_si_split_ids(results_dir: Path) -> Tuple[Set[str], Set[str]]:
    """Return the ids for si-maj and si-min test splits."""
    si_maj_path = results_dir / "SI-MAJ" / "test.csv"
    si_min_path = results_dir / "SI-MIN" / "test.csv"
    
    if not si_maj_path.exists():
        raise FileNotFoundError(f"Split file not found: {si_maj_path}")
    if not si_min_path.exists():
        raise FileNotFoundError(f"Split file not found: {si_min_path}")

    return load_ids(si_maj_path), load_ids(si_min_path)


def split_pickle_data(data: Dict[str, object], train_dev_ids: Set[str], test_ids: Set[str]) -> Tuple[Dict[str, object], Dict[str, object], List[str]]:
    """Split the pickle dictionary into train_dev and test dictionaries.

    Args:
        data: The pickle root dictionary mapping sample ids to sample payloads.
        train_dev_ids: Set of ids that belong to train+dev.
        test_ids: Set of ids that belong to test.

    Returns:
        Tuple of (train_dev_dict, test_dict, missing_id_list).
    """
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
        help=f"Output path for train_dev_sd pickle (default: {DEFAULT_TRAIN_DEV_OUTPUT})",
    )
    parser.add_argument(
        "--test-output",
        type=Path,
        default=DEFAULT_TEST_OUTPUT,
        help=f"Output path for test_sd pickle (default: {DEFAULT_TEST_OUTPUT})",
    )
    parser.add_argument(
        "--test-si-maj-output",
        type=Path,
        default=DEFAULT_TEST_SI_MAJ_OUTPUT,
        help=f"Output path for speaker-independent major test (P6-MJ) (default: {DEFAULT_TEST_SI_MAJ_OUTPUT})",
    )
    parser.add_argument(
        "--test-si-min-output",
        type=Path,
        default=DEFAULT_TEST_SI_MIN_OUTPUT,
        help=f"Output path for speaker-independent minor test (P6-MN) (default: {DEFAULT_TEST_SI_MIN_OUTPUT})",
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
        raise exceptions.ValidationException(f"Expected pickle root to be a dict, got {type(data)}")

    train_dev_ids, test_ids = load_sd_split_ids(args.results_dir)
    overlap_ids = train_dev_ids & test_ids
    if overlap_ids:
        raise exceptions.ValidationException(
            f"Found ids in both train_dev and test splits: {sorted(overlap_ids)[:10]}"
        )

    si_maj_ids, si_min_ids = load_si_split_ids(args.results_dir)

    train_dev_data, test_data, missing_ids = split_pickle_data(data, train_dev_ids, test_ids)

    # Filter out SI keys from missing_ids because they are intentionally not in SD splits
    missing_ids = [k for k in missing_ids if k not in si_maj_ids and k not in si_min_ids]

    si_maj = {}
    si_min = {}
    for k in si_maj_ids:
        if k in data:
            si_maj[k] = data[k]
        else:
            missing_ids.append(k)
            
    for k in si_min_ids:
        if k in data:
            si_min[k] = data[k]
        else:
            missing_ids.append(k)

    # Ensure P6 samples are not present in train_dev/test SD splits
    def remove_speaker_from_split(split: Dict[str, object], speaker_prefix: str):
        keys_to_remove = [k for k in split.keys() if k.startswith(speaker_prefix + "_")]
        for k in keys_to_remove:
            split.pop(k, None)

    remove_speaker_from_split(train_dev_data, "P6")
    remove_speaker_from_split(test_data, "P6")

    # Save SD splits (train_dev and test) and SI splits (P6-MJ and P6-MN)
    save_pickle(train_dev_data, args.train_dev_output)
    save_pickle(test_data, args.test_output)
    save_pickle(si_maj, args.test_si_maj_output)
    save_pickle(si_min, args.test_si_min_output)

    print(f"Loaded samples: {len(data)}")
    print(f"Saved train_dev_sd samples: {len(train_dev_data)} -> {args.train_dev_output}")
    print(f"Saved test_sd samples: {len(test_data)} -> {args.test_output}")
    print(f"Saved test_si-maj (P6-MJ) samples: {len(si_maj)} -> {args.test_si_maj_output}")
    print(f"Saved test_si-min (P6-MN) samples: {len(si_min)} -> {args.test_si_min_output}")

    if missing_ids:
        preview = ", ".join(missing_ids[:10])
        suffix = "..." if len(missing_ids) > 10 else ""
        print(f"[WARN] {len(missing_ids)} ids were not found in any split csv: {preview}{suffix}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())