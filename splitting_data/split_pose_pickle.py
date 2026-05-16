"""
Split pose_bisindo.pkl into train_dev and test pickle files.

The script reads the split CSV files in splitting_data/results and filters the
input pickle dictionary by sample id.
"""

from __future__ import annotations

import argparse
import csv
import copy
import pickle
import random
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

# Number of SI sentences required per speaker
NUM_SI_SENTENCES = 30

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


def extract_speaker_data(data: Dict[str, object], speaker: str) -> Dict[str, object]:
    """Return entries whose keys start with speaker prefix (e.g. 'P06').

    Args:
        data: The pickle root dictionary.
        speaker: Speaker id without trailing underscore (e.g. 'P06').

    Returns:
        A dictionary of samples for that speaker.
    """
    prefix = speaker if speaker.endswith("_") else speaker + "_"
    return {k: v for k, v in data.items() if k.startswith(prefix)}


def create_dummy_speaker(
    data: Dict[str, object], target_speaker: str, donor_prefixes: List[str], n_sentences: int = NUM_SI_SENTENCES
) -> Dict[str, object]:
    """Create dummy samples for `target_speaker` by sampling donor samples.

    New keys will be like 'P06_S001_R01' .. 'P06_S030_R01'. Donor samples are deep-copied.

    Args:
        data: Source data to sample donors from.
        target_speaker: Target speaker id (e.g. 'P06').
        donor_prefixes: List of speaker prefixes to draw donors from (e.g. ['P01','P02']).
        n_sentences: Number of dummy sentences to generate.

    Returns:
        A dict of generated samples keyed by new ids.

    Raises:
        exceptions.ConfigurationException: If no donor samples are available.
    """
    donors = [k for k in data.keys() if any(k.startswith(d + "_") for d in donor_prefixes)]
    if not donors:
        raise exceptions.ConfigurationException(f"No donor samples found for prefixes: {donor_prefixes}")

    result: Dict[str, object] = {}
    used_keys = set(data.keys())
    for i in range(1, n_sentences + 1):
        new_id = f"{target_speaker}_S{i:03}_R01"
        # avoid overwriting existing keys
        if new_id in used_keys:
            logger.debug("Skipping creation of %s because it already exists", new_id)
            continue
        donor = random.choice(donors)
        result[new_id] = copy.deepcopy(data[donor])

    return result


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
        help=f"Output path for speaker-independent major test (P06) (default: {DEFAULT_TEST_SI_MAJ_OUTPUT})",
    )
    parser.add_argument(
        "--test-si-min-output",
        type=Path,
        default=DEFAULT_TEST_SI_MIN_OUTPUT,
        help=f"Output path for speaker-independent minor test (P07) (default: {DEFAULT_TEST_SI_MIN_OUTPUT})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for dummy sample generation (reproducible).",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # Reproducibility for dummy generation
    if args.seed is not None:
        random.seed(args.seed)
        logger.info("Random seed set to %s", args.seed)

    if not args.input_pickle.exists():
        raise FileNotFoundError(f"Input pickle not found: {args.input_pickle}")

    with args.input_pickle.open("rb") as handle:
        data = pickle.load(handle)

    if not isinstance(data, dict):
        raise exceptions.ValidationException(f"Expected pickle root to be a dict, got {type(data)}")

    train_dev_ids, test_ids = load_split_ids(args.results_dir)
    overlap_ids = train_dev_ids & test_ids
    if overlap_ids:
        raise exceptions.ValidationException(
            f"Found ids in both train_dev and test splits: {sorted(overlap_ids)[:10]}"
        )

    train_dev_data, test_data, missing_ids = split_pickle_data(data, train_dev_ids, test_ids)

    # Prepare speaker-specific SI sets for P06 (maj) and P07 (min)
    si_maj = extract_speaker_data(data, "P06")
    si_min = extract_speaker_data(data, "P07")

    # If there are fewer than NUM_SI_SENTENCES samples for those speakers, create dummy samples
    if len(si_maj) < NUM_SI_SENTENCES:
        # donors: P01, P02, P03
        try:
            dummy = create_dummy_speaker(data, "P06", ["P01", "P02", "P03"], n_sentences=NUM_SI_SENTENCES)
            # merge existing (if any) and dummy, but prefer real if present
            for k, v in dummy.items():
                if k not in si_maj:
                    si_maj[k] = v
        except ValueError:
            # no donors available; leave si_maj as-is
            pass

    if len(si_min) < NUM_SI_SENTENCES:
        # donors: P04, P05
        try:
            dummy = create_dummy_speaker(data, "P07", ["P04", "P05"], n_sentences=NUM_SI_SENTENCES)
            for k, v in dummy.items():
                if k not in si_min:
                    si_min[k] = v
        except ValueError:
            pass

    # Ensure P06/P07 samples are not present in train_dev/test SD splits
    def remove_speaker_from_split(split: Dict[str, object], speaker_prefix: str):
        keys_to_remove = [k for k in split.keys() if k.startswith(speaker_prefix + "_")]
        for k in keys_to_remove:
            split.pop(k, None)

    remove_speaker_from_split(train_dev_data, "P06")
    remove_speaker_from_split(train_dev_data, "P07")
    remove_speaker_from_split(test_data, "P06")
    remove_speaker_from_split(test_data, "P07")

    # Save SD splits (train_dev and test) and SI splits (P06 and P07)
    save_pickle(train_dev_data, args.train_dev_output)
    save_pickle(test_data, args.test_output)
    save_pickle(si_maj, args.test_si_maj_output)
    save_pickle(si_min, args.test_si_min_output)

    print(f"Loaded samples: {len(data)}")
    print(f"Saved train_dev_sd samples: {len(train_dev_data)} -> {args.train_dev_output}")
    print(f"Saved test_sd samples: {len(test_data)} -> {args.test_output}")
    print(f"Saved test_si-maj (P06) samples: {len(si_maj)} -> {args.test_si_maj_output}")
    print(f"Saved test_si-min (P07) samples: {len(si_min)} -> {args.test_si_min_output}")

    if missing_ids:
        preview = ", ".join(missing_ids[:10])
        suffix = "..." if len(missing_ids) > 10 else ""
        print(f"[WARN] {len(missing_ids)} ids were not found in any split csv: {preview}{suffix}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())