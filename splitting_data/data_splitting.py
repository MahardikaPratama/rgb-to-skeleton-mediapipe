"""Create train/dev and SI test split files from the BISINDO Excel dataset.

Signer-Independent (SI) only scenario:

- Train  : P1-P5, repetitions R1, R2, R3, R4  (4 of 5)  -> 600
- Dev     : P1-P5, repetition  R5              (1 of 5)  -> 150
- Test    : P6 only, evaluated per-gloss as MJ vs MN      -> 57

There is intentionally no held-out test split here. R5 is folded
into training so the majority/minority gloss ratio matches the original
data and no signer is held back from P1-P5.

Outputs written under ``results/``:

- ``SI/train.csv`` / ``SI/train_list.txt``
- ``SI/dev.csv``   / ``SI/dev_list.txt``
- ``SI-MAJ/test.csv`` / ``SI-MAJ/test_list.txt``   (P6-MJ)
- ``SI-MIN/test.csv`` / ``SI-MIN/test_list.txt``   (P6-MN)

NOTE: the ``SI/`` folder holds the SI train/dev splits (P1-P5).
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import exceptions
from src.utils.logger import get_logger


DEFAULT_EXCEL_PATH = Path("./raw_data/Gloss dan Tanda Dataset.xlsx")
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_SI_DIR = DEFAULT_RESULTS_DIR / "SI"
DEFAULT_SI_MAJ_DIR = DEFAULT_RESULTS_DIR / "SI-MAJ"
DEFAULT_SI_MIN_DIR = DEFAULT_RESULTS_DIR / "SI-MIN"

# SI-only: R1-R4 is folded into training; R5 is dev. P6 is the separate test.
TRAIN_REPETITIONS = {"R1", "R2", "R3", "R4"}
DEV_REPETITIONS = {"R5"}

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser.

    Returns:
        Configured argparse parser.
    """
    parser = argparse.ArgumentParser(
        description="Create train/dev and SI test split files from the Gloss dan Tanda Excel dataset."
    )
    parser.add_argument(
        "--excel-path",
        type=Path,
        default=DEFAULT_EXCEL_PATH,
        help=f"Path to the source Excel file (default: {DEFAULT_EXCEL_PATH})",
    )
    parser.add_argument(
        "--pickle-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "pickle" / "pose_bisindo.pkl",
        help="Path to the source pickle file that contains sample ids.",
    )
    parser.add_argument(
        "--id-column",
        type=str,
        default="ID",
        help="Column name containing sample ids (default: ID)",
    )
    parser.add_argument(
        "--gloss-column",
        type=str,
        default="Gloss",
        help="Column name containing gloss labels (default: Gloss)",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="Kalimat",
        help="Column name containing the sentence text (default: Kalimat)",
    )
    return parser


def parse_sample_id(sample_id: str) -> Tuple[str, str, str]:
    """Parse a sample id in the form ``PX_SXX_RX`` or ``PX_SXX_MJ/MN``.

    Args:
        sample_id: Raw sample id string.

    Returns:
        Tuple of ``(signer, sentence_id, repetition_or_variation)``.

    Raises:
        exceptions.ValidationException: If the id cannot be parsed.
    """
    normalized = str(sample_id).strip()
    parts = [part.strip() for part in normalized.split("_") if part.strip()]
    if len(parts) < 3:
        raise exceptions.ValidationException(f"Invalid sample id format: {sample_id}")

    signer = parts[0]
    sentence_id = parts[1]
    repetition_str = parts[2]

    return signer, sentence_id, repetition_str


def normalize_sentence(value: object) -> str:
    """Normalize a sentence-like value for stable Excel lookups."""
    val_str = " ".join(str(value).strip().upper().split())
    if val_str.startswith("S") and len(val_str) > 1 and val_str[1:].isdigit():
        return f"S{int(val_str[1:]):02}"
    return val_str


def sentence_sort_order(sentence_id: str) -> int:
    """Return a numeric sort order derived from an Sxx/Sxxx sentence id."""

    digits = "".join(character for character in str(sentence_id) if character.isdigit())
    if not digits:
        return 999
    return int(digits)


def load_source_dataframe(
    excel_path: Path,
    pickle_path: Path,
    id_column: str,
    gloss_column: str,
    text_column: str,
) -> pd.DataFrame:
    """Load and normalize the source dataset using Excel labels and pickle ids.

    Args:
        excel_path: Path to the Excel file.
        id_column: Name of the id column.
        gloss_column: Name of the gloss column.
        text_column: Name of the text column.

    Returns:
        A normalized dataframe with ``id``, ``gloss``, ``text``, ``signer``,
        ``sentence_id``, and ``repetition`` columns.
    """
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    source_df = pd.read_excel(excel_path)
    required_columns = [id_column, gloss_column, text_column]
    missing_columns = [column for column in required_columns if column not in source_df.columns]
    if missing_columns:
        raise exceptions.ValidationException(
            f"Missing required Excel columns: {missing_columns}. Available columns: {list(source_df.columns)}"
        )

    gloss_lookup: Dict[str, str] = {}
    text_lookup: Dict[str, str] = {}
    for _, row in source_df.dropna(subset=[id_column]).iterrows():
        sentence_id = normalize_sentence(row[id_column])
        if not sentence_id:
            continue
        gloss_lookup[sentence_id] = str(row[gloss_column]).strip() if pd.notna(row[gloss_column]) else ""
        text_lookup[sentence_id] = str(row[text_column]).strip() if pd.notna(row[text_column]) else ""

    if not pickle_path.exists():
        raise FileNotFoundError(f"Pickle file not found: {pickle_path}")

    with pickle_path.open("rb") as handle:
        data = pickle.load(handle)

    if not isinstance(data, dict):
        raise exceptions.ValidationException(f"Expected pickle root to be a dict, got {type(data)}")

    rows: List[Dict[str, object]] = []
    for sample_id in sorted(data.keys()):
        signer, sentence_id, repetition = parse_sample_id(sample_id)
        rows.append(
            {
                "id": sample_id,
                "gloss": gloss_lookup.get(sentence_id, ""),
                "text": text_lookup.get(sentence_id, ""),
                "signer": signer,
                "sentence_id": sentence_id,
                "repetition": repetition,
            }
        )

    dataframe = pd.DataFrame(rows)
    if dataframe.empty:
        raise exceptions.ValidationException("No valid rows were loaded from the Excel file.")

    dataframe["sentence_order"] = dataframe["sentence_id"].map(sentence_sort_order).astype(int)
    return dataframe.sort_values(["sentence_order", "repetition", "id"]).drop(columns=["sentence_order"]).reset_index(drop=True)


def split_train_dev_dataframe(dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataframe into train and dev sets (SI-only; P6 is the separate test).

    Args:
        dataframe: Normalized source dataframe.

    Returns:
        Tuple of train and dev dataframes.
    """
    train_df = dataframe[dataframe["repetition"].isin(TRAIN_REPETITIONS)].copy()
    dev_df = dataframe[dataframe["repetition"].isin(DEV_REPETITIONS)].copy()

    if train_df.empty or dev_df.empty:
        raise exceptions.ValidationException("Train or dev split is empty. Check repetition indices.")

    return train_df, dev_df


def write_split_files(output_dir: Path, split_name: str, dataframe: pd.DataFrame, list_name: str) -> None:
    """Write the split outputs to disk.

    Args:
        output_dir: Directory that receives the files.
        split_name: Base name for the CSV output.
        dataframe: Split dataframe to serialize.
        list_name: Name of the companion text list file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{split_name}.csv"
    list_path = output_dir / list_name

    dataframe[["id", "gloss"]].to_csv(csv_path, index=False, sep="|")
    dataframe[["id", "gloss", "text"]].to_csv(list_path, index=False, sep="|")


def main() -> int:
    """Execute the split pipeline."""
    args = build_parser().parse_args()

    excel_path = args.excel_path if args.excel_path.is_absolute() else SCRIPT_DIR / args.excel_path
    pickle_path = args.pickle_path if args.pickle_path.is_absolute() else PROJECT_ROOT / args.pickle_path

    dataframe = load_source_dataframe(excel_path, pickle_path, args.id_column, args.gloss_column, args.text_column)
    logger.info("Loaded %s normalized rows from %s and %s", len(dataframe), excel_path, pickle_path)

    train_df, dev_df = split_train_dev_dataframe(dataframe)
    logger.info("Train/dev split sizes -> train: %s, dev: %s", len(train_df), len(dev_df))

    si_maj_df = dataframe[(dataframe["signer"] == "P6") & (dataframe["repetition"] == "MJ")].copy()
    si_min_df = dataframe[(dataframe["signer"] == "P6") & (dataframe["repetition"] == "MN")].copy()
    logger.info("SI split sizes -> P6-MJ: %s, P6-MN: %s", len(si_maj_df), len(si_min_df))

    write_split_files(DEFAULT_SI_DIR, "train", train_df, "train_list.txt")
    write_split_files(DEFAULT_SI_DIR, "dev", dev_df, "dev_list.txt")
    write_split_files(DEFAULT_SI_MAJ_DIR, "test", si_maj_df, "test_list.txt")
    write_split_files(DEFAULT_SI_MIN_DIR, "test", si_min_df, "test_list.txt")

    logger.info("Outputs written to %s, %s, and %s", DEFAULT_SI_DIR, DEFAULT_SI_MAJ_DIR, DEFAULT_SI_MIN_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())