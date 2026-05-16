"""Create SD and SI split files from the BISINDO Excel dataset.

The script reads the source Excel sheet, parses sample ids in the form
``Pxx_Sxxx_Rxx``, and writes the following outputs:

- ``SD/sd_train_list.txt``
- ``SD/sd_dev_list.txt``
- ``SD/sd_test_list.txt``
- ``SD/train.csv``
- ``SD/dev.csv``
- ``SD/test.csv``
- ``SI-MAJ/si-maj_test_list.txt``
- ``SI-MAJ/test.csv``
- ``SI-MIN/si-min_test_list.txt``
- ``SI-MIN/test.csv``

Speaker-independent outputs prefer real data when speaker rows already exist.
If speaker rows are missing, the script creates dummy rows by sampling from the
configured donor speakers.
"""

from __future__ import annotations

import argparse
import random
import pickle
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import exceptions
from src.utils.logger import get_logger


DEFAULT_EXCEL_PATH = Path("./raw_data/Gloss dan Tanda Dataset.xlsx")
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_SD_DIR = DEFAULT_RESULTS_DIR / "SD"
DEFAULT_SI_MAJ_DIR = DEFAULT_RESULTS_DIR / "SI-MAJ"
DEFAULT_SI_MIN_DIR = DEFAULT_RESULTS_DIR / "SI-MIN"

TRAIN_REPETITIONS = {1, 2, 3}
DEV_REPETITIONS = {4}
TEST_REPETITIONS = {5}
NUM_SI_SENTENCES = 30

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser.

    Returns:
        Configured argparse parser.
    """
    parser = argparse.ArgumentParser(
        description="Create SD and SI split files from the Gloss dan Tanda Excel dataset."
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
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for deterministic dummy speaker generation.",
    )
    return parser


def parse_sample_id(sample_id: str) -> Tuple[str, str, int]:
    """Parse a sample id in the form ``Pxx_Sxxx_Rxx``.

    Args:
        sample_id: Raw sample id string.

    Returns:
        Tuple of ``(signer, sentence_id, repetition)``.

    Raises:
        exceptions.ValidationException: If the id cannot be parsed.
    """
    normalized = str(sample_id).strip()
    parts = [part.strip() for part in normalized.split("_") if part.strip()]
    if len(parts) < 3:
        raise exceptions.ValidationException(f"Invalid sample id format: {sample_id}")

    signer = parts[0]
    sentence_id = parts[1]
    repetition_digits = "".join(character for character in parts[2] if character.isdigit())
    if not repetition_digits:
        raise exceptions.ValidationException(f"Invalid repetition component in sample id: {sample_id}")

    return signer, sentence_id, int(repetition_digits)


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

    mapping: Dict[str, str] = {}
    mapping_text: Dict[str, str] = {}
    for _, row in source_df.dropna(subset=[id_column, gloss_column]).iterrows():
        key = str(row[id_column]).strip()
        mapping[key] = str(row[gloss_column]).strip()
        if pd.notna(row[text_column]):
            mapping_text[key] = str(row[text_column]).strip()

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
                "gloss": mapping.get(sentence_id, mapping.get(sample_id, sample_id)),
                "text": mapping_text.get(sentence_id, mapping_text.get(sample_id, "")),
                "signer": signer,
                "sentence_id": sentence_id,
                "repetition": repetition,
            }
        )

    dataframe = pd.DataFrame(rows)
    if dataframe.empty:
        raise exceptions.ValidationException("No valid rows were loaded from the Excel file.")

    dataframe["repetition"] = dataframe["repetition"].astype(int)
    return dataframe.sort_values(["gloss", "repetition", "id"]).reset_index(drop=True)


def split_sd_dataframe(dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the dataframe into SD train, dev, and test sets.

    Args:
        dataframe: Normalized source dataframe.

    Returns:
        Tuple of train, dev, and test dataframes.
    """
    train_df = dataframe[dataframe["repetition"].isin(TRAIN_REPETITIONS)].copy()
    dev_df = dataframe[dataframe["repetition"].isin(DEV_REPETITIONS)].copy()
    test_df = dataframe[dataframe["repetition"].isin(TEST_REPETITIONS)].copy()

    if train_df.empty or dev_df.empty or test_df.empty:
        raise exceptions.ValidationException("One or more SD splits are empty. Check repetition indices.")

    return train_df, dev_df, test_df


def build_si_dataframe(
    dataframe: pd.DataFrame,
    target_signer: str,
    donor_signers: Sequence[str],
    seed: int | None,
) -> pd.DataFrame:
    """Build a speaker-independent dataframe for a target signer.

    Existing rows for the target signer are kept first. If fewer than
    ``NUM_SI_SENTENCES`` rows are present, dummy rows are created by sampling
    from the donor signers.

    Args:
        dataframe: Normalized source dataframe.
        target_signer: Signer id to construct, e.g. ``P06``.
        donor_signers: Signer ids to sample from when filling missing rows.
        seed: Optional deterministic seed.

    Returns:
        A dataframe containing exactly ``NUM_SI_SENTENCES`` rows when enough
        donor data is available.

    Raises:
        exceptions.ConfigurationException: If there are no donor rows to sample.
    """
    rng = random.Random(seed)
    target_rows = dataframe[dataframe["signer"] == target_signer].copy()
    target_rows = target_rows.sort_values(["sentence_id", "repetition", "id"]).drop_duplicates(subset=["id"])

    if len(target_rows) >= NUM_SI_SENTENCES:
        return target_rows.head(NUM_SI_SENTENCES).reset_index(drop=True)

    donor_rows = dataframe[dataframe["signer"].isin(donor_signers)].copy()
    if donor_rows.empty:
        raise exceptions.ConfigurationException(
            f"No donor rows found for {target_signer}. Donors: {list(donor_signers)}"
        )

    output_rows = target_rows.to_dict(orient="records")
    existing_ids = {str(row["id"]) for row in output_rows}
    existing_sentence_ids = {str(row["sentence_id"]) for row in output_rows}

    while len(output_rows) < NUM_SI_SENTENCES:
        donor_row = donor_rows.iloc[rng.randrange(len(donor_rows))].to_dict()
        donor_sentence_id = str(donor_row["sentence_id"])
        if donor_sentence_id in existing_sentence_ids:
            continue

        dummy_id = f"{target_signer}_{donor_sentence_id}_R01"
        if dummy_id in existing_ids:
            continue

        output_rows.append(
            {
                "id": dummy_id,
                "gloss": donor_row["gloss"],
                "text": donor_row["text"],
                "signer": target_signer,
                "sentence_id": donor_sentence_id,
                "repetition": 1,
            }
        )
        existing_ids.add(dummy_id)
        existing_sentence_ids.add(donor_sentence_id)

    output_dataframe = pd.DataFrame(output_rows)
    return output_dataframe.sort_values(["id"]).reset_index(drop=True)


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

    if args.seed is not None:
        random.seed(args.seed)
        logger.info("Random seed set to %s", args.seed)

    dataframe = load_source_dataframe(excel_path, pickle_path, args.id_column, args.gloss_column, args.text_column)
    logger.info("Loaded %s normalized rows from %s and %s", len(dataframe), excel_path, pickle_path)

    train_df, dev_df, test_df = split_sd_dataframe(dataframe)
    logger.info("SD split sizes -> train: %s, dev: %s, test: %s", len(train_df), len(dev_df), len(test_df))

    si_maj_df = build_si_dataframe(dataframe, "P06", ["P01", "P02", "P03"], args.seed)
    si_min_df = build_si_dataframe(dataframe, "P07", ["P04", "P05"], args.seed)
    logger.info("SI split sizes -> P06: %s, P07: %s", len(si_maj_df), len(si_min_df))

    write_split_files(DEFAULT_SD_DIR, "train", train_df, "train_list.txt")
    write_split_files(DEFAULT_SD_DIR, "dev", dev_df, "dev_list.txt")
    write_split_files(DEFAULT_SD_DIR, "test", test_df, "test_list.txt")
    write_split_files(DEFAULT_SI_MAJ_DIR, "test", si_maj_df, "test_list.txt")
    write_split_files(DEFAULT_SI_MIN_DIR, "test", si_min_df, "test_list.txt")

    logger.info("Outputs written to %s, %s, and %s", DEFAULT_SD_DIR, DEFAULT_SI_MAJ_DIR, DEFAULT_SI_MIN_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())