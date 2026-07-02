"""Dataset splitting pipeline untuk BISINDO — skenario Signer-Independent (SI).

Menjalankan file ini mencakup dua tahap sekaligus:
  Tahap 1 — Buat berkas split CSV + list TXT dari label CSV dan ID pickle master.
  Tahap 2 — Filter pickle master menjadi berkas pickle terpartisi sesuai split.

Skenario SI:
  - Train  : P1-P5, repetisi R1, R2, R3, R4  -> 600 sampel
  - Dev    : P1-P5, repetisi R5              -> 150 sampel
  - Test   : P6, dievaluasi per variasi MJ vs MN -> masing-masing 57 sampel

Keluaran berkas CSV/TXT (di bawah ``data/results/``):
  data/results/SI/train.csv          data/results/SI/train_list.txt
  data/results/SI/dev.csv            data/results/SI/dev_list.txt
  data/results/SI-MAJ/test.csv       data/results/SI-MAJ/test_list.txt
  data/results/SI-MIN/test.csv       data/results/SI-MIN/test_list.txt

Keluaran berkas Pickle (di bawah ``data/pickle/``):
  pose_bisindo_train_dev_si.pkl
  pose_bisindo_test_si-maj.pkl
  pose_bisindo_test_si-min.pkl

Penggunaan:
  python data_splitting.py
  python data_splitting.py --csv-path data/daftar-kalimat-dan-gloss.csv \\
      --pickle-path data/pickle/pose_bisindo.pkl
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import exceptions
from src.utils.logger import get_logger

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
DEFAULT_CSV_PATH    = PROJECT_ROOT / "data" / "daftar-kalimat-dan-gloss.csv"
DEFAULT_PICKLE_PATH = PROJECT_ROOT / "data" / "pickle" / "pose_bisindo.pkl"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "data" / "results"

DEFAULT_SI_DIR      = DEFAULT_RESULTS_DIR / "SI"
DEFAULT_SI_MAJ_DIR  = DEFAULT_RESULTS_DIR / "SI-MAJ"
DEFAULT_SI_MIN_DIR  = DEFAULT_RESULTS_DIR / "SI-MIN"

DEFAULT_TRAIN_DEV_OUTPUT   = PROJECT_ROOT / "data" / "pickle" / "pose_bisindo_train_dev_si.pkl"
DEFAULT_TEST_SI_MAJ_OUTPUT = PROJECT_ROOT / "data" / "pickle" / "pose_bisindo_test_si-maj.pkl"
DEFAULT_TEST_SI_MIN_OUTPUT = PROJECT_ROOT / "data" / "pickle" / "pose_bisindo_test_si-min.pkl"

# Skenario SI — pembagian berdasarkan repetisi
TRAIN_REPETITIONS: Set[str] = {"R1", "R2", "R3", "R4"}
DEV_REPETITIONS:   Set[str] = {"R5"}

logger = get_logger(__name__)


# ===========================================================================
# CLI
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dataset splitting pipeline BISINDO (SI scenario) — CSV split + Pickle filter."
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help=f"Berkas CSV label kalimat-gloss (default: {DEFAULT_CSV_PATH})",
    )
    parser.add_argument(
        "--pickle-path",
        type=Path,
        default=DEFAULT_PICKLE_PATH,
        help=f"Berkas Pickle master (default: {DEFAULT_PICKLE_PATH})",
    )
    parser.add_argument(
        "--train-dev-output",
        type=Path,
        default=DEFAULT_TRAIN_DEV_OUTPUT,
        help=f"Keluaran Pickle train+dev SI (default: {DEFAULT_TRAIN_DEV_OUTPUT})",
    )
    parser.add_argument(
        "--test-si-maj-output",
        type=Path,
        default=DEFAULT_TEST_SI_MAJ_OUTPUT,
        help=f"Keluaran Pickle test SI-MAJ / P6-MJ (default: {DEFAULT_TEST_SI_MAJ_OUTPUT})",
    )
    parser.add_argument(
        "--test-si-min-output",
        type=Path,
        default=DEFAULT_TEST_SI_MIN_OUTPUT,
        help=f"Keluaran Pickle test SI-MIN / P6-MN (default: {DEFAULT_TEST_SI_MIN_OUTPUT})",
    )
    return parser


# ===========================================================================
# Tahap 1 — Pembangkitan berkas split CSV / TXT
# ===========================================================================

def parse_sample_id(sample_id: str) -> Tuple[str, str, str]:
    """Urai ID sampel ``PX_SXX_RX`` atau ``PX_SXX_MJ/MN`` menjadi tiga komponen.

    Returns:
        Tuple ``(signer, sentence_id, repetition)``.

    Raises:
        exceptions.ValidationException: Jika format ID tidak valid.
    """
    parts = [p.strip() for p in str(sample_id).strip().split("_") if p.strip()]
    if len(parts) < 3:
        raise exceptions.ValidationException(f"Format ID tidak valid: {sample_id}")
    return parts[0], parts[1], parts[2]


def sentence_sort_order(sentence_id: str) -> int:
    """Kembalikan urutan numerik dari ID kalimat Sxx."""
    digits = "".join(c for c in str(sentence_id) if c.isdigit())
    return int(digits) if digits else 999


def load_label_lookup(csv_path: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Baca berkas CSV label dan kembalikan dua kamus lookup.

    Args:
        csv_path: Berkas CSV berformat semicolon-delimited dengan kolom
                  ``ID Kalimat``, ``Kalimat Bahasa Indonesia``, ``Gloss``.

    Returns:
        Tuple ``(gloss_lookup, text_lookup)`` yang di-key oleh ID kalimat (S01..S30).
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Berkas CSV tidak ditemukan: {csv_path}")

    source_df = pd.read_csv(csv_path, sep=";", dtype=str).fillna("")
    gloss_lookup: Dict[str, str] = {}
    text_lookup:  Dict[str, str] = {}

    for _, row in source_df.iterrows():
        sid = str(row["ID Kalimat"]).strip()
        if not sid:
            continue
        gloss_lookup[sid] = str(row["Gloss"]).strip()
        text_lookup[sid]  = str(row["Kalimat Bahasa Indonesia"]).strip()

    return gloss_lookup, text_lookup


def build_dataframe(
    pickle_data: Dict[str, object],
    gloss_lookup: Dict[str, str],
    text_lookup: Dict[str, str],
) -> pd.DataFrame:
    """Bangun DataFrame ternormalisasi dari kunci Pickle + label CSV.

    Returns:
        DataFrame dengan kolom ``id``, ``gloss``, ``text``, ``signer``,
        ``sentence_id``, ``repetition``.
    """
    rows: List[Dict[str, object]] = []
    for sample_id in sorted(pickle_data.keys()):
        signer, sentence_id, repetition = parse_sample_id(sample_id)
        rows.append({
            "id":          sample_id,
            "gloss":       gloss_lookup.get(sentence_id, ""),
            "text":        text_lookup.get(sentence_id, ""),
            "signer":      signer,
            "sentence_id": sentence_id,
            "repetition":  repetition,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise exceptions.ValidationException("Tidak ada baris valid yang dimuat dari Pickle.")

    df["sentence_order"] = df["sentence_id"].map(sentence_sort_order).astype(int)
    return (
        df.sort_values(["sentence_order", "repetition", "id"])
          .drop(columns=["sentence_order"])
          .reset_index(drop=True)
    )


def split_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Bagi DataFrame menjadi empat partisi: train, dev, SI-MAJ, SI-MIN.

    Returns:
        Tuple ``(train_df, dev_df, si_maj_df, si_min_df)``.
    """
    train_df  = df[df["repetition"].isin(TRAIN_REPETITIONS)].copy()
    dev_df    = df[df["repetition"].isin(DEV_REPETITIONS)].copy()
    si_maj_df = df[(df["signer"] == "P6") & (df["repetition"] == "MJ")].copy()
    si_min_df = df[(df["signer"] == "P6") & (df["repetition"] == "MN")].copy()

    if train_df.empty or dev_df.empty:
        raise exceptions.ValidationException("Partisi train atau dev kosong. Periksa indeks repetisi.")

    return train_df, dev_df, si_maj_df, si_min_df


def write_split_files(output_dir: Path, split_name: str, df: pd.DataFrame, list_name: str) -> None:
    """Tulis berkas CSV (id|gloss) dan list TXT (id|gloss|text) ke disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df[["id", "gloss"]].to_csv(output_dir / f"{split_name}.csv", index=False, sep="|")
    df[["id", "gloss", "text"]].to_csv(output_dir / list_name, index=False, sep="|")
    logger.info("  -> %s (%d sampel)", output_dir / f"{split_name}.csv", len(df))


# ===========================================================================
# Tahap 2 — Pemfilteran Pickle
# ===========================================================================

def ids_from_csv(csv_path: Path) -> Set[str]:
    """Baca sekumpulan ID dari berkas CSV pipe-delimited dengan kolom ``id``."""
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="|")
        if not reader.fieldnames or "id" not in reader.fieldnames:
            raise exceptions.ValidationException(f"Kolom 'id' tidak ditemukan di: {csv_path}")
        return {str(row["id"]).strip() for row in reader if row.get("id")}


def filter_and_save_pickle(
    source_data: Dict[str, object],
    ids_to_keep: Set[str],
    output_path: Path,
    split_label: str,
    exclude_prefix: str = "",
) -> int:
    """Saring kamus Pickle berdasarkan sekumpulan ID dan simpan ke disk.

    Args:
        source_data:    Kamus Pickle master.
        ids_to_keep:    Himpunan ID yang akan disimpan.
        output_path:    Jalur keluaran berkas Pickle.
        split_label:    Label partisi untuk pesan log.
        exclude_prefix: Jika diisi, hapus semua kunci yang dimulai dengan prefiks ini
                        (pencegahan kebocoran data).

    Returns:
        Jumlah sampel yang disimpan.
    """
    filtered = {k: v for k, v in source_data.items() if k in ids_to_keep}

    if exclude_prefix:
        filtered = {k: v for k, v in filtered.items() if not k.startswith(exclude_prefix + "_")}

    missing = ids_to_keep - set(source_data.keys())
    if missing:
        preview = ", ".join(sorted(missing)[:10])
        suffix  = "..." if len(missing) > 10 else ""
        logger.warning("[WARN] %d ID tidak ditemukan di Pickle master (%s): %s%s",
                       len(missing), split_label, preview, suffix)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as fh:
        pickle.dump(filtered, fh)

    logger.info("  -> %s (%d sampel)", output_path, len(filtered))
    return len(filtered)


# ===========================================================================
# Main
# ===========================================================================

def main() -> int:
    args        = build_parser().parse_args()
    csv_path    = args.csv_path    if args.csv_path.is_absolute()    else PROJECT_ROOT / args.csv_path
    pickle_path = args.pickle_path if args.pickle_path.is_absolute() else PROJECT_ROOT / args.pickle_path

    # ------------------------------------------------------------------
    # Muat sumber data
    # ------------------------------------------------------------------
    logger.info("Memuat label dari: %s", csv_path)
    gloss_lookup, text_lookup = load_label_lookup(csv_path)

    logger.info("Memuat Pickle master dari: %s", pickle_path)
    if not pickle_path.exists():
        raise FileNotFoundError(f"Pickle master tidak ditemukan: {pickle_path}")
    with pickle_path.open("rb") as fh:
        master_data = pickle.load(fh)
    if not isinstance(master_data, dict):
        raise exceptions.ValidationException(f"Pickle master harus berupa dict, bukan {type(master_data)}")

    logger.info("Total sampel di Pickle master: %d", len(master_data))

    # ------------------------------------------------------------------
    # Tahap 1: Bagi menjadi DataFrame dan tulis berkas CSV/TXT
    # ------------------------------------------------------------------
    logger.info("\n=== Tahap 1: Pembangkitan berkas split CSV/TXT ===")
    df = build_dataframe(master_data, gloss_lookup, text_lookup)
    train_df, dev_df, si_maj_df, si_min_df = split_dataframe(df)

    logger.info("Ukuran partisi -> train: %d | dev: %d | SI-MAJ: %d | SI-MIN: %d",
                len(train_df), len(dev_df), len(si_maj_df), len(si_min_df))

    write_split_files(DEFAULT_SI_DIR,     "train", train_df,  "train_list.txt")
    write_split_files(DEFAULT_SI_DIR,     "dev",   dev_df,    "dev_list.txt")
    write_split_files(DEFAULT_SI_MAJ_DIR, "test",  si_maj_df, "test_list.txt")
    write_split_files(DEFAULT_SI_MIN_DIR, "test",  si_min_df, "test_list.txt")

    # ------------------------------------------------------------------
    # Tahap 2: Filter Pickle master → berkas Pickle terpartisi
    # ------------------------------------------------------------------
    logger.info("\n=== Tahap 2: Pemfilteran berkas Pickle ===")

    train_dev_ids = (
        ids_from_csv(DEFAULT_SI_DIR / "train.csv") |
        ids_from_csv(DEFAULT_SI_DIR / "dev.csv")
    )
    si_maj_ids = ids_from_csv(DEFAULT_SI_MAJ_DIR / "test.csv")
    si_min_ids = ids_from_csv(DEFAULT_SI_MIN_DIR / "test.csv")

    filter_and_save_pickle(master_data, train_dev_ids, args.train_dev_output,
                           "train_dev_si", exclude_prefix="P6")
    filter_and_save_pickle(master_data, si_maj_ids, args.test_si_maj_output, "SI-MAJ")
    filter_and_save_pickle(master_data, si_min_ids, args.test_si_min_output, "SI-MIN")

    logger.info("\nSelesai. Semua keluaran tersimpan.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
