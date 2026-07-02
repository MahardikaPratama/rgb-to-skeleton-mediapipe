#!/usr/bin/env python3
"""
app.py — Unified entry point for APE-1: Skeleton Extraction Component (BISINDO).

Combines two main workflows into a single console-based interface:
  [1] Skeleton Extraction  -- convert RGB video -> skeleton data (Pickle, Excel)
  [2] Dataset Splitting    -- split master Pickle -> train/dev/test partitions (SI)
  [3] Full Pipeline        -- run both steps sequentially

Usage:
  python app.py
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# -----------------------------------------------------------------
# ANSI color constants (no extra dependencies)
# -----------------------------------------------------------------
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
WHITE  = "\033[97m"
BLUE   = "\033[94m"


def _supports_color() -> bool:
    """Return True if the terminal supports ANSI color codes."""
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def c(text: str, *codes: str) -> str:
    """Wrap text with ANSI color codes if the terminal supports them."""
    if not _supports_color():
        return text
    return "".join(codes) + text + RESET


def clear():
    os.system("cls" if os.name == "nt" else "clear")


# -----------------------------------------------------------------
# Banner & Display
# -----------------------------------------------------------------

BANNER = r"""
  APE-1: Skeleton Extraction Component
  BISINDO - RGB-to-Skeleton Pipeline
"""


def print_banner():
    print(c(BANNER, BOLD, CYAN))
    print(c("  " + "=" * 42, DIM))
    print()


def print_menu():
    print(c("  MAIN MENU", BOLD, WHITE))
    print(c("  " + "-" * 42, DIM))
    print()
    print(f"  {c('[1]', BOLD, GREEN)}  Skeleton Extraction")
    print(f"       {c('Convert RGB video -> Pickle + Excel', DIM)}")
    print()
    print(f"  {c('[2]', BOLD, YELLOW)}  Dataset Splitting")
    print(f"       {c('Split master Pickle -> train/dev/test partitions', DIM)}")
    print()
    print(f"  {c('[3]', BOLD, BLUE)}  Full Pipeline")
    print(f"       {c('Run Extraction then Splitting sequentially', DIM)}")
    print()
    print(f"  {c('[0]', BOLD, RED)}  Exit")
    print()
    print(c("  " + "-" * 42, DIM))


def print_section(title: str, color: str = CYAN):
    print()
    print(c(f"  === {title} " + "=" * max(0, 36 - len(title)), BOLD, color))
    print()


def print_done(msg: str = "Done."):
    print()
    print(c(f"  [OK] {msg}", GREEN, BOLD))
    print()


def print_error(msg: str):
    print()
    print(c(f"  [ERR] {msg}", RED, BOLD))
    print()


def prompt(text: str, color: str = CYAN) -> str:
    return input(c(f"  > {text}: ", BOLD, color)).strip()


def pause():
    input(c("\n  Press [Enter] to return to the menu...", DIM))


# -----------------------------------------------------------------
# Flow [1] -- Skeleton Extraction
# -----------------------------------------------------------------

def run_extraction():
    from src.core.pipeline import SkeletonPipeline

    print_section("SKELETON EXTRACTION", GREEN)

    default_input = str(PROJECT_ROOT / "data" / "raw")
    input_path_str = prompt(f"Video path / folder [{default_input}]", GREEN)
    if not input_path_str:
        input_path_str = default_input

    pickle_name = prompt("Pickle output filename [pose_bisindo]", GREEN)
    if not pickle_name:
        pickle_name = "pose_bisindo_sample"

    input_path = Path(input_path_str)
    if not input_path.exists():
        print_error(f"Path not found: {input_path}")
        return

    print()
    print(c(f"  Input  : {input_path}", DIM))
    print(c(f"  Pickle : {pickle_name}.pkl", DIM))
    print()
    confirm = prompt("Start extraction? [y/N]", YELLOW)
    if confirm.lower() not in ("y", "yes"):
        print(c("  Cancelled.", DIM))
        return

    print()
    try:
        pipeline = SkeletonPipeline(pickle_filename=pickle_name)
        if input_path.is_dir():
            pipeline.process_folder(str(input_path))
        else:
            pipeline.process_video(str(input_path))
        print_done("Skeleton extraction complete.")
    except Exception as exc:
        print_error(str(exc))


# -----------------------------------------------------------------
# Flow [2] -- Dataset Splitting
# -----------------------------------------------------------------

def run_splitting():
    import importlib.util

    print_section("DATASET SPLITTING", YELLOW)

    default_csv    = str(PROJECT_ROOT / "data" / "daftar-kalimat-dan-gloss.csv")
    default_pickle = str(PROJECT_ROOT / "data" / "pickle" / "pose_bisindo_sample.pkl")

    csv_str    = prompt(f"Label CSV path [{default_csv}]", YELLOW)
    pickle_str = prompt(f"Master Pickle path [{default_pickle}]", YELLOW)

    if not csv_str:
        csv_str = default_csv
    if not pickle_str:
        pickle_str = default_pickle

    csv_path    = Path(csv_str)
    pickle_path = Path(pickle_str)

    if not csv_path.exists():
        print_error(f"CSV file not found: {csv_path}")
        return
    if not pickle_path.exists():
        print_error(f"Pickle file not found: {pickle_path}")
        return

    print()
    print(c(f"  CSV    : {csv_path}", DIM))
    print(c(f"  Pickle : {pickle_path}", DIM))
    print()
    confirm = prompt("Start dataset splitting? [y/N]", YELLOW)
    if confirm.lower() not in ("y", "yes"):
        print(c("  Cancelled.", DIM))
        return

    print()
    try:
        # Dynamically import data_splitting from root
        spec = importlib.util.spec_from_file_location(
            "data_splitting", PROJECT_ROOT / "data_splitting.py"
        )
        module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(module)  # type: ignore[union-attr]

        gloss_lookup, text_lookup = module.load_label_lookup(csv_path)
        import pickle as _pkl
        with pickle_path.open("rb") as fh:
            master_data = _pkl.load(fh)

        df = module.build_dataframe(master_data, gloss_lookup, text_lookup)
        train_df, dev_df, si_maj_df, si_min_df = module.split_dataframe(df)

        print(c(f"  Train  : {len(train_df)} samples", DIM))
        print(c(f"  Dev    : {len(dev_df)} samples", DIM))
        print(c(f"  SI-MAJ : {len(si_maj_df)} samples", DIM))
        print(c(f"  SI-MIN : {len(si_min_df)} samples", DIM))
        print()

        module.write_split_files(module.DEFAULT_SI_DIR,     "train", train_df,  "train_list.txt")
        module.write_split_files(module.DEFAULT_SI_DIR,     "dev",   dev_df,    "dev_list.txt")
        module.write_split_files(module.DEFAULT_SI_MAJ_DIR, "test",  si_maj_df, "test_list.txt")
        module.write_split_files(module.DEFAULT_SI_MIN_DIR, "test",  si_min_df, "test_list.txt")

        train_dev_ids = (
            module.ids_from_csv(module.DEFAULT_SI_DIR / "train.csv") |
            module.ids_from_csv(module.DEFAULT_SI_DIR / "dev.csv")
        )
        si_maj_ids = module.ids_from_csv(module.DEFAULT_SI_MAJ_DIR / "test.csv")
        si_min_ids = module.ids_from_csv(module.DEFAULT_SI_MIN_DIR / "test.csv")

        module.filter_and_save_pickle(master_data, train_dev_ids, module.DEFAULT_TRAIN_DEV_OUTPUT, "train_dev_si", exclude_prefix="P6")
        module.filter_and_save_pickle(master_data, si_maj_ids, module.DEFAULT_TEST_SI_MAJ_OUTPUT, "SI-MAJ")
        module.filter_and_save_pickle(master_data, si_min_ids, module.DEFAULT_TEST_SI_MIN_OUTPUT, "SI-MIN")

        print_done("Dataset splitting complete.")
    except Exception as exc:
        print_error(str(exc))


# -----------------------------------------------------------------
# Flow [3] -- Full Pipeline
# -----------------------------------------------------------------

def run_full_pipeline():
    print_section("FULL PIPELINE", BLUE)
    print(c("  Step 1/2 -- Skeleton Extraction", BOLD, GREEN))
    run_extraction()
    print(c("  Step 2/2 -- Dataset Splitting", BOLD, YELLOW))
    run_splitting()
    print_done("Full pipeline complete.")


# -----------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------

def main():
    while True:
        clear()
        print_banner()
        print_menu()

        choice = prompt("Select menu [0-3]")

        if choice == "1":
            run_extraction()
            pause()
        elif choice == "2":
            run_splitting()
            pause()
        elif choice == "3":
            run_full_pipeline()
            pause()
        elif choice == "0":
            clear()
            print(c("\n  Goodbye!\n", BOLD, CYAN))
            sys.exit(0)
        else:
            print_error(f"Invalid choice '{choice}'. Enter 0, 1, 2, or 3.")
            pause()


if __name__ == "__main__":
    main()
