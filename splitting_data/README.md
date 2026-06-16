# Dataset Splitting — BISINDO

This directory contains scripts to produce Speaker-Dependent (SD) and
Speaker-Independent (SI) splits for the BISINDO dataset. The implementation
follows the project's coding standards and is intended for reproducible
research workflows.

## Contents

- `data_splitting.py` — generate split CSVs and companion list files using the
  Excel label mapping and the master `pose_bisindo.pkl` sample index.
- `split_pose_pickle.py` — filter the master pickle into SD and SI pickle
  files according to the produced CSV splits.
- `raw_data/` — source Excel file(s).
- `results/` — output folder where all split files are written.

## Inputs

- Excel labels (default): `./raw_data/Gloss dan Tanda Dataset.xlsx`
- Master pickle (default): `../data/pickle/pose_bisindo.pkl`

## Outputs

All outputs are written under `splitting_data/results/`:

- SD (Speaker-Dependent)
  - `results/SD/train.csv` (pipe-delimited: `id|gloss`)
  - `results/SD/dev.csv`
  - `results/SD/test.csv`
  - `results/SD/sd_train_list.txt` (pipe-delimited: `id|gloss|text`)
  - `results/SD/sd_dev_list.txt`
  - `results/SD/sd_test_list.txt`

- SI (Speaker-Independent)
  - `results/SI-MAJ/test.csv` and `results/SI-MAJ/test_list.txt` (P6-MJ)
  - `results/SI-MIN/test.csv` and `results/SI-MIN/test_list.txt` (P6-MN)

- Pickle outputs (produced by `split_pose_pickle.py`):
  - `../data/pickle/pose_bisindo_train_dev_sd.pkl`
  - `../data/pickle/pose_bisindo_test_sd.pkl`
  - `../data/pickle/pose_bisindo_test_si-maj.pkl`
  - `../data/pickle/pose_bisindo_test_si-min.pkl`

Each list file uses a consistent sentence code format so that the same
sentence (`Sxx`) appears identical across different signers. Example line:

```
P1_S01_R1|AYAH SAMA IBU MANA|Di mana ayah sama Ibu?
```

## Usage

1. Generate CSV splits and list files (uses Excel + pickle):

```bash
python splitting_data/data_splitting.py
```

2. Produce filtered pickle files from the CSV splits:

```bash
python splitting_data/split_pose_pickle.py
```

## Rules and Notes

- SD split is repetition-based:
  - train: R1, R2, R3
  - dev: R4
  - test: R5
- SI-MAJ uses signer `P6` with repetition code `MJ` (Mayoritas); SI-MIN uses signer `P6` with repetition code `MN` (Minoritas).

## Maintenance

This module follows the project's coding standards: clear function contracts,
typed signatures, structured logging, and custom exceptions (see
`src/utils/`). When modifying splitting behavior, update unit tests and the
README accordingly.

## Contact

For questions about the split rules or dataset provenance, contact the
repository maintainers.