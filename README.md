# 🤟 BISINDO Video-to-Skeleton Preprocessing Pipeline

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.14-green.svg)](https://mediapipe.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A standardized, high-performance preprocessing pipeline designed to convert **BISINDO (Indonesian Sign Language)** RGB videos into structured skeleton keypoints. This project utilizes **MediaPipe Holistic** to extract an 86-keypoint representation (Isharah Format) optimized for Sign Language Recognition (SLR) research.

---

## 📑 Table of Contents

- [📖 Project Overview](#-project-overview)
- [⚙️ Functional Requirements](#%EF%B8%8F-functional-requirements)
- [🔄 Data Simulation Flow](#-data-simulation-flow)
- [🧪 Simulation Example (E2E)](#-simulation-example-e2e)
- [📍 Keypoint Layout (86 Points)](#-keypoint-layout-86-points)
- [📂 Project Structure](#-project-structure)
- [🛠️ Installation & Setup](#installation--setup)
- [💻 Usage](#-usage)
- [📄 License](#-license)

---

## 📖 Project Overview

This pipeline automates the transformation of raw sign language videos into deep learning-ready datasets. It follows the **Single Responsibility Principle (SRP)**, ensuring scalability and ease of integration into academic research.

### ✨ Key Features

- **86-Keypoint Extraction**: Captures hands, mouth, and upper body pose.

- **Filename-Based Sample Keys**: Uses the renamed video filename stem directly as the sample key in pickle outputs.
- **Automated Data Splitting**: Synchronized with `splitting_data` results to generate `train_dev` and `test` datasets.
- **Multi-Format Export**: Generates serialized Pickle files and subject-specific Excel files.

---

## ⚙️ Functional Requirements

The system is designed to meet the following technical requirements for robust data preprocessing:

| ID | Requirement | Description |
| :--- | :--- | :--- |
| **FR-01** | **Holistic Extraction** | Extract 86 specific landmarks using MediaPipe Holistic API. |
| **FR-02** | **Coordinate Normalization** | Convert pixel coordinates to normalized values (0.0 to 1.0). |
| **FR-03** | **Filename-Based Keys** | Use the renamed video filename stem directly as the sample key in pickle outputs. |
| **FR-04** | **Set Separation** | Automatically route data to `train_dev` or `test` sets based on CSV split maps. |
| **FR-05** | **Rotation Handling** | Respect video metadata to handle portrait/landscape orientations correctly. |
| **FR-06** | **Analytical Export** | Generate per-subject Excel workbooks for manual coordinate verification. |

---

## 🔄 Data Simulation Flow

The pipeline operates in three distinct stages:

### 🏗️ Stage 1: Input Organization

- **Input**: Renamed video directory + `splitting_data/results/*.csv`.

- **Process**: Folder traversal, split assignment lookup, and direct filename-based keying.
- **Output**: Processing queue keyed by the video filename stem.

### 🧬 Stage 2: Feature Extraction

- **Input**: RGB Video Frames.

- **Process**: Model inference, landmark selection (GL, GR, GM, GP), and XY normalization.
- **Output**: Temporal Numpy array `(T, 86, 2)`.

### 💾 Stage 3: Data Serialization

- **Input**: Extracted Numpy keypoints.

- **Process**: Dictionary aggregation using the video filename stem and Pickle serialization / Excel tabularization.
- **Output**: `.pkl` and `.xlsx` files in the `data/` directory.

---

## 🧪 Simulation Example (E2E)

This example demonstrates how a single raw video is transformed into structured outputs.

### 1. Initial Input

- **Video File**: `data/raw/S06/P1_S06_R1.mp4`

- **Split Config (`splitting_data/results/train.csv`)**:

    ```text
    id|gloss
    P1_S06_R1|AYAH SAMA IBU MANA
    ```

### 2. Execution Command

```bash
python main.py --input data/raw/ --pickle-name pose_bisindo
```

### 3. Internal Processing

- **Rename Result**: `ACHMAD_AYAH SAMA IBU MANA_01.mp4` is renamed to `P1_S06_R1.mp4`.

- **Resulting Key**: `P1_S06_R1`.
- **Set Lookup**: Found in `train.csv` → Target: `train_dev`.

### 4. Final Outputs

#### 📁 Pickle Output (`data/pickle/pose_bisindo_train_dev.pkl`)

The data is appended to a dictionary:

```python
{
    "P1_S06_R1": {
        "keypoints": array([[0.51, 0.23], [0.52, 0.24], ...]) # Shape (T, 86, 2)
    },
    ...
}
```

#### 📊 Excel Output (`data/excel/P1.xlsx`)

A sheet named `P1_S06_R1` is created with the following structure:

| kode_video | frame | keypoint_id | x | y |
| :--- | :--- | :--- | :--- | :--- |
| P1_S06_R1 | 0 | 0 | 0.5123 | 0.2345 |
| P1_S06_R1 | 0 | 1 | 0.5145 | 0.2367 |
| ... | ... | ... | ... | ... |

---

## 📍 Keypoint Layout (86 Points)

| Region | Code | Index | Count | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Left Hand** | `GL` | 0 – 20 | 21 | Full hand landmarks (0-20) |
| **Right Hand** | `GR` | 21 – 41 | 21 | Full hand landmarks (0-20) |
| **Mouth** | `GM` | 42 – 60 | 19 | Selected lip contour landmarks |
| **Pose** | `GP` | 61 – 85 | 25 | Upper body pose (0-24) |

---

## 📂 Project Structure

This repository has been refactored: most runtime logic resides under `src/`.
Refer to `CODING_STANDARDS.md` for architecture, conventions, and development
guidelines. The high-level layout is:

```bash
rgb-to-skeleton-mediapipe/
├── main.py                  # CLI entrypoint (orchestrates pipeline)
├── data/
│   ├── raw/                 # Raw videos and rename helper script
│   ├── pickle/              # Serialized pickle outputs
│   └── excel/               # Excel exports per subject
├── splitting_data/          # Data split utilities and results (CSV, lists)
│   ├── data_splitting.py    # Create SD / SI split CSVs and lists
│   └── split_pose_pickle.py # Filter master pickle using split CSVs
└── src/                     # Core library (importable package)
    ├── config/              # Configuration, paths, and keypoint layout
    ├── core/                # Pipeline orchestration and CLI
    ├── extractor/           # MediaPipe Holistic extractor
    ├── processor/           # Keypoint selection and validators
    ├── converter/           # Pickle / Excel converters
    └── utils/               # Logging, exceptions, helpers
```

---

## 🛠️ Installation & Setup

Prerequisite: this project expects a working Conda environment. We recommend
installing Miniconda (lightweight Conda distribution). Follow the official
installation guide for your platform:

<https://www.anaconda.com/docs/getting-started/miniconda/install/overview>

After Miniconda is installed, run the following commands to set up the
project environment and install dependencies:

```bash
git clone https://github.com/MahardikaPratama/rgb-to-skeleton-mediapipe.git
cd rgb-to-skeleton-mediapipe

# Create the Conda environment from the provided spec
conda env create -f environment.yml

# Activate the created environment
conda activate rgb-skeleton

# Install any additional pip-only packages (optional)
pip install -r requirements.txt
```

Notes:

- Use the `conda` commands above in a shell where Miniconda is available.
- If you prefer `mamba`, it can be used as a faster drop-in replacement for
    the `conda env create` step.

## Developer Guide

- Read `CODING_STANDARDS.md` for architecture decisions, coding conventions,
    type and docstring requirements, and testing guidance. This document is the
    authoritative source for contributions and code reviews.
- Use the `src/` package when importing library components in scripts or tests.
    Example:

```python
from src.core.pipeline import SkeletonPipeline
```

- The project enforces strict typing and Google-style docstrings. Run the
    repository checks described in `CODING_STANDARDS.md` before submitting PRs.

## Quick Commands

- Generate SD / SI CSVs and list files:

```bash
python splitting_data/data_splitting.py --seed 42
```

- Rename raw folders and videos before extraction:

```bash
python data/raw/rename_folder_file.py --dry-run
python data/raw/rename_folder_file.py
```

- Produce filtered pickle files from the CSV splits:

```bash
python splitting_data/split_pose_pickle.py --seed 42
```

---

## 💻 Usage

```bash
python main.py --input <path_to_videos_or_folder> --pickle-name <name>
```

Example:

```bash
# Process a single video
python main.py --input data/raw/S06/P1_S06_R1.mp4 --pickle-name pose_bisindo

# Process a folder of videos recursively
python main.py --input data/raw/ --pickle-name pose_bisindo
```

---

## 📄 License

MIT License.
