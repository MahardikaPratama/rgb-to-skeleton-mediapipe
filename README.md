# 🤟 BISINDO Video-to-Skeleton Preprocessing Pipeline
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.14-green.svg)](https://mediapipe.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A standardized, high-performance preprocessing pipeline designed to convert **BISINDO (Indonesian Sign Language)** RGB videos into structured skeleton keypoints. This project utilizes **MediaPipe Holistic** to extract an 86-keypoint representation (Isharah Format) optimized for Sign Language Recognition (SLR) research.

---

## 📑 Table of Contents
- [📖 Project Overview](#-project-overview)
- [🔄 Data Simulation Flow](#-data-simulation-flow)
- [📍 Keypoint Layout (86 Points)](#-keypoint-layout-86-points)
- [📂 Project Structure](#-project-structure)
- [🛠️ Installation & Setup](#-installation--setup)
- [💻 Usage](#-usage)
- [📝 Technical Notes](#-technical-notes)
- [📄 License](#-license)

---

## 📖 Project Overview

This pipeline automates the transformation of raw sign language videos into deep learning-ready datasets. It features a modular architecture following the **Single Responsibility Principle (SRP)**, ensuring scalability and ease of integration into academic research or industrial applications.

### ✨ Key Features
*   **86-Keypoint Extraction**: Captures hands, mouth, and upper body pose.
*   **Automated Data Splitting**: Synchronized with `splitting_data` results to generate dedicated `train_dev` and `test` datasets.
*   **Multi-Format Export**: Generates serialized Pickle files for training and subject-specific Excel files for analytical verification.
*   **Resolution Agnostic**: Handles varying video aspect ratios through normalized coordinate extraction.

---

## 🔄 Data Simulation Flow

The pipeline operates in three distinct stages, transforming raw video pixels into structured numerical data.

### 🏗️ Stage 1: Metadata Orchestration
*   **Input**: Directory of RGB videos and Splitting CSVs (`train.csv`, `dev.csv`, `test.csv`).
*   **Process**: 
    *   Path parsing to extract `SignerID`, `SentenceID`, and `RepetitionID`.
    *   Standardization of IDs into `Pxx_Sxxx_Rxx` format.
    *   Lookup in the splitting map to determine the target dataset set (`train_dev` or `test`).
*   **Output**: Metadata-enriched processing queue.

### 🧬 Stage 2: Feature Extraction
*   **Input**: Individual RGB frames.
*   **Process**:
    *   Conversion to RGB space and MediaPipe Holistic model inference.
    *   Extraction of 86 specific landmarks across 4 regions.
    *   Coordinate normalization (0.0 - 1.0) and dimensionality reduction (X, Y).
*   **Output**: Temporal Numpy array of shape `(Frames, 86, 2)`.

### 💾 Stage 3: Data Serialization & Export
*   **Input**: Extracted Numpy keypoints.
*   **Process**:
    *   **Pickle Export**: Appending data to either `_train_dev.pkl` or `_test.pkl` based on split metadata.
    *   **Excel Export**: Mapping coordinates to tabular format and saving to subject-specific workbooks (e.g., `P01.xlsx`).
*   **Output**: Serialized datasets in `data/pickle/` and analytical reports in `data/excel/`.

---

## 📍 Keypoint Layout (86 Points)

The pipeline adheres to the standardized 86-keypoint layout, ensuring compatibility with state-of-the-art SLR models.

| Region | Code | Index | Count | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Left Hand** | `GL` | 0 – 20 | 21 | Full hand landmarks (0-20) |
| **Right Hand** | `GR` | 21 – 41 | 21 | Full hand landmarks (0-20) |
| **Mouth** | `GM` | 42 – 60 | 19 | Selected lip contour landmarks |
| **Pose** | `GP` | 61 – 85 | 25 | Upper body pose (0-24) |

---

## 📂 Project Structure

```text
rgb-to-skeleton-mediapipe/
├── main.py                 # Entry point: Orchestrates the entire pipeline
├── splitting_data/         # Results from data splitting scripts
│   └── results/            # train.csv, dev.csv, test.csv
├── src/
│   ├── config/             # Configuration: Paths, Settings, Mappings
│   ├── core/               # Core Logic: Pipeline orchestration, CLI, Metadata
│   ├── extractor/          # Extraction: MediaPipe Holistic implementation
│   ├── converter/          # Conversion: Pickle and Excel exporters
│   └── processor/          # Processing: Keypoint validation and selection
├── data/
│   ├── raw/                # Input: Place raw RGB videos here
│   ├── pickle/             # Output: pose_bisindo_train_dev.pkl, pose_bisindo_test.pkl
│   └── excel/              # Output: Pxx.xlsx (Subject-specific analytics)
└── notebooks/              # Analysis: Data distribution and augmentation tests
```

---

## 🛠️ Installation & Setup

### 1. Prerequisites
- **Python 3.10** (Highly Recommended)
- **Conda** or **venv**

### 2. Environment Setup
```bash
# Clone the repository
git clone https://github.com/MahardikaPratama/rgb-to-skeleton-mediapipe.git
cd rgb-to-skeleton-mediapipe

# Create and activate environment
conda env create -f environment.yml
conda activate rgb-skeleton
```

---

## 💻 Usage

The pipeline is operated via a simple Command Line Interface (CLI).

### 🚀 Basic Command
```bash
python main.py --input data/raw/ --pickle-name pose_bisindo
```

### ⚙️ CLI Arguments
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--input`, `-i` | (Required) | Path to a single video file or a folder of videos. |
| `--pickle-name` | `pose_bisindo` | Base name for the output pickle files. |

---

## 📝 Technical Notes

- **MediaPipe Version**: Ensure `mediapipe <= 0.10.14` for consistent Holistic landmark detection.
- **Coordinate System**: All output coordinates are normalized relative to frame width and height.
- **Data Splitting**: The pipeline automatically detects the set (`train_dev` or `test`) for each video based on the `video_id` found in `splitting_data/results/*.csv`.

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
