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
- [🛠️ Installation & Setup](#-installation--setup)
- [💻 Usage](#-usage)
- [📄 License](#-license)

---

## 📖 Project Overview

This pipeline automates the transformation of raw sign language videos into deep learning-ready datasets. It follows the **Single Responsibility Principle (SRP)**, ensuring scalability and ease of integration into academic research.

### ✨ Key Features
*   **86-Keypoint Extraction**: Captures hands, mouth, and upper body pose.
*   **Automated Data Splitting**: Synchronized with `splitting_data` results to generate `train_dev` and `test` datasets.
*   **Multi-Format Export**: Generates serialized Pickle files and subject-specific Excel files.

---

## ⚙️ Functional Requirements

The system is designed to meet the following technical requirements for robust data preprocessing:

| ID | Requirement | Description |
| :--- | :--- | :--- |
| **FR-01** | **Holistic Extraction** | Extract 86 specific landmarks using MediaPipe Holistic API. |
| **FR-02** | **Coordinate Normalization** | Convert pixel coordinates to normalized values (0.0 to 1.0). |
| **FR-03** | **Metadata Mapping** | Map raw filenames and folder structures to standardized `Pxx_Sxxx_Rxx` IDs. |
| **FR-04** | **Set Separation** | Automatically route data to `train_dev` or `test` sets based on CSV split maps. |
| **FR-05** | **Rotation Handling** | Respect video metadata to handle portrait/landscape orientations correctly. |
| **FR-06** | **Analytical Export** | Generate per-subject Excel workbooks for manual coordinate verification. |

---

## 🔄 Data Simulation Flow

The pipeline operates in three distinct stages:

### 🏗️ Stage 1: Metadata Orchestration
*   **Input**: Video directory + `splitting_data/results/*.csv`.
*   **Process**: Path parsing, ID standardization, and set assignment lookup.
*   **Output**: Metadata-enriched processing queue.

### 🧬 Stage 2: Feature Extraction
*   **Input**: RGB Video Frames.
*   **Process**: Model inference, landmark selection (GL, GR, GM, GP), and XY normalization.
*   **Output**: Temporal Numpy array `(T, 86, 2)`.

### 💾 Stage 3: Data Serialization
*   **Input**: Extracted Numpy keypoints.
*   **Process**: Dictionary aggregation and Pickle serialization / Excel tabularization.
*   **Output**: `.pkl` and `.xlsx` files in the `data/` directory.

---

## 🧪 Simulation Example (E2E)

This example demonstrates how a single raw video is transformed into structured outputs.

### 1. Initial Input
*   **Video File**: `data/raw/AYAH SAMA IBU MANA/ACHMAD_AYAH SAMA IBU MANA_01.mp4`
*   **Split Config (`splitting_data/results/train.csv`)**:
    ```text
    id|gloss
    P01_S001_R01|AYAH SAMA IBU MANA
    ```

### 2. Execution Command
```bash
python main.py --input data/raw/ --pickle-name pose_bisindo
```

### 3. Internal Processing
*   **ID Mapping**: `ACHMAD` → `P01`, `AYAH SAMA IBU MANA` → `S001`, `01` → `R01`.
*   **Resulting ID**: `P01_S001_R01`.
*   **Set Lookup**: Found in `train.csv` → Target: `train_dev`.

### 4. Final Outputs
#### 📁 Pickle Output (`data/pickle/pose_bisindo_train_dev.pkl`)
The data is appended to a dictionary:
```python
{
    "P01_S001_R01": {
        "keypoints": array([[0.51, 0.23], [0.52, 0.24], ...]) # Shape (T, 86, 2)
    },
    ...
}
```

#### 📊 Excel Output (`data/excel/P01.xlsx`)
A sheet named `P01_S001_R01` is created with the following structure:
| kode_video | frame | keypoint_id | x | y |
| :--- | :--- | :--- | :--- | :--- |
| P01_S001_R01 | 0 | 0 | 0.5123 | 0.2345 |
| P01_S001_R01 | 0 | 1 | 0.5145 | 0.2367 |
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

```text
rgb-to-skeleton-mediapipe/
├── main.py                 # Entry point
├── splitting_data/         # Data splitting results
├── src/
│   ├── config/             # Paths, Settings, Mappings
│   ├── core/               # Pipeline, CLI, Metadata
│   ├── extractor/          # MediaPipe Holistic
│   └── converter/          # Pickle & Excel exporters
└── data/                   # Raw videos and extracted outputs
```

---

## 🛠️ Installation & Setup
```bash
git clone https://github.com/MahardikaPratama/rgb-to-skeleton-mediapipe.git
cd rgb-to-skeleton-mediapipe
conda env create -f environment.yml
conda activate rgb-skeleton
pip install -r requirements.txt
```

---

## 💻 Usage
```bash
python main.py --input <path_to_videos> --pickle-name <name>
```

---

## 📄 License
MIT License.
