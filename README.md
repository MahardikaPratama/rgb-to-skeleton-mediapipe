
# RGB-to-Skeleton MediaPipe Pipeline

This repository provides a command-line pipeline for converting RGB video data into an 86-keypoint skeleton representation using **MediaPipe Holistic**. The system is designed for research in sign language recognition and human motion analysis, supporting output in **Pickle** (`.pkl`) and **Excel** (`.xlsx`) formats.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Keypoint Structure](#keypoint-structure)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage (CLI Only)](#usage-cli-only)
- [Output Formats](#output-formats)
- [Configuration](#configuration)
- [Data Format](#data-format)
- [System Architecture](#system-architecture)
- [Technical Notes](#technical-notes)
- [License](#license)

---

## Project Overview

This pipeline is intended for use in **Continuous Sign Language Recognition (CSLR)** and related human motion research. It is fully compatible with the Isharah dataset format and is optimized for batch processing of large video datasets.

The pipeline performs the following steps:

1. Sequentially reads RGB video frames.
2. Detects body, hand, and face landmarks using MediaPipe Holistic.
3. Extracts 86 keypoints per frame, following the Isharah layout.
4. Saves results in a single aggregated **Pickle** (`.pkl`) file and individual **Excel** (`.xlsx`) files for detailed analysis.

---

## Keypoint Structure

The extracted skeleton consists of **86 keypoints** per frame, structured as follows:

| Region      | Abbreviation | Index Range | Count | MediaPipe Source          |
|-------------|--------------|-------------|-------|--------------------------|
| Left Hand   | GL           | 0–20        | 21    | `left_hand_landmarks`    |
| Right Hand  | GR           | 21–41       | 21    | `right_hand_landmarks`   |
| Mouth       | GM           | 42–60       | 19    | `face_landmarks` (subset)|
| Pose (upper)| GP           | 61–85       | 25    | `pose_landmarks`         |

Keypoints are selected sequentially from the MediaPipe output. The index mapping is defined in `src/config/settings.py`.

---

## Directory Structure

```text
rgb-to-skeleton-mediapipe/
│
├── main.py                         # CLI entry point
├── requirements.txt
├── README.md
├── notebooks/                      # Data analysis notebooks
│
├── src/
│   ├── config/                     # Configuration modules
│   │   ├── paths.py                # Directory paths
│   │   └── settings.py             # Pipeline settings
│   │
│   ├── core/                       # Core pipeline logic
│   │   ├── pipeline.py             # Orchestration
│   │   └── cli.py                  # Argument parsing
│   │
│   ├── extractor/
│   │   └── holistic_86.py          # MediaPipe extraction
│   │
│   └── converter/
│       ├── to_pickle.py            # Aggregated dataset export
│       └── to_excel.py             # Individual video export
│
└── data/
    ├── raw/                        # Input RGB videos (place videos here)
    ├── pickle/                     # Output aggregated .pkl
    └── excel/                      # Output individual .xlsx
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/username/rgb-to-skeleton-mediapipe.git
cd rgb-to-skeleton-mediapipe
```

### 2. Set Up the Conda Environment

> **Note:** Python 3.10 is required. MediaPipe Holistic is only supported up to `mediapipe==0.10.14`.

```bash
conda env create -f environment.yml
conda activate rgb-skeleton
```

---

## Usage (CLI Only)

### Process a Single Video

```bash
python main.py --input data/raw/angry.mp4
```

### Process an Entire Directory

```bash
python main.py --input data/raw/
```

### Options

| Flag              | Description                                    |
|-------------------|------------------------------------------------|
| `--label` / `-l`  | Integer label for the dataset                  |
| `--pickle-name`   | Custom .pkl filename                           |
| `--no-pickle`     | Skip .pkl export                               |
| `--no-excel`      | Skip .xlsx export                              |

---

## Output Formats

### 1. Pickle — `data/pickle/<name>.pkl`

An aggregated dictionary containing all processed samples.

```python
import pickle
with open("data/pickle/your_dataset.pkl", "rb") as f:
    data = pickle.load(f)

# Structure: { "VIDEO_ID": {"keypoints": np.ndarray(T, 86, 2)}, ... }
```

### 2. Excel — `data/excel/<subpath>/<video_id>.xlsx`

Individual spreadsheets for frame-by-frame coordinate analysis.

---

## Configuration

Settings are in `src/config/settings.py` and `src/config/paths.py`.

```python
# settings.py
TOTAL_KEYPOINTS  = 86
USE_3D_COORDINATES = False
SAVE_PICKLE = True
SAVE_EXCEL  = True
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
