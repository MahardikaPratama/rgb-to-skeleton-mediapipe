# RGB to Skeleton — MediaPipe Pipeline

Convert RGB videos into an 86-keypoint skeleton representation using **MediaPipe Holistic**. This pipeline generates skeleton data in `.npy`, `.json`, and `.pkl` formats, and can render preview videos featuring skeleton-only outputs or skeletons overlaid on the original video.

---

## Table of Contents

- [About the Project](#about-the-project)
- [Keypoint Structure (Isharah Layout)](#keypoint-structure-isharah-layout)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [RGB to Skeleton Studio (UI Application)](#rgb-to-skeleton-studio-ui-application)
  - [Command-Line Interface (CLI)](#command-line-interface-cli)
- [Generated Outputs](#generated-outputs)
- [Configuration](#configuration)
- [Data Format](#data-format)
- [Code Architecture](#code-architecture)
- [Technical Notes](#technical-notes)

---

## About the Project

This pipeline is designed for **Continuous Sign Language Recognition (CSLR)** research and is compatible with the **Isharah** dataset format.

The process involves:

1. Reading RGB videos frame by frame
2. Detecting body landmarks using MediaPipe Holistic
3. Extracting the 86 relevant keypoints
4. Saving the outputs in NumPy, JSON, and Pickle formats
5. Generating video previews (optional)

---

## Keypoint Structure (Isharah Layout)

Total keypoints: **86**

| Name              | Abbreviation | Index  | Count | MediaPipe Source          |
| ----------------- | ------------ | ------ | ----- | ------------------------- |
| Left Hand         | GL           | `0–20` | 21    | `left_hand_landmarks`     |
| Right Hand        | GR           | `21–41`| 21    | `right_hand_landmarks`    |
| Mouth             | GM           | `42–60`| 19    | `face_landmarks` (subset) |
| Pose (upper body) | GP           | `61–85`| 25    | `pose_landmarks`          |

```python
import numpy as np

left_hand_idx  = np.arange(0,  21)   # GL
right_hand_idx = np.arange(21, 42)   # GR
mouth_idx      = np.arange(42, 61)   # GM
pose_idx       = np.arange(61, 86)   # GP
```

### Landmark Selection (Sequential)

All regions are selected sequentially directly from the MediaPipe output (landmarks 0 to N-1):

| Region     | MediaPipe Source       | Extracted Landmarks          |
| ---------- | ---------------------- | ---------------------------- |
| Left Hand  | `left_hand_landmarks`  | Landmarks 0 – 20 (all 21)    |
| Right Hand | `right_hand_landmarks` | Landmarks 0 – 20 (all 21)    |
| Mouth      | `face_landmarks`       | Landmarks 0 – 18 (first 19)  |
| Pose       | `pose_landmarks`       | Landmarks 0 – 24 (first 25)  |

The count per region is obtained directly from the range in `config.py` — there are no hardcoded index lists.

---

## Directory Structure

```text
rgb-to-skeleton-mediapipe/
│
├── main.py                         # CLI entry point
├── requirements.txt
├── README.md
│
├── src/
│   ├── config.py                   # Global configuration
│   │
│   ├── extractor/
│   │   └── holistic_86.py         # Keypoint extraction from video
│   │
│   ├── processor/
│   │   └── keypoint_selector.py   # Keypoint shape validation
│   │
│   ├── converter/
│   │   ├── to_json.py             # Export to JSON
│   │   └── to_pickle.py           # Export to Pickle
│   │
│   └── visualizer/
│       ├── draw_skeleton.py       # Render skeleton frames
│       └── preview_generator.py   # Generate preview videos
│
└── data/
    ├── raw/                        # Input RGB videos (place videos here)
    ├── skeleton/                   # Output .npy
    ├── json/                       # Output .json
    ├── pickle/                     # Output .pkl
    └── preview/
        ├── skeleton_only/          # Skeleton preview (dark background)
        └── skeleton_rgb/           # Skeleton overlaid on original video
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/username/rgb-to-skeleton-mediapipe.git
cd rgb-to-skeleton-mediapipe
```

### 2. Create a Conda Environment

> ⚠️ **Python 3.10 is required.** The MediaPipe `mp.solutions.holistic` module is only available in `mediapipe <= 0.10.14`, and this version is most stable on Python 3.10.

```bash
conda env create -f environment.yml
conda activate rgb-skeleton
```

To update the environment if `environment.yml` changes:

```bash
conda env update -f environment.yml --prune
```

### Version Notes

| Package   | Version   | Reason                                                  |
| --------- | --------- | ------------------------------------------------------- |
| Python    | `3.10`    | Compatible with mediapipe 0.10.14                       |
| mediapipe | `0.10.14` | Last version to include `mp.solutions.holistic`         |
| numpy     | `< 2.0`   | Compatible with mediapipe 0.10.14                       |

---

## Usage

> 🚀 **Want Much Faster Processing?** If you are processing thousands of videos, it is **highly recommended** to run the pipeline on Google Colab (using a GPU). Please refer to our **[Google Colab Execution Guide](COLAB_GUIDE.md)** for more details.

This application can be executed locally in two ways:

1.  **RGB to Skeleton Studio (UI Application)**: For interactive use via a web browser.
2.  **Command-Line Interface (CLI)**: For batch processing and script integration.

### RGB to Skeleton Studio (UI Application)

This Studio provides a graphical interface to manage videos, run the skeleton extraction process, monitor status, and visually inspect the results. It is built using Streamlit (frontend) and FastAPI (backend).

**1. Install Dependencies**

Ensure all packages in `requirements.txt` are installed.

```bash
pip install -r requirements.txt
```

**2. Run the Backend (FastAPI)**

Open a new terminal and start the API server:

```bash
uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
```

This server will handle processing requests from the Streamlit interface.

**3. Run the Frontend (Streamlit)**

Open another terminal and start the Streamlit application:

```bash
streamlit run RGB_to_Skeleton_Studio.py
```

The application will automatically open in your browser, typically at `http://localhost:8501`.

**Application Features:**

- **Dashboard**: Displays data summaries and recent jobs.
- **Processing**: Select videos (single, multiple, or by folder) and initiate the extraction process.
- **Jobs**: View the status of all jobs (pending, running, done, failed).
- **Results**: Browse, visualize, and download skeleton results (`.npy`).
- **Previews**: View rendered preview videos or generate GIF animations on the fly.
- **Data Management**: Upload new videos to the `data/raw` folder.

### Command-Line Interface (CLI)

#### Single Video

```bash
python main.py --input data/raw/angry.mp4
```

#### Entire Folder

Process all video files in a directory sequentially:

```bash
python main.py --input data/raw/
```

Process all video files in a folder without generating previews, overlays, or skeleton-only videos, and save them into `baseline_bisindo.pkl`:
```bash
python main.py --input data/raw/ --pickle-name baseline_bisindo.pkl --no-preview --no-overlay --no-skeleton-only
```

Supported video formats: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`

### Add a Label (for Pickle output)

```bash
python main.py --input data/raw/angry.mp4 --label 3
```

### Additional Options

| Flag                 | Description                                         |
| -------------------- | --------------------------------------------------- |
| `--no-npy`           | Do not save `.npy` files                            |
| `--no-json`          | Do not save `.json` files                           |
| `--no-pickle`        | Do not save `.pkl` files                            |
| `--no-preview`       | Do not generate any preview videos                  |
| `--no-overlay`       | Do not generate overlay previews (skeleton + RGB)   |
| `--no-skeleton-only` | Do not generate skeleton-only previews              |

**Example: Process a folder, save only `.npy`, and disable previews:**

```bash
python main.py --input data/raw/ --no-json --no-pickle --no-preview
```

**Example: Process a single video, but skip generating the overlay:**

```bash
python main.py --input data/raw/one.mp4 --no-overlay
```

---

## Generated Outputs

### 1. NumPy Array — `data/skeleton/<name>.npy`

```python
import numpy as np

data = np.load("data/skeleton/angry.npy")
print(data.shape)  # (T, 86, 3)
# T = number of frames
# 86 = number of keypoints
# 3 = coordinates (x, y, z)
```

Access by region:

```python
left_hand  = data[:, 0:21,  :]   # GL
right_hand = data[:, 21:42, :]   # GR
mouth      = data[:, 42:61, :]   # GM
pose       = data[:, 61:86, :]   # GP
```

### 2. JSON — `data/json/<name>.json`

```json
{
  "video_name": "angry",
  "num_frames": 72,
  "num_keypoints": 86,
  "dimensions": 3,
  "keypoints": [[[...], ...], ...]
}
```

### 3. Pickle — `data/pickle/<name>.pkl`

```python
import pickle

with open("data/pickle/angry.pkl", "rb") as f:
    obj = pickle.load(f)

# obj["video_name"] → str
# obj["data"]       → np.ndarray (T, 86, 3) float32
# obj["label"]      → int or None
```

### 4. Video Previews — `data/preview/`

- **`skeleton_only/`** — Skeleton rendered on a dark background
- **`skeleton_rgb/`** — Skeleton overlaid on the original video frames

Colors per region:

| Region     | Color  |
| ---------- | ------ |
| Left Hand  | Cyan   |
| Right Hand | Yellow |
| Mouth      | Red    |
| Pose       | Green  |

---

## Configuration

All configuration settings are located in `src/config.py`. This file contains the **public layout** and pipeline parameters — not the detailed landmark selection logic.

```python
# === Keypoint Layout (Isharah Format) ===
TOTAL_KEYPOINTS  = 86
LEFT_HAND_RANGE  = (0,  21)   # GL
RIGHT_HAND_RANGE = (21, 42)   # GR
MOUTH_RANGE      = (42, 61)   # GM
POSE_RANGE       = (61, 86)   # GP

# === MediaPipe ===
MEDIAPIPE_CONFIG = {
    "model_complexity": 1,       # 0=fast, 1=balanced, 2=accurate
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5,
}

# === Coordinates ===
USE_3D_COORDINATES = False   # False = only (x, y) coordinates

# === Output ===
SAVE_NUMPY  = True
SAVE_JSON   = True
SAVE_PICKLE = True

# === Preview ===
PREVIEW_FPS        = 25
PREVIEW_RESOLUTION = (640, 480)   # default fallback
DRAW_CONNECTIONS   = True
DRAW_JOINTS        = True
JOINT_RADIUS       = 4
LINE_THICKNESS     = 2
```

---

## Data Format

Coordinates are saved in **normalized MediaPipe** format:

- `x` : horizontal, range `[0.0, 1.0]` from left to right
- `y` : vertical, range `[0.0, 1.0]` from top to bottom
- `z` : depth (relative to the hips/wrists), available only for pose and hands

Keypoints that are not detected (e.g., when a hand is out of the frame) will be filled with `[0.0, 0.0, 0.0]`.

---

## Code Architecture

```text
Input Video
     │
     ▼
Holistic86Extractor          (src/extractor/holistic_86.py)
  └─ MediaPipe Holistic → 86 keypoints per frame
  └─ Output: np.ndarray (T, 86, 3)
     │
     ▼
KeypointSelector             (src/processor/keypoint_selector.py)
  └─ Validate shape (T, 86, C)
     │
     ├──────────────────────────────►  JSONConverter   → .json
     ├──────────────────────────────►  PickleConverter → .pkl
     ├──────────────────────────────►  np.save()       → .npy
     │
     └──────────────────────────────►  PreviewGenerator
                                          ├── skeleton_only → dark background video
                                          └── overlay       → video + skeleton
```

---

## Technical Notes

- The pipeline **does not perform coordinate normalization**. Coordinates are saved exactly as provided by MediaPipe (range 0–1).
- If MediaPipe fails to detect a landmark in a frame (e.g., a hand is out of view), the coordinates will be set to `[0.0, 0.0, 0.0]`.
- The resolution of the preview video matches the original input video's resolution. If the resolution cannot be determined, it will fall back to `PREVIEW_RESOLUTION` in the config.

---

## License

MIT License. See the [LICENSE](LICENSE) file for more information.
