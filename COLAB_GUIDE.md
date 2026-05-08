# 🚀 Guide to Running the BISINDO Pipeline on Google Colab

Processing large video datasets locally can be slow and resource-intensive. Google Colab provides a powerful virtual environment to execute this pipeline efficiently. Although MediaPipe Holistic primarily utilizes the CPU in Python environments, Colab's infrastructure offers superior I/O speeds and stable processing for batch execution.

This guide explains how to execute this project on Google Colab and how to mount Google Drive for persistent storage.

---

## 1️⃣ Google Drive Preparation

Because the Google Colab file system is ephemeral (deleted after the session ends), you must store your dataset and source code in **Google Drive**.

1. Create a project folder in your Drive (e.g., `BISINDO_Project`).
2. Upload the entire `rgb-to-skeleton-mediapipe` repository into that folder.
3. Place your raw `.mp4` videos in `data/raw/`.

**Structure:**
```text
MyDrive/
└── BISINDO_Project/
    └── rgb-to-skeleton-mediapipe/
        ├── data/
        │   ├── raw/              <-- Input Videos
        │   └── results/          <-- Splitting CSVs (Essential)
        ├── src/
        └── main.py
```

---

## 2️⃣ Create a Google Colab Notebook

1. Open [Google Colab](https://colab.research.google.com/).
2. Create a **New notebook**.
3. **Runtime Configuration**: 
   - While this pipeline runs on the **CPU**, using a **High-RAM** runtime (if available in Colab Pro) can help with very large datasets.
   - Go to **Runtime > Change runtime type** and ensure it is set to **None** (CPU) to avoid unnecessary GPU quota usage.

---

## 3️⃣ Mount Google Drive to Colab

Run the following cell to access your Drive files:

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## 4️⃣ Navigate to the Project Directory

```python
import os

# Update this path to match your folder structure
project_path = '/content/drive/MyDrive/BISINDO_Project/rgb-to-skeleton-mediapipe'
os.chdir(project_path)

# Verify location
!pwd
!ls -la
```

---

## 5️⃣ Install Dependencies

MediaPipe requires `version <= 0.10.14` for stable holistic landmark detection.

```bash
!pip install -r requirements.txt
```

---

## 6️⃣ Running the Pipeline

Once the environment is ready, execute `main.py`. The pipeline will automatically read your splitting configuration from `splitting_data/results/` and sort the data into the appropriate pickle files.

### Standard Execution
```bash
!python main.py --input data/raw/ --pickle-name pose_bisindo
```

*Note: This command will generate both Pickle datasets (train_dev & test) and subject-specific Excel files.*

---

## 💡 Colab Execution Tips & Tricks

1.  **CPU Processing**: MediaPipe Holistic for Python is currently optimized for CPU. Running on Colab is still significantly faster than most local setups due to optimized virtualization.
2.  **Persistent Storage**: Since you are working directly within the mounted `/content/drive` path, all outputs in `data/pickle/` and `data/excel/` are saved in real-time. You won't lose data if the session expires.
3.  **Large Datasets**: If you have thousands of videos, Colab may still take several hours. The free tier allows up to **12 hours** of execution.
4.  **Avoid Disconnection**: Keep the browser tab active. You can occasionally interact with the notebook (e.g., creating a new cell) to prevent the "Idle" timeout.

---

## 📄 License
This project is licensed under the MIT License.
