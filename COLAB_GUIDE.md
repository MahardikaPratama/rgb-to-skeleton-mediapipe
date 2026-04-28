# 🚀 Guide to Running the BISINDO Pipeline on Google Colab (GPU)

If you have thousands of videos, processing them locally using a CPU can take days. Therefore, running this pipeline on Google Colab with hardware acceleration is highly recommended.

This guide explains how to execute this project on Google Colab, as well as how to mount Google Drive to ensure your data is automatically and securely stored.

---

## 1️⃣ Google Drive Preparation

Because the Google Colab file system is ephemeral (it will be automatically deleted when the runtime ends), you must store your dataset and source code in **Google Drive**.

1. Open your [Google Drive](https://drive.google.com/).
2. Create a new folder, for example, `BISINDO_Project`.
3. Upload the entire source code folder of this project (including the `rgb-to-skeleton-mediapipe` directory) into the `BISINDO_Project` folder.
4. Upload your raw `.mp4` videos into the `BISINDO_Project/rgb-to-skeleton-mediapipe/data/raw/` directory.

Your Google Drive directory structure should look like this:
```text
MyDrive/
└── BISINDO_Project/
    └── rgb-to-skeleton-mediapipe/
        ├── data/
        │   └── raw/              <-- Place your .mp4 videos here
        ├── src/
        ├── README.md
        ├── requirements.txt
        └── main.py
```

---

## 2️⃣ Create a Google Colab Notebook

1. Open [Google Colab](https://colab.research.google.com/).
2. Create a new notebook by selecting **File > New notebook**.
3. **Important! Enable the GPU:**
   - Click the **Runtime > Change runtime type** menu.
   - Under **Hardware accelerator**, select **T4 GPU** (or any other available GPU).
   - Click **Save**.

---

## 3️⃣ Mount Google Drive to Colab

To access the files in your Google Drive from the Colab notebook, execute the following code cell:

```python
from google.colab import drive
drive.mount('/content/drive')
```
*When you run this cell, an authorization pop-up will appear. Follow the on-screen instructions to grant access to your Google Drive account.*

---

## 4️⃣ Navigate to the Project Directory

Once your Drive is successfully mounted, navigate to your project folder using the following code block:

```python
import os

# Change the path below according to your folder's location in Google Drive
project_path = '/content/drive/MyDrive/BISINDO_Project/rgb-to-skeleton-mediapipe'
os.chdir(project_path)

# Verify if we are in the correct directory
!pwd
!ls -la
```

---

## 5️⃣ Install Dependencies

MediaPipe requires a specific version limit (`<= 0.10.14`) to ensure its holistic detection features (especially hand and lip tracking) function optimally within this pipeline.

Run this command to install the required dependencies:

```bash
# Install libraries according to the project's requirements
!pip install -r requirements.txt
```

---

## 6️⃣ Running the Pipeline (Execution)

After the preparation is complete, you can directly execute `main.py` using a terminal command (prepended with `!`). Executing the script in Colab takes advantage of optimized read/write speeds and cloud-based processors.

### Example: Running the Full Pipeline

If your raw videos are in `data/raw/` and you want to extract all of them into Pickle, JSON, and Excel formats by default, run this cell:

```bash
!python main.py --input data/raw/ --pickle-name dataset_bisindo_colab.pkl
```

### ⚡ Saving Execution Time (Fast Mode)
To significantly speed up the data extraction process for large datasets, **it is highly recommended to disable video previews**. Colab will take a considerable amount of time if it has to re-render overlay and skeleton videos into `.mp4` format.

Add the `--no-preview` parameter:

```bash
!python main.py --input data/raw/ --pickle-name dataset_bisindo_colab.pkl --no-preview
```

### Additional Options (Disabling Other Outputs)
Just like running the pipeline on a local terminal, you can set flags to disable JSON or Excel outputs if they are not needed, saving storage capacity on your Google Drive:
```bash
!python main.py --input data/raw/ \
    --pickle-name dataset_bisindo_colab.pkl \
    --no-preview \
    --no-json \
    --no-excel
```

---

## 💡 Colab Execution Tips & Tricks

1. **Keep the Browser Tab Active:** 
   Google Colab will automatically terminate the runtime if your browser tab is left idle for too long. Monitor the progress bar occasionally to keep the connection alive.
2. **MediaPipe in Python Limitations:** 
   Note that the MediaPipe Python library currently relies heavily on the CPU. However, the primary bottleneck in video processing (OpenCV reading/writing video frames) operates far more efficiently in the Google Colab virtual machine environment.
3. **Direct Storage in Google Drive:** 
   Because you have mounted and changed your directory (`cd`) directly into Google Drive, all output data (Pickle, JSON, Excel) will automatically sync and be permanently saved to your Google Drive account in real-time. Your data is safe even if Google Colab disconnects.
4. **Execution Time Limit (Max 12 Hours):**
   The free version of Google Colab has a maximum runtime limit of approximately 12 hours per session. For datasets containing tens to hundreds of videos, this should be more than enough time to complete the process before disconnection.
