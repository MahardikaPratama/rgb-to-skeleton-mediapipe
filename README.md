# RGB to Skeleton — MediaPipe Pipeline

Konversi video RGB menjadi representasi skeleton 86-keypoint menggunakan **MediaPipe Holistic**. Pipeline ini menghasilkan data skeleton dalam format `.npy`, `.json`, dan `.pkl`, serta dapat meng-generate video preview berupa skeleton-only maupun skeleton overlay di atas video asli.

---

## Daftar Isi

- [Tentang Proyek](#tentang-proyek)
- [Struktur Keypoint (Isharah Layout)](#struktur-keypoint-isharah-layout)
- [Struktur Direktori](#struktur-direktori)
- [Instalasi](#instalasi)
- [Cara Penggunaan](#cara-penggunaan)
  - [Satu Video](#satu-video)
  - [Satu Folder](#satu-folder)
  - [Opsi Tambahan](#opsi-tambahan)
- [Output yang Dihasilkan](#output-yang-dihasilkan)
- [Konfigurasi](#konfigurasi)
- [Format Data](#format-data)
- [Visualisasi](#visualisasi)
- [Arsitektur Kode](#arsitektur-kode)

---

## Tentang Proyek

Pipeline ini dirancang untuk penelitian **Continuous Sign Language Recognition (CSLR)** dan kompatibel dengan format dataset **Isharah**.

Proses yang dilakukan:

1. Membaca video RGB frame per frame
2. Mendeteksi landmark tubuh menggunakan MediaPipe Holistic
3. Memilih 86 keypoint yang relevan
4. Menyimpan keluaran dalam format NumPy, JSON, dan Pickle
5. Menghasilkan video preview (opsional)

---

## Struktur Keypoint (Isharah Layout)

Total keypoint: **86**

| Nama              | Singkatan | Indeks  | Jumlah | Sumber MediaPipe          |
| ----------------- | --------- | ------- | ------ | ------------------------- |
| Left Hand         | GL        | `0–20`  | 21     | `left_hand_landmarks`     |
| Right Hand        | GR        | `21–41` | 21     | `right_hand_landmarks`    |
| Mouth             | GM        | `42–60` | 19     | `face_landmarks` (subset) |
| Pose (upper body) | GP        | `61–85` | 25     | `pose_landmarks`          |

```python
import numpy as np

left_hand_idx  = np.arange(0,  21)   # GL
right_hand_idx = np.arange(21, 42)   # GR
mouth_idx      = np.arange(42, 61)   # GM
pose_idx       = np.arange(61, 86)   # GP
```

### Pemilihan Landmark (Sequential)

Semua region dipilih secara berurut langsung dari output MediaPipe (landmark 0 hingga N-1):

| Region     | Sumber MediaPipe       | Landmark yang diambil        |
| ---------- | ---------------------- | ---------------------------- |
| Left Hand  | `left_hand_landmarks`  | landmark 0 – 20 (semua 21)   |
| Right Hand | `right_hand_landmarks` | landmark 0 – 20 (semua 21)   |
| Mouth      | `face_landmarks`       | landmark 0 – 18 (19 pertama) |
| Pose       | `pose_landmarks`       | landmark 0 – 24 (25 pertama) |

Count per region didapat langsung dari range di `config.py` — tidak ada daftar indeks hardcoded.

---

## Struktur Direktori

```
rgb-to-skeleton-mediapipe/
│
├── main.py                         # Entry point CLI
├── requirements.txt
├── README.md
│
├── src/
│   ├── config.py                   # Semua konfigurasi global
│   │
│   ├── extractor/
│   │   └── holistic_86.py         # Ekstraksi keypoint dari video
│   │
│   ├── processor/
│   │   └── keypoint_selector.py   # Validasi shape keypoint
│   │
│   ├── converter/
│   │   ├── to_json.py             # Ekspor ke JSON
│   │   └── to_pickle.py           # Ekspor ke Pickle
│   │
│   └── visualizer/
│       ├── draw_skeleton.py       # Render frame skeleton
│       └── preview_generator.py   # Buat video preview
│
└── data/
    ├── raw/                        # Input video RGB (taruh video di sini)
    ├── skeleton/                   # Output .npy
    ├── json/                       # Output .json
    ├── pickle/                     # Output .pkl
    └── preview/
        ├── skeleton_only/          # Preview skeleton (latar hitam)
        └── skeleton_rgb/           # Preview skeleton overlay video asli
```

---

## Instalasi

### 1. Clone repository

```bash
git clone https://github.com/username/rgb-to-skeleton-mediapipe.git
cd rgb-to-skeleton-mediapipe
```

### 2. Buat Conda Environment

> ⚠️ **Wajib menggunakan Python 3.10.** MediaPipe `mp.solutions.holistic` hanya tersedia di `mediapipe <= 0.10.14`, dan versi ini paling stabil di Python 3.10.

```bash
conda env create -f environment.yml
conda activate rgb-skeleton
```

Untuk update environment jika `environment.yml` berubah:

```bash
conda env update -f environment.yml --prune
```

### Catatan Versi

| Paket     | Versi     | Alasan                                               |
| --------- | --------- | ---------------------------------------------------- |
| Python    | `3.10`    | Kompatibel dengan mediapipe 0.10.14                  |
| mediapipe | `0.10.14` | Versi terakhir yang memiliki `mp.solutions.holistic` |
| numpy     | `< 2.0`   | Kompatibel dengan mediapipe 0.10.14                  |

---

## Cara Penggunaan

Aplikasi ini dapat dijalankan melalui dua cara:

1.  **Aplikasi UI (RGB to Skeleton Studio)**: Untuk penggunaan interaktif melalui web browser.
2.  **Command-Line Interface (CLI)**: Untuk pemrosesan batch dan integrasi skrip.

### RGB to Skeleton Studio (Aplikasi UI)

Studio ini menyediakan antarmuka grafis untuk mengelola video, menjalankan proses ekstraksi skeleton, memantau status, dan melihat hasil secara visual. Dibangun menggunakan Streamlit (frontend) dan FastAPI (backend).

**1. Instalasi Dependensi**

Pastikan semua paket di `requirements.txt` terinstal.

```bash
pip install -r requirements.txt
```

**2. Jalankan Backend (FastAPI)**

Buka terminal baru dan jalankan server API:

```bash
uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload
```

Server ini akan menangani permintaan pemrosesan dari antarmuka Streamlit.

**3. Jalankan Frontend (Streamlit)**

Buka terminal lain dan jalankan aplikasi Streamlit:

```bash
streamlit run RGB_to_Skeleton_Studio.py
```

Aplikasi akan terbuka secara otomatis di browser Anda, biasanya di `http://localhost:8501`.

**Fitur Aplikasi:**

- **Dashboard**: Menampilkan ringkasan data dan job terakhir.
- **Processing**: Memilih video (tunggal, ganda, atau per folder) dan memulai proses ekstraksi.
- **Jobs**: Melihat status semua job (pending, running, done, failed).
- **Results**: Menjelajahi, memvisualisasikan, dan mengunduh hasil skeleton (`.npy`).
- **Previews**: Menampilkan video preview yang sudah di-render atau membuat animasi GIF secara langsung.
- **Data Management**: Mengunggah video baru ke folder `data/raw`.

### Command-Line Interface (CLI)

#### Satu Video

```bash
python main.py --input data/raw/marah.mp4
```

#### Satu Folder

Memproses semua file video dalam folder secara berurutan:

```bash
python main.py --input data/raw/
```

Format video yang didukung: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`

### Tambahkan Label (untuk Pickle)

```bash
python main.py --input data/raw/marah.mp4 --label 3
```

### Opsi Tambahan

| Flag                 | Keterangan                                          |
| -------------------- | --------------------------------------------------- |
| `--no-npy`           | Tidak menyimpan file `.npy`                         |
| `--no-json`          | Tidak menyimpan file `.json`                        |
| `--no-pickle`        | Tidak menyimpan file `.pkl`                         |
| `--no-preview`       | Tidak menghasilkan video preview sama sekali        |
| `--no-overlay`       | Tidak menghasilkan preview overlay (skeleton + RGB) |
| `--no-skeleton-only` | Tidak menghasilkan preview skeleton saja            |

**Contoh: proses folder, simpan hanya .npy, tanpa preview:**

```bash
python main.py --input data/raw/ --no-json --no-pickle --no-preview
```

**Contoh: proses satu video, proses tapi tanpa overlay:**

```bash
python main.py --input data/raw/satu.mp4 --no-overlay
```

---

## Output yang Dihasilkan

### 1. NumPy Array — `data/skeleton/<nama>.npy`

```python
import numpy as np

data = np.load("data/skeleton/marah.npy")
print(data.shape)  # (T, 86, 3)
# T = jumlah frame
# 86 = jumlah keypoint
# 3 = koordinat (x, y, z)
```

Akses per region:

```python
left_hand  = data[:, 0:21,  :]   # GL
right_hand = data[:, 21:42, :]   # GR
mouth      = data[:, 42:61, :]   # GM
pose       = data[:, 61:86, :]   # GP
```

### 2. JSON — `data/json/<nama>.json`

```json
{
  "video_name": "marah",
  "num_frames": 72,
  "num_keypoints": 86,
  "dimensions": 3,
  "keypoints": [[[...], ...], ...]
}
```

### 3. Pickle — `data/pickle/<nama>.pkl`

```python
import pickle

with open("data/pickle/marah.pkl", "rb") as f:
    obj = pickle.load(f)

# obj["video_name"] → str
# obj["data"]       → np.ndarray (T, 86, 3) float32
# obj["label"]      → int or None
```

### 4. Video Preview — `data/preview/`

- **`skeleton_only/`** — Skeleton dirender di atas latar gelap
- **`skeleton_rgb/`** — Skeleton di-overlay di atas frame video asli

Warna per region:

| Region     | Warna  |
| ---------- | ------ |
| Left Hand  | Cyan   |
| Right Hand | Kuning |
| Mouth      | Merah  |
| Pose       | Hijau  |

---

## Konfigurasi

Semua konfigurasi ada di `src/config.py`. File ini hanya berisi **layout publik** dan parameter pipeline — bukan detail seleksi landmark.

```python
# === Layout Keypoint (Isharah Format) ===
TOTAL_KEYPOINTS  = 86
LEFT_HAND_RANGE  = (0,  21)   # GL
RIGHT_HAND_RANGE = (21, 42)   # GR
MOUTH_RANGE      = (42, 61)   # GM
POSE_RANGE       = (61, 86)   # GP

# === MediaPipe ===
MEDIAPIPE_CONFIG = {
    "model_complexity": 1,       # 0=cepat, 1=seimbang, 2=akurat
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5,
}

# === Koordinat ===
USE_3D_COORDINATES = True   # False = hanya (x, y)

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

## Format Data

Koordinat disimpan dalam format **normalized MediaPipe**:

- `x` : horizontal, range `[0.0, 1.0]` dari kiri ke kanan
- `y` : vertical, range `[0.0, 1.0]` dari atas ke bawah
- `z` : kedalaman (relatif terhadap pinggul/wrist), hanya tersedia di pose dan tangan

Keypoint yang tidak terdeteksi (misalnya tangan tidak terlihat di frame) akan diisi dengan `[0.0, 0.0, 0.0]`.

---

## Arsitektur Kode

```
Input Video
     │
     ▼
Holistic86Extractor          (src/extractor/holistic_86.py)
  └─ MediaPipe Holistic → 86 keypoints per frame
  └─ Output: np.ndarray (T, 86, 3)
     │
     ▼
KeypointSelector             (src/processor/keypoint_selector.py)
  └─ Validasi shape (T, 86, C)
     │
     ├──────────────────────────────►  JSONConverter   → .json
     ├──────────────────────────────►  PickleConverter → .pkl
     ├──────────────────────────────►  np.save()       → .npy
     │
     └──────────────────────────────►  PreviewGenerator
                                          ├── skeleton_only → video latar gelap
                                          └── overlay       → video + skeleton
```

---

## Catatan Teknis

- Pipeline **tidak melakukan normalisasi koordinat**. Koordinat disimpan apa adanya dari MediaPipe (range 0–1).
- Jika MediaPipe gagal mendeteksi suatu landmark pada sebuah frame (misalnya tangan tidak terlihat), koordinat akan diisi `[0.0, 0.0, 0.0]`.
- Resolusi video preview mengikuti resolusi asli video input. Jika resolusi tidak dapat dibaca, akan fallback ke `PREVIEW_RESOLUTION` di config.

---

## Lisensi

MIT License. Lihat file [LICENSE](LICENSE).
