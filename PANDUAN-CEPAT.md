# 🚀 Panduan Cepat (Quick Start)

Panduan ini akan membantu Anda menjalankan proses pengubahan video RGB menjadi pose skeleton untuk dataset BISINDO dari awal (clone repositori) hingga selesai (menghasilkan file pickle dan excel). Silakan ikuti langkah-langkah di bawah ini.

## 1. Clone Repositori

Pertama, unduh kode sumber dari GitHub:
```bash
git clone https://github.com/MahardikaPratama/rgb-to-skeleton-mediapipe.git
cd rgb-to-skeleton-mediapipe
```

## 2. Instalasi & Setup Miniconda

Sistem ini membutuhkan environment Python yang spesifik (Python 3.10 dan MediaPipe 0.10.14). Kami sangat menyarankan penggunaan Miniconda untuk mengelola dependensi.

1. Instal Miniconda dengan mengikuti panduan resminya: [Miniconda Installation Guide](https://www.anaconda.com/docs/getting-started/miniconda/install/overview)
2. Buat environment baru menggunakan file konfigurasi yang telah disediakan:
    ```bash
    conda env create -f environment.yml
    ```
3. Aktifkan environment yang telah dibuat:
    ```bash
    conda activate rgb-skeleton
    ```

## 3. Standarisasi Nama Folder dan File (Rename)

Folder data mentah umumnya masih menggunakan format nama lama (legacy), seperti `ACHMAD_AKU CIUM BADAN DIA_01.mp4`. Anda **wajib** menstandarisasi nama folder dan file sebelum melanjutkan ke proses ekstraksi.

1. Pastikan folder data mentah Anda sudah tersedia, misalnya diletakkan di dalam direktori `data/raw/`.
2. Jalankan script pengganti nama (rename):
    ```bash
    # Lakukan simulasi (dry-run) terlebih dahulu untuk memverifikasi perubahan:
    python rename_folder_file.py --input data/raw --dry-run
    
    # Jika hasil simulasi sudah sesuai, jalankan tanpa --dry-run untuk mengubah nama file secara permanen:
    python rename_folder_file.py --input data/raw
    ```

## 4. (Penting!) Menggunakan Checkpoint Pickle

Proyek ini mendukung fitur *checkpoint* untuk file pickle. Jika proses ekstraksi terhenti, Anda tidak perlu mengulang dari awal. Sistem akan otomatis melakukan *replace* atau *append* (menambahkan data baru ke file lama) jika mendeteksi keberadaan file pickle dengan nama yang sama.

**Panduan Penggunaan:**
1. Unduh file checkpoint terbaru dari tautan Google Drive berikut:
   👉 [Tautan Unduh Checkpoint](https://drive.google.com/drive/folders/1-K5yotG0PPatvr7l0L81dYc2YHr5jclB)
2. Masukkan file `.pkl` hasil unduhan tersebut ke dalam folder `data/pickle/` (silakan buat folder tersebut secara manual jika belum tersedia).
3. Saat menjalankan proses ekstraksi (pada langkah 5), pastikan Anda menggunakan nama pickle (parameter `--pickle-name`) yang sama persis dengan nama file yang baru Anda pindahkan. Data yang baru diproses akan otomatis digabungkan.

## 5. Proses Ekstraksi Skeleton (Jalankan Main)

Setelah data video mentah distandarisasi dan file checkpoint disiapkan (jika ada), Anda dapat langsung mengekstrak landmark/skeleton menggunakan `main.py`.

* **Untuk memproses satu video spesifik:**
  ```bash
  python main.py --input data/raw/S06/P1_S06_R1.mp4 --pickle-name pose_bisindo
  ```

* **Untuk memproses seluruh folder secara otomatis (bulk process):**
  ```bash
  python main.py --input data/raw/ --pickle-name pose_bisindo
  ```

Output dari proses ini akan tersimpan di dalam folder `data/pickle/` (berformat `.pkl`) dan `data/excel/` (berformat tabular `.xlsx` untuk keperluan pengecekan manual).

## 6. (Opsional) Membagi Dataset (Data Splitting)

Jika Anda membutuhkan pembagian data ke dalam subset Train, Validation, dan Test:

1. Buat file CSV untuk pembagian data:
   ```bash
   python splitting_data/data_splitting.py --seed 42
   ```
2. Lakukan filtering pada hasil pickle utama menjadi file terpisah untuk train/dev/test:
   ```bash
   python splitting_data/split_pose_pickle.py --seed 42
   ```

Selamat! Data pose skeleton Anda telah siap untuk digunakan dalam pelatihan model Deep Learning.
