# Analisis Kebutuhan Fungsional Aplikasi

Analisis kebutuhan fungsional bertujuan untuk mengidentifikasi fitur-fitur utama yang harus dimiliki oleh Aplikasi Pengolahan Kerangka (*Skeleton Processing Application*) agar dapat mendukung alur standarisasi dataset BISINDO secara menyeluruh. Aplikasi ini terdiri dari dua subsistem utama yang bekerja secara berurutan, yaitu **Subsistem Ekstraksi Kerangka** (modul `src/`) dan **Subsistem Pembagian Dataset** (modul `splitting_data/`).

---

## A. Subsistem Ekstraksi Kerangka (`src/`)

**Tabel 1. Kebutuhan Fungsional Subsistem Ekstraksi Kerangka**

| No | ID Fitur | Kebutuhan Fungsional |
|----|----------|----------------------|
| 1 | FR-A01 | Aplikasi mampu menerima masukan berupa satu *file* video maupun sebuah direktori yang berisi banyak *file* video, yang dikonfigurasi melalui argumen antarmuka baris perintah (`--input`). |
| 2 | FR-A02 | Aplikasi mampu melakukan penelusuran direktori secara rekursif untuk mendeteksi seluruh *file* video yang tersimpan pada berbagai tingkatan subdirektori. |
| 3 | FR-A03 | Aplikasi mampu mengekstrak tepat **86 titik kerangka** (*keypoints*) per *frame* dari video masukan menggunakan model MediaPipe Holistic, yang terdiri dari 21 titik tangan kiri, 21 titik tangan kanan, 19 titik mulut, dan 25 titik tubuh (*pose*). |
| 4 | FR-A04 | Aplikasi mampu mengekspor data kerangka secara agregat ke dalam format **Pickle** (`.pkl`), di mana setiap sampel diidentifikasi dengan ID video sebagai kunci dan matriks koordinat 2D berdimensi `(T, 86, 2)` sebagai nilainya. |
| 5 | FR-A05 | Aplikasi mampu mengekspor data kerangka analitik ke dalam format **Excel** (`.xlsx`) yang diorganisasikan satu berkas per Penanda (contoh: `P1.xlsx`), di mana setiap *sheet* mewakili satu sesi video dengan kolom `kode_video`, `frame`, `keypoint_id`, `x`, dan `y`. |
| 6 | FR-A06 | Aplikasi mampu menentukan berkas Pickle tujuan secara otomatis berdasarkan pemetaan pembagian data, yaitu memisahkan video kelompok *train/dev* dan *test* ke dalam berkas Pickle yang berbeda. |

---

## B. Subsistem Pembagian Dataset (`splitting_data/`)

**Tabel 2. Kebutuhan Fungsional Subsistem Pembagian Dataset**

| No | ID Fitur | Kebutuhan Fungsional |
|----|----------|----------------------|
| 7 | FR-B01 | Aplikasi mampu memuat data referensi label dari berkas Excel (`Gloss dan Tanda Dataset.xlsx`) dan memetakannya ke setiap ID sampel dari berkas Pickle master untuk menghasilkan dataset berlabel (*labeled dataset*). |
| 8 | FR-B02 | Aplikasi mampu membagi dataset menjadi partisi **train** (repetisi R1–R4, 600 sampel) dan **dev** (repetisi R5, 150 sampel) khusus untuk penanda P1–P5 dalam skenario *Signer-Independent* (SI). |
| 9 | FR-B03 | Aplikasi mampu menghasilkan dua varian data uji SI yang terpisah untuk penanda P6: **SI-MAJ** (variasi Mayoritas, kode `MJ`) dan **SI-MIN** (variasi Minoritas, kode `MN`), masing-masing sebanyak 57 sampel. |
| 10 | FR-B04 | Aplikasi mampu menulis hasil pembagian dalam format CSV (*pipe-delimited*) dan berkas teks pendamping ke direktori `results/` dengan struktur `SI/`, `SI-MAJ/`, dan `SI-MIN/`. |
| 11 | FR-B05 | Aplikasi mampu memfilter berkas Pickle master menjadi berkas-berkas Pickle terpartisi sesuai skenario eksperimen: `pose_bisindo_train_dev_si.pkl`, `pose_bisindo_test_si-maj.pkl`, dan `pose_bisindo_test_si-min.pkl`. |
| 12 | FR-B06 | Aplikasi mampu mencegah kebocoran data (*data leakage*) dengan memastikan sampel penanda P6 tidak pernah masuk ke dalam partisi *train* atau *dev* meskipun terdapat inkonsistensi pada berkas CSV pembagian. |
