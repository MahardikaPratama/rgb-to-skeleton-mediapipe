# Dokumentasi Reverse Engineering: BISINDO Video-to-Skeleton Preprocessing Pipeline
*(Diadaptasi berdasarkan struktur Laporan TA Referensi - APE)*

---

## III.6.6. Penyusunan Skenario Preprocessing (Eksperimen)

Penyusunan skenario eksperimen pada proyek ini difokuskan pada standarisasi konversi video RGB BISINDO (Bahasa Isyarat Indonesia) menjadi data *skeleton* menggunakan MediaPipe Holistic. Perancangan skenario mempertimbangkan tiga aspek utama: format data masukan (video mp4), kerumitan ekstraksi titik-titik *landmark* (86 *keypoints*), dan format data keluaran yang bervariasi (JSON, Pickle, dan Excel) untuk mendukung analisis lanjutan temporal maupun spasial. Hasil dari tahap ini adalah tersusunnya alur *preprocessing* yang sistematis mulai dari pembacaan direktori rekursif hingga ekspor *dataset* yang siap digunakan untuk *training* model pembelajaran mesin.

## III.6.7. Pengembangan Aplikasi Pipeline Preprocessing (APP)

APP (Aplikasi Pipeline Preprocessing) dikembangkan menggunakan *environment* Python dengan pustaka utama OpenCV dan MediaPipe. Proses pengembangan mengadopsi prinsip *Single Responsibility Principle* (SRP) di mana kode monolitik dipecah menjadi modul-modul spesifik. Model pengembangan ini memastikan bahwa logika orkestrasi utama (*core*), konfigurasi (*config*), dan konversi data (*converter*) terpisah dengan jelas. Karakteristik modular ini sesuai dengan pengembangan APP yang dilakukan secara terstruktur dan terukur, sehingga memudahkan pemeliharaan (pembaruan versi dependensi) serta skalabilitas (penambahan format *export* baru).

Sebelum masuk ke tahap implementasi, dilakukan analisis kebutuhan fungsional. Berdasarkan fungsionalitas tersebut, dirancang struktur aplikasi yang direpresentasikan dalam alur kerja pemrosesan video menjadi data kerangka.

## III.6.8. Eksekusi Pipeline

Tahap eksekusi dilakukan dengan menjalankan *pipeline* terhadap keseluruhan dataset video masukan yang tersimpan di dalam direktori `data/raw/`. Eksekusi mencakup beberapa tahapan: iterasi dan pencarian file `.mp4` secara rekursif, ekstraksi *metadata* berdasarkan penamaan file (`[NAME]_[SENTENCE]_[REPETITION].mp4`), proses inferensi menggunakan MediaPipe untuk mendapatkan 86 *keypoints* dari setiap *frame*, dan konversi data mentah menjadi bentuk struktural. Eksekusi dievaluasi berdasarkan kelengkapan *dataset* keluaran pada masing-masing format (JSON, Pickle, Excel) dengan konsistensi penamaan identitas seperti `Pxx_Sxxx_Rxx`.

---

## IV.6. Pengembangan APP

Pengembangan APP diawali dengan pemaparan hasil tahap analisis berupa kebutuhan fungsional aplikasi. Pada bagian ini juga dijelaskan rancangan alur kerja serta spesifikasi desain luaran (*output*). Selain itu, dipaparkan proses implementasi pengembangan beserta hasil pengujian yang telah dilakukan.

### IV.6.1. Analisis Kebutuhan Fungsional Aplikasi

Analisis kebutuhan fungsional bertujuan untuk mengidentifikasi fitur utama yang harus dimiliki Aplikasi Pipeline Preprocessing (APP) untuk mendukung proses standarisasi dataset.

**Tabel 1. Kebutuhan Fungsional APP**

| No | ID Fitur | Kebutuhan Fungsional |
|---|---|---|
| **A. Pembacaan dan Ekstraksi Data** | | |
| 1 | FR-01 | Aplikasi mampu melakukan pembacaan *file* `.mp4` secara rekursif di dalam direktori `data/raw/`. |
| 2 | FR-02 | Aplikasi mampu mengekstrak *metadata* *Subject*, *Sentence*, dan *Repetition* dari format nama *file* menjadi ID standar (`Pxx_Sxxx_Rxx`). |
| 3 | FR-03 | Aplikasi harus dapat mengekstrak 86 titik *landmark* menggunakan model MediaPipe Holistic. |
| **B. Konversi dan Ekspor Data** | | |
| 4 | FR-04 | Aplikasi mampu mengekspor data *keypoints* tunggal per video ke dalam format JSON. |
| 5 | FR-05 | Aplikasi mampu mengagregasi seluruh data *keypoints* ke dalam format Pickle berskala besar. |
| 6 | FR-06 | Aplikasi mampu mengekspor *keypoints* analitik ke format *multisheet* Excel per *Subject*. |

### IV.6.2. Desain Aplikasi

Bagian desain menjelaskan tahapan alur kerja pemrosesan dari video mentah hingga menjadi *dataset* struktural, serta rincian desain *output* yang dihasilkan.

#### IV.6.2.1. Desain Alur Kerja Aplikasi

Alur kerja APP terdiri dari beberapa tahapan utama:
1. **Pemindaian Direktori**: Sistem memindai `data/raw/` untuk mendeteksi *file* `.mp4`.
2. **Metadata Parsing**: Sistem memetakan nama *file* menggunakan `mappings.py` (misal: "ACHMAD" -> "P01").
3. **Ekstraksi Landmark**: Setiap *frame* dari video diproses menggunakan MediaPipe untuk mendapatkan *array* koordinat dimensi 3 (*x, y, z*).
4. **Data Formatting**: Matriks titik koordinat diubah bentuknya (*reshape*).
5. **Konversi (Ekspor)**: Data didistribusikan oleh modul *converter* (`to_json.py`, `to_pickle.py`, `to_excel.py`) ke lokasi *output* yang telah dikonfigurasi pada `paths.py`.

#### IV.6.2.2. Desain Output Aplikasi

Aplikasi ini dirancang untuk menghasilkan tiga tipe *output*:
1. **File JSON**: Menghasilkan satu *file* independen per video (contoh: `P01_S001_R01.json`), berisi atribut `video_id`, `num_frames`, `num_keypoints`, `dimensions`, dan *array* `keypoints`.
2. **File Pickle**: Struktur basis data agregat berupa struktur *dictionary* dengan kunci berupa `video_id` dan nilai berupa matriks *ndarray* dari keseluruhan *frame*.
3. **File Excel**: Disusun secara *Subject-centric* (contoh: `P01.xlsx`), di mana masing-masing *sheet* mewakili satu video. Kolom yang dihasilkan meliputi `kode_video`, `frame`, `keypoint_id`, `x`, dan `y`.

### IV.6.3. Pengembangan Aplikasi

APP dibangun menggunakan arsitektur berbasis *Single Responsibility Principle*. Struktur kode dibagi ke dalam direktori-direktori fungsional seperti `src/config/`, `src/core/`, `src/converter/`, `src/extractor/`, dan `src/processor/`.

Pendekatan teknis utama dalam implementasi ini adalah penggunaan modul `metadata.py` di dalam lapisan *Core Logic* untuk menangani abstraksi nama tanpa perlu menuliskannya secara *hardcode* dalam pembaca direktori.

Sebagai contoh, konfigurasi pemetaan *Subject* dan kalimat yang diatur pada `src/config/mappings.py` dikelola secara dinamis:

```python
# Contoh representasi struktural pada APP
PERSON_MAP = {
    "ACHMAD": "P01",
    "ANDRI": "P02",
    "BALQIS": "P03",
    "HADI": "P04",
    "HENDI": "P05"
}
```

Setiap konverter dalam direktori `src/converter/` (seperti `to_excel.py`) mewarisi antarmuka standar yang hanya bertanggung jawab penuh atas logika tulis menulis sesuai dengan ekstensinya, menghindari duplikasi memori selama proses *dump*.

### IV.6.4. Pengujian Aplikasi

Pengujian pada APP bertujuan untuk mengevaluasi akurasi logika *parsing metadata* dan konsistensi matriks *keypoints* yang diekstraksi dari MediaPipe ke dalam format akhir. 

**Tabel 2. Test Case APP**

| Case ID | ID Fitur | Item Case | Expected Result |
|---|---|---|---|
| TC-01 | FR-01 & FR-02 | Aplikasi mengekstrak dan menstandarkan ID video dari format string yang rumit. | *Metadata* berhasil dipetakan menjadi pola string standar `Pxx_Sxxx_Rxx` sesuai referensi *dictionary*. |
| TC-02 | FR-03 | Aplikasi mengeksekusi ekstraksi menggunakan MediaPipe pada video valid. | Menghasilkan matriks (T, 86, 3) di mana T adalah total *frame* dan 86 adalah jumlah titik kerangka *Holistic*. |
| TC-03 | FR-04 | Aplikasi menyimpan matriks koordinat tunggal ke dalam format JSON. | *File* JSON berhasil dibuat, dapat dibaca (*parsed*), dan sesuai dengan dimensi array *keypoints*. |
| TC-04 | FR-06 | Aplikasi merekapitulasi matriks koordinat dari *Subject* "ACHMAD" ke Excel. | Terbentuk file `P01.xlsx` yang memiliki beberapa *sheet* di dalamnya berdasarkan jumlah pengulangan (*repetition*). |
