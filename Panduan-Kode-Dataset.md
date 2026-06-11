# Panduan Penamaan Folder dan File Dataset BISINDO

## 1. Struktur Kode Dataset

| Simbol | Keterangan |
| ------ | ---------- |
| P      | Penutur    |
| S      | Kalimat    |
| R      | Repetisi   |

---

## 2. Kode Penutur

| Nama   | Kode |
| ------ | ---- |
| ACHMAD | P1   |
| ANDRI  | P2   |
| FADIL  | P3   |
| HADI   | P4   |
| HENDI  | P5   |
| DELIA  | P6   |

---

## 3. Kode Kalimat

| Kode | Kalimat                                     |
| ---- | ------------------------------------------- |
| S01  | AKU CIUM BADAN DIA                          |
| S02  | AKU LIHAT ADA ULAR MASUK KELAS              |
| S03  | AKU NILAI JELEK                             |
| S04  | AKU PUSING AKU HARUS PERIKSA MANA           |
| S05  | APA KAMU PERNAH BACA BUKU BAHASA INGGRIS    |
| S06  | AYAH SAMA IBU MANA                          |
| S07  | BADAN AKU GEMUK TAPI BADAN ADIK KURUS       |
| S08  | BUKU AKU SOBEK GEGARA DIA                   |
| S09  | DIA ANAK BAIK SAMPAI BANYAK ORANG SUKA      |
| S10  | DIA MENGEJEK AKU                            |
| S11  | GAKBOLEH PULANG SEKARANG KAMU               |
| S12  | IBU AKU PUNYA KUCING SAMA IKAN              |
| S13  | KAKAK AKU KASIH HADIAH BUAT AKU             |
| S14  | KAMU BELAJAR BISINDO KAPAN                  |
| S15  | KAMU PERGI MANA                             |
| S16  | KAMU PUNYA ANGGOTA KELUARGA BERAPA          |
| S17  | KENAPA KAMU GAK MASUK KULIAH KEMARIN        |
| S18  | KITA ISTIRAHAT JAM BERAPA                   |
| S19  | MANA IBU KAMU BAIK-BAIK ATAU TIDAK          |
| S20  | NAMA ISYARAT KAMU APA                       |
| S21  | OBAT BISA BELI TOKO MANA                    |
| S22  | ORANG JAHAT SANA PUKUL AKU BERULANG         |
| S23  | POLISI SANA PUKUL PENCURI                   |
| S24  | RUMAH DIMANA KAMU                           |
| S25  | SANA BERITA SUDAH BANYAK RIBUAN ORANG LIHAT |
| S26  | SANA ENAK NASI PADANG TAPI MAHAL            |
| S27  | SANA TOILET KOTOR                           |
| S28  | SEPATU DIA KOTOR                            |
| S29  | TONG-SAMPAH ADA SEMUT BANYAK                |
| S30  | ULAR SANA MAKAN KAMBING                     |

---

## 4. Kode Pengulangan

| Pengulangan | Kode |
| ----------- | ---- |
| 01          | R1   |
| 02          | R2   |
| 03          | R3   |
| 04          | R4   |
| 05          | R5   |

---

## 5. Aturan Penamaan File

### 5.1 Penutur P1–P5

Untuk penutur:

* P1 (ACHMAD)
* P2 (ANDRI)
* P3 (FADIL)
* P4 (HADI)
* P5 (HENDI)

Format nama file:

```text
PX_SXX_RX.mp4
```

Keterangan:

```text
PX   = Kode Penutur
SXX  = Kode Kalimat
RX   = Kode Repetisi
```

Contoh:

```text
P1_S01_R1.mp4
P1_S01_R2.mp4
P1_S01_R3.mp4
P1_S01_R4.mp4
P1_S01_R5.mp4
```

---

### 5.2 Penutur P6 (DELIA)

Penutur P6 memiliki dua variasi data:

* MJ = Mayoritas
* MN = Minoritas

Format nama file:

```text
P6_SXX_MJ.mp4
P6_SXX_MN.mp4
```

Contoh:

```text
P6_S01_MJ.mp4
P6_S01_MN.mp4

P6_S15_MJ.mp4
P6_S15_MN.mp4

P6_S30_MJ.mp4
P6_S30_MN.mp4
```

---

## 6. Contoh Rename File

### Nama File Awal

```text
ACHMAD_AKU CIUM BADAN DIA_01.mp4
ACHMAD_AKU CIUM BADAN DIA_02.mp4
ACHMAD_AKU CIUM BADAN DIA_03.mp4

DELIA_AKU CIUM BADAN DIA_MJ.mp4
DELIA_AKU CIUM BADAN DIA_MN.mp4
```

### Nama File Setelah Rename

```text
P1_S01_R1.mp4
P1_S01_R2.mp4
P1_S01_R3.mp4

P6_S01_MJ.mp4
P6_S01_MN.mp4
```

---

## 7. Struktur Folder Dataset

Direkomendasikan menggunakan struktur berikut:

```text
Dataset/
├── S01/
├── S02/
├── S03/
...
└── S30/
```

---

## 8. Ringkasan Format

### Penutur P1–P5

```text
PX_SXX_RX.mp4
```

Contoh:

```text
P3_S12_R4.mp4
P5_S30_R2.mp4
```

### Penutur P6

```text
P6_SXX_MJ.mp4
P6_SXX_MN.mp4
```

Contoh:

```text
P6_S22_MJ.mp4
P6_S22_MN.mp4
```
