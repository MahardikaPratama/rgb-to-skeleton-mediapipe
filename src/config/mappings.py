"""
Metadata Mappings Configuration

This module defines standard dictionaries for mapping raw strings
(e.g. folder names, speaker names) to standardized system IDs (Pxx, Sxxx).
"""

# ==========================================================
# METADATA MAPPINGS
# ==========================================================

PERSON_MAP = {
    "ACHMAD": "P01",
    "ANDRI": "P02",
    "BALQIS": "P03",
    "HADI": "P04",
    "HENDI": "P05",
}

SENTENCE_FOLDERS = [
    "AKU CIUM BAU BADAN DIA",
    "AKU LIHAT ADA ULAR MASUK KELAS",
    "AKU NILAI JELEK",
    "AKU PUSING SERING, AKU HARUS PERIKSA MANA",
    "APA KAMU PERNAH BACA NOVEL B.INGGRIS",
    "AYAH SAMA IBU MANA",
    "BADAN AKU GEMUK TAPI BADAN ADIK KURUS",
    "BUKU AKU SOBEK GEGARA DIA",
    "DIA ANAK BAIK SAMPAI BANYAK ORANG SUKA",
    "DIA MENGEJEK AKU",
    "GAK BOLEH PULANG SEKARANG KAMU",
    "GIMANA IBUMU BAIK-BAIK ATAU TIDAK",
    "IBU AKU PUNYA KUCING SAMA IKAN",
    "KAKAK AKU KASIH HADIAH BUAT AKU",
    "KAMU BELAJAR BISINDO KAPAN",
    "KAMU PERGI KEMANA",
    "KAMU PUNYA ANGGOTA KELUARGA BERAPA",
    "KENAPA KAMU GAK MASUK KULIAH KEMARIN",
    "KITA ISTIRAHAT JAM BERAPA",
    "OBAT BISA BELI TOKO OBAT MANA",
    "ORANG JAHAT SANA PUKUL AKU BERULANG",
    "POLISI SANA PUKUL PENCURI",
    "RUMAH DIMANA KAMU",
    "SANA BERITA SUDAH BANYAK RIBUAN ORANG LIHAT",
    "SANA ENAK NASI PADANG TAPI MAHAL",
    "SANA TOILET KOTOR",
    "SEPATU DIA KOTOR",
    "TONG SAMPAH ADA SEMUT BANYAK",
    "ULANG TAHUN SELAMAT",
    "ULAR SANA MAKAN KAMBING",
]

# Create mapping S001 - S030 alphabetically
SENTENCE_MAP = {folder: f"S{i:03d}" for i, folder in enumerate(sorted(SENTENCE_FOLDERS), 1)}
