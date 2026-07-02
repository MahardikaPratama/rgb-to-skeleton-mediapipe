"""Rename BISINDO raw dataset folders and videos into the standard format.

The current raw dataset still uses legacy sentence folder names and legacy
video filenames such as ``ACHMAD_AKU CIUM BADAN DIA_01.mp4``.

This script converts them to the standardized structure described in
``panduan.md``:

* folders: ``S01`` .. ``S30``
* videos for P1-P5: ``PX_SXX_RX.mp4``
* videos for P6 (DELIA): ``P6_SXX_MJ.mp4`` / ``P6_SXX_MN.mp4``

Empty folders are skipped safely, so the script can also be used when DELIA
data is not present yet.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path


LOGGER = logging.getLogger(__name__)

STANDARD_FOLDER_PATTERN = re.compile(r"^S\d{2}$", re.IGNORECASE)
STANDARD_VIDEO_PATTERN = re.compile(
	r"^(?P<speaker>P\d|P6)_(?P<sentence>S\d{2})_(?P<variant>R\d|MJ|MN)\.mp4$",
	re.IGNORECASE,
)
LEGACY_VIDEO_PATTERN = re.compile(
	r"^(?P<speaker>[A-Z]+)_(?P<sentence>.+)_(?P<suffix>\d{2}|MJ|MN)\.mp4$",
	re.IGNORECASE,
)

SPEAKER_TO_CODE = {
	"ACHMAD": "P1",
	"ANDRI": "P2",
	"FADIL": "P3",
	"HADI": "P4",
	"HENDI": "P5",
	"DELIA": "P6",
}

SENTENCE_FOLDER_TO_CODE = {
	"AKU CIUM BADAN DIA": "S01",
	"AKU LIHAT ADA ULAR MASUK KELAS": "S02",
	"AKU NILAI JELEK": "S03",
	"AKU PUSING AKU HARUS PERIKSA MANA": "S04",
	"APA KAMU PERNAH BACA BUKU BAHASA INGGRIS": "S05",
	"AYAH SAMA IBU MANA": "S06",
	"BADAN AKU GEMUK TAPI BADAN ADIK KURUS": "S07",
	"BUKU AKU SOBEK GEGARA DIA": "S08",
	"DIA ANAK BAIK SAMPAI BANYAK ORANG SUKA": "S09",
	"DIA MENGEJEK AKU": "S10",
	"GAKBOLEH PULANG SEKARANG KAMU": "S11",
	"IBU AKU PUNYA KUCING SAMA IKAN": "S12",
	"KAKAK AKU KASIH HADIAH BUAT AKU": "S13",
	"KAMU BELAJAR BISINDO KAPAN": "S14",
	"KAMU PERGI MANA": "S15",
	"KAMU PUNYA ANGGOTA KELUARGA BERAPA": "S16",
	"KENAPA KAMU GAK MASUK KULIAH KEMARIN": "S17",
	"KITA ISTIRAHAT JAM BERAPA": "S18",
	"MANA IBU KAMU BAIK-BAIK ATAU TIDAK": "S19",
	"NAMA ISYARAT KAMU APA": "S20",
	"OBAT BISA BELI TOKO MANA": "S21",
	"ORANG JAHAT SANA PUKUL AKU BERULANG": "S22",
	"POLISI SANA PUKUL PENCURI": "S23",
	"RUMAH DIMANA KAMU": "S24",
	"SANA BERITA SUDAH BANYAK RIBUAN ORANG LIHAT": "S25",
	"SANA ENAK NASI PADANG TAPI MAHAL": "S26",
	"SANA TOILET KOTOR": "S27",
	"SEPATU DIA KOTOR": "S28",
	"TONG-SAMPAH ADA SEMUT BANYAK": "S29",
	"ULAR SANA MAKAN KAMBING": "S30",
}


def normalize_key(text: str) -> str:
	"""Return a stable uppercase lookup key for legacy names."""

	return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9]+", " ", text.upper())).strip()


SENTENCE_LOOKUP = {
	normalize_key(alias): code for alias, code in SENTENCE_FOLDER_TO_CODE.items()
}


def resolve_sentence_code(folder_name: str) -> str | None:
	"""Return the standardized sentence code for a folder name."""

	folder_name = folder_name.strip()
	if STANDARD_FOLDER_PATTERN.fullmatch(folder_name):
		return folder_name.upper()

	return SENTENCE_LOOKUP.get(normalize_key(folder_name))


def resolve_speaker_code(raw_speaker: str) -> str | None:
	"""Return the standardized speaker code for a legacy speaker name."""

	return SPEAKER_TO_CODE.get(normalize_key(raw_speaker))


def build_target_filename(speaker_code: str, sentence_code: str, suffix: str) -> str | None:
	"""Build the standardized filename for a single video."""

	normalized_suffix = suffix.upper()

	if speaker_code == "P6":
		if normalized_suffix not in {"MJ", "MN"}:
			return None
		return f"{speaker_code}_{sentence_code}_{normalized_suffix}.mp4"

	if normalized_suffix.isdigit():
		return f"{speaker_code}_{sentence_code}_R{int(normalized_suffix)}.mp4"

	if re.fullmatch(r"R\d", normalized_suffix):
		return f"{speaker_code}_{sentence_code}_{normalized_suffix}.mp4"

	return None


def rename_file(source: Path, target: Path, dry_run: bool) -> None:
	"""Rename one file if the target does not already exist."""

	if source == target:
		return

	if target.exists():
		LOGGER.warning("Skip file because target already exists: %s", target.name)
		return

	LOGGER.info("File: %s -> %s", source.name, target.name)
	if not dry_run:
		source.rename(target)


def rename_folder(source: Path, target: Path, dry_run: bool) -> Path:
	"""Rename one sentence folder and return the new folder path."""

	if source == target:
		return source

	if target.exists():
		LOGGER.warning("Skip folder because target already exists: %s", target.name)
		return source

	LOGGER.info("Folder: %s -> %s", source.name, target.name)
	if dry_run:
		return target

	source.rename(target)
	return target


def rename_dataset(root_dir: Path, dry_run: bool = False) -> None:
	"""Rename all legacy folders and videos under ``root_dir``."""

	if not root_dir.exists():
		raise FileNotFoundError(f"Root directory does not exist: {root_dir}")

	for folder in sorted(root_dir.iterdir(), key=lambda path: path.name.lower()):
		if not folder.is_dir():
			continue

		mp4_files = sorted(file.name for file in folder.iterdir() if file.suffix.lower() == ".mp4")
		if not mp4_files:
			LOGGER.info("Skip empty folder: %s", folder.name)
			continue

		sentence_code = resolve_sentence_code(folder.name)
		if sentence_code is None:
			LOGGER.warning("Skip unknown sentence folder: %s", folder.name)
			continue

		current_folder = rename_folder(folder, folder.with_name(sentence_code), dry_run)

		for file_name in mp4_files:
			source_file = current_folder / file_name

			standardized_match = STANDARD_VIDEO_PATTERN.fullmatch(file_name)
			if standardized_match:
				if standardized_match.group("sentence").upper() != sentence_code:
					LOGGER.warning(
						"Skip file because sentence code does not match folder: %s",
						file_name,
					)
				continue

			legacy_match = LEGACY_VIDEO_PATTERN.fullmatch(file_name)
			if legacy_match is None:
				LOGGER.warning("Skip unrecognized filename: %s", file_name)
				continue

			speaker_code = resolve_speaker_code(legacy_match.group("speaker"))
			if speaker_code is None:
				LOGGER.warning("Skip unknown speaker in file: %s", file_name)
				continue

			target_filename = build_target_filename(
				speaker_code=speaker_code,
				sentence_code=sentence_code,
				suffix=legacy_match.group("suffix"),
			)
			if target_filename is None:
				LOGGER.warning("Skip unsupported suffix in file: %s", file_name)
				continue

			rename_file(source_file, current_folder / target_filename, dry_run)


def parse_args() -> argparse.Namespace:
	"""Parse command-line arguments."""

	parser = argparse.ArgumentParser(description="Rename BISINDO raw dataset folders and videos.")
	parser.add_argument(
		"--input",
		type=Path,
		default=Path(__file__).resolve().parent,
		help="Input folder that contains the legacy sentence folders (e.g. 'raw' or 'raw_add').",
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Show the rename plan without changing any file.",
	)
	return parser.parse_args()


def main() -> None:
	"""Run the rename process from the command line."""

	args = parse_args()
	logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

	target_dir = args.input
	if not target_dir.exists():
		# Fallback to checking inside the data directory if given a folder name like "raw_add"
		data_dir = Path(__file__).resolve().parent.parent
		fallback_dir = data_dir / target_dir
		if fallback_dir.exists():
			target_dir = fallback_dir

	rename_dataset(target_dir, dry_run=args.dry_run)


if __name__ == "__main__":
	main()
