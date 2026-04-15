import streamlit as st
import requests
import os
import platform
from datetime import datetime
import numpy as np

from src.config import (
	RAW_VIDEO_DIR,
	SKELETON_DIR,
	JSON_DIR,
	PICKLE_DIR,
	PREVIEW_SKELETON_DIR,
	PREVIEW_OVERLAY_DIR,
)


API_DEFAULT = "http://127.0.0.1:8000"

st.set_page_config(page_title="RGB to Skeleton Studio", layout="wide")

st.title("RGB to Skeleton Studio")

api_base = st.sidebar.text_input("API base URL", API_DEFAULT)

st.markdown("---")

def _safe_count(path):
	if not os.path.isdir(path):
		return 0
	return len(os.listdir(path))


def _get_pkg_version(module_name):
	try:
		module = __import__(module_name)
		return getattr(module, "__version__", "unknown")
	except Exception:
		return "not installed"


@st.cache_data(ttl=5)
def _fetch_jobs(api_url):
	try:
		r = requests.get(f"{api_url}/jobs", timeout=5)
		r.raise_for_status()
		return r.json(), None
	except Exception as e:
		return [], str(e)


@st.cache_data(ttl=10)
def _quality_snapshot(skeleton_dir):
	if not os.path.isdir(skeleton_dir):
		return {
			"num_files": 0,
			"total_frames": 0,
			"avg_frames": 0,
			"invalid_frames": 0,
			"invalid_ratio": 0.0,
			"frames_per_video": [],
			"errors": [],
		}

	npy_files = sorted(
		[f for f in os.listdir(skeleton_dir) if f.endswith(".npy")]
	)

	total_frames = 0
	invalid_frames = 0
	frames_per_video = []
	errors = []

	for fname in npy_files:
		fpath = os.path.join(skeleton_dir, fname)
		try:
			data = np.load(fpath)
			if data.ndim != 3 or data.shape[1] != 86:
				errors.append(f"{fname}: invalid shape {getattr(data, 'shape', None)}")
				continue

			frames = int(data.shape[0])
			total_frames += frames

			# Invalid frame: all keypoints x,y,z are zero
			inv = int(np.sum(np.all(data == 0, axis=(1, 2))))
			invalid_frames += inv

			frames_per_video.append({"video": os.path.splitext(fname)[0], "frames": frames})
		except Exception as e:
			errors.append(f"{fname}: {e}")

	num_files = len(frames_per_video)
	avg_frames = (total_frames / num_files) if num_files > 0 else 0
	invalid_ratio = (invalid_frames / total_frames * 100.0) if total_frames > 0 else 0.0

	return {
		"num_files": num_files,
		"total_frames": total_frames,
		"avg_frames": avg_frames,
		"invalid_frames": invalid_frames,
		"invalid_ratio": invalid_ratio,
		"frames_per_video": frames_per_video,
		"errors": errors,
	}


def _recent_files(path, suffix, limit=5):
	if not os.path.isdir(path):
		return []
	files = [f for f in os.listdir(path) if f.endswith(suffix)]
	files.sort(key=lambda x: os.path.getmtime(os.path.join(path, x)), reverse=True)
	return files[:limit]


# Basic stats from filesystem (local view)
num_raw = _safe_count(RAW_VIDEO_DIR)
num_npy = _safe_count(SKELETON_DIR)
num_json = _safe_count(JSON_DIR)
num_pkl = _safe_count(PICKLE_DIR)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Raw videos", num_raw)
col2.metric(".npy files", num_npy)
col3.metric(".json files", num_json)
col4.metric(".pkl files", num_pkl)

st.markdown("## Pipeline Health")

# API health check
api_ok = False
api_msg = "OK"
try:
	ping = requests.get(f"{api_base}/videos", timeout=3)
	ping.raise_for_status()
	api_ok = True
except Exception as e:
	api_msg = str(e)

hc1, hc2, hc3 = st.columns(3)
hc1.metric("API Status", "UP" if api_ok else "DOWN")
hc2.metric("Python", platform.python_version())
hc3.metric("NumPy", _get_pkg_version("numpy"))

hc4, hc5, hc6 = st.columns(3)
hc4.metric("OpenCV", _get_pkg_version("cv2"))
hc5.metric("MediaPipe", _get_pkg_version("mediapipe"))
hc6.metric("Requests", _get_pkg_version("requests"))

if not api_ok:
	st.warning(f"API unreachable: {api_msg}")

st.caption(
	f"Raw dir exists: {os.path.isdir(RAW_VIDEO_DIR)} | "
	f"Skeleton dir exists: {os.path.isdir(SKELETON_DIR)} | "
	f"Preview dir exists: {os.path.isdir(PREVIEW_SKELETON_DIR)}"
)

st.markdown("## Job Monitor (Ringkas)")
jobs, jobs_err = _fetch_jobs(api_base)
if jobs_err:
	st.info(f"Jobs endpoint belum tersedia / gagal diakses: {jobs_err}")
else:
	pending = sum(1 for j in jobs if j.get("status") == "pending")
	running = sum(1 for j in jobs if j.get("status") == "running")
	done = sum(1 for j in jobs if j.get("status") == "done")
	failed = sum(1 for j in jobs if j.get("status") == "failed")

	j1, j2, j3, j4 = st.columns(4)
	j1.metric("Pending", pending)
	j2.metric("Running", running)
	j3.metric("Done", done)
	j4.metric("Failed", failed)

	if jobs:
		recent_jobs = list(reversed(jobs))[:5]
		st.write("5 job terakhir:")
		st.table(recent_jobs)
	else:
		st.write("Belum ada jobs.")

st.markdown("## Recent Outputs")

recent_npy = _recent_files(SKELETON_DIR, ".npy", limit=5)
recent_skel_vid = _recent_files(PREVIEW_SKELETON_DIR, ".mp4", limit=3)
recent_overlay_vid = _recent_files(PREVIEW_OVERLAY_DIR, ".mp4", limit=3)

r1, r2, r3 = st.columns(3)
with r1:
	st.write("Recent skeleton (.npy)")
	if recent_npy:
		for f in recent_npy:
			ts = datetime.fromtimestamp(os.path.getmtime(os.path.join(SKELETON_DIR, f)))
			st.write(f"- `{f}` ({ts.strftime('%Y-%m-%d %H:%M:%S')})")
	else:
		st.write("- belum ada")

with r2:
	st.write("Recent preview skeleton-only")
	if recent_skel_vid:
		for f in recent_skel_vid:
			st.write(f"- `{f}`")
	else:
		st.write("- belum ada")

with r3:
	st.write("Recent preview overlay")
	if recent_overlay_vid:
		for f in recent_overlay_vid:
			st.write(f"- `{f}`")
	else:
		st.write("- belum ada")

st.markdown("## Dataset Quality Snapshot")
quality = _quality_snapshot(SKELETON_DIR)

q1, q2, q3, q4 = st.columns(4)
q1.metric("Skeleton files", quality["num_files"])
q2.metric("Total frames", quality["total_frames"])
q3.metric("Avg frames/video", f"{quality['avg_frames']:.1f}")
q4.metric("Invalid frame ratio", f"{quality['invalid_ratio']:.2f}%")

if quality["frames_per_video"]:
	st.write("Distribusi jumlah frame per video")
	chart_data = {
		row["video"]: row["frames"]
		for row in quality["frames_per_video"][:20]
	}
	st.bar_chart(chart_data)

if quality["errors"]:
	with st.expander("Detail error loading .npy"):
		for err in quality["errors"]:
			st.write(f"- {err}")

st.markdown("## Quick Links")
st.write("Gunakan sidebar `Pages` di kiri atas (Streamlit) untuk membuka halaman:")
st.write("- Processing: Jalankan pipeline untuk 1 video atau batch")
st.write("- Jobs: Lihat antrian dan status pemrosesan (jika diaktifkan)")
st.write("- Results: Visualisasi detail hasil per video")
st.write("- Data Management: Upload / hapus video di data/raw")

st.markdown("---")
st.write("Dashboard ini hanya menampilkan ringkasan. Buka halaman lain lewat menu multipage Streamlit.")
