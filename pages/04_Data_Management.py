import streamlit as st
import os
from src.config import RAW_VIDEO_DIR

st.title("Data Management")

st.markdown("---")

st.subheader("Upload video baru")

uploaded = st.file_uploader("Pilih video", type=["mp4", "avi", "mov", "mkv", "webm"])

if uploaded is not None:
    os.makedirs(RAW_VIDEO_DIR, exist_ok=True)
    save_path = os.path.join(RAW_VIDEO_DIR, uploaded.name)
    with open(save_path, "wb") as f:
        f.write(uploaded.getbuffer())
    st.success(f"Saved to {save_path}")

st.markdown("---")

st.subheader("Daftar video di data/raw")

if not os.path.isdir(RAW_VIDEO_DIR):
    st.info("Folder data/raw belum ada.")
else:
    files = sorted(os.listdir(RAW_VIDEO_DIR))
    if not files:
        st.info("Belum ada file di data/raw.")
    else:
        for name in files:
            col1, col2 = st.columns([4, 1])
            col1.write(name)
            if col2.button("Hapus", key=f"del_{name}"):
                path = os.path.join(RAW_VIDEO_DIR, name)
                try:
                    os.remove(path)
                    st.success(f"Deleted {name}")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Failed to delete {name}: {e}")
