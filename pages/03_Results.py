import streamlit as st
import numpy as np
import os
from src.config import SKELETON_DIR
from src.visualizer.plot_skeleton import plot_frame

st.title("Results & Visualization")

st.markdown("---")

if not os.path.isdir(SKELETON_DIR):
    st.info("No skeleton directory found.")
else:
    npy_files = [f for f in os.listdir(SKELETON_DIR) if f.endswith('.npy')]
    if not npy_files:
        st.info("No .npy skeleton files found.")
    else:
        fname = st.selectbox("Pilih file skeleton (.npy)", npy_files)
        npy_path = os.path.join(SKELETON_DIR, fname)
        data = np.load(npy_path)
        st.write(f"Shape: {data.shape}")

        frame_idx = st.slider("Frame index", 0, max(0, data.shape[0]-1), 0)
        scale_mode = st.radio("Coordinate mode", ["normalized", "pixel 256x256"], index=0)
        if scale_mode == "normalized":
            fig = plot_frame(data, frame_idx=frame_idx, show=False)
        else:
            fig = plot_frame(data, frame_idx=frame_idx, show=False, scale_to=(256, 256))

        st.pyplot(fig)
