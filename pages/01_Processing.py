import streamlit as st
import requests
import os
import time

from src.config import RAW_VIDEO_DIR

API_DEFAULT = "http://127.0.0.1:8000"

st.set_page_config(layout="wide")

st.title("Batch Processing")

api_base = st.sidebar.text_input("API base URL", API_DEFAULT)

st.sidebar.header("Options")
no_npy = st.sidebar.checkbox("Skip .npy", value=False)
no_json = st.sidebar.checkbox("Skip .json", value=False)
no_pickle = st.sidebar.checkbox("Skip .pkl", value=False)
no_preview = st.sidebar.checkbox("Skip preview", value=False)
no_overlay = st.sidebar.checkbox("Skip overlay", value=False)
no_skel_only = st.sidebar.checkbox("Skip skeleton-only preview", value=False)
label = st.sidebar.number_input("Label (optional)", value=0, min_value=0, step=1)

st.markdown("---")

# Fetch available videos and folders
try:
    r_videos = requests.get(f"{api_base}/videos", timeout=5)
    r_videos.raise_for_status()
    all_videos = r_videos.json().get("videos", [])

    r_folders = requests.get(f"{api_base}/folders", timeout=5)
    r_folders.raise_for_status()
    all_folders = r_folders.json().get("folders", [])
except Exception as e:
    st.error(f"Failed to fetch data from API: {e}")
    all_videos = []
    all_folders = []

if not all_videos:
    st.warning(f"No videos found in `{RAW_VIDEO_DIR}`. Put videos in that folder and restart the app.")
else:
    # --- UI for selecting input ---
    st.header("1. Select Input")
    input_mode = st.radio(
        "Input Mode",
        ("Single Video", "Multiple Videos", "Folder"),
        horizontal=True,
        label_visibility="collapsed"
    )

    selected_videos = []
    if input_mode == "Single Video":
        video = st.selectbox("Select a video", all_videos)
        if video:
            selected_videos = [video]
    elif input_mode == "Multiple Videos":
        selected_videos = st.multiselect("Select videos", all_videos)
    elif input_mode == "Folder":
        folder = st.selectbox("Select a folder", all_folders, help="'.' is the root `data/raw` folder.")
        recursive = st.checkbox("Include subfolders (recursive)", value=True)
        if folder:
            # Filter videos based on the selected folder
            if recursive:
                selected_videos = [v for v in all_videos if v.startswith(folder)]
            else:
                selected_videos = [v for v in all_videos if os.path.dirname(v) == folder]

    # --- UI for showing selected files ---
    st.header("2. Review Files")
    if selected_videos:
        st.info(f"**{len(selected_videos)}** video(s) selected for processing.")
        with st.expander("Show selected files"):
            st.json(selected_videos)
    else:
        st.info("No videos selected.")

    # --- UI for starting the job ---
    st.header("3. Start Processing")
    if st.button("Create Batch Job", disabled=not selected_videos):
        payload = {
            "videos": selected_videos,
            "label": int(label) if label else None,
            "no_npy": no_npy,
            "no_json": no_json,
            "no_pickle": no_pickle,
            "no_preview": no_preview,
            "no_overlay": no_overlay,
            "no_skeleton_only": no_skel_only,
        }
        with st.spinner("Creating batch job..."):
            try:
                r = requests.post(f"{api_base}/jobs/batch", json=payload, timeout=60)
                r.raise_for_status()
                resp = r.json()
                st.success(f"Successfully created {len(resp)} jobs.")
                st.info("Navigate to the 'Jobs' page to see the status.")
                
                # Store job IDs to highlight them on the Jobs page
                st.session_state['last_job_ids'] = [job['id'] for job in resp]
                
                time.sleep(1) # Give a moment before suggesting a switch
                st.info("Switching to Jobs page in 3 seconds...")
                time.sleep(3)
                st.switch_page("pages/02_Jobs.py")

            except Exception as e:
                st.error(f"Failed to create batch job: {e}")
                try:
                    st.error(f"API Response: {r.text}")
                except NameError:
                    pass
