import streamlit as st
import requests

API_DEFAULT = "http://127.0.0.1:8000"

st.title("Jobs & Progress")

api_base = st.sidebar.text_input("API base URL", API_DEFAULT)

st.markdown("---")


def render_jobs():
    try:
        r = requests.get(f"{api_base}/jobs", timeout=5)
        r.raise_for_status()
        jobs = r.json()
    except Exception as e:
        st.error(f"Failed to fetch jobs: {e}")
        return

    if not jobs:
        st.info("No jobs yet. Create one from the Processing page.")
        return

    st.write("Daftar job pemrosesan video:")
    st.table(jobs)

render_jobs()

st.info("Tekan tombol Rerun di pojok kanan atas atau reload halaman untuk refresh status jobs.")
