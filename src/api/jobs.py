"""
Background Jobs Module

This module manages the execution of background tasks and
maintains an in-memory job queue for the API.
"""

import os
from typing import Dict, Any
from src.config import RAW_VIDEO_DIR
from src.core.pipeline import SkeletonPipeline
from src.api.models import ProcessRequest

# Simple in-memory job queue (per-process)
JOBS: Dict[str, Dict[str, Any]] = {}

def run_job(job_id: str, video_path: str, output_subpath: str, req: ProcessRequest) -> None:
    """
    Background worker for executing a single video pipeline job.

    Args:
        job_id (str): The unique identifier for the job.
        video_path (str): The relative path to the video file.
        output_subpath (str): Subpath mapping for mirroring the folder structure.
        req (ProcessRequest): The request object containing user preferences.
    """
    job = JOBS.get(job_id)
    if not job:
        return
    job["status"] = "running"

    full_video_path = os.path.join(RAW_VIDEO_DIR, video_path)
    try:
        pipeline = SkeletonPipeline(
            save_npy=not req.no_npy,
            save_json=not req.no_json,
            save_pickle=not req.no_pickle,
            save_excel=not req.no_excel,
            generate_preview=not req.no_preview,
            generate_overlay=not req.no_overlay,
            generate_skeleton_only=not req.no_skeleton_only,
        )
        pipeline.process_video(full_video_path, label=req.label, output_subpath=output_subpath)
        job["status"] = "done"
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
