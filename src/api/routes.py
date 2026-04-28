"""
API Endpoints Module

This module defines the REST endpoints for interacting with the pipeline.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
import os
import uuid
import traceback
from pathlib import Path

from src.config import (
    RAW_VIDEO_DIR,
    SKELETON_DIR,
    JSON_DIR,
    PREVIEW_SKELETON_DIR,
    PREVIEW_OVERLAY_DIR,
)
from src.core.pipeline import SkeletonPipeline
from src.core.metadata import parse_video_id
from src.api.models import ProcessRequest, BatchProcessRequest, JobStatus
from src.api.jobs import JOBS, run_job

router = APIRouter()


@router.get("/videos")
def list_videos():
    """
    Lists all supported video files in the root of the raw directory.
    """
    if not os.path.isdir(RAW_VIDEO_DIR):
        return {"videos": []}
    videos = sorted([
        f for f in os.listdir(RAW_VIDEO_DIR)
        if os.path.splitext(f)[1].lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    ])
    return {"videos": videos}


@router.get("/folders")
def list_folders():
    """
    Lists all subdirectories within the raw video directory recursively.
    """
    if not os.path.isdir(RAW_VIDEO_DIR):
        return {"folders": []}
    
    folders = []
    for root, dirs, files in os.walk(RAW_VIDEO_DIR):
        for name in dirs:
            dir_path = os.path.join(root, name)
            relative_path = os.path.relpath(dir_path, RAW_VIDEO_DIR)
            if relative_path != '.':
                folders.append(relative_path)
    
    folders.insert(0, ".")
    return {"folders": sorted(folders)}


@router.post("/process")
def process_video(req: ProcessRequest):
    """
    Synchronously processes a single video based on user configuration.
    """
    video_path = os.path.join(RAW_VIDEO_DIR, req.video)
    if not os.path.isfile(video_path):
        raise HTTPException(status_code=404, detail=f"Video not found: {req.video}")

    pipeline = SkeletonPipeline(
        save_npy=not req.no_npy,
        save_json=not req.no_json,
        save_pickle=not req.no_pickle,
        save_excel=not req.no_excel,
        generate_preview=not req.no_preview,
        generate_overlay=not req.no_overlay,
        generate_skeleton_only=not req.no_skeleton_only,
    )

    try:
        keypoints = pipeline.process_video(video_path, label=req.label)
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=str(e) + "\n" + tb)

    video_id = parse_video_id(Path(video_path))
    resp = {
        "video": req.video,
        "video_id": video_id,
        "npy": os.path.join(SKELETON_DIR, f"{video_id}.npy") if pipeline.save_npy else None,
        "json": os.path.join(JSON_DIR, f"{video_id}.json") if pipeline.save_json else None,
        "pkl": pipeline.last_pickle_path if pipeline.save_pickle else None,
        "pkl_sample_id": pipeline.last_pickle_sample_id if pipeline.save_pickle else None,
        "preview_skeleton": os.path.join(PREVIEW_SKELETON_DIR, f"{video_id}_skeleton.mp4") if (pipeline.gen_preview and pipeline.gen_skel_only) else None,
        "preview_overlay": os.path.join(PREVIEW_OVERLAY_DIR, f"{video_id}_overlay.mp4") if (pipeline.gen_preview and pipeline.gen_overlay) else None,
        "frames_extracted": int(keypoints.shape[0]),
        "keypoint_dims": int(keypoints.shape[2]) if keypoints.ndim == 3 else None,
    }
    return resp


@router.post("/jobs", response_model=JobStatus)
def create_job(req: ProcessRequest, background_tasks: BackgroundTasks):
    """
    Create an asynchronous background job for processing a single video.
    """
    video_path = os.path.join(RAW_VIDEO_DIR, req.video)
    if not os.path.isfile(video_path):
        raise HTTPException(status_code=404, detail=f"Video not found: {req.video}")

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "id": job_id,
        "video": req.video,
        "status": "pending",
        "error": None,
        "output_subpath": ""
    }

    background_tasks.add_task(run_job, job_id, req.video, "", req)

    return JobStatus(**JOBS[job_id])


@router.post("/jobs/batch", response_model=list[JobStatus])
def create_batch_job(req: BatchProcessRequest, background_tasks: BackgroundTasks):
    """
    Create multiple asynchronous background jobs from a list of video paths.
    """
    created_jobs = []
    for video_rel_path in req.videos:
        full_video_path = os.path.join(RAW_VIDEO_DIR, video_rel_path)
        if not os.path.isfile(full_video_path):
            continue

        job_id = str(uuid.uuid4())
        output_subpath = os.path.dirname(video_rel_path)

        job_data = {
            "id": job_id,
            "video": video_rel_path,
            "status": "pending",
            "error": None,
            "output_subpath": output_subpath
        }
        JOBS[job_id] = job_data
        
        # We need to construct a standard ProcessRequest from BatchProcessRequest
        req_single = ProcessRequest(
            video=video_rel_path,
            label=req.label,
            no_npy=req.no_npy,
            no_json=req.no_json,
            no_pickle=req.no_pickle,
            no_excel=req.no_excel,
            no_preview=req.no_preview,
            no_overlay=req.no_overlay,
            no_skeleton_only=req.no_skeleton_only,
        )
        background_tasks.add_task(run_job, job_id, video_rel_path, output_subpath, req_single)
        created_jobs.append(JobStatus(**job_data))

    if not created_jobs:
        raise HTTPException(
            status_code=404,
            detail="No valid videos found in the provided list."
        )

    return created_jobs


@router.get("/jobs", response_model=list[JobStatus])
def list_jobs():
    """
    List all recorded jobs.
    """
    return [JobStatus(**job) for job in sorted(JOBS.values(), key=lambda j: j.get("video"))]


@router.get("/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str):
    """
    Get the status of a specific job by its ID.
    """
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatus(**job)
