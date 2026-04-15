from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import traceback
import uuid
from typing import Optional, Dict, Any

from src.config import (
    RAW_VIDEO_DIR,
    SKELETON_DIR,
    JSON_DIR,
    PICKLE_DIR,
    PREVIEW_SKELETON_DIR,
    PREVIEW_OVERLAY_DIR,
)
from src.pipeline import SkeletonPipeline


app = FastAPI(title="RGB→Skeleton API")

# Allow local UI to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ProcessRequest(BaseModel):
    video: str
    label: Optional[int] = None
    no_npy: bool = False
    no_json: bool = False
    no_pickle: bool = False
    no_preview: bool = False
    no_overlay: bool = False
    no_skeleton_only: bool = False


class BatchProcessRequest(BaseModel):
    videos: list[str]  # Relative paths from RAW_VIDEO_DIR
    label: Optional[int] = None
    no_npy: bool = False
    no_json: bool = False
    no_pickle: bool = False
    no_preview: bool = False
    no_overlay: bool = False
    no_skeleton_only: bool = False


@app.get("/videos")
def list_videos():
    if not os.path.isdir(RAW_VIDEO_DIR):
        return {"videos": []}
    videos = sorted([
        f for f in os.listdir(RAW_VIDEO_DIR)
        if os.path.splitext(f)[1].lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    ])
    return {"videos": videos}


@app.get("/folders")
def list_folders():
    """Lists all subdirectories within the raw video directory."""
    if not os.path.isdir(RAW_VIDEO_DIR):
        return {"folders": []}
    
    folders = []
    for root, dirs, files in os.walk(RAW_VIDEO_DIR):
        # For each directory, create its path relative to RAW_VIDEO_DIR
        for name in dirs:
            dir_path = os.path.join(root, name)
            relative_path = os.path.relpath(dir_path, RAW_VIDEO_DIR)
            if relative_path != '.':
                folders.append(relative_path)
    
    # Add the root directory itself
    folders.insert(0, ".")
    return {"folders": sorted(folders)}


@app.post("/process")
def process_video(req: ProcessRequest):
    video_path = os.path.join(RAW_VIDEO_DIR, req.video)
    if not os.path.isfile(video_path):
        raise HTTPException(status_code=404, detail=f"Video not found: {req.video}")

    pipeline = SkeletonPipeline(
        save_npy=not req.no_npy,
        save_json=not req.no_json,
        save_pickle=not req.no_pickle,
        generate_preview=not req.no_preview,
        generate_overlay=not req.no_overlay,
        generate_skeleton_only=not req.no_skeleton_only,
    )

    try:
        keypoints = pipeline.process_video(video_path, label=req.label)
    except Exception as e:
        tb = traceback.format_exc()
        raise HTTPException(status_code=500, detail=str(e) + "\n" + tb)

    base = os.path.splitext(os.path.basename(video_path))[0]
    resp = {
        "video": req.video,
        "npy": os.path.join(SKELETON_DIR, f"{base}.npy") if pipeline.save_npy else None,
        "json": os.path.join(JSON_DIR, f"{base}.json") if pipeline.save_json else None,
        "pkl": os.path.join(PICKLE_DIR, f"{base}.pkl") if pipeline.save_pickle else None,
        "preview_skeleton": os.path.join(PREVIEW_SKELETON_DIR, f"{base}_skeleton.mp4") if (pipeline.gen_preview and pipeline.gen_skel_only) else None,
        "preview_overlay": os.path.join(PREVIEW_OVERLAY_DIR, f"{base}_overlay.mp4") if (pipeline.gen_preview and pipeline.gen_overlay) else None,
        "frames_extracted": int(keypoints.shape[0]),
        "keypoint_dims": int(keypoints.shape[2]) if keypoints.ndim == 3 else None,
    }
    return resp


# ------------------------------------------------------------
# Simple in-memory job queue (per-process)
# ------------------------------------------------------------


class JobStatus(BaseModel):
    id: str
    video: str
    status: str  # pending, running, done, failed
    error: Optional[str] = None
    output_subpath: Optional[str] = None


JOBS: Dict[str, Dict[str, Any]] = {}


def _run_job(job_id: str, video_path: str, output_subpath: str, req: BaseModel) -> None:
    """Background worker for a single video job."""
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
            generate_preview=not req.no_preview,
            generate_overlay=not req.no_overlay,
            generate_skeleton_only=not req.no_skeleton_only,
        )
        pipeline.process_video(full_video_path, label=req.label, output_subpath=output_subpath)
        job["status"] = "done"
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)


@app.post("/jobs", response_model=JobStatus)
def create_job(req: ProcessRequest, background_tasks: BackgroundTasks):
    """Create a background job for processing a single video."""
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

    background_tasks.add_task(_run_job, job_id, req.video, "", req)

    return JobStatus(**JOBS[job_id])


@app.post("/jobs/batch", response_model=list[JobStatus])
def create_batch_job(req: BatchProcessRequest, background_tasks: BackgroundTasks):
    """Create multiple background jobs from a list of video paths."""
    created_jobs = []
    for video_rel_path in req.videos:
        full_video_path = os.path.join(RAW_VIDEO_DIR, video_rel_path)
        if not os.path.isfile(full_video_path):
            # Silently skip non-existent files for now, or could collect errors
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
        
        # Pass video_rel_path and output_subpath to the worker
        background_tasks.add_task(_run_job, job_id, video_rel_path, output_subpath, req)
        created_jobs.append(JobStatus(**job_data))

    if not created_jobs:
        raise HTTPException(
            status_code=404,
            detail="No valid videos found in the provided list."
        )

    return created_jobs


@app.get("/jobs", response_model=list[JobStatus])
def list_jobs():
    return [JobStatus(**job) for job in sorted(JOBS.values(), key=lambda j: j.get("video"))]


@app.get("/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatus(**job)

