"""
API Data Models Module

This module defines the Pydantic schemas for request and response
objects used within the FastAPI application.
"""

from pydantic import BaseModel
from typing import Optional

class ProcessRequest(BaseModel):
    """
    Schema for a single video processing request.
    """
    video: str
    label: Optional[int] = None
    no_npy: bool = False
    no_json: bool = False
    no_pickle: bool = False
    no_excel: bool = False
    no_preview: bool = False
    no_overlay: bool = False
    no_skeleton_only: bool = False


class BatchProcessRequest(BaseModel):
    """
    Schema for a batch video processing request.
    """
    videos: list[str]  # Relative paths from RAW_VIDEO_DIR
    label: Optional[int] = None
    no_npy: bool = False
    no_json: bool = False
    no_pickle: bool = False
    no_excel: bool = False
    no_preview: bool = False
    no_overlay: bool = False
    no_skeleton_only: bool = False


class JobStatus(BaseModel):
    """
    Schema representing the status of a background processing job.
    """
    id: str
    video: str
    status: str  # pending, running, done, failed
    error: Optional[str] = None
    output_subpath: Optional[str] = None
