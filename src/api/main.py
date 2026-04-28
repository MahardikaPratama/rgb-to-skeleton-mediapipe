"""
API Initialization Module

This module initializes the FastAPI application, configures middleware,
and registers all the routing endpoints.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import router

app = FastAPI(title="RGB→Skeleton API")

# Allow local UI to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
