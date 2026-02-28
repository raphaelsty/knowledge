"""
FastAPI data server — serves generated data from PostgreSQL.

Endpoints:
    GET /api/folder_tree — folder tree structure
    GET /api/sources     — source filter list
    GET /api/health      — health check

Run:
    uvicorn sources.api:app --port 3001
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .database import ensure_schema, load_generated

app = FastAPI(title="Knowledge Data API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    ensure_schema()


@app.get("/api/folder_tree")
def folder_tree():
    data = load_generated("folder_tree")
    if data is None:
        return JSONResponse(status_code=404, content={"error": "folder_tree not found"})
    return data


@app.get("/api/sources")
def sources():
    data = load_generated("sources")
    if data is None:
        return JSONResponse(status_code=404, content={"error": "sources not found"})
    return data


@app.get("/api/health")
def health():
    return {"status": "ok"}
