"""
FastAPI server for the Knowledge Search Engine.

This module provides REST API endpoints for searching documents.
"""

import datetime
import json
import pickle

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI application
app = FastAPI(
    title="Knowledge Search Engine",
    description="Personal knowledge graph with neural search.",
    version="1.0.0",
)

# Configure CORS for frontend access
ALLOWED_ORIGINS = [
    "https://raphaelsty.github.io",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:3000",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class Knowledge:
    """
    Wrapper class for the knowledge pipeline.

    Manages the lifecycle of the search pipeline, providing methods for
    searching documents.
    """

    def __init__(self) -> None:
        self.pipeline = None

    def start(self) -> "Knowledge":
        """Load the serialized pipeline from disk."""
        with open("database/pipeline.pkl", "rb") as f:
            self.pipeline = pickle.load(f)
        return self

    def get_latest_documents(self, count: int) -> list[dict]:
        """Get the most recently added documents."""
        return self.pipeline.get_latest_documents(count=count)

    def search(self, q: str, tags: str) -> list[dict]:
        """Search for documents matching a query."""
        return self.pipeline.search(q=q, tags=tags)


# Global knowledge instance
knowledge = Knowledge()


@app.get("/sources")
def get_sources() -> list:
    """Return available source filters extracted from database URLs."""
    try:
        with open("docs/sources.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


@app.get("/latest/{count}")
def get_latest(count: int) -> dict:
    """Get the most recently added documents."""
    documents = knowledge.get_latest_documents(count=count)
    return {"documents": documents}


@app.get("/search/{sort}/{tags}/{q}")
def search(tags: str, sort: bool, q: str) -> dict:
    """Search for documents with optional sorting and tag filtering."""
    tags = tags != "null"
    documents = knowledge.search(q=q, tags=tags)

    if bool(sort):
        documents = [
            document
            for _, document in sorted(
                [(document["date"], document) for document in documents],
                key=lambda doc: datetime.datetime.strptime(doc[0], "%Y-%m-%d"),
                reverse=True,
            )
        ]

    return {"documents": documents}


@app.on_event("startup")
def start() -> Knowledge:
    """Initialize the knowledge pipeline on server startup."""
    return knowledge.start()
