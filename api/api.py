"""
FastAPI server for the Knowledge Search Engine.

This module provides REST API endpoints for searching documents, visualizing
the knowledge graph, and interacting with an LLM for document recommendations.
"""

import datetime
import pickle

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse, StreamingResponse
from openai import OpenAI

# Initialize FastAPI application
app = FastAPI(
    title="Knowledge Search Engine",
    description="Personal knowledge graph with neural search and visualization.",
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
    searching documents and generating graph visualizations.

    Attributes
    ----------
    pipeline : Pipeline | None
        The loaded knowledge pipeline, or None before initialization.
    """

    def __init__(self) -> None:
        self.pipeline = None

    def start(self) -> "Knowledge":
        """
        Load the serialized pipeline from disk.

        Returns
        -------
        Knowledge
            Self reference for method chaining.
        """
        with open("database/pipeline.pkl", "rb") as f:
            self.pipeline = pickle.load(f)
        return self

    def get_latest_documents(self, count: int) -> list[dict]:
        """
        Get the most recently added documents.

        Parameters
        ----------
        count : int
            Number of documents to return.

        Returns
        -------
        list[dict]
            List of recent documents sorted by date.
        """
        return self.pipeline.get_latest_documents(count=count)

    def search(self, q: str, tags: str) -> list[dict]:
        """
        Search for documents matching a query.

        Parameters
        ----------
        q : str
            Search query string.
        tags : str
            Whether to filter by tags ("true" or other).

        Returns
        -------
        list[dict]
            List of matching documents ranked by relevance.
        """
        return self.pipeline.search(q=q, tags=tags)

    def plot(
        self,
        q: str,
        k_tags: int,
        k_yens: int = 1,
        k_walk: int = 3,
    ) -> dict:
        """
        Generate knowledge graph data for visualization.

        Parameters
        ----------
        q : str
            Search query to build the graph around.
        k_tags : int
            Maximum number of tags to include.
        k_yens : int, default=1
            Number of shortest paths between tags.
        k_walk : int, default=3
            Number of neighbors for random walks.

        Returns
        -------
        dict
            Dictionary with 'nodes' and 'links' for graph visualization.
        """
        nodes, links = self.pipeline.plot(
            q=q,
            k_tags=k_tags,
            k_yens=k_yens,
            k_walk=k_walk,
        )
        return {"nodes": nodes, "links": links}


# Global knowledge instance
knowledge = Knowledge()


@app.get("/latest/{count}")
def get_latest(count: int) -> dict:
    """
    Get the most recently added documents.

    Parameters
    ----------
    count : int
        Number of documents to return.

    Returns
    -------
    dict
        Dictionary containing 'documents' list.
    """
    documents = knowledge.get_latest_documents(count=count)
    return {"documents": documents}


@app.get("/search/{sort}/{tags}/{q}")
def search(tags: str, sort: bool, q: str) -> dict:
    """
    Search for documents with optional sorting and tag filtering.

    Parameters
    ----------
    tags : str
        Tag filter mode ("null" for no filter, any other value to enable).
    sort : bool
        Whether to sort results by date (newest first).
    q : str
        Search query string.

    Returns
    -------
    dict
        Dictionary containing 'documents' list.
    """
    tags = tags != "null"
    documents = knowledge.search(q=q, tags=tags)

    # Sort by date if requested
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


@app.get("/plot/{k_tags}/{q}", response_class=ORJSONResponse)
def plot(k_tags: int, q: str) -> dict:
    """
    Generate knowledge graph visualization data.

    Parameters
    ----------
    k_tags : int
        Maximum number of tags to include in the graph.
    q : str
        Search query to build the graph around.

    Returns
    -------
    dict
        Dictionary with 'nodes' and 'links' for D3/ForceGraph visualization.
    """
    return knowledge.plot(q=q, k_tags=k_tags)


@app.get("/expand/{node_id}", response_class=ORJSONResponse)
def expand_node(node_id: str) -> dict:
    """
    Expand a node to show its neighbors.

    Used for progressive graph exploration where users click nodes
    to reveal their connections.

    Parameters
    ----------
    node_id : str
        The tag name of the node to expand.

    Returns
    -------
    dict
        Dictionary with 'nodes' and 'links' for the expanded subgraph.
    """
    nodes, links = knowledge.pipeline.graph.expand(node_id=node_id)
    return {"nodes": nodes, "links": links}


@app.on_event("startup")
def start() -> Knowledge:
    """
    Initialize the knowledge pipeline on server startup.

    Returns
    -------
    Knowledge
        The initialized knowledge instance.
    """
    return knowledge.start()


async def async_chat(query: str, content: str):
    """
    Stream LLM responses for document recommendations.

    Uses GPT-4 to analyze retrieved documents and provide natural language
    recommendations based on the user's query.

    Parameters
    ----------
    query : str
        The user's search query.
    content : str
        Formatted string of retrieved document metadata.

    Yields
    ------
    str
        Incrementally built response text as tokens arrive.
    """
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant designed to output JSON.",
            },
            {
                "role": "user",
                "content": (
                    f"Hi, answer in comprehensible english, do not reply with json, "
                    f"among the set of documents retrieved, which documents are related "
                    f"to my query: {query}, set of documents: {content}."
                ),
            },
        ],
        max_tokens=200,
        stream=True,
    )

    answer = ""
    for chunk in response:
        if chunk.choices[0].finish_reason == "stop":
            break

        token = chunk.choices[0].delta.content
        if token is None:
            break

        answer += token
        yield answer.strip()


@app.get("/chat/{k_tags}/{q}")
async def chat(k_tags: int, q: str) -> StreamingResponse:
    """
    Get LLM-powered document recommendations.

    Searches for relevant documents and streams GPT-4's analysis
    of which documents best match the query.

    Parameters
    ----------
    k_tags : int
        Not used (kept for API compatibility).
    q : str
        Search query for finding relevant documents.

    Returns
    -------
    StreamingResponse
        Streaming text response from the LLM.
    """
    documents = knowledge.search(q=q, tags=False)

    # Format documents for LLM context
    content = ""
    for document in documents:
        content += "title: " + document["title"] + "\n"
        content += "summary: " + document["summary"][:30] + "\n"
        content += "tags: " + (", ".join(document["tags"] + document["extra-tags"]) + "\n")
        content += "url: " + document["url"] + "\n\n"

    # Truncate to fit context window
    content = "title: ".join(content[:3000].split("title:")[:-1])

    return StreamingResponse(
        async_chat(query=q, content=content),
        media_type="text/plain",
    )
