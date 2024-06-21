import datetime
import pickle
import typing

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse, StreamingResponse
from openai import OpenAI

app = FastAPI(
    description="Personnal knowledge graph.",
    title="FactGPT",
    version="0.0.1",
)

origins = [
    "https://raphaelsty.github.io",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Knowledge:
    """This class is a wrapper around the pipeline."""

    def __init__(self) -> None:
        self.pipeline = None

    def start(self):
        """Load the pipeline."""
        with open("database/pipeline.pkl", "rb") as f:
            self.pipeline = pickle.load(f)
        return self

    def search(
        self,
        q: str,
        tags: str,
    ) -> typing.Dict:
        """Returns the documents."""
        return self.pipeline.search(q=q, tags=tags)

    def plot(
        self,
        q: str,
        k_tags: int,
        k_yens: int = 1,
        k_walk: int = 3,
    ) -> typing.Dict:
        """Returns the graph."""
        nodes, links = self.pipeline.plot(
            q=q,
            k_tags=k_tags,
            k_yens=k_yens,
            k_walk=k_walk,
        )
        return {"nodes": nodes, "links": links}


knowledge = Knowledge()


async def async_chat(query: str, content: str):
    """Re-rank the documents using ChatGPT."""
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {
                "role": "system",
                "content": """
                You are a helpful assistant designed to output JSON.
                """,
            },
            {
                "role": "user",
                "content": f"Hi, answer in comprehensible english, do not reply with json, among the set of documents retrieved, which documents are related to my query: {query}, set of documents: {content}.",
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


@app.get("/search/{sort}/{tags}/{k_tags}/{q}")
def search(k_tags: int, tags: str, sort: bool, q: str):
    """Search for documents."""
    tags = tags != "null"
    documents = knowledge.search(q=q, tags=tags)
    if bool(sort):
        documents = [
            document
            for _, document in sorted(
                [(document["date"], document) for document in documents],
                key=lambda document: datetime.datetime.strptime(
                    document[0], "%Y-%m-%d"
                ),
                reverse=True,
            )
        ]
    return {"documents": documents}


@app.get("/plot/{k_tags}/{q}", response_class=ORJSONResponse)
def plot(k_tags: int, q: str):
    """Plot tags."""
    return knowledge.plot(q=q, k_tags=k_tags)


@app.on_event("startup")
def start():
    """Intialiaze the pipeline."""
    return knowledge.start()


@app.get("/chat/{k_tags}/{q}")
async def chat(k_tags: int, q: str):
    """LLM recommendation."""
    documents = knowledge.search(q=q, tags=False)
    content = ""
    for document in documents:
        content += "title: " + document["title"] + "\n"
        content += "summary: " + document["summary"][:30] + "\n"
        content += "targs: " + (
            ", ".join(document["tags"] + document["extra-tags"]) + "\n"
        )
        content += "url: " + document["url"] + "\n\n"
    content = "title: ".join(content[:3000].split("title:")[:-1])
    return StreamingResponse(
        async_chat(query=q, content=content), media_type="text/plain"
    )
