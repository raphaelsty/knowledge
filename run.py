import json
import os
import pickle

import yaml

from knowledge_database import (
    github,
    hackernews,
    pipeline,
    semanlink,
    tags,
    twitter,
    zotero,
)

with open("sources.yml", "r") as f:
    sources = yaml.load(f, Loader=yaml.FullLoader)

twitter_token = os.environ.get("TWITTER_TOKEN")
hackernews_username = os.environ.get("HACKERNEWS_USERNAME")
hackernews_password = os.environ.get("HACKERNEWS_PASSWORD")
zotero_library_id = os.environ.get("ZOTERO_LIBRARY_ID")
zotero_api_key = os.environ.get("ZOTERO_API_KEY")

data = {}

if os.path.exists("database/database.json"):
    with open("database/database.json", "r") as f:
        data = json.load(f)

# Twitter
if twitter_token is not None and sources.get("twitter") is not None:
    print("Twitter knowledge.")
    for user_id_username in sources["twitter"]:
        user_id = user_id_username[0]
        username = user_id_username[1]

        knowledge = twitter.Twitter(
            username=username, user_id=user_id, token=twitter_token
        )
        knowledge = {
            url: document for url, document in knowledge().items() if url not in data
        }
        print(f"Found {len(knowledge)} new Twitter documents.")
        data = {**data, **knowledge}
else:
    print("No Twitter token.")

# Github
if sources.get("github") is not None:
    print("Github knowledge.")
    for user in sources["github"]:
        knowledge = github.Github(user=user)
        knowledge = {
            url: document for url, document in knowledge().items() if url not in data
        }
        print(f"Found {len(knowledge)} new Github documents.")
        data = {**data, **knowledge}


# Hackernews
if hackernews_username is not None and hackernews_password is not None:
    print("Hackernews knowledge.")
    knowledge = hackernews.HackerNews(
        username=hackernews_username,
        password=hackernews_password,
    )
    knowledge = {
        url: document for url, document in knowledge().items() if url not in data
    }
    print(f"Found {len(knowledge)} new Hackernews documents.")
    data = {**data, **knowledge}
else:
    print("No Hackernews credentials.")

# Zotero
if zotero_library_id is not None and zotero_api_key is not None:
    print("Zotero knowledge.")
    knowledge = zotero.Zotero(
        library_id=zotero_library_id,
        library_type="group",
        api_key=zotero_api_key,
    )
    knowledge = {
        url: document for url, document in knowledge().items() if url not in data
    }
    print(f"Found {len(knowledge)} new Zotero documents.")
    data = {**data, **knowledge}
else:
    print("No Zotero credentials.")

# Semanlink
if sources["semanlink"]:
    print("Semanlink knowledge.")
    knowledge = semanlink.Semanlink(
        urls=[
            "https://raw.githubusercontent.com/fpservant/semanlink-kdmkb/master/files/sldocs-2023-01-26.ttl",
            "https://raw.githubusercontent.com/fpservant/semanlink-kdmkb/master/files/sltags-2020-11-18.ttl",
        ]
    )
    knowledge = {
        url: document for url, document in knowledge().items() if url not in data
    }
    print(f"Found {len(knowledge)} new semanlink documents.")
    data = {**data, **knowledge}
else:
    print("Semanlink disabled.")

if len(data) == 0:
    raise ValueError("No data.")

# Sanity check.
for url, document in data.items():
    for field in ["title", "tags", "summary", "date"]:
        if document.get(field, None) is None:
            document[field] = ""

print("Adding extra tags.")
data = tags.get_extra_tags(data=data)

print("Saving database.")
with open("database/database.json", "w") as f:
    json.dump(data, f, indent=4)

excluded_tags = {
    "twitter": True,
    "github": True,
    "semanlink": True,
    "hackernews": True,
    "arxiv doc": True,
}

print("Exporting tree of tags.")
triples = tags.get_tags_triples(data=data, excluded_tags=excluded_tags)
with open("database/triples.json", "w") as f:
    json.dump(triples, f, indent=4)

print("Serializing pipeline.")
knowledge_pipeline = pipeline.Pipeline(
    documents=data,
    triples=triples,
    excluded_tags=excluded_tags,
)
with open("database/pipeline.pkl", "wb") as f:
    pickle.dump(knowledge_pipeline, f)
