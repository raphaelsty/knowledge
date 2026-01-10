"""
HuggingFace module for extracting liked models, datasets, and spaces.

This module fetches liked items from a HuggingFace user's profile and
extracts metadata from README files.
"""

import datetime
import re

import requests
import trafilatura
from huggingface_hub import HfApi

__all__ = ["HuggingFace"]


class HuggingFace:
    """
    Extract knowledge from HuggingFace liked repositories.

    Fetches liked models, datasets, and spaces from a user's HuggingFace
    profile, extracting titles and summaries from repository README files.

    Parameters
    ----------
    token : str, optional
        HuggingFace User Access Token. If not provided, relies on
        local authentication via `huggingface-cli login`.

    Example
    -------
    >>> from knowledge_database import huggingface
    >>>
    >>> hf = huggingface.HuggingFace(token="hf_xxxxx")
    >>> documents = hf()
    >>>
    >>> for url, doc in documents.items():
    ...     print(f"{doc['title']}: {doc['tags']}")
    """

    def __init__(self, token: str = None):
        self.token = token
        self.api = HfApi(token=self.token)

    def __call__(self) -> dict[str, dict]:
        """
        Fetch liked repositories and extract document metadata.

        Returns
        -------
        dict[str, dict]
            Dictionary mapping HuggingFace URLs to document metadata containing:
            - title: Repository name with HuggingFace emoji prefix
            - summary: First ~50 tokens of README content
            - date: Current date
            - tags: ["huggingface", <repo_type>]
        """
        data: dict[str, dict] = {}

        try:
            likes = self.api.list_liked_repos()
        except Exception as e:
            print(f"Error fetching likes: {e}")
            return data

        # Process liked models
        if hasattr(likes, "models"):
            for model in likes.models:
                repo_id = model.repo_id if hasattr(model, "repo_id") else str(model)
                url = f"https://huggingface.co/{repo_id}"
                raw_url = f"https://huggingface.co/{repo_id}/resolve/main/README.md"
                self._process_entry(data, url, raw_url, repo_id, "model")

        # Process liked datasets
        if hasattr(likes, "datasets"):
            for dataset in likes.datasets:
                repo_id = dataset.repo_id if hasattr(dataset, "repo_id") else str(dataset)
                dataset_url = f"https://huggingface.co/datasets/{repo_id}"
                branch = self._get_default_branch(repo_id, "dataset")
                raw_url = f"https://huggingface.co/datasets/{repo_id}/resolve/{branch}/README.md"
                self._process_entry(data, dataset_url, raw_url, repo_id, "dataset")

        # Process liked spaces
        if hasattr(likes, "spaces"):
            for space in likes.spaces:
                repo_id = space.repo_id if hasattr(space, "repo_id") else str(space)
                url = f"https://huggingface.co/spaces/{repo_id}"
                branch = self._get_default_branch(repo_id, "space")
                raw_url = f"https://huggingface.co/spaces/{repo_id}/resolve/{branch}/README.md"
                self._process_entry(data, url, raw_url, repo_id, "space")

        return data

    def _get_default_branch(self, repo_id: str, repo_type: str) -> str:
        """
        Get the default branch name for a repository.

        Parameters
        ----------
        repo_id : str
            Repository identifier (e.g., "username/repo-name").
        repo_type : str
            Type of repository: "model", "dataset", or "space".

        Returns
        -------
        str
            Default branch name, or "main" if unable to determine.
        """
        try:
            repo_info = self.api.repo_info(repo_id=repo_id, repo_type=repo_type)
            branch = getattr(repo_info, "default_branch", None)
            return branch if branch else "main"
        except Exception:
            return "main"

    def _process_entry(
        self,
        data: dict,
        data_url: str,
        summarization_url: str,
        title_suffix: str,
        tag_type: str,
    ) -> None:
        """
        Process a single repository and add it to the data dictionary.

        Parameters
        ----------
        data : dict
            Dictionary to store the processed entry.
        data_url : str
            URL to use as the document key.
        summarization_url : str
            URL to fetch README content from.
        title_suffix : str
            Repository name to include in the title.
        tag_type : str
            Type tag to add: "model", "dataset", or "space".
        """
        print(f"Processing {tag_type}: {title_suffix}")
        summary = self.get_summary(summarization_url)

        data[data_url] = {
            "title": f"HuggingFace {title_suffix}",
            "tags": ["huggingface", tag_type],
            "summary": summary,
            "date": datetime.datetime.today().strftime("%Y-%m-%d"),
        }

    @staticmethod
    def get_summary(url: str, num_tokens: int = 50) -> str:
        """
        Extract summary from a HuggingFace README URL.

        Handles YAML frontmatter commonly found in HuggingFace READMEs
        and extracts clean text content.

        Parameters
        ----------
        url : str
            URL to the raw README.md file.
        num_tokens : int, default=50
            Number of words to include in the summary.

        Returns
        -------
        str
            First N words of the README content, or empty string on failure.
        """
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            content = response.text

            # Handle YAML frontmatter (common in HuggingFace READMEs)
            if content.strip().startswith("---"):
                # Remove YAML block between --- markers
                cleaned_text = re.sub(r"^---\n.*?\n---", "", content, count=1, flags=re.DOTALL)
                # Strip remaining markdown syntax
                cleaned_text = re.sub(r"[#*`]", "", cleaned_text)
                cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
            else:
                # Fall back to trafilatura for HTML content
                core_text = trafilatura.extract(content)
                if not core_text:
                    cleaned_text = re.sub(r"\s+", " ", content).strip()
                else:
                    cleaned_text = re.sub(r"\s+", " ", core_text).strip()

            tokens = cleaned_text.split()
            first_n_tokens = tokens[:num_tokens]

            return " ".join(first_n_tokens)

        except requests.exceptions.RequestException as e:
            print(f"Could not fetch {url}: {e}")
            return ""
        except Exception as e:
            print(f"An error occurred while processing {url}: {e}")
            return ""
