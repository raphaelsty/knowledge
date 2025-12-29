import datetime
import re

import requests
import trafilatura
from huggingface_hub import HfApi

__all__ = ["HuggingFace"]


class HuggingFace:
    """HuggingFace liked models, datasets, and spaces.

    Parameters
    ----------
    token : str, optional
        HuggingFace User Access Token. If not provided, it relies on
        local authentication (huggingface-cli login).
    """

    def __init__(self, token: str = None):
        self.token = token
        self.api = HfApi(token=self.token)

    def __call__(self):
        """Get liked content on HuggingFace."""
        data = {}

        try:
            likes = self.api.list_liked_repos()
        except Exception as e:
            print(f"Error fetching likes: {e}")
            return data

        # Process Models
        if hasattr(likes, "models"):
            for model in likes.models:
                repo_id = model.repo_id if hasattr(model, "repo_id") else str(model)
                url = f"https://huggingface.co/{repo_id}"

                # Fetch raw README for summary
                raw_url = f"https://huggingface.co/{repo_id}/resolve/main/README.md"
                self._process_entry(data, url, raw_url, repo_id, "model")

        # Process Datasets
        if hasattr(likes, "datasets"):
            for dataset in likes.datasets:
                repo_id = (
                    dataset.repo_id if hasattr(dataset, "repo_id") else str(dataset)
                )
                dataset_url = f"https://huggingface.co/datasets/{repo_id}"

                # specific logic to find branch for datasets
                branch = self._get_default_branch(repo_id, "dataset")
                raw_url = f"https://huggingface.co/datasets/{repo_id}/resolve/{branch}/README.md"

                self._process_entry(data, dataset_url, raw_url, repo_id, "dataset")

        # Process Spaces
        if hasattr(likes, "spaces"):
            for space in likes.spaces:
                repo_id = space.repo_id if hasattr(space, "repo_id") else str(space)
                url = f"https://huggingface.co/spaces/{repo_id}"

                # specific logic to find branch for spaces
                branch = self._get_default_branch(repo_id, "space")
                raw_url = f"https://huggingface.co/spaces/{repo_id}/resolve/{branch}/README.md"

                self._process_entry(data, url, raw_url, repo_id, "space")

        return data

    def _get_default_branch(self, repo_id, repo_type):
        """Helper to safely get default branch."""
        try:
            repo_info = self.api.repo_info(repo_id=repo_id, repo_type=repo_type)
            return repo_info.default_branch if repo_info.default_branch else "main"
        except Exception:
            return "main"

    def _process_entry(self, data, data_url, summarization_url, title_suffix, tag_type):
        """Helper to fetch summary and populate data dict."""
        print(f"Processing {tag_type}: {title_suffix}")
        summary = self.get_summary(summarization_url)

        data[data_url] = {
            "title": f"🤗 HuggingFace {title_suffix}",
            "tags": ["huggingface", tag_type],
            "summary": summary,
            "date": datetime.datetime.today().strftime("%Y-%m-%d"),
        }

    @staticmethod
    def get_summary(url, num_tokens=50):
        """
        Fetches the content from a URL.
        If it's a raw Markdown file with YAML frontmatter, it strips the metadata.
        Otherwise, it attempts to extract text using trafilatura.
        """
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            content = response.text

            # Check for YAML frontmatter (starts with ---)
            if content.strip().startswith("---"):
                # Regex to remove the first block enclosed in ---
                # DOTALL allows . to match newlines
                cleaned_text = re.sub(
                    r"^---\n.*?\n---", "", content, count=1, flags=re.DOTALL
                )

                # Since it is markdown, we might want to strip common markdown syntax
                # for a cleaner summary (optional, but makes it readable)
                cleaned_text = re.sub(r"[#*`]", "", cleaned_text)
                cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

            else:
                # Fallback to trafilatura for non-markdown/HTML content
                core_text = trafilatura.extract(content)
                if not core_text:
                    # If trafilatura fails (e.g. on raw text), use raw content
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
