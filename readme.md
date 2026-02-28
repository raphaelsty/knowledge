<div align="center">

# Knowledge

</div>

<p align="center">
<a href="https://raphaelsty.github.io/knowledge/"><strong>Personal Knowledge Base</strong></a>
</p>

<p align="center">
<img src="web/img/demo.gif" alt="Demonstration GIF" style="width:100%; border-radius:10px; box-shadow:0 4px 8px rgba(0,0,0,0.1);">
</p>

**Knowledge** is a web application that automatically transforms the digital footprint into a personal search engine. It fetches content you interact with from various platforms—**GitHub**, **HackerNews**, **Zotero**, **HuggingFace likes**, **X/Twitter**—and organizes it into a navigable knowledge graph.

---

## 🌟 Features

- **🤖 Automatic Aggregation:** Daily, automated extraction of GitHub stars, HackerNews upvotes, and Zotero library.

- **🔍 Powerful Search:** A built-in search engine to instantly find any item you've saved or interacted with.

- **🕸️ Knowledge Graph:** Navigate bookmarks through a graph of automatically extracted topics and their connections.

My Personal Knowledge Base is available at [raphaelsty.github.io/knowledge](https://raphaelsty.github.io/knowledge/).

---

## 🛠️ How It Works

A GitHub Actions workflow runs once a day to perform the following tasks:

1.  **Extracts Content** from specified accounts:
    - GitHub Stars
    - HackerNews Upvotes
    - Zotero Records
    - HuggingFace Likes
    - X/Twitter Bookmarks & Likes
2.  **Processes and Stores Data** in the `web/data/` directory:
    - `database.json`: Contains all the raw records.
3.  **Deploys Updates**:
    - The frontend on GitHub Pages is refreshed with the latest data.

The search engine is powered by a [ColBERT](https://github.com/lightonai/pylate-rs) model served by a Rust API, with a static frontend hosted on GitHub Pages.

## 🚀 Getting Started: Installation & Deployment

Follow these steps to deploy your own instance of Knowledge.

### 1\. Fork & Clone

First, fork this repository to your own GitHub account and then clone it to your local machine.

### 2\. Configuration

#### A. Configure Secrets

The application requires API keys and credentials to function. These must be set as **Repository secrets** in your forked repository's settings (`Settings` > `Secrets and variables` > `Actions`).

<br>

<table style="width:100%; border-collapse: collapse;">
<thead>
<tr>
<th style="text-align:left; padding:8px; border-bottom: 1px solid \#ddd;">Secret</th>
<th style="text-align:left; padding:8px; border-bottom: 1px solid \#ddd;">Service</th>
<th style="text-align:center; padding:8px; border-bottom: 1px solid \#ddd;">Required</th>
<th style="text-align:left; padding:8px; border-bottom: 1px solid \#ddd;">Description</th>
</tr>
</thead>
<tbody>
<tr>
<td style="padding:8px; border-bottom: 1px solid \#ddd;"><code>ZOTERO_API_KEY</code></td>
<td style="padding:8px; border-bottom: 1px solid \#ddd;"><a href="https://www.zotero.org/settings/keys">Zotero Settings</a></td>
<td style="text-align:center; padding:8px; border-bottom: 1px solid \#ddd;">Optional</td>
<td style="padding:8px; border-bottom: 1px solid \#ddd;">An API key to access your Zotero library.</td>
</tr>
<tr>
<td style="padding:8px; border-bottom: 1px solid \#ddd;"><code>ZOTERO_LIBRARY_ID</code></td>
<td style="padding:8px; border-bottom: 1px solid \#ddd;"><a href="https://www.zotero.org">Zotero</a></td>
<td style="text-align:center; padding:8px; border-bottom: 1px solid \#ddd;">Optional</td>
<td style="padding:8px; border-bottom: 1px solid \#ddd;">The ID of the Zotero group library you want to index.</td>
</tr>
<tr>
<td style="padding:8px; border-bottom: 1px solid \#ddd;"><code>HACKERNEWS_USERNAME</code></td>
<td style="padding:8px; border-bottom: 1px solid \#ddd;"><a href="https://news.ycombinator.com">Hacker News</a></td>
<td style="text-align:center; padding:8px; border-bottom: 1px solid \#ddd;">Optional</td>
<td style="padding:8px; border-bottom: 1px solid \#ddd;">HackerNews username to fetch upvoted posts.</td>
</tr>
<tr>
<td style="padding:8px;"><code>HACKERNEWS_PASSWORD</code></td>
<td style="padding:8px;"><a href="https://news.ycombinator.com/">Hacker News</a></td>
<td style="text-align:center; padding:8px;">Optional</td>
<td style="padding:8px;">HackerNews password.</td>
</tr>
<tr>
<td style="padding:8px;"><code>HUGGINGFACE_TOKEN</code></td>
<td style="padding:8px;"><a href="https://huggingface.co/settings/tokens">HuggingFace</a></td>
<td style="text-align:center; padding:8px;">Optional</td>
<td style="padding:8px;">Token to fetch your HuggingFace liked models and datasets.</td>
</tr>
<tr>
<td style="padding:8px;"><code>TWITTER_AUTH_TOKEN</code></td>
<td style="padding:8px;"><a href="https://x.com">X/Twitter</a></td>
<td style="text-align:center; padding:8px;">Optional</td>
<td style="padding:8px;">Browser <code>auth_token</code> cookie for X. See the <a href="#-xtwitter-integration">X/Twitter section</a> below.</td>
</tr>
<tr>
<td style="padding:8px;"><code>TWITTER_CT0</code></td>
<td style="padding:8px;"><a href="https://x.com">X/Twitter</a></td>
<td style="text-align:center; padding:8px;">Optional</td>
<td style="padding:8px;">Browser <code>ct0</code> cookie (CSRF token) for X. See the <a href="#-xtwitter-integration">X/Twitter section</a> below.</td>
</tr>
</tbody>
</table>

#### B. Specify Sources

Next, edit the `sources.yml` file at the root of the repository to configure your data sources.

```yml
github:
  - "raphaelsty"
  - "gbolmier"
  - "MaxHalford"

twitter:
  username: "raphaelsrty"
  min_likes: 10
  max_pages: 2

huggingface: True
```

- **github**: List of GitHub usernames whose starred repositories you want to track.
- **twitter**: X/Twitter configuration. Set `username` to your handle, `min_likes` to filter bookmarks, and `max_pages` to control how many pages of recent likes to fetch per run (~100 likes per page). Remove this block entirely to skip X/Twitter.
- **huggingface**: Set to `True` to fetch your HuggingFace liked models and datasets (requires `HUGGINGFACE_TOKEN` secret).

### 3\. Deployment

#### A. Deploy to a VPS

The recommended deployment uses Docker Compose on a VPS (e.g. Hetzner CX32: 4 vCPU, 8GB RAM, ~€7.49/month).

1.  Point your domain's DNS A record to the server IP.
2.  Clone the repository on the server and run:
    ```sh
    DOMAIN=your-domain.com POSTGRES_PASSWORD=a-strong-password make deploy-build
    ```
    Caddy handles HTTPS automatically via Let's Encrypt.

#### B. Set up GitHub Pages

1.  Go to your forked repository's settings (`Settings` > `Pages`).
2.  Under `Build and deployment`, select the **Source** as `GitHub Actions` (the `web/` folder is not supported by branch-based deployment, which only allows `/` or `/docs`).

---

## 💻 Local Development

Start all services locally with Docker Compose:

```sh
make up
```

This starts PostgreSQL, the search API, data API, events API, and a local web server on port 3000.

---

## 🔌 Zotero Integration

The Zotero integration allows you to save academic papers, articles, and other documents, which will then be automatically indexed by your search engine.

- **Browser Extension:** Use the Zotero Connector extension for your browser to easily save documents from the web.

- **Mobile App:** The Zotero mobile app lets you add documents on the go. Any uploads will be indexed within a few hours.

  <div style="display: flex; justify-content: space-around; align-items: center; gap: 10px;">
  <img src="./web/img/arxiv_1.png" alt="Zotero mobile app" style="width: 30%;">
  <img src="./web/img/arxiv_2.png" alt="Zotero mobile app" style="width: 30%;">
  <img src="./web/img/arxiv_3.png" alt="Zotero mobile app" style="width: 30%;">
  </div>

---

## 🐦 X/Twitter Integration

> This source is **entirely optional**. If you don't need it, simply remove the `twitter` block from `sources.yml` and skip this section.

The X/Twitter integration fetches your **bookmarked tweets** (filtered by a minimum like count) and your **liked tweets**. It uses [Twikit](https://github.com/d60/twikit), which connects to X's internal API — **no paid API key required**.

The setup is a bit of a trick: since X blocks automated logins from servers (Cloudflare protection), authentication relies on **browser cookies** rather than username/password.

### How to get your cookies

1. Log into [x.com](https://x.com) in your browser.
2. Open DevTools (**F12** or **Cmd+Option+I**).
3. Go to **Application** > **Cookies** > `https://x.com`.
4. Copy the values for `auth_token` and `ct0`.
5. Add them as GitHub repository secrets: `TWITTER_AUTH_TOKEN` and `TWITTER_CT0`.

**On macOS with Safari**, you can skip the manual step — the pipeline automatically extracts cookies from Safari when running locally (requires Full Disk Access for your terminal). This means `uv run python run.py` works out of the box on your Mac with no environment variables needed.

### Cookie expiration

The `auth_token` cookie typically lasts **about a year**. The `ct0` token may expire sooner. When the CI starts failing on the Twitter step, simply grab fresh cookies from your browser and update the GitHub secrets.

### Configuration

In `sources.yml`:

```yml
twitter:
  username: "your_handle" # Your X screen name
  min_likes: 10 # Minimum likes for bookmarked tweets to be included
  max_pages: 2 # Pages of recent likes to fetch per run (~100/page)
```

- **Bookmarks**: All pages are always fetched (typically small). Filtered by `min_likes`.
- **Likes**: Only the `max_pages` most recent pages are fetched. This keeps daily CI runs fast while still catching new activity. For an initial backfill of your full like history, temporarily increase `max_pages` to `200` and run locally.

---

## 📜 License

This project is licensed under the **GNU General Public License v3.0**.

Knowledge Copyright (C) 2023-2025 Raphaël Sourty

TODO: Finish first version of the tree. Put tags in the leafs. Then write a program which call chatgpt and reorganize the tree automatically. Then assign a document to the leaf where it has the most tags that matchs. Then find a way to update the index with new documents and re-create the tree once every week. Assert that if re-creation failed, then we keep the previous tree. Then integrate next-plaid and enable filtering search based on source.
Improve tweet handling, only kept the main tweet of a thread or something like this, potentially extract arxiv / github / hackernews / article from the tweet and save it within the database. Then find a way to be gentle with twitter, the idea is to parse up to 5 tweets if all news then parse a bit more without re-parsing already parsed tweets.
