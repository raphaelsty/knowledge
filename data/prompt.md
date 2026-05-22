# Personality Source Discovery Prompt

Use this prompt to research and configure all available public data sources for a new personality. The goal is maximum coverage — every public trace of their work, writing, talks, and interactions.

---

## Prompt

You are configuring a public knowledge base for **{NAME}** ({ROLE/BIO}).

Their known accounts:
- GitHub: @{github_username}
- Twitter/X: @{twitter_handle}

### Step 1: Find all public accounts

Search for their presence on each platform. Use these techniques:

**HackerNews username:**
- Search for their blog/website URL on HN: `https://hn.algolia.com/api/v1/search?query={their_blog_domain}&hitsPerPage=20`
- The author who submitted the most posts linking to their blog is likely them
- Verify by checking the HN user's "about" field: `https://hn.algolia.com/api/v1/users/{username}`
- Also try common username patterns: their name, initials, GitHub handle

**Google Scholar ID:**
- Search: `site:scholar.google.com "{NAME}"` + their institution/topic
- The user ID is in the URL: `citations?user=XXXXXXXXXXX`
- Verify by checking the listed affiliation and publications match

**Semantic Scholar ID:**
- API: `https://api.semanticscholar.org/graph/v1/author/search?query={NAME}&limit=5`
- Verify by checking paper titles match known work

**Reddit username:**
- Search for their blog URL on Reddit: `site:reddit.com "{blog_domain}"`
- Or try their GitHub handle, real name variations
- Verify by checking post history matches their domain

**Twitter/X handle:**
- DuckDuckGo lite is the most reliable scrape-friendly search endpoint:
  `curl -sX POST https://lite.duckduckgo.com/lite/ -d "q={NAME}+twitter"`
  then regex-match `(twitter\.com|x\.com)/[A-Za-z0-9_]+` and take the first
  non-junk handle. (Junk = Twitter UI paths like `intent`, `share`, `home`, or
  brand accounts like `MetaAI`.)
- A ready-to-run version of this lives at
  `sources/utils/find_twitter_handles.py` — runs over every user with no
  `sources.twitter` and prints candidates; pass `--apply` to auto-set the
  top one (review first).
- Common-name fallback: try `{github_username}` as a Twitter handle (often
  identical), or `{first_initial}{lastname}`, `{firstname}_{lastname}`,
  `{firstname}{lastname}{job_year}`.
- Verify by visiting `x.com/{handle}` — the bio + linked website usually
  confirms the match.

**Stack Overflow user ID:**
- API: `https://api.stackexchange.com/2.3/users?inname={NAME}&site=stackoverflow&sort=reputation`
- Also check stats.stackexchange.com (Cross Validated) for ML researchers
- Verify by checking their top answers match their expertise

**YouTube:**
- Check if they have a channel: `https://www.youtube.com/@{handle}`
- Search for talks: `"{NAME}" talk OR interview OR conference OR keynote`
- Note their channel ID for the RSS feed source

**Blog/Website:**
- Check their GitHub profile bio for links
- Check `{github_username}.github.io` for a blog
- Check for RSS/Atom feeds: `/feed`, `/feed.xml`, `/index.xml`, `/atom.xml`, `/rss.xml`
- Check for sitemap: `/sitemap.xml` (gives full historical coverage)
- Check Substack: `{name}.substack.com`

**Wikipedia:**
- Search: `https://en.wikipedia.org/wiki/{Name}_(computer_scientist)` or similar
- The References and External links sections contain curated high-value URLs

**DBLP:**
- Search: `https://dblp.org/search/publ/api?q=author:{NAME}&format=json`
- Most comprehensive for CS publications

### Step 2: Estimate coverage

For each source found, estimate the document count:

| Source | Username/URL | Est. docs | Verified? |
|--------|-------------|----------|-----------|
| GitHub stars | @{user} | ? | |
| GitHub repos | @{user} | ? | |
| GitHub gists | @{user} | ? | |
| Blog RSS | {url} | ? | |
| Blog sitemap | {url} | ? | |
| Google Scholar | {user_id} | ? | |
| Semantic Scholar | {author_id} | ? | |
| DBLP | {author_name} | ? | |
| arXiv | {author_name} | ? | |
| YouTube channel | @{handle} | ? | |
| YouTube search | "{NAME}" talk | ? | |
| HN comments | @{hn_user} | ? | |
| HN submissions | @{hn_user} | ? | |
| Reddit | u/{reddit_user} | ? | |
| Stack Overflow | {user_id} | ? | |
| Wikipedia | {page_title} | ? | |
| Twitter/X | @{handle} | ? | |

### Step 3: Build the config

Generate the `sources` block for `data/personalities.json`:

```json
{
  "slug": "{slug}",
  "name": "{NAME}",
  "indexName": "{slug}",
  "description": "{one-line bio}",
  "avatar": "https://avatars.githubusercontent.com/u/{github_id}?v=4",
  "links": {
    "github": "https://github.com/{user}",
    "twitter": "https://x.com/{handle}"
  },
  "sources": {
    "github": ["{github_user}"],
    "github_repos": ["{github_user}"],
    "github_gists": ["{github_user}"],
    "blog": [
      {"url": "{rss_feed_url}", "tags": ["blog"]}
    ],
    "sitemap": [
      {"url": "{sitemap_url}", "tags": ["blog"], "url_filter": "/blog/"}
    ],
    "scholar": {"user_id": "{scholar_id}", "max_pages": 3, "min_citations": 10},
    "semantic_scholar": {"author_id": "{s2_id}", "max_papers": 200, "min_citations": 5},
    "dblp": {"author": "{full_name}", "max_results": 200},
    "arxiv": {"author": "{full_name}", "max_results": 200},
    "youtube": ["@{yt_handle}"],
    "youtube_search": {
      "queries": ["\"{NAME}\" talk", "\"{NAME}\" interview", "\"{NAME}\" conference"],
      "must_contain": ["{lowercase_lastname}", "{key_project}"],
      "max_results": 30
    },
    "hn_comments": {"username": "{hn_user}", "max_items": 500},
    "hn_submissions": {"username": "{hn_user}", "max_items": 500},
    "reddit": {"username": "{reddit_user}", "max_pages": 5},
    "stackoverflow": {"username": "{so_display_name}", "min_score": 1},
    "wikipedia": ["{Wikipedia_Page_Title}"]
  }
}
```

### Step 4: Validate

Only include sources that exist and are verified. Remove any source block where:
- The account doesn't exist or belongs to a different person
- The API returns 0 results
- The username is ambiguous (common names) — use `author_id` instead of name search

### Step 5: Watch for common pitfalls

- **arXiv author search**: Always quote the full name (`"Max Halford"` not `Max Halford`) to avoid matching other authors with the same first name
- **YouTube search**: Always set `must_contain` with the person's full name to avoid false positives (e.g. "Halford" matching "Rob Halford")
- **Google Scholar**: The `user_id` from the URL is the only reliable identifier — name search can fail
- **HN username**: Never guess — always verify by searching for their blog URL submissions or checking the profile's "about" field
- **DBLP / Semantic Scholar**: Common names return papers by different people — cross-check with known publication titles
- **Stack Overflow**: Low-rep accounts (<100) are usually not worth indexing
- **Wikipedia**: Only useful for well-known people who have a dedicated page

### Available source modules (20 total)

| Module | Config key | Auth | Dedup |
|--------|-----------|------|-------|
| GitHub stars | `github` | No | URL |
| GitHub repos | `github_repos` | No | URL |
| GitHub gists | `github_gists` | No | URL |
| Blog RSS | `blog` | No | URL |
| Sitemap | `sitemap` | No | URL |
| Google Scholar | `scholar` | No | Scholar page URL |
| Semantic Scholar | `semantic_scholar` | No | arXiv > DOI > S2 |
| DBLP | `dblp` | No | arXiv > DOI > DBLP |
| arXiv | `arxiv` | No | arXiv abs/ URL |
| YouTube channels | `youtube` | No | Video URL |
| YouTube search | `youtube_search` | No | Video URL |
| HN comments | `hn_comments` | No | Story URL |
| HN submissions | `hn_submissions` | No | Story URL |
| Reddit | `reddit` | No | External URL |
| Stack Overflow | `stackoverflow` | No | Question URL |
| Wikipedia | `wikipedia` | No | External URL |
| Twitter/X | `twitter` | Cookies | Tweet + extracted URLs |
| HackerNews auth | `hackernews` | Password | Post URL |
| HuggingFace | `huggingface` | Token | Repo URL |
| Zotero | `zotero` | API key | Item URL |

Paper deduplication: Scholar, S2, DBLP, and arXiv all produce canonical URLs (arXiv abs/ preferred, then DOI). Running all four for the same person won't create duplicates.
