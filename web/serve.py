"""SPA-aware static file server with local admin API.

Routing:
  /              → index.html (welcome page + OAuth landing)
  /profile       → profile.html
  /search        → search.html
  /admin         → admin.html  (localhost only)
  /admin/api/*   → JSON admin API (localhost only)
  /{slug}        → search.html (per-personality search view)
  everything else → physical files, falling back to index.html
"""

import http.server
import json
import os
import re
import sys
from urllib.parse import parse_qs, urlparse

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 3001
DIR = os.path.dirname(os.path.abspath(__file__))
SEARCH_PAGE = os.path.join(DIR, "search.html")
PROFILE_PAGE = os.path.join(DIR, "profile.html")
ADMIN_PAGE = os.path.join(DIR, "admin.html")

PROFILE_RE = re.compile(r"^/profile/?$")
SEARCH_RE = re.compile(r"^/search/?$")
ADMIN_API_RE = re.compile(r"^/admin/api/(.+?)/?$")
SLUG_RE = re.compile(r"^/[a-zA-Z0-9_-]+/?$")

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://knowledge:knowledge@localhost:5433/knowledge")

try:
    import psycopg
    import psycopg.rows

    HAS_DB = True
except ImportError:
    HAS_DB = False


def _ensure_source_runs_table() -> None:
    """Create `pipeline_source_runs` lazily so the admin panel works
    even if `make run` hasn't been executed since the schema was added.

    Idempotent — safe to call on every admin request, but we call it
    once at startup and silently skip on failure.
    """
    if not HAS_DB:
        return
    try:
        from sources.sql import create_pipeline_source_runs_table

        create_pipeline_source_runs_table(DATABASE_URL)
    except Exception as exc:
        print(f"  Note: could not bootstrap pipeline_source_runs: {exc}")


_ensure_source_runs_table()


import datetime  # noqa: E402


def _json_default(obj):
    if isinstance(obj, datetime.datetime | datetime.date):
        return obj.isoformat()
    if isinstance(obj, datetime.timedelta):
        return obj.total_seconds()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def _admin_query(endpoint: str, params: dict):
    if not HAS_DB:
        raise RuntimeError("psycopg not installed — run: uv run python3 web/serve.py")

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            if endpoint == "overview":
                # Latest index-health verdict per user (only when the
                # `index_health_checks` table exists — it's lazily
                # bootstrapped, and a brand-new install won't have any
                # rows yet).
                cur.execute(
                    """
                    SELECT to_regclass('public.index_health_checks') IS NOT NULL
                       AS exists
                    """
                )
                _row = cur.fetchone() or {}
                idx_table_exists = bool(_row.get("exists"))
                idx_stats = {
                    "checked": 0,
                    "healthy": 0,
                    "unhealthy": 0,
                    "stale_max_age_hours": None,
                }
                if idx_table_exists:
                    cur.execute(
                        """
                        WITH latest AS (
                            SELECT DISTINCT ON (user_id)
                                user_id, status, checked_at
                            FROM index_health_checks
                            ORDER BY user_id, checked_at DESC
                        )
                        SELECT
                            COUNT(*)                                  AS checked,
                            COUNT(*) FILTER (WHERE status = 'healthy') AS healthy,
                            COUNT(*) FILTER (WHERE status <> 'healthy') AS unhealthy,
                            EXTRACT(EPOCH FROM (NOW() - MIN(checked_at))) / 3600.0
                                                                       AS oldest_age_hours
                        FROM latest
                        """
                    )
                    row = cur.fetchone()
                    if row:
                        idx_stats = {
                            "checked": int(row["checked"] or 0),
                            "healthy": int(row["healthy"] or 0),
                            "unhealthy": int(row["unhealthy"] or 0),
                            "stale_max_age_hours": (
                                float(row["oldest_age_hours"]) if row["oldest_age_hours"] is not None else None
                            ),
                        }

                cur.execute(
                    """
                    SELECT
                        (SELECT COUNT(*) FROM users) AS total_users,
                        (SELECT COUNT(*) FROM users WHERE vip) AS vip_users,
                        COUNT(r.id) AS total_runs_7d,
                        COUNT(r.id) FILTER (WHERE r.status = 'success') AS success_runs_7d,
                        COUNT(r.id) FILTER (WHERE r.status = 'failed') AS failed_runs_7d,
                        (SELECT COUNT(*) FROM pipeline_runs WHERE status = 'running') AS running_now,
                        COALESCE(SUM(r.duration_secs) / 3600.0, 0) AS total_duration_hours,
                        COALESCE(SUM(r.new_documents), 0) AS total_new_docs_7d
                    FROM pipeline_runs r
                    WHERE r.started_at > NOW() - INTERVAL '7 days'
                    """
                )
                base = cur.fetchone() or {}
                base["index_health"] = idx_stats
                return base

            elif endpoint == "users":
                q = params.get("q", [""])[0]
                cur.execute(
                    """
                    WITH latest AS (
                        SELECT DISTINCT ON (user_id)
                            user_id, id AS run_id, status, trigger, started_at,
                            finished_at, duration_secs, new_documents, total_documents,
                            error, stage
                        FROM pipeline_runs
                        ORDER BY user_id, started_at DESC
                    )
                    SELECT
                        u.id, u.username, u.name, u.vip, u.avatar, u.category,
                        l.run_id, l.status, l.trigger, l.started_at, l.finished_at,
                        l.duration_secs, l.new_documents, l.total_documents,
                        l.error, l.stage
                    FROM users u
                    LEFT JOIN latest l ON l.user_id = u.id
                    WHERE (%s = '' OR u.username ILIKE '%%' || %s || '%%'
                                   OR u.name    ILIKE '%%' || %s || '%%')
                    ORDER BY u.vip DESC, l.started_at DESC NULLS LAST, u.name
                    LIMIT 200
                    """,
                    (q, q, q),
                )
                rows = cur.fetchall()
                result = []
                for row in rows:
                    user = {
                        "id": row["id"],
                        "username": row["username"],
                        "name": row["name"],
                        "vip": row["vip"],
                        "avatar": row["avatar"],
                        "category": row["category"],
                    }
                    if row["run_id"] is not None:
                        user["last_run"] = {
                            "id": row["run_id"],
                            "status": row["status"],
                            "trigger": row["trigger"],
                            "started_at": row["started_at"],
                            "finished_at": row["finished_at"],
                            "duration_secs": row["duration_secs"],
                            "new_documents": row["new_documents"],
                            "total_documents": row["total_documents"],
                            "error": row["error"],
                            "stage": row["stage"],
                        }
                    else:
                        user["last_run"] = None
                    result.append(user)
                return result

            else:
                m = re.match(r"^users/([^/]+)/runs$", endpoint)
                if m:
                    username = m.group(1)
                    cur.execute(
                        """
                        SELECT
                            r.id, r.trigger, r.status, r.stage,
                            r.started_at, r.finished_at, r.duration_secs,
                            r.new_documents, r.total_documents, r.error, r.timings
                        FROM pipeline_runs r
                        JOIN users u ON u.id = r.user_id
                        WHERE u.username = %s
                        ORDER BY r.started_at DESC
                        LIMIT 50
                        """,
                        (username,),
                    )
                    return cur.fetchall()
                m2 = re.match(r"^sources/health$", endpoint)
                if m2:
                    days = int(params.get("days", ["7"])[0])
                    cur.execute(
                        """
                        SELECT
                            source,
                            COUNT(*)                                            AS total_runs,
                            COUNT(*) FILTER (WHERE status = 'success')          AS success_runs,
                            COUNT(*) FILTER (WHERE status = 'failed')           AS failed_runs,
                            COUNT(*) FILTER (WHERE status = 'skipped')          AS skipped_runs,
                            COUNT(DISTINCT user_id)                             AS users_touched,
                            COUNT(DISTINCT user_id) FILTER (WHERE status = 'failed') AS users_failing,
                            COALESCE(SUM(new_documents), 0)                     AS total_new_docs,
                            COALESCE(AVG(duration_secs) FILTER (WHERE status = 'success'), 0) AS avg_duration_ok,
                            MAX(started_at) FILTER (WHERE status = 'failed')    AS last_failure_at,
                            MAX(started_at) FILTER (WHERE status = 'success')   AS last_success_at
                        FROM pipeline_source_runs
                        WHERE started_at > NOW() - make_interval(days => %s)
                        GROUP BY source
                        ORDER BY (COUNT(*) FILTER (WHERE status = 'failed')) DESC,
                                 total_runs DESC
                        """,
                        (days,),
                    )
                    rows = cur.fetchall()
                    return rows

                m2 = re.match(r"^sources/([^/]+)/failures$", endpoint)
                if m2:
                    src = m2.group(1)
                    days = int(params.get("days", ["7"])[0])
                    cur.execute(
                        """
                        SELECT
                            psr.id, psr.detail, psr.error, psr.started_at,
                            psr.duration_secs,
                            u.username, u.name, u.avatar, u.vip
                        FROM pipeline_source_runs psr
                        JOIN users u ON u.id = psr.user_id
                        WHERE psr.source = %s
                          AND psr.status = 'failed'
                          AND psr.started_at > NOW() - make_interval(days => %s)
                        ORDER BY psr.started_at DESC
                        LIMIT 200
                        """,
                        (src, days),
                    )
                    rows = cur.fetchall()
                    # Group by error message for readability.
                    from collections import defaultdict as _dd

                    by_err: dict = _dd(list)
                    for row in rows:
                        by_err[(row["error"] or "").strip()].append(row)
                    groups = []
                    for err_msg, err_rows in sorted(by_err.items(), key=lambda x: -len(x[1])):
                        seen: set = set()
                        users = []
                        for r in err_rows:
                            if r["username"] not in seen:
                                seen.add(r["username"])
                                users.append(
                                    {
                                        "username": r["username"],
                                        "name": r["name"],
                                        "avatar": r["avatar"],
                                        "vip": r["vip"],
                                    }
                                )
                        groups.append(
                            {
                                "message": err_msg,
                                "count": len(err_rows),
                                "users": users,
                                "sample_runs": [
                                    {
                                        "username": r["username"],
                                        "name": r["name"],
                                        "detail": r["detail"],
                                        "started_at": r["started_at"],
                                        "duration_secs": r["duration_secs"],
                                    }
                                    for r in err_rows[:5]
                                ],
                            }
                        )
                    return {"source": src, "total_failures": len(rows), "error_groups": groups}

                m2 = re.match(r"^users/([^/]+)/source-runs$", endpoint)
                if m2:
                    username = m2.group(1)
                    cur.execute(
                        """
                        WITH latest AS (
                            SELECT DISTINCT ON (psr.source, psr.detail)
                                psr.source, psr.detail, psr.status, psr.error,
                                psr.started_at, psr.finished_at, psr.duration_secs,
                                psr.new_documents
                            FROM pipeline_source_runs psr
                            JOIN users u ON u.id = psr.user_id
                            WHERE u.username = %s
                              AND psr.started_at > NOW() - INTERVAL '30 days'
                            ORDER BY psr.source, psr.detail, psr.started_at DESC
                        )
                        SELECT * FROM latest
                        ORDER BY
                            CASE status WHEN 'failed' THEN 0 WHEN 'success' THEN 1 ELSE 2 END,
                            source, detail
                        """,
                        (username,),
                    )
                    return cur.fetchall()

                m2 = re.match(r"^failures$", endpoint)
                if m2:
                    limit = int(params.get("limit", ["50"])[0])
                    cur.execute(
                        """
                        SELECT
                            r.id, r.trigger, r.started_at, r.finished_at,
                            r.duration_secs, r.error, r.stage,
                            u.username, u.name, u.avatar, u.vip, u.category
                        FROM pipeline_runs r
                        JOIN users u ON u.id = r.user_id
                        WHERE r.status = 'failed'
                          AND r.started_at > NOW() - INTERVAL '7 days'
                        ORDER BY r.started_at DESC
                        LIMIT %s
                        """,
                        (limit,),
                    )
                    return cur.fetchall()
                raise ValueError(f"Unknown admin API endpoint: {endpoint!r}")


class SPAHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIR, **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _serve_html(self, html_path: str) -> None:
        try:
            with open(html_path, "rb") as f:
                body = f.read()
        except OSError:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data, status: int = 200) -> None:
        body = json.dumps(data, default=_json_default).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _is_local(self) -> bool:
        return self.client_address[0] in ("127.0.0.1", "::1", "")

    def do_GET(self):
        clean = self.path.split("?", 1)[0]

        # ── Admin routes (localhost only) ──────────────────────────
        # Only intercept the admin landing page and `/admin/api/*`.
        # Static admin assets (`/admin/index.js`, `/admin/style.css`)
        # must fall through to SimpleHTTPRequestHandler so they're
        # served as files, not the admin HTML.
        is_admin_page = clean == "/admin" or clean == "/admin/"
        is_admin_api = ADMIN_API_RE.match(clean) is not None
        if is_admin_page or is_admin_api:
            if not self._is_local():
                self.send_error(403, "Admin is local-only")
                return

            m = ADMIN_API_RE.match(clean)
            if m:
                endpoint = m.group(1)
                params = parse_qs(urlparse(self.path).query)
                try:
                    self._json(_admin_query(endpoint, params))
                except Exception as exc:
                    self._json({"error": str(exc)}, 500)
                return

            self._serve_html(ADMIN_PAGE)
            return

        # ── Regular SPA routes ─────────────────────────────────────
        if PROFILE_RE.match(clean):
            self._serve_html(PROFILE_PAGE)
            return
        if SEARCH_RE.match(clean):
            self._serve_html(SEARCH_PAGE)
            return
        if SLUG_RE.match(clean) and not os.path.exists(self.translate_path(clean)):
            self._serve_html(SEARCH_PAGE)
            return
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            if os.path.exists(os.path.join(path, "index.html")):
                return super().do_GET()
            self.path = "/index.html"
        elif not os.path.exists(path):
            self.path = "/index.html"
        return super().do_GET()


if __name__ == "__main__":
    if HAS_DB:
        print(f"  Admin:  http://localhost:{PORT}/admin")
    else:
        print("  Note: psycopg not found — run via `make web` for admin API")
    with http.server.HTTPServer(("", PORT), SPAHandler) as srv:
        print(f"  Web:    http://localhost:{PORT} (SPA mode)")
        srv.serve_forever()
