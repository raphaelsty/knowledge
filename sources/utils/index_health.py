"""Audit + (re)build the single `__all__` search index.

There is ONE search index, `__all__`, maintained incrementally by the
indexer daemon (see ``sources.utils.build_all_index.sync_all_index``).
This module provides the operator-facing audit + full-rebuild helpers:

  * ``all_index_status`` — compare the index's doc count against the sum
    of every VIP's PG document count; reports whether it's behind.
  * ``rebuild_all_index`` — delegate to ``build_all_index`` for a full
    (re)sync. Reserved for the structural-break / manual path.

Exposed via ``python -m sources.utils.index_health all-status`` and
``all-rebuild`` (and ``make all-status`` / ``make index-all``). The
per-personality index machinery that used to live here was removed when
the project collapsed to the single `__all__` index.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request

import psycopg

# The single cross-personality search index.
ALL_INDEX_NAME = "__all__"

DEFAULT_DATABASE_URL = "postgresql://knowledge:knowledge@localhost:5433/knowledge"
DEFAULT_API_URL = "http://localhost:8080"


# ── classification ────────────────────────────────────────────────────


def _get_index_info(api_url: str, name: str, timeout: int = 10) -> tuple[int | None, dict | str]:
    """GET /indices/{name}. Return (status, payload) where payload is
    the decoded JSON on 200, the response body on error."""
    try:
        with urllib.request.urlopen(f"{api_url}/indices/{name}", timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        return e.code, body
    except Exception as e:
        return None, str(e)


# ── `__all__` audit + rebuild ────────────────────────────────────────


def vip_document_total(database_url: str) -> int:
    """Sum of `documents` rows owned by VIP users. The source-of-truth
    target for `__all__`'s document count."""
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM documents d JOIN users u ON u.id = d.user_id WHERE u.vip = TRUE")
            (n,) = cur.fetchone()
    return int(n)


def all_index_doc_count(api_url: str) -> int | None:
    """Return `num_documents` of `__all__`, or ``None`` if missing/error."""
    status, payload = _get_index_info(api_url.rstrip("/"), ALL_INDEX_NAME)
    if status == 200 and isinstance(payload, dict):
        return int(payload.get("num_documents") or 0)
    return None


def all_index_status(database_url: str, api_url: str) -> dict:
    """Snapshot for the startup audit and end-of-run decision.

    Returns ``{"vip_total", "all_count", "stale", "reason"}`` where
    ``stale`` is True when the index has *fewer* docs than the VIP
    total (the user's spec: "if it contains less documents that the
    number of total documents"). A missing index counts as stale.
    """
    vip_total = vip_document_total(database_url)
    all_count = all_index_doc_count(api_url)
    if all_count is None:
        return {
            "vip_total": vip_total,
            "all_count": None,
            "stale": vip_total > 0,
            "reason": "index missing or unreachable",
        }
    stale = all_count < vip_total
    return {
        "vip_total": vip_total,
        "all_count": all_count,
        "stale": stale,
        "reason": (f"all={all_count} < vip_total={vip_total}" if stale else f"all={all_count} ≥ vip_total={vip_total}"),
    }


def rebuild_all_index(database_url: str, api_url: str, admin_key: str | None = None) -> int:
    """Rebuild `__all__` via the staging+promote path.

    The cross-personality index is a *derivative*: every chunk it
    holds can be re-derived from `documents`, so a full rebuild is
    the simplest correct path. The build helper writes into a
    staging index and promotes via ArcSwap at the end, so search
    against `__all__` keeps working through the whole rebuild.

    Imported lazily so `run.py` startup doesn't pay the psycopg +
    HTTP-machinery cost until a rebuild actually fires.
    """
    # Re-export the build helper's env contract. Keep current process
    # env intact by restoring on the way out.
    prev = {k: os.environ.get(k) for k in ("DATABASE_URL", "API_URL", "ADMIN_API_KEY")}
    os.environ["DATABASE_URL"] = database_url
    os.environ["API_URL"] = api_url
    if admin_key is not None:
        os.environ["ADMIN_API_KEY"] = admin_key

    try:
        from sources.utils.build_all_index import main as build_all_main

        return int(build_all_main())
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ── CLI ──────────────────────────────────────────────────────────────


def _cli() -> int:
    ap = argparse.ArgumentParser(
        prog="python -m sources.utils.index_health",
        description="Audit / rebuild the single `__all__` search index.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("all-status", help="print the `__all__` audit snapshot")

    a = sub.add_parser("all-rebuild", help="fully (re)sync the `__all__` index")
    a.add_argument("--if-stale", action="store_true", help="no-op when not stale")

    args = ap.parse_args()
    database_url = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    api_url = os.environ.get("API_URL", DEFAULT_API_URL)
    admin_key = os.environ.get("ADMIN_API_KEY") or None

    if args.cmd == "all-status":
        snap = all_index_status(database_url, api_url)
        flag = "STALE" if snap["stale"] else "ok"
        all_count = snap["all_count"] if snap["all_count"] is not None else "missing"
        print(f"  __all__ {flag}  ({snap['reason']})  vip_total={snap['vip_total']}  all={all_count}")
        return 1 if snap["stale"] else 0

    if args.cmd == "all-rebuild":
        if args.if_stale:
            snap = all_index_status(database_url, api_url)
            if not snap["stale"]:
                print(f"  __all__ up-to-date ({snap['reason']}) — skipping rebuild")
                return 0
            print(f"  __all__ is stale ({snap['reason']}) — rebuilding")
        return rebuild_all_index(database_url, api_url, admin_key)

    return 2


if __name__ == "__main__":
    raise SystemExit(_cli())
