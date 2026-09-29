"""Mainspace page-to-page link graph from the MediaWiki SQL dumps.

Four tables are needed:
- `page`: ids, titles and redirect flags;
- `redirect`: where each redirect page points;
- `linktarget`: the namespace and title each link points to;
- `pagelinks`: source page id -> linktarget id.

Only links between mainspace pages are kept, as (source page id, target
page id). A link to a redirect is resolved to the redirect's target,
following up to MAX_REDIRECT_HOPS hops. Links from redirect pages (a
redirect's only link is its own target), self-links and red links (targets
that don't exist) are dropped, and duplicates are removed.

Titles are matched as raw dump bytes (see `sql_dump`). The edge list is
built in memory: fine for Simple Wikipedia (millions of links), but
English Wikipedia (1B+ links) would need restricting to the pages being
scored, or an out-of-core build.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.ingest.sql_dump import iter_rows, sql_str

MAINSPACE = b"0"
MAX_REDIRECT_HOPS = 2
_CHUNK = 1_000_000


def load_pages(path: Path) -> tuple[dict[bytes, int], set[int]]:
    """Mainspace {raw title: page id}, and the ids that are redirects."""
    title_to_id: dict[bytes, int] = {}
    redirects: set[int] = set()
    for page_id, namespace, title, is_redirect in iter_rows(
        path, ("page_id", "page_namespace", "page_title", "page_is_redirect")
    ):
        if namespace == MAINSPACE:
            title_to_id[sql_str(title)] = int(page_id)
            if is_redirect == b"1":
                redirects.add(int(page_id))
    return title_to_id, redirects


def load_redirect_targets(path: Path, title_to_id: dict[bytes, int], redirects: set[int]) -> dict[int, int]:
    """{mainspace redirect page id: final non-redirect mainspace target id}."""
    direct: dict[int, int] = {}
    for rd_from, namespace, title in iter_rows(path, ("rd_from", "rd_namespace", "rd_title")):
        source = int(rd_from)
        if source in redirects and namespace == MAINSPACE:
            target = title_to_id.get(sql_str(title))
            if target is not None:
                direct[source] = target
    resolved: dict[int, int] = {}
    for source, target in direct.items():
        for _ in range(MAX_REDIRECT_HOPS - 1):
            if target not in redirects:
                break
            target = direct.get(target, target)
        if target not in redirects and target != source:
            resolved[source] = target
    return resolved


def load_link_targets(
    path: Path, title_to_id: dict[bytes, int], redirects: set[int], redirect_targets: dict[int, int]
) -> np.ndarray:
    """Dense array: linktarget id -> final mainspace page id, or -1."""
    ids: list[int] = []
    pages: list[int] = []
    for lt_id, namespace, title in iter_rows(path, ("lt_id", "lt_namespace", "lt_title")):
        if namespace != MAINSPACE:
            continue
        page = title_to_id.get(sql_str(title))
        if page is not None and page in redirects:
            page = redirect_targets.get(page)
        if page is not None:
            ids.append(int(lt_id))
            pages.append(page)
    target_of = np.full(max(ids, default=0) + 1, -1, dtype=np.int64)
    target_of[ids] = pages
    return target_of


def _edge_keys(sources: list[int], target_ids: list[int], target_of: np.ndarray, redirects: np.ndarray) -> np.ndarray:
    source = np.array(sources, dtype=np.int64)
    target_id = np.array(target_ids, dtype=np.int64)
    target = np.full(len(source), -1, dtype=np.int64)
    known = target_id < len(target_of)
    target[known] = target_of[target_id[known]]
    keep = (target >= 0) & (target != source) & ~np.isin(source, redirects)
    return (source[keep] << 32) | target[keep]


def build_link_graph(
    page_path: Path, redirect_path: Path, linktarget_path: Path, pagelinks_path: Path
) -> tuple[np.ndarray, np.ndarray]:
    """(source, target) page-id arrays of unique mainspace links, sorted by source."""
    title_to_id, redirects = load_pages(page_path)
    redirect_targets = load_redirect_targets(redirect_path, title_to_id, redirects)
    target_of = load_link_targets(linktarget_path, title_to_id, redirects, redirect_targets)
    del title_to_id
    redirect_ids = np.array(sorted(redirects), dtype=np.int64)

    chunks: list[np.ndarray] = []
    sources: list[int] = []
    target_ids: list[int] = []
    for pl_from, from_namespace, target_id in iter_rows(
        pagelinks_path, ("pl_from", "pl_from_namespace", "pl_target_id")
    ):
        if from_namespace == MAINSPACE:
            sources.append(int(pl_from))
            target_ids.append(int(target_id))
            if len(sources) >= _CHUNK:
                chunks.append(_edge_keys(sources, target_ids, target_of, redirect_ids))
                sources, target_ids = [], []
    chunks.append(_edge_keys(sources, target_ids, target_of, redirect_ids))
    keys = np.unique(np.concatenate(chunks))
    return keys >> 32, keys & 0xFFFFFFFF
