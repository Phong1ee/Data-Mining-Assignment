"""
Build a citation graph over the assignment corpus and extract graph features.

Uses OpenAlex (DOI batch lookup) for reference edges. Results are cached under
`data/` so repeated training runs do not hammer the API.
"""

from __future__ import annotations

import json
import pickle
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

OPENALEX_WORKS = "https://api.openalex.org/works"
S2_HEX_ID = re.compile(r"semanticscholar\.org/paper/([0-9a-f]{40})\b", re.I)
DEFAULT_CACHE = Path(__file__).resolve().parent / "data" / "citation_graph_cache.pkl"
BATCH_SIZE = 40
REQUEST_DELAY = 0.12


def parse_doi(cell: str) -> str | None:
    s = (cell or "").strip()
    if not s:
        return None
    low = s.lower()
    if low.startswith("10."):
        return s
    if "doi.org/" in low:
        part = s.split("doi.org/", 1)[-1].split("?", 1)[0].strip().rstrip("/")
        return urllib.parse.unquote(part) if part.startswith("10.") else None
    return None


def parse_s2_hex(cell: str) -> str | None:
    m = S2_HEX_ID.search(cell or "")
    return m.group(1).lower() if m else None


def openalex_short_id(url_or_id: str) -> str:
    s = (url_or_id or "").rstrip("/")
    return s.split("/")[-1]


def _http_get_json(url: str, headers: dict[str, str], timeout: float = 60.0) -> dict[str, Any] | None:
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError, TimeoutError):
        return None


def _openalex_headers() -> dict[str, str]:
    return {"User-Agent": "mailto:student@localhost (DataMiningAssignment citation graph)"}


def fetch_openalex_works(
    dois: list[str],
    cache_path: Path = DEFAULT_CACHE,
) -> dict[str, dict[str, Any]]:
    """Return doi -> {openalex_id, referenced_ids} using on-disk cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cached: dict[str, dict[str, Any]] = {}
    if cache_path.is_file():
        with cache_path.open("rb") as f:
            blob = pickle.load(f)
        cached = blob.get("openalex_by_doi", {})

    missing = [d for d in dois if d not in cached]
    headers = _openalex_headers()

    for i in range(0, len(missing), BATCH_SIZE):
        batch = missing[i : i + BATCH_SIZE]
        filt = "doi:" + "|".join(urllib.parse.quote(d, safe="") for d in batch)
        url = f"{OPENALEX_WORKS}?filter={filt}&per-page={BATCH_SIZE}&select=id,doi,referenced_works"
        data = _http_get_json(url, headers)
        if data:
            for work in data.get("results") or []:
                doi_url = work.get("doi") or ""
                if not doi_url:
                    continue
                doi = doi_url.replace("https://doi.org/", "").strip()
                refs = [openalex_short_id(r) for r in (work.get("referenced_works") or [])]
                cached[doi] = {
                    "openalex_id": openalex_short_id(work.get("id", "")),
                    "referenced_ids": refs,
                }
        for d in batch:
            cached.setdefault(d, {"openalex_id": None, "referenced_ids": []})
        time.sleep(REQUEST_DELAY)

        with cache_path.open("wb") as f:
            pickle.dump({"openalex_by_doi": cached}, f)

    return {d: cached[d] for d in dois if d in cached}


def build_citation_graph(
    df: pd.DataFrame,
    cache_path: Path = DEFAULT_CACHE,
) -> nx.DiGraph:
    """
    Build a directed citation graph on paper `id` nodes.
    Edge u -> v means paper u cites paper v (u lists v in references).
    """
    df = df.copy()
    df["norm_doi"] = df["doi"].map(parse_doi)
    dois = sorted({d for d in df["norm_doi"].dropna().unique()})
    oa = fetch_openalex_works(dois, cache_path=cache_path)

    doi_to_id: dict[str, int] = {}
    for _, row in df.iterrows():
        d = row["norm_doi"]
        if pd.notna(d) and d not in doi_to_id:
            doi_to_id[str(d)] = int(row["id"])

    oa_to_id: dict[str, int] = {}
    for doi, meta in oa.items():
        oid = meta.get("openalex_id")
        if oid and doi in doi_to_id:
            oa_to_id[oid] = doi_to_id[doi]

    G = nx.DiGraph()
    for _, row in df.iterrows():
        nid = int(row["id"])
        attrs: dict[str, Any] = {
            "title": row.get("title"),
            "venue": row.get("venue"),
            "year": row.get("year"),
            "norm_doi": row.get("norm_doi"),
        }
        if "Label" in row and pd.notna(row.get("Label")):
            attrs["label"] = int(row["Label"])
        G.add_node(nid, **attrs)

    for doi, meta in oa.items():
        src = doi_to_id.get(doi)
        if src is None:
            continue
        for ref_oa in meta.get("referenced_ids") or []:
            tgt = oa_to_id.get(ref_oa)
            if tgt is not None and src != tgt:
                G.add_edge(src, tgt)

    return G


def graph_structural_features(G: nx.DiGraph, node_ids: list[int]) -> np.ndarray:
    """in-degree, out-degree, PageRank (3 features per node)."""
    n = len(node_ids)
    idx = {nid: i for i, nid in enumerate(node_ids)}
    indeg = np.zeros(n, dtype=np.float32)
    outdeg = np.zeros(n, dtype=np.float32)
    for nid in node_ids:
        if nid in G:
            i = idx[nid]
            indeg[i] = G.in_degree(nid)
            outdeg[i] = G.out_degree(nid)
    try:
        pr = nx.pagerank(G, alpha=0.85, max_iter=100)
        pagerank = np.array([pr.get(nid, 0.0) for nid in node_ids], dtype=np.float32)
    except nx.PowerIterationFailedConvergence:
        pagerank = np.zeros(n, dtype=np.float32)
    max_in = max(indeg.max(), 1.0)
    max_out = max(outdeg.max(), 1.0)
    max_pr = max(pagerank.max(), 1e-9)
    return np.column_stack([indeg / max_in, outdeg / max_out, pagerank / max_pr])


def neighbor_label_features(
    G: nx.DiGraph,
    labels_by_id: dict[int, int],
    train_ids: set[int],
    node_ids: list[int],
    n_classes: int = 5,
) -> np.ndarray:
    """
    For each node, distribution of labels among train neighbors
    (papers that cite it or are cited by it).
    """
    n = len(node_ids)
    feats = np.zeros((n, n_classes), dtype=np.float32)
    idx = {nid: i for i, nid in enumerate(node_ids)}

    for nid in node_ids:
        neighbor_labels: list[int] = []
        if nid not in G:
            continue
        for pred in G.predecessors(nid):
            if pred in train_ids and pred in labels_by_id:
                neighbor_labels.append(labels_by_id[pred])
        for succ in G.successors(nid):
            if succ in train_ids and succ in labels_by_id:
                neighbor_labels.append(labels_by_id[succ])
        if not neighbor_labels:
            continue
        row = idx[nid]
        for lab in neighbor_labels:
            c = int(lab) - 1
            if 0 <= c < n_classes:
                feats[row, c] += 1.0
        feats[row] /= len(neighbor_labels)
    return feats


def build_feature_matrix(
    G: nx.DiGraph,
    labels_by_id: dict[int, int],
    train_ids: set[int],
    node_ids: list[int],
    n_classes: int = 5,
) -> np.ndarray:
    """Structural + neighbor-label graph features."""
    struct = graph_structural_features(G, node_ids)
    neigh = neighbor_label_features(G, labels_by_id, train_ids, node_ids, n_classes=n_classes)
    return np.hstack([struct, neigh]).astype(np.float32)


def graph_summary(G: nx.DiGraph) -> dict[str, Any]:
    n = G.number_of_nodes()
    e = G.number_of_edges()
    return {
        "nodes": n,
        "edges": e,
        "density": e / (n * (n - 1)) if n > 1 else 0.0,
        "weakly_connected_components": nx.number_weakly_connected_components(G) if n else 0,
    }
