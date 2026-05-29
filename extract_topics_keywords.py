"""
Extract topics and top-3 keywords (with scores) for papers in the assignment CSVs.

- **Bare DOI / doi.org links**: OpenAlex Works API (topics + keywords with scores).
- **Semantic Scholar URLs** (40-char hex id): OpenAlex first (DOI from S2, or title
  search), then Semantic Scholar fallback (s2FieldsOfStudy + text keywords).

Set SEMANTIC_SCHOLAR_API_KEY in the environment for higher S2 rate limits.

Results are written directly to a CSV file (no disk cache).

Usage:
  python extract_topics_keywords.py --doi 10.1007/978-3-319-41540-6_20
  python extract_topics_keywords.py --inputs CSV/public_test.csv --output CSV/public_test_topics_keywords.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from extract_abstracts import (
    OPENALEX_BASE,
    S2_BASE,
    http_get_json,
    openalex_headers,
    parse_doi_cell,
    s2_headers,
    title_jaccard,
)

OPENALEX_WORKS = f"{OPENALEX_BASE}/works"
S2_BATCH_URL = f"{S2_BASE}/batch"
S2_FIELDS = "title,abstract,paperId,externalIds,s2FieldsOfStudy,fieldsOfStudy,tldr"
SELECT_FIELDS = "id,doi,title,display_name,primary_topic,topics,keywords"
BATCH_SIZE = 40
S2_BATCH_SIZE = 100
TOP_K_KEYWORDS = 3
TOP_K_TOPICS = 3

_STOPWORDS = frozenset(
    "a an the and or of for to in on at by with from as is are was were be been being "
    "into over per via de la le et al this that these those it its we our their can may "
    "not".split()
)

OUTPUT_COLUMNS = [
    "id",
    "title",
    "venue",
    "year",
    "authors",
    "doi",
    "Label",
    "resolved_doi",
    "openalex_id",
    "semantic_scholar_id",
    "primary_topic",
    "primary_topic_score",
    "topic_1",
    "topic_1_score",
    "topic_2",
    "topic_2_score",
    "topic_3",
    "topic_3_score",
    "keyword_1",
    "keyword_1_score",
    "keyword_2",
    "keyword_2_score",
    "keyword_3",
    "keyword_3_score",
    "topics_json",
    "keywords_json",
    "fetch_source",
]


class _RateLimit:
    delay_s = 0.11
    s2_delay_s = 0.35


def _sleep_openalex() -> None:
    time.sleep(_RateLimit.delay_s)


def _sleep_s2() -> None:
    time.sleep(_RateLimit.s2_delay_s)


def _empty_meta() -> dict[str, Any]:
    return {
        "openalex_id": None,
        "semantic_scholar_id": None,
        "primary_topic": None,
        "primary_topic_score": None,
        "topic_1": None,
        "topic_1_score": None,
        "topic_2": None,
        "topic_2_score": None,
        "topic_3": None,
        "topic_3_score": None,
        "keyword_1": None,
        "keyword_1_score": None,
        "keyword_2": None,
        "keyword_2_score": None,
        "keyword_3": None,
        "keyword_3_score": None,
        "topics_json": None,
        "keywords_json": None,
        "fetch_source": None,
        "resolved_doi": None,
    }


def openalex_work_url_for_doi(doi: str) -> str:
    enc = urllib.parse.quote(f"https://doi.org/{doi}", safe="")
    return f"{OPENALEX_WORKS}/{enc}?select={SELECT_FIELDS}"


def parse_openalex_topics_keywords(work: dict[str, Any] | None) -> dict[str, Any]:
    """Flatten OpenAlex work JSON into topic/keyword columns."""
    out = _empty_meta()
    if not work or not work.get("id"):
        return out

    out["openalex_id"] = work.get("id")
    topics = list(work.get("topics") or [])
    primary = work.get("primary_topic")
    if primary and not any(t.get("id") == primary.get("id") for t in topics):
        topics.insert(0, primary)
    elif primary and not topics:
        topics = [primary]

    topics = sorted(topics, key=lambda t: float(t.get("score") or 0), reverse=True)
    if primary:
        out["primary_topic"] = primary.get("display_name")
        out["primary_topic_score"] = primary.get("score")

    for i, t in enumerate(topics[:TOP_K_TOPICS], start=1):
        out[f"topic_{i}"] = t.get("display_name")
        out[f"topic_{i}_score"] = t.get("score")
    out["topics_json"] = json.dumps(
        [
            {"id": t.get("id"), "display_name": t.get("display_name"), "score": t.get("score")}
            for t in topics
        ],
        ensure_ascii=False,
    )

    keywords = sorted(
        work.get("keywords") or [],
        key=lambda k: float(k.get("score") or 0),
        reverse=True,
    )
    for i, kw in enumerate(keywords[:TOP_K_KEYWORDS], start=1):
        out[f"keyword_{i}"] = kw.get("display_name")
        out[f"keyword_{i}_score"] = kw.get("score")
    out["keywords_json"] = json.dumps(
        [
            {"id": kw.get("id"), "display_name": kw.get("display_name"), "score": kw.get("score")}
            for kw in keywords[:TOP_K_KEYWORDS]
        ],
        ensure_ascii=False,
    )
    return out


def _s2_topic_categories(paper: dict[str, Any]) -> list[str]:
    """Unique field-of-study categories from S2, preferring the internal classifier."""
    categories: list[str] = []
    seen: set[str] = set()
    for source in ("s2-fos-model", "external"):
        for item in paper.get("s2FieldsOfStudy") or []:
            if not isinstance(item, dict):
                continue
            if item.get("source") != source:
                continue
            cat = (item.get("category") or "").strip()
            if cat and cat not in seen:
                seen.add(cat)
                categories.append(cat)
    for fos in paper.get("fieldsOfStudy") or []:
        cat = (fos or "").strip() if isinstance(fos, str) else ""
        if cat and cat not in seen:
            seen.add(cat)
            categories.append(cat)
    return categories


def _keywords_from_s2_text(text: str, top_k: int = TOP_K_KEYWORDS) -> list[tuple[str, float]]:
    """
    Top terms from S2 title/abstract (S2 has no keyword list with scores).
    Returns (term, normalized_score) pairs.
    """
    tokens = re.findall(r"[a-z][a-z0-9\-]{2,}", (text or "").lower())
    counts: dict[str, int] = {}
    for t in tokens:
        if t in _STOPWORDS or len(t) < 3:
            continue
        counts[t] = counts.get(t, 0) + 1
    if not counts:
        return []
    ranked = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:top_k]
    max_c = ranked[0][1] if ranked else 1
    return [(w, round(c / max_c, 6)) for w, c in ranked]


def parse_s2_topics_keywords(paper: dict[str, Any] | None) -> dict[str, Any]:
    """Map Semantic Scholar paper fields into the same CSV columns as OpenAlex."""
    out = _empty_meta()
    if not paper or not paper.get("paperId"):
        return out

    out["semantic_scholar_id"] = paper.get("paperId")
    ext = paper.get("externalIds") or {}
    doi = ext.get("DOI") or ext.get("doi")
    if doi:
        out["resolved_doi"] = doi.replace("https://doi.org/", "").strip()

    categories = _s2_topic_categories(paper)
    topic_scores = [1.0, 0.85, 0.70]
    topic_records: list[dict[str, Any]] = []
    for i, cat in enumerate(categories[:TOP_K_TOPICS]):
        score = topic_scores[i] if i < len(topic_scores) else 0.5
        out[f"topic_{i + 1}"] = cat
        out[f"topic_{i + 1}_score"] = score
        topic_records.append({"display_name": cat, "score": score, "source": "s2FieldsOfStudy"})
    if categories:
        out["primary_topic"] = categories[0]
        out["primary_topic_score"] = 1.0
    out["topics_json"] = json.dumps(topic_records, ensure_ascii=False)

    text = f"{paper.get('title') or ''} {paper.get('abstract') or ''}"
    tldr = paper.get("tldr")
    if isinstance(tldr, dict) and tldr.get("text"):
        text += " " + str(tldr["text"])
    kw_pairs = _keywords_from_s2_text(text)
    kw_records: list[dict[str, Any]] = []
    for i, (kw, score) in enumerate(kw_pairs, start=1):
        out[f"keyword_{i}"] = kw
        out[f"keyword_{i}_score"] = score
        kw_records.append({"display_name": kw, "score": score, "source": "s2_text"})
    out["keywords_json"] = json.dumps(kw_records, ensure_ascii=False)
    return out


def fetch_openalex_by_doi(doi: str) -> tuple[dict[str, Any] | None, str | None]:
    url = openalex_work_url_for_doi(doi)
    data, code = http_get_json(url, openalex_headers())
    if data and data.get("id"):
        return data, "openalex_doi"
    if code == 404:
        return None, None
    return data, None


def fetch_openalex_by_title(title: str, min_jaccard: float = 0.35) -> tuple[dict[str, Any] | None, str | None]:
    t = (title or "").strip()
    if len(t) < 8:
        return None, None
    q = urllib.parse.urlencode({"search": t, "per-page": "3", "select": SELECT_FIELDS})
    url = f"{OPENALEX_WORKS}?{q}"
    data, _ = http_get_json(url, openalex_headers())
    if not data:
        return None, None
    for hit in data.get("results") or []:
        hit_title = (hit.get("title") or hit.get("display_name") or "").strip()
        if title_jaccard(t, hit_title) >= min_jaccard:
            return hit, "openalex_title_search"
    return None, None


def _doi_from_s2_paper(paper: dict[str, Any] | None) -> str | None:
    if not paper:
        return None
    ext = paper.get("externalIds") or {}
    doi = ext.get("DOI") or ext.get("doi")
    if not doi:
        return None
    return doi.replace("https://doi.org/", "").strip()


def _openalex_work_usable(work: dict[str, Any] | None) -> bool:
    if not work or not work.get("id"):
        return False
    meta = parse_openalex_topics_keywords(work)
    return bool(meta.get("topic_1") or meta.get("keyword_1"))


def fetch_openalex_for_s2_row(
    title: str,
    s2_paper: dict[str, Any] | None,
    doi_works: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any] | None, str | None]:
    """Try OpenAlex for a Semantic Scholar paper (DOI from S2 metadata, then title)."""
    work: dict[str, Any] | None = None
    source: str | None = None

    resolved_doi = _doi_from_s2_paper(s2_paper)
    if resolved_doi:
        work = doi_works.get(resolved_doi)
        if work and work.get("id"):
            source = "openalex_batch"
        else:
            work, source = fetch_openalex_by_doi(resolved_doi)
            _sleep_openalex()
            if source:
                source = "openalex_s2_doi"

    if not _openalex_work_usable(work) and title:
        work, source = fetch_openalex_by_title(title)
        _sleep_openalex()

    if _openalex_work_usable(work):
        return work, source
    return None, None


def _http_post_json(url: str, body: dict[str, Any], headers: dict[str, str]) -> dict[str, Any] | None:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception:
        return None


def fetch_s2_paper(s2_hex_id: str, retries: int = 2) -> dict[str, Any] | None:
    q = urllib.parse.urlencode({"fields": S2_FIELDS})
    url = f"{S2_BASE}/{s2_hex_id}?{q}"
    for attempt in range(retries + 1):
        data, code = http_get_json(url, s2_headers())
        if data and data.get("paperId"):
            return data
        if code == 429 and attempt < retries:
            time.sleep(12.0 * (attempt + 1))
            continue
        break
    return None


def batch_fetch_s2_papers(s2_ids: list[str]) -> dict[str, dict[str, Any]]:
    """POST /paper/batch — returns hex_id -> paper dict."""
    result: dict[str, dict[str, Any]] = {}
    headers = s2_headers()
    for i in range(0, len(s2_ids), S2_BATCH_SIZE):
        batch = s2_ids[i : i + S2_BATCH_SIZE]
        q = urllib.parse.urlencode({"fields": S2_FIELDS})
        url = f"{S2_BATCH_URL}?{q}"
        payload = {"ids": batch}
        data = _http_post_json(url, payload, headers)
        if isinstance(data, list):
            for hid, paper in zip(batch, data):
                if isinstance(paper, dict) and paper.get("paperId"):
                    result[hid] = paper
        _sleep_s2()
    return result


def fetch_openalex_for_row(doi_cell: str) -> dict[str, Any]:
    """OpenAlex lookup for DOI-like cells only."""
    kind, value = parse_doi_cell(doi_cell)
    work: dict[str, Any] | None = None
    source: str | None = None

    if kind == "doi" and value:
        work, source = fetch_openalex_by_doi(value)
        _sleep_openalex()

    meta = parse_openalex_topics_keywords(work)
    meta["fetch_source"] = source
    if work and work.get("doi"):
        meta["resolved_doi"] = work["doi"].replace("https://doi.org/", "")
    elif kind == "doi" and value:
        meta["resolved_doi"] = value
    return meta


def batch_fetch_openalex_by_dois(dois: list[str]) -> dict[str, dict[str, Any]]:
    works: dict[str, dict[str, Any]] = {}
    headers = openalex_headers()
    for i in range(0, len(dois), BATCH_SIZE):
        batch = dois[i : i + BATCH_SIZE]
        filt = "doi:" + "|".join(urllib.parse.quote(d, safe="") for d in batch)
        q = urllib.parse.urlencode({"filter": filt, "per-page": str(BATCH_SIZE), "select": SELECT_FIELDS})
        url = f"{OPENALEX_WORKS}?{q}"
        data, _ = http_get_json(url, headers)
        if data:
            for work in data.get("results") or []:
                doi_url = work.get("doi") or ""
                if not doi_url:
                    continue
                doi = doi_url.replace("https://doi.org/", "").strip()
                works[doi] = work
        _sleep_openalex()
    return works


def collect_ids_from_inputs(paths: list[Path]) -> tuple[list[str], list[str]]:
    dois: set[str] = set()
    s2_ids: set[str] = set()
    for p in paths:
        with p.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                kind, value = parse_doi_cell(row.get("doi", ""))
                if kind == "doi" and value:
                    dois.add(value)
                elif kind == "s2_corpus_id" and value:
                    s2_ids.add(value)
    return sorted(dois), sorted(s2_ids)


def enrich_row(
    row: dict[str, str],
    doi_works: dict[str, dict[str, Any]],
    s2_papers: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    kind, value = parse_doi_cell(row.get("doi", ""))
    title = row.get("title", "") or ""

    # Semantic Scholar URL → OpenAlex first, then S2 fallback
    if kind == "s2_corpus_id" and value:
        paper = s2_papers.get(value)
        if not paper:
            paper = fetch_s2_paper(value)
            _sleep_s2()

        work, oa_source = fetch_openalex_for_s2_row(title, paper, doi_works)
        if work:
            meta = parse_openalex_topics_keywords(work)
            meta["fetch_source"] = oa_source
            if work.get("doi"):
                meta["resolved_doi"] = work["doi"].replace("https://doi.org/", "")
            elif paper:
                meta["resolved_doi"] = _doi_from_s2_paper(paper)
            if paper and paper.get("paperId"):
                meta["semantic_scholar_id"] = paper.get("paperId")
            return {**row, **meta}

        meta = parse_s2_topics_keywords(paper)
        meta["fetch_source"] = (
            "semantic_scholar" if meta.get("topic_1") or meta.get("keyword_1") else None
        )
        return {**row, **meta}

    # DOI → OpenAlex (batch or single)
    doi = value if kind == "doi" else None
    work = doi_works.get(doi) if doi else None
    if work and work.get("id"):
        meta = parse_openalex_topics_keywords(work)
        meta["fetch_source"] = "openalex_batch"
        meta["resolved_doi"] = doi
    else:
        meta = fetch_openalex_for_row(row.get("doi", ""))

    return {**row, **meta}


def output_fieldnames(sample_row: dict[str, Any]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for col in OUTPUT_COLUMNS:
        if col in sample_row and col not in seen:
            names.append(col)
            seen.add(col)
    for col in sample_row:
        if col not in seen:
            names.append(col)
            seen.add(col)
    return names


def extract_to_csv(
    inputs: list[Path],
    output_path: Path,
    doi_works: dict[str, dict[str, Any]],
    s2_papers: dict[str, dict[str, Any]],
) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    ok = 0
    writer: csv.DictWriter | None = None

    with output_path.open("w", newline="", encoding="utf-8") as fout:
        for input_path in inputs:
            with input_path.open(newline="", encoding="utf-8") as fin:
                rows = list(csv.DictReader(fin))
            print(f"Processing {input_path.name} ({len(rows)} rows)...")
            for row in rows:
                out = enrich_row(row, doi_works, s2_papers)
                if writer is None:
                    writer = csv.DictWriter(
                        fout, fieldnames=output_fieldnames(out), extrasaction="ignore"
                    )
                    writer.writeheader()
                writer.writerow(out)
                fout.flush()
                total += 1
                if out.get("topic_1") or out.get("keyword_1"):
                    ok += 1
                if total % 50 == 0:
                    print(f"  … {total} rows written")

    return total, ok


def main() -> None:
    root = Path(__file__).resolve().parent
    default_inputs = [
        root / "CSV" / "train.csv",
        root / "CSV" / "public_test.csv",
        root / "CSV" / "private_test.csv",
    ]

    ap = argparse.ArgumentParser(description="Extract topics and top-3 keywords to CSV.")
    ap.add_argument("--doi", help="Fetch a single DOI via OpenAlex and print JSON")
    ap.add_argument("--s2-id", help="Fetch a single Semantic Scholar hex id and print JSON")
    ap.add_argument("--inputs", nargs="*", type=Path, default=default_inputs)
    ap.add_argument(
        "--output",
        type=Path,
        default=root / "CSV" / "papers_topics_keywords.csv",
    )
    ap.add_argument("--delay", type=float, default=_RateLimit.delay_s, help="OpenAlex delay (seconds)")
    ap.add_argument("--s2-delay", type=float, default=_RateLimit.s2_delay_s, help="S2 delay (seconds)")
    args = ap.parse_args()
    _RateLimit.delay_s = args.delay
    _RateLimit.s2_delay_s = args.s2_delay

    if args.doi:
        meta = fetch_openalex_for_row(args.doi)
        print(json.dumps(meta, indent=2, ensure_ascii=False))
        return

    if args.s2_id:
        s2_hex = args.s2_id.strip()
        paper = fetch_s2_paper(s2_hex)
        work, oa_source = fetch_openalex_for_s2_row("", paper, {})
        if work:
            meta = parse_openalex_topics_keywords(work)
            meta["fetch_source"] = oa_source
        else:
            meta = parse_s2_topics_keywords(paper)
            meta["fetch_source"] = "semantic_scholar"
        if paper and paper.get("paperId"):
            meta["semantic_scholar_id"] = paper.get("paperId")
        print(json.dumps(meta, indent=2, ensure_ascii=False))
        return

    inputs = [p.resolve() for p in args.inputs if p.resolve().is_file()]
    if not inputs:
        print("No input CSV files found.", file=sys.stderr)
        sys.exit(1)

    all_dois, all_s2 = collect_ids_from_inputs(inputs)
    print(f"OpenAlex: batch-fetching {len(all_dois)} DOIs...")
    doi_works = batch_fetch_openalex_by_dois(all_dois)
    print(f"Semantic Scholar: batch-fetching {len(all_s2)} paper ids...")
    s2_papers = batch_fetch_s2_papers(all_s2) if all_s2 else {}

    extra_dois: set[str] = set()
    for paper in s2_papers.values():
        d = _doi_from_s2_paper(paper)
        if d and d not in doi_works:
            extra_dois.add(d)
    if extra_dois:
        print(f"OpenAlex: batch-fetching {len(extra_dois)} DOIs resolved from S2 metadata...")
        doi_works.update(batch_fetch_openalex_by_dois(sorted(extra_dois)))

    total, ok = extract_to_csv(inputs, args.output.resolve(), doi_works, s2_papers)
    print(f"Wrote {total} rows to {args.output} ({ok} with topic or keyword data)")


if __name__ == "__main__":
    main()
