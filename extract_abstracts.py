"""
Fetch paper abstracts for rows in public_test.csv / private_test.csv.

The `doi` column may contain:
  - A bare DOI (e.g. 10.24963/kr.2020/66)
  - A Semantic Scholar paper URL (hex id after /paper/)
  - Other HTTP URLs (OpenAlex title search fallback using the `title` column)

Uses Semantic Scholar Graph API first, then OpenAlex (DOI URL or title search).
Polite delays help stay within default rate limits; set SEMANTIC_SCHOLAR_API_KEY for higher throughput.

Usage:
  python extract_abstracts.py
  python extract_abstracts.py --delay 3.5 --inputs CSV/public_test.csv CSV/private_test.csv
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

S2_PAPER_FIELDS = "title,abstract"
S2_BASE = "https://api.semanticscholar.org/graph/v1/paper"
OPENALEX_BASE = "https://api.openalex.org"

# 40-char hex after /paper/ in semanticscholar.org URLs
S2_HEX_ID = re.compile(r"semanticscholar\.org/paper/([0-9a-f]{40})\b", re.I)

_STOP_TITLE = frozenset(
    "a an the and or of for to in on at by with from as is are was were be been being into over per via de la le et al".split()
)


def http_get_json(
    url: str, headers: dict[str, str], timeout: float = 60.0
) -> tuple[dict[str, Any] | None, int | None]:
    """Returns (json_or_none, http_status_or_none). status is None on transport/parse errors."""
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            code = getattr(resp, "status", 200) or 200
            if not raw.strip():
                return None, code
            return json.loads(raw), code
    except urllib.error.HTTPError as e:
        return None, int(e.code)
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError):
        return None, None


def _title_tokens(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", s.lower()) if len(t) > 2 and t not in _STOP_TITLE}


def title_jaccard(a: str, b: str) -> float:
    ta, tb = _title_tokens(a), _title_tokens(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def is_plausible_abstract(text: str) -> bool:
    s = (text or "").strip()
    if len(s) < 55:
        return False
    low = s.lower()
    if low.startswith("sponsorship:") or low.startswith("funding:") or low.startswith("grant "):
        return False
    words = s.split()
    if len(words) < 10:
        return False
    letters = sum(c.isalpha() for c in s)
    if letters < 40:
        return False
    return True


def s2_headers() -> dict[str, str]:
    h = {"User-Agent": "DataMiningAssignment/1.0 (abstract extraction)"}
    import os

    key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    if key:
        h["x-api-key"] = key
    return h


def openalex_headers() -> dict[str, str]:
    # OpenAlex asks for contact; mailto in User-Agent is recommended.
    return {"User-Agent": "mailto:student@localhost (DataMiningAssignment abstract script)"}


def reconstruct_openalex_abstract(inv: dict[str, list[int]] | None) -> str | None:
    if not inv:
        return None
    slots: list[tuple[int, str]] = []
    for word, indices in inv.items():
        for pos in indices:
            slots.append((pos, word))
    if not slots:
        return None
    slots.sort(key=lambda x: x[0])
    return " ".join(w for _, w in slots)


def http_get_text(url: str, headers: dict[str, str], timeout: float = 60.0) -> tuple[str | None, int | None]:
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            code = getattr(resp, "status", 200) or 200
            return raw, code
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", errors="replace")
            return body, int(e.code)
        except Exception:
            return None, int(e.code)
    except (urllib.error.URLError, TimeoutError):
        return None, None


def extract_abstract_from_html(html_text: str) -> str | None:
    if not html_text:
        return None
    patterns = [
        r'<meta[^>]+name=["\']citation_abstract["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+name=["\']dc\.Description["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+name=["\']DC\.Description["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']+)["\']',
    ]
    for patt in patterns:
        m = re.search(patt, html_text, re.I | re.S)
        if m:
            text = html.unescape(m.group(1).strip())
            if is_plausible_abstract(text):
                return re.sub(r"\s+", " ", text)

    m = re.search(
        r'<(?:section|div)[^>]+class=["\'][^"\']*abstract[^"\']*["\'][^>]*>(.*?)</(?:section|div)>',
        html_text,
        re.I | re.S,
    )
    if m:
        text = re.sub(r'<[^>]+>', ' ', m.group(1)).strip()
        text = html.unescape(text)
        text = re.sub(r"\s+", " ", text)
        if is_plausible_abstract(text):
            return text
    return None


def parse_doi_cell(cell: str) -> tuple[str, str | None]:
    """
    Returns (kind, value) where kind is:
      doi | s2_corpus_id | other_url | empty
    """
    s = (cell or "").strip()
    if not s:
        return "empty", None
    low = s.lower()
    if low.startswith("10."):
        return "doi", s
    if "doi.org/" in low:
        part = s.split("doi.org/", 1)[-1]
        part = part.split("?", 1)[0].strip().rstrip("/")
        return ("doi", urllib.parse.unquote(part)) if part.startswith("10.") else ("other_url", s)
    m = S2_HEX_ID.search(s)
    if m:
        return "s2_corpus_id", m.group(1).lower()
    if low.startswith("http://") or low.startswith("https://"):
        return "other_url", s
    # Rare: bare corpus id?
    if re.fullmatch(r"[0-9a-f]{40}", low):
        return "s2_corpus_id", low
    if "/" not in s and "." in s and not s.startswith("http"):
        return "doi", s
    return "other_url", s


def fetch_semantic_scholar(kind: str, value: str) -> tuple[str | None, str | None, bool]:
    """
    Returns (abstract, source, s2_rate_limited).
    If S2 returns 429, waits once and retries; s2_rate_limited is True if the final attempt was 429.
    """
    if kind == "doi":
        q = urllib.parse.quote(value, safe="")
        url = f"{S2_BASE}/DOI:{q}?fields={S2_PAPER_FIELDS}"
    elif kind == "s2_corpus_id":
        url = f"{S2_BASE}/{value}?fields={S2_PAPER_FIELDS}"
    else:
        return None, None, False

    rate_limited = False
    for attempt in range(2):
        data, code = http_get_json(url, s2_headers())
        if data:
            ab = (data.get("abstract") or "").strip()
            if ab:
                return ab, "semantic_scholar", False
        if code == 429:
            rate_limited = True
            if attempt == 0:
                time.sleep(22.0)
                continue
        break
    return None, None, rate_limited


def fetch_openalex_by_doi(doi: str) -> tuple[str | None, str | None]:
    enc = urllib.parse.quote(f"https://doi.org/{doi}", safe="")
    url = f"{OPENALEX_BASE}/works/{enc}"
    data, _code = http_get_json(url, openalex_headers())
    if not data:
        return None, None
    inv = data.get("abstract_inverted_index")
    ab = reconstruct_openalex_abstract(inv)
    if ab:
        return ab.strip(), "openalex_doi"

    # Fallback: some OpenAlex DOI records omit `abstract_inverted_index` even though
    # the publisher landing page contains an abstract.
    html_text, _ = http_get_text(f"https://doi.org/{doi}", openalex_headers())
    if html_text:
        ab = extract_abstract_from_html(html_text)
        if ab:
            return ab.strip(), "doi_html_fallback"
    return None, None


def fetch_openalex_search_title(title: str, min_jaccard: float = 0.28) -> tuple[str | None, str | None]:
    t = (title or "").strip()
    if len(t) < 8:
        return None, None
    q = urllib.parse.urlencode({"search": t, "per_page": "1"})
    url = f"{OPENALEX_BASE}/works?{q}"
    data, _code = http_get_json(url, openalex_headers())
    if not data:
        return None, None
    results = data.get("results") or []
    if not results:
        return None, None
    hit_title = (results[0].get("title") or "").strip()
    if title_jaccard(t, hit_title) < min_jaccard:
        return None, None
    inv = results[0].get("abstract_inverted_index")
    ab = reconstruct_openalex_abstract(inv)
    if ab and is_plausible_abstract(ab):
        return ab.strip(), "openalex_title_search"
    return None, None


def fetch_abstract_for_row(doi_cell: str, title: str) -> tuple[str | None, str | None]:
    kind, value = parse_doi_cell(doi_cell)

    if kind == "empty":
        return None, None

    if kind in ("doi", "s2_corpus_id"):
        ab, src, _s2_429 = fetch_semantic_scholar(kind, value or "")
        if ab:
            return ab, src
        if kind == "doi" and value:
            ab, src = fetch_openalex_by_doi(value)
            if ab:
                return ab, src

    if kind == "other_url" and value and "semanticscholar.org" in value.lower():
        m = S2_HEX_ID.search(value)
        if m:
            ab, src, _s2_429 = fetch_semantic_scholar("s2_corpus_id", m.group(1).lower())
            if ab:
                return ab, src

    ab, src = fetch_openalex_search_title(title)
    if ab:
        return ab, src

    return None, None


def process_file(
    path: Path,
    out_path: Path,
    delay_s: float,
    start_row: int,
    max_rows: int | None,
) -> None:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        extra = ["abstract", "abstract_source"]
        out_fields = fieldnames + [c for c in extra if c not in fieldnames]
        rows = list(reader)

    if max_rows is not None:
        rows = rows[: max(0, max_rows)]

    done = 0
    with out_path.open("w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=out_fields, extrasaction="ignore")
        writer.writeheader()
        for i, row in enumerate(rows):
            if i < start_row:
                writer.writerow({**row, "abstract": row.get("abstract", ""), "abstract_source": row.get("abstract_source", "")})
                continue

            doi_cell = row.get("doi", "") or ""
            title = row.get("title", "") or ""
            ab, src = fetch_abstract_for_row(doi_cell, title)

            row_out = {**row, "abstract": ab or "", "abstract_source": src or ""}
            writer.writerow(row_out)
            fout.flush()
            done += 1
            if delay_s > 0:
                time.sleep(delay_s)

    print(f"Wrote {done} rows (from row {start_row + 1}) to {out_path}")


def main() -> None:
    root = Path(__file__).resolve().parent
    default_inputs = [root / "CSV" / "public_test.csv", root / "CSV" / "private_test.csv"]

    ap = argparse.ArgumentParser(description="Extract abstracts via DOI / Semantic Scholar / OpenAlex.")
    ap.add_argument(
        "--inputs",
        nargs="*",
        type=Path,
        default=default_inputs,
        help="Input CSV paths (default: CSV/public_test.csv CSV/private_test.csv)",
    )
    ap.add_argument(
        "--delay",
        type=float,
        default=3.1,
        help="Seconds to sleep between rows (rate limiting). Default 3.1",
    )
    ap.add_argument("--start-row", type=int, default=0, help="0-based row index in CSV body to start fetching")
    ap.add_argument("--max-rows", type=int, default=None, help="Process at most this many data rows")
    ap.add_argument(
        "--suffix",
        default="_with_abstracts",
        help="Output filename suffix before .csv (default _with_abstracts)",
    )
    args = ap.parse_args()

    for inp in args.inputs:
        inp = inp.resolve()
        if not inp.is_file():
            print(f"Skip missing file: {inp}", file=sys.stderr)
            continue
        stem = inp.stem
        out = inp.with_name(f"{stem}{args.suffix}.csv")
        process_file(inp, out, delay_s=args.delay, start_row=args.start_row, max_rows=args.max_rows)


if __name__ == "__main__":
    main()
