"""Debug helper for testing abstract fetches on specific inputs.

Use this script to inspect DOI/title lookups without reprocessing the full dataset.

Examples:
  python debug_extract.py --doi 10.1007/978-3-031-15707-3_13 --title "Policy-aware autonomous agents"
  python debug_extract.py --input CSV/public_test.csv --rows 1 3 5
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from extract_abstracts import (
    fetch_abstract_for_row,
    parse_doi_cell,
    fetch_semantic_scholar,
    fetch_semantic_scholar_publisher_fallback,
    fetch_openalex_by_doi,
    fetch_openalex_search_title,
    extract_semantic_scholar_doi_link,
    extract_abstract_from_html,
    http_get_text,
    browser_headers,
)


def print_debug(title: str, value: Any) -> None:
    print(f"=== {title} ===")
    if isinstance(value, (dict, list)):
        print(json.dumps(value, indent=2, ensure_ascii=False))
    else:
        print(value)
    print()


def debug_single_input(doi: str | None, title: str | None) -> None:
    doi = doi or ""
    title = title or ""
    kind, parsed = parse_doi_cell(doi)
    print_debug("Parsed DOI cell", {"kind": kind, "value": parsed})

    if kind in ("doi", "s2_corpus_id") and parsed:
        print_debug("Semantic Scholar API lookup", fetch_semantic_scholar(kind, parsed))
        if kind == "doi":
            print_debug("OpenAlex DOI lookup", fetch_openalex_by_doi(parsed))

    if kind == "other_url" and parsed and "semanticscholar.org" in parsed.lower():
        print_debug("Semantic Scholar publisher fallback", fetch_semantic_scholar_publisher_fallback(parsed))
        html_text, _, _ = http_get_text(parsed, browser_headers())
        doi_link = extract_semantic_scholar_doi_link(html_text or "")
        print_debug("Semantic Scholar DOI page link", doi_link)

    if parsed and kind in ("doi", "other_url"):
        print_debug("OpenAlex search by title", fetch_openalex_search_title(title))

    print_debug("Final abstract fetch", fetch_abstract_for_row(doi, title))


def debug_csv_rows(path: Path, rows: list[int]) -> None:
    with path.open(newline="", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    for idx in rows:
        if idx < 1 or idx > len(reader):
            print(f"Row {idx} is out of range (1..{len(reader)})")
            continue

        row = reader[idx - 1]
        print(f"\n=== Row {idx} ===")
        print_debug("Row data", {k: row.get(k) for k in row})
        debug_single_input(row.get("doi", ""), row.get("title", ""))


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug abstract extraction for specific inputs.")
    parser.add_argument("--input", type=Path, help="CSV input file to debug")
    parser.add_argument("--rows", type=int, nargs="*", help="1-based row numbers to inspect")
    parser.add_argument("--doi", help="DOI value or URL to debug")
    parser.add_argument("--title", help="Title to use for title-based fallback")
    args = parser.parse_args()

    if args.input and args.rows:
        debug_csv_rows(args.input, args.rows)
    elif args.doi or args.title:
        debug_single_input(args.doi, args.title)
    else:
        parser.error("Provide --input and --rows, or --doi/--title for a single case.")


if __name__ == "__main__":
    main()
