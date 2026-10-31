"""Command-line interface.

Examples:
    python -m malaria_insights validate data/my_khis_export.csv
    python -m malaria_insights ask data/my_khis_export.csv "Which sub-counties in Kisumu had the highest test positivity last year?"
    python -m malaria_insights brief data/my_khis_export.csv "Nyando" --out reports/
    python -m malaria_insights rank data/my_khis_export.csv test_positivity --county Kisumu
"""
from __future__ import annotations

import argparse
import json
import sys

from . import analysis
from .brief import build_brief
from .data import DataValidationError, load_indicators
from .llm import AnthropicClient, ask


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="malaria_insights", description="LLM-assisted malaria indicator analysis")
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="Check a dataset and report problems")
    v.add_argument("csv")

    a = sub.add_parser("ask", help="Ask a question in plain language (needs ANTHROPIC_API_KEY)")
    a.add_argument("csv")
    a.add_argument("question")

    b = sub.add_parser("brief", help="Write a one-page sub-county brief")
    b.add_argument("csv")
    b.add_argument("subcounty")
    b.add_argument("--out", default="reports")
    b.add_argument("--llm", action="store_true", help="Use the LLM for the narrative")

    r = sub.add_parser("rank", help="Rank sub-counties by an indicator (no LLM)")
    r.add_argument("csv")
    r.add_argument("indicator")
    r.add_argument("--year")
    r.add_argument("--county")
    r.add_argument("--top", type=int, default=10)

    args = p.parse_args(argv)
    try:
        df, report = load_indicators(args.csv)
    except DataValidationError as e:
        print(f"Data error: {e}", file=sys.stderr)
        return 2

    try:
        return _run(args, df, report)
    except (RuntimeError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _run(args, df, report) -> int:
    if args.cmd == "validate":
        print(report.as_text())
    elif args.cmd == "rank":
        print(json.dumps(analysis.rank_subcounties(df, args.indicator, args.year, args.county, args.top), indent=2))
    elif args.cmd == "brief":
        client = AnthropicClient() if args.llm else None
        print(f"Brief written to {build_brief(df, args.subcounty, args.out, client)}")
    elif args.cmd == "ask":
        print(ask(df, args.question, AnthropicClient()).render())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
