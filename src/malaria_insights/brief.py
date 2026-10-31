"""Generate a one-page sub-county malaria brief (Markdown, optional chart).

Two modes:
* offline (default): a deterministic template filled with computed values;
* llm: the same computed values are passed to the model, which writes a short
  narrative. The figures table is always produced by code, never by the model.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .analysis import subcounty_trend
from .data import INDICATORS


def _fmt(v, indicator: str = ""):
    if v is None:
        return "n/a"
    if indicator == "opd_cases":
        return f"{v:,.0f}"
    return f"{v:,.1f}"


def figures(df: pd.DataFrame, subcounty: str) -> list[dict]:
    present = [i for i in INDICATORS if i in df.columns]
    return [subcounty_trend(df, subcounty, i) for i in present]


def table_markdown(trends: list[dict]) -> str:
    years = [s["fiscal_year"] for s in trends[0]["series"]]
    head = "| Indicator | " + " | ".join(years) + " | Change |\n|---|" + "---|" * (len(years) + 1)
    rows = []
    for t in trends:
        vals = " | ".join(_fmt(s["value"], t["indicator"]) for s in t["series"])
        pct = t["percent_change_first_to_last"]
        ch = "n/a" if pct is None else f"{pct:+.1f}%"
        rows.append(f"| {t['indicator_label']} | {vals} | {ch} |")
    return head + "\n" + "\n".join(rows)


def offline_narrative(trends: list[dict]) -> str:
    lines = []
    for t in trends:
        pct = t["percent_change_first_to_last"]
        if pct is None:
            continue
        direction = "rose" if pct > 0 else "fell" if pct < 0 else "was unchanged"
        first, last = t["series"][0], t["series"][-1]
        ind = t["indicator"]
        lines.append(f"- {t['indicator_label']} {direction} from {_fmt(first['value'], ind)} "
                     f"({first['fiscal_year']}) to {_fmt(last['value'], ind)} ({last['fiscal_year']}), "
                     f"a change of {pct:+.1f}%.")
    return "\n".join(lines) or "- Not enough years of data to describe trends."


def chart(trends: list[dict], path: Path) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    n = len(trends)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 2.6))
    axes = [axes] if n == 1 else axes
    for ax, t in zip(axes, trends):
        xs = [s["fiscal_year"] for s in t["series"]]
        ys = [s["value"] for s in t["series"]]
        ax.plot(xs, ys, marker="o", color="#2E75B6")
        ax.set_title(t["indicator_label"], fontsize=8, wrap=True)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def build_brief(df: pd.DataFrame, subcounty: str, out_dir: str | Path = ".",
                client=None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trends = figures(df, subcounty)
    name, county = trends[0]["subcounty"], trends[0]["county"]
    slug = name.lower().replace(" ", "_")
    narrative = offline_narrative(trends)
    if client is not None:
        prompt = ("Write 3-4 short bullet points for a county malaria programme manager "
                  "interpreting these computed trends. Use only these numbers; do not add any. "
                  "Suggest one practical follow-up question for the programme.\n\n"
                  + json.dumps(trends))
        resp = client.create("You write concise, factual programme briefs.",
                             [{"role": "user", "content": prompt}], [])
        narrative = "".join(b.get("text", "") for b in resp.get("content", []) if b.get("type") == "text")
    png = chart(trends, out_dir / f"{slug}_trends.png")
    md = [f"# Malaria brief: {name} sub-county, {county} County", "",
          "## Key figures", "", table_markdown(trends), "",
          "## Interpretation", "", narrative, ""]
    if png:
        md += [f"![Trends]({png.name})", ""]
    md += ["---", "*Figures computed directly from the input dataset. "
           "Interpretation should be reviewed by programme staff before use.*"]
    path = out_dir / f"{slug}_brief.md"
    path.write_text("\n".join(md), encoding="utf-8")
    return path
