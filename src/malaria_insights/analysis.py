"""Deterministic analysis functions.

These are the only functions that touch the numbers. The language model never
computes statistics itself: it calls these tools and narrates their output, so
every figure in an answer can be traced to a row of the input table.
"""
from __future__ import annotations

import pandas as pd

from .data import INDICATORS


def _check_indicator(df: pd.DataFrame, indicator: str) -> None:
    if indicator not in INDICATORS:
        raise ValueError(f"Unknown indicator '{indicator}'. Choose from: {', '.join(INDICATORS)}")
    if indicator not in df.columns:
        raise ValueError(f"Indicator '{indicator}' is not in this dataset")


def _match(df: pd.DataFrame, column: str, value: str) -> pd.DataFrame:
    sel = df[df[column].str.lower() == value.strip().lower()]
    if sel.empty:
        options = ", ".join(sorted(df[column].unique())[:15])
        raise ValueError(f"No {column} named '{value}'. Examples: {options}")
    return sel


def list_units(df: pd.DataFrame) -> dict:
    """Counties, sub-counties and years available."""
    return {
        "counties": sorted(df["county"].unique().tolist()),
        "subcounties": sorted(df["subcounty"].unique().tolist()),
        "fiscal_years": sorted(df["fiscal_year"].unique().tolist()),
        "indicators": [c for c in INDICATORS if c in df.columns],
    }


def subcounty_trend(df: pd.DataFrame, subcounty: str, indicator: str) -> dict:
    """Values of one indicator over time for one sub-county, with change."""
    _check_indicator(df, indicator)
    sel = _match(df, "subcounty", subcounty).sort_values("fiscal_year")
    series = [
        {"fiscal_year": r.fiscal_year, "value": None if pd.isna(getattr(r, indicator)) else round(float(getattr(r, indicator)), 2)}
        for r in sel.itertuples()
    ]
    vals = [s["value"] for s in series if s["value"] is not None]
    change = None
    pct = None
    if len(vals) >= 2:
        change = round(vals[-1] - vals[0], 2)
        pct = round(100 * change / vals[0], 1) if vals[0] else None
    return {
        "subcounty": sel["subcounty"].iloc[0],
        "county": sel["county"].iloc[0],
        "indicator": indicator,
        "indicator_label": INDICATORS[indicator],
        "series": series,
        "absolute_change_first_to_last": change,
        "percent_change_first_to_last": pct,
    }


def rank_subcounties(df: pd.DataFrame, indicator: str, fiscal_year: str | None = None,
                     county: str | None = None, top_n: int = 10, ascending: bool = False) -> dict:
    """Rank sub-counties by an indicator for one year (default: latest)."""
    _check_indicator(df, indicator)
    sub = df if county is None else _match(df, "county", county)
    year = fiscal_year or sorted(sub["fiscal_year"].unique())[-1]
    sub = sub[sub["fiscal_year"] == year].dropna(subset=[indicator])
    if sub.empty:
        raise ValueError(f"No data for {indicator} in {year}")
    sub = sub.sort_values(indicator, ascending=ascending).head(max(1, int(top_n)))
    return {
        "indicator": indicator,
        "indicator_label": INDICATORS[indicator],
        "fiscal_year": year,
        "county_filter": county,
        "order": "ascending" if ascending else "descending",
        "ranking": [
            {"rank": i + 1, "subcounty": r.subcounty, "county": r.county, "value": round(float(getattr(r, indicator)), 2)}
            for i, r in enumerate(sub.itertuples())
        ],
    }


def flag_rising(df: pd.DataFrame, indicator: str, threshold_pct: float = 10.0,
                county: str | None = None) -> dict:
    """Sub-counties where the indicator rose by more than threshold_pct
    between the first and last available years."""
    _check_indicator(df, indicator)
    sub = df if county is None else _match(df, "county", county)
    years = sorted(sub["fiscal_year"].unique())
    if len(years) < 2:
        raise ValueError("Need at least two fiscal years to assess change")
    first, last = years[0], years[-1]
    a = sub[sub["fiscal_year"] == first].set_index("subcounty")[indicator]
    b = sub[sub["fiscal_year"] == last].set_index("subcounty")[indicator]
    joined = pd.concat([a.rename("first"), b.rename("last")], axis=1).dropna()
    joined = joined[joined["first"] > 0]
    joined["pct_change"] = 100 * (joined["last"] - joined["first"]) / joined["first"]
    flagged = joined[joined["pct_change"] > threshold_pct].sort_values("pct_change", ascending=False)
    return {
        "indicator": indicator,
        "indicator_label": INDICATORS[indicator],
        "from_year": first,
        "to_year": last,
        "threshold_pct": threshold_pct,
        "flagged": [
            {"subcounty": s, "first": round(float(r["first"]), 2), "last": round(float(r["last"]), 2),
             "pct_change": round(float(r["pct_change"]), 1)}
            for s, r in flagged.iterrows()
        ],
        "n_assessed": int(len(joined)),
    }


def county_summary(df: pd.DataFrame, county: str, fiscal_year: str | None = None) -> dict:
    """Simple descriptive summary of all indicators for a county in a year."""
    sub = _match(df, "county", county)
    year = fiscal_year or sorted(sub["fiscal_year"].unique())[-1]
    sub = sub[sub["fiscal_year"] == year]
    out = {"county": sub["county"].iloc[0], "fiscal_year": year, "n_subcounties": int(len(sub)), "indicators": {}}
    for ind in INDICATORS:
        if ind in sub.columns and sub[ind].notna().any():
            col = sub[ind].dropna()
            stats = {"min": round(float(col.min()), 2), "median": round(float(col.median()), 2),
                     "max": round(float(col.max()), 2)}
            if ind == "opd_cases":
                stats["total"] = round(float(col.sum()), 0)
            out["indicators"][ind] = stats
    return out


TOOLS = {
    "list_units": list_units,
    "subcounty_trend": subcounty_trend,
    "rank_subcounties": rank_subcounties,
    "flag_rising": flag_rising,
    "county_summary": county_summary,
}
