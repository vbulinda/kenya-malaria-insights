"""Loading and validating sub-county malaria indicator data.

Expected input: a CSV with one row per sub-county per fiscal year, as exported
from the Kenya Health Information System (KHIS / DHIS2) or compiled from
programme reports. Column names are case-insensitive; common aliases are
mapped automatically (see COLUMN_ALIASES).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

REQUIRED = ["county", "subcounty", "fiscal_year"]
INDICATORS = {
    "opd_cases": "Outpatient malaria cases (count)",
    "confirmed_per_1000": "Confirmed malaria cases per 1,000 population",
    "test_positivity": "Malaria test positivity rate (%)",
    "microscopy_positivity": "Microscopy positivity rate (%)",
}

COLUMN_ALIASES = {
    "sub_county": "subcounty",
    "sub-county": "subcounty",
    "sub county": "subcounty",
    "year": "fiscal_year",
    "fy": "fiscal_year",
    "opd": "opd_cases",
    "opd_malaria_cases": "opd_cases",
    "confirmed_cases_per_1000": "confirmed_per_1000",
    "incidence_per_1000": "confirmed_per_1000",
    "tpr": "test_positivity",
    "test_positivity_rate": "test_positivity",
    "microscopy_positivity_rate": "microscopy_positivity",
    "slide_positivity": "microscopy_positivity",
}


class DataValidationError(ValueError):
    """Raised when the input table cannot be used safely."""


@dataclass
class ValidationReport:
    rows: int
    subcounties: int
    years: list[str]
    indicators_present: list[str]
    warnings: list[str] = field(default_factory=list)

    def as_text(self) -> str:
        lines = [
            f"Rows: {self.rows}",
            f"Sub-counties: {self.subcounties}",
            f"Fiscal years: {', '.join(self.years)}",
            f"Indicators: {', '.join(self.indicators_present)}",
        ]
        lines += [f"WARNING: {w}" for w in self.warnings]
        return "\n".join(lines)


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {}
    for c in df.columns:
        key = str(c).strip().lower()
        cols[c] = COLUMN_ALIASES.get(key, key.replace(" ", "_"))
    return df.rename(columns=cols)


def load_indicators(path: str | Path) -> tuple[pd.DataFrame, ValidationReport]:
    """Load a CSV of sub-county indicators and return (table, report).

    The function never invents or fills values: missing cells stay missing and
    are reported, so every number the assistant later quotes comes from the
    user's own data.
    """
    path = Path(path)
    if not path.exists():
        raise DataValidationError(f"File not found: {path}")
    df = _normalise_columns(pd.read_csv(path))

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise DataValidationError(f"Missing required column(s): {', '.join(missing)}")
    present = [c for c in INDICATORS if c in df.columns]
    if not present:
        raise DataValidationError(
            "No indicator columns found. Expected at least one of: " + ", ".join(INDICATORS)
        )

    for c in ("county", "subcounty", "fiscal_year"):
        df[c] = df[c].astype(str).str.strip()
    warnings: list[str] = []
    for c in present:
        before = df[c].isna().sum()
        df[c] = pd.to_numeric(df[c], errors="coerce")
        bad = df[c].isna().sum() - before
        if bad:
            warnings.append(f"{bad} non-numeric value(s) in '{c}' were treated as missing")
        if df[c].isna().any():
            warnings.append(f"'{c}' has {int(df[c].isna().sum())} missing value(s)")
        if c in ("test_positivity", "microscopy_positivity"):
            out = df[(df[c] < 0) | (df[c] > 100)]
            if len(out):
                warnings.append(f"{len(out)} '{c}' value(s) outside 0-100%")
        elif (df[c] < 0).any():
            warnings.append(f"Negative values found in '{c}'")

    dupes = df.duplicated(subset=["county", "subcounty", "fiscal_year"]).sum()
    if dupes:
        warnings.append(f"{dupes} duplicate sub-county/year row(s)")

    years = sorted(df["fiscal_year"].unique().tolist())
    report = ValidationReport(
        rows=len(df),
        subcounties=df["subcounty"].nunique(),
        years=years,
        indicators_present=present,
        warnings=warnings,
    )
    return df, report
