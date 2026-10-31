# Data

Place your sub-county indicator file here (e.g. `khis_lake_endemic_fy2223_2425.csv`).
CSV files in this folder are **git-ignored** so programme data is never pushed to a public repository.

## Expected columns

| Column | Required | Description |
|---|---|---|
| `county` | yes | County name |
| `subcounty` | yes | Sub-county name (aliases: `sub_county`, `sub-county`) |
| `fiscal_year` | yes | e.g. `2024/25` (aliases: `year`, `fy`) |
| `opd_cases` | at least one indicator | Outpatient malaria cases |
| `confirmed_per_1000` | | Confirmed malaria cases per 1,000 population |
| `test_positivity` | | Test positivity rate, % (alias: `tpr`) |
| `microscopy_positivity` | | Microscopy positivity rate, % |

One row per sub-county per fiscal year. Missing values are allowed and are reported, never filled in.

Source suggestion: Kenya Health Information System (KHIS, DHIS2) malaria indicators, exported at sub-county level.
