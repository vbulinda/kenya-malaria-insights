"""LLM agent that answers questions by calling deterministic analysis tools.

Design choices (see README, "Responsible AI"):
* The model never sees raw rows beyond what a tool returns, and never does
  arithmetic itself; all numbers come from analysis.py.
* Every answer is checked: numbers in the final text that do not appear in any
  tool result are flagged as "unverified" so a human can review them.
* No personal data is involved: inputs are aggregate sub-county indicators.

The client calls the Anthropic Messages API over HTTPS using only the Python
standard library, so the project has no SDK dependency. Any object with a
``create(system, messages, tools) -> dict`` method can be passed instead
(used in the tests with a scripted fake model).
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field

import pandas as pd

from .analysis import TOOLS

SYSTEM_PROMPT = """You are a careful malaria data analyst supporting Kenyan county \
and national malaria programme staff. Answer questions ONLY using results returned \
by the tools provided. Do not estimate, recall or invent figures. If the data cannot \
answer a question, say so and suggest what data would be needed. Quote figures with \
their fiscal year and indicator name. Keep answers short and practical, and end with \
one line noting any data limitations (e.g., missing years, reporting completeness)."""

TOOL_SCHEMAS = [
    {
        "name": "list_units",
        "description": "List the counties, sub-counties, fiscal years and indicators available in the dataset.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "subcounty_trend",
        "description": "Time series of one indicator for one sub-county, with absolute and percent change from first to last year.",
        "input_schema": {
            "type": "object",
            "properties": {
                "subcounty": {"type": "string"},
                "indicator": {"type": "string", "enum": ["opd_cases", "confirmed_per_1000", "test_positivity", "microscopy_positivity"]},
            },
            "required": ["subcounty", "indicator"],
        },
    },
    {
        "name": "rank_subcounties",
        "description": "Rank sub-counties by an indicator for a fiscal year (default latest), optionally within one county.",
        "input_schema": {
            "type": "object",
            "properties": {
                "indicator": {"type": "string", "enum": ["opd_cases", "confirmed_per_1000", "test_positivity", "microscopy_positivity"]},
                "fiscal_year": {"type": "string"},
                "county": {"type": "string"},
                "top_n": {"type": "integer", "minimum": 1, "maximum": 50},
                "ascending": {"type": "boolean"},
            },
            "required": ["indicator"],
        },
    },
    {
        "name": "flag_rising",
        "description": "Sub-counties where an indicator rose by more than threshold_pct percent between the first and last fiscal years.",
        "input_schema": {
            "type": "object",
            "properties": {
                "indicator": {"type": "string", "enum": ["opd_cases", "confirmed_per_1000", "test_positivity", "microscopy_positivity"]},
                "threshold_pct": {"type": "number"},
                "county": {"type": "string"},
            },
            "required": ["indicator"],
        },
    },
    {
        "name": "county_summary",
        "description": "Min, median and max of each indicator across a county's sub-counties for a fiscal year (default latest).",
        "input_schema": {
            "type": "object",
            "properties": {"county": {"type": "string"}, "fiscal_year": {"type": "string"}},
            "required": ["county"],
        },
    },
]


class AnthropicClient:
    """Minimal Messages API client (standard library only)."""

    def __init__(self, api_key: str | None = None, model: str | None = None, max_tokens: int = 1024):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError("Set the ANTHROPIC_API_KEY environment variable to use 'ask' or 'brief --llm'. "
                               "The validate, rank and brief commands work without it.")
        self.model = model or os.environ.get("MALARIA_INSIGHTS_MODEL", "claude-sonnet-5-5")
        self.max_tokens = max_tokens

    def create(self, system: str, messages: list, tools: list) -> dict:
        body = json.dumps({
            "model": self.model, "max_tokens": self.max_tokens,
            "system": system, "messages": messages, "tools": tools,
        }).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages", data=body, method="POST",
            headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            raise RuntimeError(f"LLM API error {e.code}: {detail}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Could not reach the LLM API: {e.reason}") from e


@dataclass
class Answer:
    text: str
    tool_calls: list = field(default_factory=list)
    unverified_numbers: list = field(default_factory=list)

    def render(self) -> str:
        out = [self.text.strip()]
        if self.tool_calls:
            out.append("\nSources (tool calls): " + "; ".join(
                f"{c['name']}({json.dumps(c['input'])})" for c in self.tool_calls))
        if self.unverified_numbers:
            out.append("CHECK: these numbers were not found in any tool output: "
                       + ", ".join(self.unverified_numbers))
        return "\n".join(out)


_NUM = re.compile(r"(?<![\w.])-?\d[\d,]*\.?\d*")


_FY = re.compile(r"\b(19|20)\d{2}/\d{2}\b")


def _numbers(text: str) -> set[str]:
    found = set()
    text = _FY.sub(" ", text)  # fiscal years like 2024/25 are labels, not figures
    for m in _NUM.findall(text):
        n = m.replace(",", "").rstrip(".")
        try:
            v = float(n)
        except ValueError:
            continue
        found.add(f"{v:g}")
    return found


def unverified_numbers(answer: str, tool_outputs: list[str]) -> list[str]:
    """Numbers in the answer that appear in no tool output.

    Years and small ordinal numbers (<= 10, e.g. "top 5") are ignored.
    """
    source = set()
    for t in tool_outputs:
        source |= _numbers(t)
    flagged = []
    for n in sorted(_numbers(answer)):
        v = float(n)
        if v.is_integer() and (abs(v) <= 10 or 1900 <= v <= 2100):
            continue
        if n not in source:
            flagged.append(n)
    return flagged


def run_tool(df: pd.DataFrame, name: str, args: dict) -> str:
    if name not in TOOLS:
        return json.dumps({"error": f"unknown tool {name}"})
    try:
        return json.dumps(TOOLS[name](df, **args))
    except (ValueError, TypeError) as e:
        return json.dumps({"error": str(e)})


def ask(df: pd.DataFrame, question: str, client, max_steps: int = 6) -> Answer:
    """Agent loop: the model may call tools up to max_steps times."""
    messages = [{"role": "user", "content": question}]
    calls, outputs = [], []
    for _ in range(max_steps):
        resp = client.create(SYSTEM_PROMPT, messages, TOOL_SCHEMAS)
        content = resp.get("content", [])
        messages.append({"role": "assistant", "content": content})
        tool_uses = [b for b in content if b.get("type") == "tool_use"]
        if not tool_uses:
            text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
            return Answer(text, calls, unverified_numbers(text, outputs))
        results = []
        for b in tool_uses:
            out = run_tool(df, b["name"], b.get("input", {}))
            calls.append({"name": b["name"], "input": b.get("input", {})})
            outputs.append(out)
            results.append({"type": "tool_result", "tool_use_id": b["id"], "content": out})
        messages.append({"role": "user", "content": results})
    return Answer("Stopped: too many tool calls without a final answer.", calls, [])
