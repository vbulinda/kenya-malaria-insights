"""Unit tests. Run with:  python -m unittest discover -s tests

The fixture is SYNTHETIC test data (fictional counties 'Alpha' and 'Beta')
used only to check the code; it is not real malaria data.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from malaria_insights import analysis  # noqa: E402
from malaria_insights.brief import build_brief  # noqa: E402
from malaria_insights.data import DataValidationError, load_indicators  # noqa: E402
from malaria_insights.llm import ask, unverified_numbers  # noqa: E402

FIXTURE = Path(__file__).parent / "fixture_synthetic.csv"


class TestData(unittest.TestCase):
    def test_load_and_aliases(self):
        df, rep = load_indicators(FIXTURE)
        self.assertIn("subcounty", df.columns)
        self.assertIn("test_positivity", df.columns)
        self.assertEqual(rep.subcounties, 3)
        self.assertEqual(rep.years, ["2022/23", "2023/24", "2024/25"])
        self.assertTrue(any("missing" in w for w in rep.warnings))

    def test_missing_required(self):
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
            f.write("county,opd_cases\nX,1\n")
        with self.assertRaises(DataValidationError):
            load_indicators(f.name)


class TestAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, _ = load_indicators(FIXTURE)

    def test_trend(self):
        t = analysis.subcounty_trend(self.df, "north", "test_positivity")
        self.assertEqual([s["value"] for s in t["series"]], [31.2, 33.8, 36.1])
        self.assertAlmostEqual(t["percent_change_first_to_last"], 15.7)

    def test_rank_latest(self):
        r = analysis.rank_subcounties(self.df, "confirmed_per_1000")
        self.assertEqual(r["fiscal_year"], "2024/25")
        self.assertEqual(r["ranking"][0]["subcounty"], "North")

    def test_flag_rising(self):
        f = analysis.flag_rising(self.df, "opd_cases", 10)
        names = [x["subcounty"] for x in f["flagged"]]
        self.assertEqual(names, ["East", "North"])

    def test_unknown_subcounty(self):
        with self.assertRaises(ValueError):
            analysis.subcounty_trend(self.df, "Nowhere", "opd_cases")


class FakeModel:
    """Scripted model: first calls a tool, then answers using its output."""
    def __init__(self, final_text):
        self.final_text = final_text
        self.calls = 0

    def create(self, system, messages, tools):
        self.calls += 1
        if self.calls == 1:
            return {"content": [{"type": "tool_use", "id": "t1", "name": "rank_subcounties",
                                 "input": {"indicator": "test_positivity", "top_n": 2}}]}
        return {"content": [{"type": "text", "text": self.final_text}]}


class TestAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df, _ = load_indicators(FIXTURE)

    def test_grounded_answer(self):
        ans = ask(self.df, "Highest TPR?", FakeModel("North had the highest test positivity (36.1%) in 2024/25."))
        self.assertEqual(ans.unverified_numbers, [])
        self.assertEqual(ans.tool_calls[0]["name"], "rank_subcounties")

    def test_hallucinated_number_flagged(self):
        ans = ask(self.df, "Highest TPR?", FakeModel("North had 44.7% positivity in 2024/25."))
        self.assertEqual(ans.unverified_numbers, ["44.7"])

    def test_unverified_ignores_years_and_ordinals(self):
        self.assertEqual(unverified_numbers("Top 5 in 2024", [json.dumps({"a": 1})]), [])


class TestBrief(unittest.TestCase):
    def test_offline_brief(self):
        df, _ = load_indicators(FIXTURE)
        with tempfile.TemporaryDirectory() as d:
            path = build_brief(df, "South", d)
            text = path.read_text()
            self.assertIn("South sub-county", text)
            self.assertIn("fell", text)


if __name__ == "__main__":
    unittest.main()
