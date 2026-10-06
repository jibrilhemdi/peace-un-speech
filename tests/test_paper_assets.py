"""Regression checks for the scoped, display-ready paper assets."""
import json
import unittest
from itertools import combinations, product
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "research_outputs"
PAPER = OUTPUT / "latex/paper"


class PaperAssetTests(unittest.TestCase):
    def test_ranked_table_matches_all_cross_block_candidates(self):
        if not (PAPER / "rq1_triplet_rankings.csv").exists():
            self.skipTest("run latex stage first")
        matrix = pd.read_parquet(OUTPUT / "prepared_data/analysis_matrix.parquet")
        data = matrix[
            matrix.role.eq("representative") & matrix.para_is_procedural.eq(0)
            & matrix.in_scope.eq(True) & matrix.forum.isin(["SC", "GA"])
            & ~matrix.country.isin(["Palestine", "Holy See"])
        ]
        self.assertEqual(len(data), 14463)
        groups = [
            ["PN0", "PN1", "PN2", "PN3", "PN4"],
            ["VD", "VS", "VC"],
            ["AB", "ANTI", "SYM"],
            ["I0", "I1", "I2", "I3", "I4"],
            ["T0", "T1", "T2", "T3", "T4"],
        ]
        features = {code: next(c for c in data if c.startswith("Y__") and c.endswith("__" + code))
                    for group in groups for code in group}
        observed = []
        for chosen_groups in combinations(groups, 3):
            for codes in product(*chosen_groups):
                hit = data[[features[c] for c in codes]].eq(1).all(axis=1)
                observed.append(("+".join(codes), int(hit.sum()),
                                 int(data.loc[hit, "meeting_id"].nunique())))
        observed.sort(key=lambda x: (-x[1], x[0]))
        audit = pd.read_csv(PAPER / "rq1_triplet_rankings.csv")
        self.assertEqual(len(observed), 710)
        self.assertEqual(audit.configuration.tolist(), [x[0] for x in observed])
        self.assertEqual(audit.paragraphs.tolist(), [x[1] for x in observed])
        self.assertEqual(audit.meetings.tolist(), [x[2] for x in observed])
        self.assertTrue((audit.evaluable_paragraphs == len(data)).all())
        self.assertTrue(((audit.share - audit.paragraphs / len(data)).abs() < 1e-12).all())

    def test_each_appendix_excerpt_has_ranked_source_and_pdf_check(self):
        if not (PAPER / "rq1_excerpt_provenance.json").exists():
            self.skipTest("run latex stage first")
        ranked = pd.read_csv(PAPER / "rq1_triplet_rankings.csv").head(10)
        excerpts = json.loads((PAPER / "rq1_excerpt_provenance.json").read_text())
        self.assertEqual(len(excerpts), 10)
        self.assertEqual([x["rank"] for x in excerpts], list(range(1, 11)))
        self.assertEqual([x["configuration"] for x in excerpts], ranked.configuration.tolist())
        self.assertEqual(len({x["unit_id"] for x in excerpts}), 10)
        for x in excerpts:
            self.assertTrue((ROOT / x["source_pdf"]).is_file())
            self.assertEqual(set(x["verified_codes"]), set(x["configuration"].split("+")))
            self.assertIn("source PDF", x["verification"])


if __name__ == "__main__":
    unittest.main()
