"""Regression checks for the scoped, display-ready paper assets."""
import json
import unittest
import tempfile
from itertools import combinations, product
from pathlib import Path

import pandas as pd
from un_speeches.research.galtung_research import load_config, population


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "research_outputs"
PAPER = OUTPUT / "latex/paper"


class PaperAssetTests(unittest.TestCase):
    def test_nonconverged_anchor_is_reported_as_not_estimable(self):
        from un_speeches.research.paper_assets import _rq2
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); paper = root / "paper"; paper.mkdir()
            (root / "rq2_provisional").mkdir(); (root / "rq3_provisional").mkdir()
            features = ["Y__positive_negative__positive_negative__PN0", "Y__violence__violence_type__VS",
                        "Y__coupling__coupling_type__SYM", "Y__relational__relational__I1", "Y__transcend__transcend__T3"]
            rows = [{"country": country, "feature": feature, "variant": "primary", "adjusted_probability_re0": p,
                     "ci_low": p-.01, "ci_high": p+.01} for feature in features for country, p in [("A", .1), ("B", .2)]]
            pd.DataFrame(rows).to_csv(root / "rq2_provisional/adjusted_probabilities.csv", index=False)
            diag = [{"feature": f, "status": "ok", "converged": True} for f in features]
            diag.append({"feature": "Y__positive_negative__positive_negative__PN3", "status": "ok", "converged": False})
            pd.DataFrame(diag).to_csv(root / "rq2_provisional/model_diagnostics.csv", index=False)
            pd.DataFrame({"country": ["A", "B"], "PC1": [0, 1]}).to_csv(root / "rq3_provisional/pca_scores.csv", index=False)
            _rq2(root, paper, plt)
            extrema = pd.read_csv(paper / "rq2_anchor_extrema.csv").set_index("code")
            self.assertTrue(pd.isna(extrema.loc["PN3", "low_probability"]))
            self.assertIn("Not estimable", (paper / "rq2_anchor_extrema.tex").read_text())

    def test_ranked_table_matches_all_cross_block_candidates(self):
        if not (PAPER / "rq1_triplet_rankings.csv").exists():
            self.skipTest("run latex stage first")
        matrix = pd.read_parquet(OUTPUT / "prepared_data/analysis_matrix.parquet")
        data = population(matrix, load_config())
        self.assertTrue(data.country.eq("Palestine").any())
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
