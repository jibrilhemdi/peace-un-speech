import unittest
from pathlib import Path

import pandas as pd

from un_speeches.research.galtung_research import load_config, mark_scope, normalize_symbol


class ScopeTests(unittest.TestCase):
    def test_symbol_normalization_and_meeting_decisions(self):
        cfg = load_config()
        paragraphs = pd.DataFrame({
            "doc_symbol": ["S/PV.9400", "S/PV.9439", "S/PV.10023 (Resumption 1)",
                           "S/PV.10068", "S/PV.10084"],
            "date": pd.to_datetime(["2023-08-21", "2023-10-16", "2025-10-23",
                                    "2025-12-16", "2025-12-29"]),
        })
        manifest = pd.DataFrame({
            "symbol": ["S/PV.9439", "S/PV.10023(Resumption1)", "S/PV.10068", "S/PV.10084"],
            "exists": [True] * 4,
            "meeting_date": ["2023-10-16", "2025-10-23", "2025-12-16", "2025-12-29"],
            "agenda": ["Palestinian question"] * 3 + ["Threats to international peace and security"],
            "agenda_palestine": [1, 1, 1, 0],
        })
        self.assertEqual(normalize_symbol("S/PV.10023 (Resumption 1)"),
                         normalize_symbol("S/PV.10023(Resumption1)"))
        result = mark_scope(paragraphs, manifest, cfg)
        self.assertEqual(result.in_scope.tolist(), [False, True, True, True, False])
        self.assertEqual(result.scope_exclusion_reason.tolist(),
                         ["before_start", "included", "included", "included", "agenda_not_palestine"])

    def test_in_window_unmatched_meeting_fails(self):
        cfg = load_config()
        paragraphs = pd.DataFrame({"doc_symbol": ["S/PV.unknown"],
                                   "date": pd.to_datetime(["2024-01-01"])})
        manifest = pd.DataFrame(columns=["symbol", "exists", "meeting_date", "agenda",
                                         "agenda_palestine"])
        with self.assertRaisesRegex(ValueError, "missing from manifest"):
            mark_scope(paragraphs, manifest, cfg)

    def test_regenerated_evidence_and_reviewer_sample_obey_scope(self):
        base = Path("research_outputs")
        prepared = base / "prepared_data/paragraphs.parquet"
        audit = base / "audit/meeting_scope.csv"
        if not prepared.exists() or not audit.exists():
            self.skipTest("run prepare and audit first")
        paragraphs = pd.read_parquet(prepared, columns=["unit_id", "doc_symbol", "date", "in_scope"])
        meetings = pd.read_csv(audit)
        included = meetings.loc[meetings.in_scope, "symbol"]
        self.assertEqual(len(included), 93)
        self.assertEqual(meetings.loc[meetings.in_scope, "date"].max(), "2025-12-16")
        self.assertTrue(meetings.loc[meetings.in_scope, "agenda"].str.contains("palestin", case=False).all())
        self.assertEqual(meetings.loc[meetings.symbol.eq("S/PV.10084"), "scope_exclusion_reason"].iloc[0],
                         "agenda_not_palestine")
        for rel in ["rq1_provisional/evidence_review.csv",
                    "rq3_provisional/representative_deviant_evidence.csv",
                    "validation/coordinator_only/coordinator_key.csv"]:
            path = base / rel
            if not path.exists():
                continue
            selected = pd.read_csv(path, usecols=["unit_id"])
            checked = selected.merge(paragraphs, on="unit_id", validate="many_to_one")
            self.assertEqual(len(checked), len(selected), rel)
            self.assertTrue(checked.in_scope.all(), rel)
            self.assertTrue(checked.date.between("2023-10-07", "2025-12-31").all(), rel)


if __name__ == "__main__":
    unittest.main()
