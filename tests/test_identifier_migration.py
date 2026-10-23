"""Source/content invariants for the explicit naming and population correction."""
import json
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
import hashlib
import contextlib
import io
from pathlib import Path

import pandas as pd
from openpyxl import Workbook, load_workbook

from un_speeches.lib.identifiers import (normalize_frame, normalize_structured, normalize_config,
                                        legacy_frame, feature_name)
from un_speeches.lib.db import Store
from un_speeches.research.galtung_research import load_config, population, prepare, codebook_specs
from un_speeches.lib import codebook as cbmod
from un_speeches.codebook.build_from_csv import convert
import yaml
import numpy as np
from scipy.optimize import minimize, rosen, rosen_der
from un_speeches.research.bfgs_update import rank_two_bfgs
from un_speeches.research.migrate_identifiers import copy_review_file
from un_speeches.research import migrate_identifiers as migration
from un_speeches.research.validation_pack import export_validation_pack


class IdentifierMigrationTests(unittest.TestCase):
    def test_rollback_restores_unsaved_work_and_retains_new_reviews_and_original_inputs(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td); project = base / "project"; backup = base / "backup"; stage = base / "stage"
            original = {"paper/main.tex": "original unsaved manuscript", "data/README.md": "original data guide",
                        "codebooks/relations.yaml": "original codebook", "research_outputs/result.csv": "original estimate",
                        "data/exports/result.csv": "original export", "data/gold/assessment.csv": "original assessment"}
            entries = []
            for rel, value in original.items():
                target = backup / rel; target.parent.mkdir(parents=True, exist_ok=True); target.write_text(value)
                active = project / rel; active.parent.mkdir(parents=True, exist_ok=True); active.write_text(value)
                digest = hashlib.sha256(target.read_bytes()).hexdigest()
                entries.append({"path": rel, "source_sha256": digest, "backup_sha256": digest, "method": "copy"})
            (backup / "manifest.json").write_text(json.dumps({"entries": entries}))
            (project / "paper/main.tex").write_text("migrated manuscript")
            (project / "data/README.md").write_text("migrated data guide")
            (project / "research_outputs/result.csv").write_text("recomputed estimate")
            (project / "research_outputs/returned_review.csv").write_text("new completed human review")
            (project / "data/gold/assessment.csv").write_text("updated original human assessment")
            (project / "codebooks/coupling.yaml").write_text("corrected codebook")
            added = project / "src/un_speeches/lib/identifiers.py"; added.parent.mkdir(parents=True); added.write_text("migration source")
            with patch.object(migration, "ROOT", project), contextlib.redirect_stdout(io.StringIO()):
                migration.rollback(backup, stage)
            self.assertEqual((project / "paper/main.tex").read_text(), "original unsaved manuscript")
            self.assertEqual((project / "data/README.md").read_text(), "original data guide")
            self.assertEqual((project / "data/gold/assessment.csv").read_text(), "updated original human assessment")
            self.assertEqual((project / "research_outputs/result.csv").read_text(), "original estimate")
            self.assertFalse((project / "codebooks/coupling.yaml").exists())
            self.assertFalse(added.exists())
            retained = next(stage.glob("rollback_preserved_*"))
            self.assertEqual((retained / "research_outputs/returned_review.csv").read_text(), "new completed human review")

    def test_bfgs_algebra_preserves_original_iteration_result(self):
        start = np.array([-1.2, 1.0, .8, 1.3])
        original = minimize(rosen, start, jac=rosen_der, method="BFGS", options={"maxiter": 500})
        expanded = minimize(rosen, start, jac=rosen_der, method=rank_two_bfgs, options={"maxiter": 500})
        self.assertEqual(original.success, expanded.success)
        self.assertEqual(original.nit, expanded.nit)
        np.testing.assert_allclose(original.x, expanded.x, rtol=1e-8, atol=1e-9)
        np.testing.assert_allclose(original.fun, expanded.fun, rtol=1e-10, atol=1e-10)

    def test_round_trip_and_idempotence_preserve_annotations_and_raw_provenance(self):
        old = pd.DataFrame({"codebook_id": ["relations", "imperialism"],
                            "dimension": ["relation_type", "imperialism"],
                            "value": ["AB", "I0"], "confidence": [0.75, 0.9],
                            "evidence": ['"imperialism and relations"', "coupling is a word"],
                            "response_json": ['{"relation_type":"AB"}', '{"imperialism":"I0"}'],
                            "request_hash": ["h1", "h2"], "prompt_hash": ["p1", "p2"],
                            "run_id": ["laguna-v2", "laguna"], "updated_at": ["2026-09-07", "2026-08-01"]})
        new = normalize_frame(old)
        self.assertEqual(new.codebook_id.tolist(), ["coupling", "relational"])
        pd.testing.assert_frame_equal(new, normalize_frame(new))
        pd.testing.assert_frame_equal(old, legacy_frame(new))
        self.assertEqual(new.source_codebook_id.tolist(), old.codebook_id.tolist())

    def test_collisions_raise_for_columns_data_config_and_json(self):
        for frame in [pd.DataFrame({"pred__relation_type": ["AB"], "pred__coupling_type": ["SYM"]}),
                      pd.DataFrame({"codebook_id": ["relations", "coupling"]}),
                      pd.DataFrame({"dimension": ["imperialism", "relational"]})]:
            with self.assertRaisesRegex(ValueError, "collision|mixed"):
                normalize_frame(frame)
        cfg = load_config(); cfg["selected_runs"]["relations"] = cfg["selected_runs"]["coupling"]
        with self.assertRaisesRegex(ValueError, "collision"):
            normalize_config(cfg)
        with self.assertRaisesRegex(ValueError, "collision"):
            normalize_structured({"relation_type": "AB", "coupling_type": "SYM"})

    def test_json_keys_and_dimension_slots_are_corrected_without_editing_evidence(self):
        original = {"annotations": [{"relation_type": "ANTI", "imperialism": "I0",
                                     "evidence": "relations and imperialism", "confidence": 0}],
                    "dimension": "relation_type", "run_id": "relations"}
        new = normalize_structured(original)
        self.assertEqual(new["annotations"][0]["coupling_type"], "ANTI")
        self.assertEqual(new["annotations"][0]["relational"], "I0")
        self.assertEqual(new["annotations"][0]["evidence"], "relations and imperialism")
        self.assertEqual(new["run_id"], "relations")
        self.assertEqual(new, normalize_structured(new))

    def test_null_zero_none_and_literal_na_stay_distinct(self):
        old = pd.DataFrame({"Y__relations__relation_type__AB": pd.Series([pd.NA, 0, 1], dtype="Int8"),
                            "value": ["NONE", "NA", "0"]})
        new = normalize_frame(old)
        self.assertEqual(new.iloc[1, 0], 0)
        self.assertTrue(pd.isna(new.iloc[0, 0]))
        pd.testing.assert_frame_equal(old, legacy_frame(new))

    def test_feature_exclusion_values_map_without_rewriting_country_values(self):
        old = pd.DataFrame({"type": ["feature", "country"],
                            "value": ["Y__relations__relation_type__AB", "relations"]})
        new = normalize_frame(old)
        self.assertEqual(new.value.tolist(), ["Y__coupling__coupling_type__AB", "relations"])
        pd.testing.assert_frame_equal(old, legacy_frame(new))

    def test_completed_reviews_preserve_ids_weights_notes_and_misleading_filename(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td); source = base / "2029_relations.csv"
            old = pd.DataFrame({"codebook_id": ["imperialism"], "gold__imperialism": ["I0"],
                                "gold__notes": ["relations, imperialism: reviewed"], "unit_id": ["001ab"],
                                "sample_id": ["S-UNCHANGED"], "weight": ["645.0333333333333"],
                                "blank": [""], "literal": ["NA"], "zero": ["0"]})
            old.to_csv(source, index=False)
            dest = copy_review_file(source, base / "corrected")
            self.assertEqual(dest.name, "2029_relational.csv")
            new = pd.read_csv(dest, dtype=str, keep_default_na=False)
            pd.testing.assert_frame_equal(old, legacy_frame(new))
            pd.testing.assert_frame_equal(old, pd.read_csv(source, dtype=str, keep_default_na=False))
            with self.assertRaises(FileExistsError):
                copy_review_file(source, base / "corrected")

    def test_workbook_completed_judgments_and_formulas_survive(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td); source = base / "validator_1.xlsx"
            wb = Workbook(); ws = wb.active; ws.title = "relations"
            ws.append(["sample_id", "gold__relation_type", "human_notes", "weight"])
            ws.append(["S-123", "AB", "Keep relations quotation", "=1/2"])
            wb.save(source)
            dest = copy_review_file(source, base / "out")
            out = load_workbook(dest)
            self.assertEqual(out["coupling"]["B1"].value, "gold__coupling_type")
            self.assertEqual(out["coupling"]["B2"].value, "AB")
            self.assertEqual(out["coupling"]["C2"].value, "Keep relations quotation")
            self.assertEqual(out["coupling"]["D2"].value, "=1/2")

    def test_central_population_adds_only_palestine_and_retains_exclusions(self):
        cfg = load_config()
        countries = ["Israel", "Palestine", "Holy See", "Palestine", "Palestine", "Palestine", "Palestine"]
        frame = pd.DataFrame({"country": countries, "role": ["representative"]*3 + ["official/briefer"] + ["representative"]*3,
                              "para_is_procedural": [0, 0, 0, 0, 1, 0, 0],
                              "in_scope": [True]*5 + [False, True], "forum": ["SC"]*6 + ["UNKNOWN"]})
        new = population(frame, cfg)
        self.assertEqual(new.country.tolist(), ["Israel", "Palestine"])
        cfg["population"]["included_observers"] = []
        self.assertEqual(population(frame, cfg).country.tolist(), ["Israel"])

    def test_readonly_store_never_initializes_or_writes_schema(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "source.db"
            with sqlite3.connect(path) as con:
                con.execute("CREATE TABLE legacy(value text)")
            con.close()
            before = path.read_bytes()
            with Store(path, readonly=True) as store:
                self.assertEqual(store.conn.execute("SELECT name FROM sqlite_master").fetchall()[0][0], "legacy")
                with self.assertRaises(sqlite3.OperationalError):
                    store.conn.execute("CREATE TABLE forbidden(x)")
            self.assertEqual(before, path.read_bytes())

    def test_legacy_sqlite_prepare_distinguishes_empty_failed_and_missing_jobs(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td); db = base / "source.db"
            with Store(db) as store:
                units = []
                for i in range(4):
                    units.append({"unit_id": f"u{i}", "speech_id": "S/PV.1#1", "para_index": i,
                        "text": "A substantive paragraph with peace and other words.", "country": "Palestine",
                        "ambassador_name": "Representative", "role": "representative", "language": "English",
                        "meeting_date": "01 January 2024", "bloc": "OBSERVER", "is_party": 1,
                        "para_is_procedural": 0, "n_words": 20, "segmented_at": "historical"})
                pd.DataFrame(units).to_sql("units", store.conn, if_exists="append", index=False)
                for i, status in enumerate(["ok", "ok", "api_error"]):
                    store.conn.execute("INSERT INTO jobs(unit_id,codebook_id,codebook_version,run_id,status,created_at,updated_at) VALUES(?,?,?,?,?,?,?)",
                        (f"u{i}", "relations", 2, "laguna-v2", status, "original-date", "original-date"))
                store.conn.execute("INSERT INTO annotations VALUES(?,?,?,?,?,?,?,?,?,?)",
                    ("u0", "relations", 2, "laguna-v2", 0, "relation_type", "AB", "peace", .9, 1))
            before = db.read_bytes()
            corpus = pd.DataFrame({"doc_symbol": ["S/PV.1"], "turn_index": [1], "meeting_number": [1],
                "speaker_header": ["Representative"], "word_count": [80], "source_pdf": ["unused.pdf"],
                "record_ids": ["1"], "meeting_provenance": ["historical"]})
            corpus.to_csv(base / "corpus.csv", index=False)
            pd.DataFrame({"symbol": ["S/PV.1"], "exists": [True], "meeting_date": ["2024-01-01"],
                "agenda": ["Palestinian question"], "agenda_palestine": [1]}).to_csv(base / "manifest.csv", index=False)
            cfg = load_config()
            cfg["source"].update(database=str(db), corpus=str(base / "corpus.csv"),
                meeting_manifest=str(base / "manifest.csv"), output=str(base / "out"))
            cfg["selected_runs"] = {"coupling": {"version": 2, "run_id": "laguna-v2"}}
            cfg["dimensions"] = {"coupling_type": ["AB", "ANTI", "SYM"]}
            cfg["core_profile_dimensions"] = {"coupling": "coupling_type"}
            data = prepare(cfg)
            values = data["matrix"]["Y__coupling__coupling_type__AB"].tolist()
            self.assertEqual(values[:2], [1, 0])
            self.assertTrue(pd.isna(values[2]) and pd.isna(values[3]))
            self.assertEqual(data["evaluations"].evaluation_status.tolist(),
                ["successful_complete", "successful_complete", "api_error", "not_evaluated"])
            cfg["validation"].update(validators=1, probability_per_codebook=1, diagnostic_per_codebook=1)
            export_validation_pack(cfg, data)
            review = base / "out/validation_member_states_palestine_v1/reviewers/validator_1_coupling.csv"
            assessment = pd.read_csv(review, dtype=str, keep_default_na=False)
            assessment.loc[0, "human_task_status"] = "CODABLE"
            assessment.loc[0, "human_code__AB"] = "PRESENT"
            assessment.to_csv(review, index=False)
            reviewed_bytes = review.read_bytes()
            export_validation_pack(cfg, data)
            self.assertEqual(reviewed_bytes, review.read_bytes())
            cached = base / "out/rq2_provisional/model_diagnostics.csv"
            cached.parent.mkdir(); cached.write_text("old estimates\n")
            prepare(cfg)
            self.assertTrue(cached.exists())  # identical inputs keep compatible results
            cfg["population"]["included_observers"] = []
            prepare(cfg)
            self.assertFalse(cached.exists())
            self.assertEqual(len(list((base / "out/incompatible_cache").glob("*/rq2_provisional/model_diagnostics.csv"))), 1)
            self.assertEqual(before, db.read_bytes())

    def test_historical_prompts_keep_original_hash_and_codebook_content(self):
        cfg = load_config(); directory = Path(cfg["source"]["codebooks"])
        for canonical in ["coupling", "relational"]:
            historical = cbmod.historical_codebook(directory.resolve(), canonical)
            active = cbmod.require_codebook(directory, canonical)
            source = cbmod.load_codebook(historical.path).codebook
            self.assertEqual(cbmod.prefix_hash(historical), cbmod.prefix_hash(source))
            self.assertEqual(active.version, historical.version)
            for a, b in zip(active.dimensions, historical.dimensions):
                self.assertEqual(a.categories, b.categories) if canonical == "relational" else self.assertEqual(
                    [(c.code, c.definition, c.value) for c in a.categories],
                    [(c.code, c.definition, c.value) for c in b.categories])

    def test_csv_builder_preserves_category_codes_after_display_rename(self):
        for name, dimension, codes in [("coupling", "coupling_type", ["AB", "ANTI", "SYM"]),
                                        ("relational", "relational", ["I0", "I1", "I2", "I3", "I4"])]:
            _, text = convert(Path("codebooks") / f"{name}.csv")
            raw = yaml.safe_load(text)
            self.assertEqual(raw["id"], name)
            self.assertEqual(raw["dimensions"][0]["name"], dimension)
            self.assertEqual([x["code"] for x in raw["dimensions"][0]["categories"]], codes)


if __name__ == "__main__":
    unittest.main()
