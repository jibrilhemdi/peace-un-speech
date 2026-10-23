import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

from un_speeches.research.assessment_scoring import binary_metrics, score
from un_speeches.research.galtung_research import load_config, codebook_specs, resample_speeches_within_country
from un_speeches.research.validation_pack import opaque_id
from un_speeches.research.research_outputs import _best_match_jaccard, export_latex, summarize_bootstrap


class ResearchPipelineTests(unittest.TestCase):
    def test_canonical_names_and_separate_violence_dimensions(self):
        cfg=load_config(); specs,rows=codebook_specs(cfg)
        self.assertIn("coupling",specs); self.assertIn("relational",specs); self.assertNotIn("relations",specs)
        dims={r["dimension"] for r in rows if r["codebook_id"]=="violence"}
        self.assertEqual(dims,{"violence_type","attribution"})

    def test_stable_sampling_id(self):
        self.assertEqual(opaque_id(7,"violence","u1"),opaque_id(7,"violence","u1"))
        self.assertNotEqual(opaque_id(7,"violence","u1"),opaque_id(7,"violence","u2"))

    def test_undefined_metrics_are_not_zero(self):
        m=binary_metrics(pd.Series(["ABSENT"]),pd.Series(["ABSENT"]))
        self.assertTrue(np.isnan(m["precision"])); self.assertTrue(np.isnan(m["recall"]))

    def test_id_join_not_row_order(self):
        key=pd.DataFrame({"sample_id":["a","b"],"prediction":[1,0]})
        human=pd.DataFrame({"sample_id":["b","a"],"human":["ABSENT","PRESENT"]})
        joined=human.merge(key,on="sample_id",validate="one_to_one")
        self.assertEqual(joined.prediction.tolist(),[0,1])

    def test_na_is_distinct_from_zero_and_multilabel_denominator(self):
        y=pd.Series([1,0,pd.NA],dtype="Int8")
        self.assertEqual(y.notna().sum(),2); self.assertEqual(y.sum(),1)
        a=pd.Series([1,1,0],dtype="Int8"); b=pd.Series([1,0,1],dtype="Int8")
        self.assertEqual(a.mean()+b.mean(),4/3)  # multi-label proportions need not sum to one

    def test_prepared_dedup_runs_joins_and_violence_na(self):
        base=Path("research_outputs/prepared_data")
        if not (base/"analysis_matrix.parquet").exists(): self.skipTest("run prepare first")
        ann=pd.read_parquet(base/"annotations_long_deduplicated.parquet")
        self.assertFalse(ann.duplicated(["unit_id","codebook_id","codebook_version","run_id","dimension","value"]).any())
        selected={("positive_negative",1,"laguna"),("violence",1,"laguna"),("coupling",2,"laguna-v2"),("relational",1,"laguna"),("transcend",1,"laguna")}
        self.assertEqual(set(map(tuple,ann[["codebook_id","codebook_version","run_id"]].drop_duplicates().to_records(index=False))),selected)
        p=pd.read_parquet(base/"paragraphs.parquet"); self.assertTrue(p.join_status.eq("both").all())
        m=pd.read_parquet(base/"analysis_matrix.parquet")
        attr=m["Y__violence__attribution__ATR-ISR"]
        violence=m[["Y__violence__violence_type__VD","Y__violence__violence_type__VS","Y__violence__violence_type__VC"]].max(axis=1)
        self.assertTrue(attr[violence.eq(0)].isna().all())

    def test_reviewer_blinding_and_blank_fields(self):
        p=Path("research_outputs/validation_member_states_palestine_v1/reviewers/validator_1.xlsx")
        if not p.exists(): self.skipTest("run validation first")
        wb=load_workbook(p,read_only=True,data_only=False)
        for name in [x for x in wb.sheetnames if x not in ("Instructions","Frozen codebook definitions")]:
            ws=wb[name]; headers=[c.value for c in next(ws.iter_rows(min_row=1,max_row=1))]
            forbidden=("prediction","confidence","country","strat","unit_id","speech_id","run_id")
            self.assertFalse(any(any(x in str(h).lower() for x in forbidden) for h in headers))
            human=[i for i,h in enumerate(headers) if str(h).startswith("human_") or h=="context_used"]
            for row in ws.iter_rows(min_row=2,values_only=True): self.assertTrue(all(row[i] in (None,"") for i in human))

    def test_sampling_metadata_and_same_reviewer_tasks(self):
        base=Path("research_outputs/validation")
        if not (base/"coordinator_only/coordinator_key.csv").exists(): self.skipTest("run validation first")
        key=pd.read_csv(base/"coordinator_only/coordinator_key.csv")
        scoped=pd.read_parquet("research_outputs/prepared_data/paragraphs.parquet",columns=["unit_id","in_scope"])
        self.assertTrue(key[["unit_id"]].merge(scoped,on="unit_id",validate="many_to_one").in_scope.all())
        prob=key[key.sampling_component.eq("probability")]; diag=key[key.sampling_component.eq("diagnostic")]
        self.assertTrue(prob.selection_probability.gt(0).all()); self.assertTrue(prob.sampling_weight.gt(0).all())
        self.assertTrue(diag.selection_probability.isna().all()); self.assertTrue(diag.sampling_weight.isna().all())
        for cb in key.codebook_id.unique():
            a=pd.read_csv(base/f"reviewers/validator_1_{cb}.csv"); b=pd.read_csv(base/f"reviewers/validator_2_{cb}.csv")
            self.assertEqual(set(a.sample_id),set(b.sample_id))

    def test_bootstrap_carries_speech_paragraphs_and_new_ids(self):
        d=pd.DataFrame({"country":["A","A","A"],"speech_id":["s1","s1","s2"],"unit_id":["u1","u2","u3"]})
        b=resample_speeches_within_country(d,np.random.default_rng(4))
        for _,g in b.groupby("speech_id"):
            source=g.source_speech_id.iloc[0]
            self.assertEqual(set(g.unit_id),set(d.loc[d.speech_id.eq(source),"unit_id"]))
        self.assertTrue(b.speech_id.str.contains("__boot").all())

    def test_model_outputs_are_provisional_and_not_attribution_profiles(self):
        p=Path("research_outputs/rq2_provisional/adjusted_probabilities.csv")
        if not p.exists(): self.skipTest("run rq2 first")
        d=pd.read_csv(p)
        self.assertTrue(d.status.eq("PROVISIONAL / UNVALIDATED").all())
        self.assertFalse(d.feature.str.contains("attribution").any())

    def test_synthetic_scoring_fixture_keeps_comparisons_separate(self):
        fixtures=Path("tests/fixtures")
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/"metrics.csv"
            score([fixtures/"synthetic_validator_1.csv",fixtures/"synthetic_validator_2.csv"],
                  fixtures/"synthetic_validation_key.csv",out)
            d=pd.read_csv(out)
            self.assertEqual(set(d.comparison),{"human-human","human-human-supplementary","human-LLM"})
            self.assertEqual(set(d.loc[d.comparison.eq("human-LLM"),"component"]),{"probability","diagnostic"})

    def test_stability_is_label_invariant(self):
        base=np.array([1,1,2,2,3])
        permuted=np.array([8,8,4,4,9])
        self.assertTrue(all(v==1.0 for _,v in _best_match_jaccard(base,permuted)))

    def test_latex_export_skips_raw_data(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/"research_outputs"; (root/"rq3_provisional").mkdir(parents=True)
            pd.DataFrame({"k":[2,3],"silhouette":[.4,.2]}).to_csv(root/"rq3_provisional/cluster_k_diagnostics.csv",index=False)
            written=export_latex(root,root/"latex",3)
            self.assertTrue((root/"latex/rq3_provisional/cluster_k_diagnostics.tex").exists())
            self.assertTrue((root/"latex/research_tables.tex").exists())

    def test_stability_summary_from_checkpoints_is_marked_incomplete(self):
        with tempfile.TemporaryDirectory() as td:
            rdir=Path(td)/"rq3_provisional"; bdir=rdir/"bootstrap_checkpoints"; bdir.mkdir(parents=True)
            base=pd.DataFrame({"country":["A","B","C","D"],"k":[2]*4,"profile_id":["P1","P1","P2","P2"]})
            base.to_csv(rdir/"cluster_assignments.csv",index=False)
            pd.DataFrame({"k":[2],"silhouette":[.5],"min_cluster_size":[2]}).to_csv(rdir/"cluster_k_diagnostics.csv",index=False)
            for rep,labels in [(0,[1,1,2,2]),(1,[2,2,1,1])]:
                pd.DataFrame({"replicate":[rep]*4,"country":["A","B","C","D"],"k":[2]*4,"profile_id":labels}).to_csv(bdir/f"replicate_{rep:04d}.csv",index=False)
            cfg=load_config(); cfg["analysis"]["bootstrap_target"]=3
            status=summarize_bootstrap(rdir,cfg,Path(td)/"cache")
            self.assertEqual(status["usable_replicates"],2)
            self.assertFalse(status["stability_evidence_complete"])
            summary=pd.read_csv(rdir/"bootstrap_stability_summary.csv")
            self.assertEqual(summary.mean_ari.iloc[0],1.0)


if __name__=="__main__": unittest.main()
