"""Blinded future-validation workbook export. Predictions only enter coordinator files."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font
from openpyxl.worksheet.datavalidation import DataValidation

from un_speeches.research.galtung_research import ROOT, STATUS, outdir, write_csv


def opaque_id(seed: int, cb: str, unit: str) -> str:
    return "S-" + hashlib.sha256(f"{seed}|{cb}|{unit}".encode()).hexdigest()[:14].upper()


def _context(paragraphs: pd.DataFrame) -> dict[tuple[str, int], tuple[str, str]]:
    out={}
    for speech,g in paragraphs.sort_values(["speech_id","para_index"]).groupby("speech_id"):
        rows=list(g[["para_index","text"]].itertuples(index=False,name=None))
        for i,(idx,_) in enumerate(rows): out[(speech,idx)]=(rows[i-1][1] if i else "", rows[i+1][1] if i+1<len(rows) else "")
    return out


def _reviewer_workbook(path: Path, sheets: dict[str,pd.DataFrame], definitions: pd.DataFrame) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite reviewer workbook: {path}")
    wb=Workbook(); ws=wb.active; ws.title="Instructions"
    instructions=[
        ["VALIDATION STATUS","PENDING — these are independent future human assessments."],
        ["Blinding","You are blinded to model outputs, not to political identity that may be evident in original text."],
        ["Unit","Judge only the focal paragraph. Surrounding context is optional and visually separated."],
        ["Task status","Choose CODABLE, INSUFFICIENT_CONTEXT, or UNCERTAIN. Blank means unreviewed."],
        ["Code fields","For every code choose PRESENT, ABSENT, or UNSURE. Blank means unreviewed, never absent."],
        ["Evidence","Quote the focal words supporting a PRESENT judgment. Record whether context was used."],
        ["Multi-label","Assess each code independently; multiple PRESENT codes are allowed where the frozen codebook permits them."],
        ["NONE","Only violence attribution NONE is substantive. Do not use NONE as a universal absence label."],
    ]
    for row in instructions: ws.append(row)
    ws.column_dimensions["A"].width=24; ws.column_dimensions["B"].width=110
    for row in ws.iter_rows():
        for c in row: c.alignment=Alignment(wrap_text=True,vertical="top")
    defs=wb.create_sheet("Frozen codebook definitions")
    definition_cols=["codebook_id","dimension","code","label","definition","includes","excludes","positive_examples","negative_examples","codebook_instructions"]
    defs.append(definition_cols)
    for row in definitions[definition_cols].itertuples(index=False,name=None): defs.append(row)
    defs.freeze_panes="A2"; defs.auto_filter.ref=defs.dimensions; defs.column_dimensions["E"].width=100
    for row in defs.iter_rows():
        for c in row: c.alignment=Alignment(wrap_text=True,vertical="top")
    for cb,df in sheets.items():
        w=wb.create_sheet(cb[:31]); w.append(list(df.columns))
        for row in df.itertuples(index=False,name=None): w.append(list(row))
        w.freeze_panes="A2"; w.auto_filter.ref=w.dimensions
        for cell in w[1]: cell.font=Font(bold=True)
        widths={"A":20,"B":70,"C":55,"D":55}
        for col,width in widths.items(): w.column_dimensions[col].width=width
        for row in w.iter_rows():
            for c in row: c.alignment=Alignment(wrap_text=True,vertical="top")
        headers={c.value:c.column for c in w[1]}
        task=DataValidation(type="list",formula1='"CODABLE,INSUFFICIENT_CONTEXT,UNCERTAIN"',allow_blank=True)
        yn=DataValidation(type="list",formula1='"PRESENT,ABSENT,UNSURE"',allow_blank=True)
        context=DataValidation(type="list",formula1='"YES,NO"',allow_blank=True)
        w.add_data_validation(task); task.add(f"{w.cell(2,headers['human_task_status']).coordinate}:{w.cell(max(2,w.max_row),headers['human_task_status']).coordinate}")
        w.add_data_validation(context); context.add(f"{w.cell(2,headers['context_used']).coordinate}:{w.cell(max(2,w.max_row),headers['context_used']).coordinate}")
        for h,col in headers.items():
            if h.startswith("human_code__"):
                w.add_data_validation(yn); yn.add(f"{w.cell(2,col).coordinate}:{w.cell(max(2,w.max_row),col).coordinate}")
    path.parent.mkdir(parents=True,exist_ok=True); wb.save(path)


def export_validation_pack(cfg: dict[str,Any], data: dict[str,pd.DataFrame]) -> None:
    seed=cfg["validation"]["seed"]; rng=np.random.default_rng(seed)
    p=data["paragraphs"].loc[data["paragraphs"].in_scope.eq(True)].copy()
    ev=data["evaluations"].loc[data["evaluations"].unit_id.isin(p.unit_id)].copy()
    ann=data["dedup"].loc[data["dedup"].unit_id.isin(p.unit_id)].copy()
    defs=data["dictionary"]
    ctxt=_context(p); coordinator=[]; samples={}
    for cb,sel in cfg["selected_runs"].items():
        dims=defs.loc[defs.codebook_id.eq(cb),"dimension"].unique(); frame_status=ev[(ev.codebook_id.eq(cb)) & ev.dimension.eq(dims[0]) & ev.evaluation_status.eq("successful_complete")]
        frame=p[p.unit_id.isin(frame_status.unit_id) & p.para_is_procedural.eq(0)].copy()
        n=min(cfg["validation"]["probability_per_codebook"],len(frame)); prob=frame.sample(n=n,random_state=seed+sum(map(ord,cb))).copy()
        prob["sampling_component"]="probability"; prob["selection_probability"]=n/len(frame); prob["sampling_weight"]=len(frame)/n
        predicted=ann[ann.codebook_id.eq(cb)]; rare=predicted.groupby(["dimension","value"]).unit_id.nunique().sort_values()
        rare_rank={uv:i for i,uv in enumerate(rare.index)}
        diag_pool=frame[~frame.unit_id.isin(prob.unit_id)].copy()
        pred_by=predicted.groupby("unit_id").agg(predicted_count=("value","size"),mean_confidence=("confidence","mean"))
        diag_pool=diag_pool.join(pred_by,on="unit_id"); diag_pool["predicted_count"]=diag_pool.predicted_count.fillna(0)
        def rank_unit(uid):
            rows=predicted[predicted.unit_id.eq(uid)]; return min([rare_rank.get((r.dimension,r.value),99999) for r in rows.itertuples()] or [99999])
        diag_pool["rare_rank"]=[rank_unit(x) for x in diag_pool.unit_id]
        diag_pool["difficulty"]=(diag_pool.predicted_count.gt(1).astype(int)*-10000 + diag_pool.mean_confidence.fillna(1)*100 + diag_pool.rare_rank)
        nd=min(cfg["validation"]["diagnostic_per_codebook"],len(diag_pool)); diag=diag_pool.sort_values(["difficulty","unit_id"]).head(nd).copy()
        diag["sampling_component"]="diagnostic"; diag["selection_probability"]=np.nan; diag["sampling_weight"]=np.nan
        sample=pd.concat([prob,diag],ignore_index=True); sample["sample_id"]=[opaque_id(seed,cb,u) for u in sample.unit_id]
        sample["context_before"]=[ctxt.get((s,i),("",""))[0] for s,i in zip(sample.speech_id,sample.para_index)]
        sample["context_after"]=[ctxt.get((s,i),("",""))[1] for s,i in zip(sample.speech_id,sample.para_index)]
        # Context was not part of the original paragraph-only model input unit, even though requests were speech-batched.
        # Requests were speech-batched, so adjacent same-speech paragraphs were in the model input.
        sample["context_differs_from_model_focal_input"]=False
        predwide=(predicted[predicted.unit_id.isin(sample.unit_id)].groupby(["unit_id","dimension"]).value.apply(lambda x:"|".join(sorted(set(x)))).unstack(fill_value=""))
        coord=sample.merge(predwide.add_prefix("prediction__"),left_on="unit_id",right_index=True,how="left")
        output_rows=predicted[predicted.unit_id.isin(sample.unit_id)].copy()
        output_rows["output_record"]=output_rows.apply(lambda r: json.dumps({"annotation_index":int(r.annotation_index),
            "dimension":r.dimension,"value":r.value,"evidence":r.evidence,
            "confidence":None if pd.isna(r.confidence) else float(r.confidence)},ensure_ascii=False),axis=1)
        coord["original_llm_outputs"]=coord.unit_id.map(output_rows.groupby("unit_id").output_record.apply(lambda x:" | ".join(x))).fillna("")
        coord["selection_stratum"]=np.where(coord.sampling_component.eq("probability"),"simple_random_evaluation_frame","diagnostic_rare_multilabel_low_confidence")
        for code in defs.loc[defs.codebook_id.eq(cb),"code"]:
            positive_ids=set(predicted.loc[predicted.value.eq(code),"unit_id"])
            coord[f"prediction_code__{code}"]=coord.unit_id.isin(positive_ids).astype("int8")
        coord["codebook_id"]=cb; coord["codebook_version"]=sel["version"]; coord["run_id"]=sel["run_id"]
        coordinator.append(coord)
        human_cols=[f"human_code__{code}" for code in defs.loc[defs.codebook_id.eq(cb),"code"]]
        review=sample[["sample_id","text","context_before","context_after"]].rename(columns={"text":"focal_paragraph"})
        for col in ["human_task_status",*human_cols,"human_evidence_quote","human_notes","context_used"]: review[col]=""
        samples[cb]=review
    coord=pd.concat(coordinator,ignore_index=True)
    coord["codebook_overlap_tasks_for_unit"]=coord.groupby("unit_id").unit_id.transform("size")
    selected=pd.read_csv(outdir(cfg,"audit")/"selected_runs.csv").set_index("codebook_id")
    coord["model"]=coord.codebook_id.map(selected.models)
    coord["prompt_hash"]=coord.codebook_id.map(selected.prompt_hashes)
    cdir=outdir(cfg,"validation","coordinator_only"); rdir=outdir(cfg,"validation","reviewers")
    write_csv(coord,cdir/"coordinator_key.csv"); coord.to_parquet(cdir/"coordinator_key.parquet",index=False)
    manifest={"validation_status":"pending","seed":seed,"probability_per_codebook":cfg["validation"]["probability_per_codebook"],
              "diagnostic_per_codebook":cfg["validation"]["diagnostic_per_codebook"],"unique_paragraphs":int(coord.unit_id.nunique()),
              "tasks":len(coord),"per_codebook":coord.groupby("codebook_id").size().to_dict(),
              "probability_support":coord[coord.sampling_component.eq("probability")].groupby("codebook_id").size().to_dict(),
              "diagnostic_support":coord[coord.sampling_component.eq("diagnostic")].groupby("codebook_id").size().to_dict(),
              "note":"Diagnostic tasks are purposive and must not be pooled unweighted with probability tasks."}
    (cdir/"sampling_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    for v in range(1,cfg["validation"]["validators"]+1):
        vsheets={cb:df.sample(frac=1,random_state=seed+v+sum(map(ord,cb))).reset_index(drop=True) for cb,df in samples.items()}
        _reviewer_workbook(rdir/f"validator_{v}.xlsx",vsheets,defs)
        for cb,df in vsheets.items():
            path=rdir/f"validator_{v}_{cb}.csv"
            if path.exists(): raise FileExistsError(f"refusing to overwrite reviewer CSV: {path}")
            write_csv(df,path)


def write_validation_diagnostics(cfg: dict[str,Any]) -> None:
    """Summarize sampling support without changing reviewer files."""
    cdir=outdir(cfg,"validation","coordinator_only")
    d=pd.read_csv(cdir/"coordinator_key.csv",low_memory=False)
    coverage=(d.groupby(["codebook_id","sampling_component","country","year_month"],dropna=False)
              .size().rename("tasks").reset_index())
    write_csv(coverage,cdir/"sampling_country_time_coverage.csv")
    rows=[]
    for cb,g in d.groupby("codebook_id"):
        codecols=[c for c in g if c.startswith("prediction_code__")]
        for component,x in g.groupby("sampling_component"):
            for c in codecols:
                pos=int(pd.to_numeric(x[c],errors="coerce").fillna(0).sum())
                rows.append({"codebook_id":cb,"sampling_component":component,"code":c.removeprefix("prediction_code__"),
                             "predicted_positive":pos,"predicted_negative":len(x)-pos,"tasks":len(x)})
    write_csv(pd.DataFrame(rows),cdir/"sampling_prediction_support.csv")
