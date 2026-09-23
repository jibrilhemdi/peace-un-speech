"""Import returned validation sheets by stable sample ID and score them after review."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

VALID_CODE={"PRESENT","ABSENT","UNSURE",""}
VALID_TASK={"CODABLE","INSUFFICIENT_CONTEXT","UNCERTAIN",""}


def load_assessment(path: Path, validator: str) -> pd.DataFrame:
    if path.suffix.lower()==".xlsx":
        frames=[]
        for sheet,df in pd.read_excel(path,sheet_name=None,dtype=str).items():
            if "sample_id" in df: df["codebook_id"]=sheet; frames.append(df)
        d=pd.concat(frames,ignore_index=True)
    else: d=pd.read_csv(path,dtype=str,keep_default_na=False)
    if d.sample_id.duplicated().any(): raise ValueError(f"duplicate sample_id in {path}")
    for c in [x for x in d if x.startswith("human_code__")]:
        bad=set(d[c].fillna("").str.upper())-VALID_CODE
        if bad: raise ValueError(f"invalid values in {c}: {bad}")
        d[c]=d[c].fillna("").str.upper()
    bad=set(d.human_task_status.fillna("").str.upper())-VALID_TASK
    if bad: raise ValueError(f"invalid task status: {bad}")
    d["validator_id"]=validator; return d


def binary_metrics(a: pd.Series,b: pd.Series) -> dict:
    z=pd.DataFrame({"a":a,"b":b}); z=z[z.a.isin(["PRESENT","ABSENT"]) & z.b.isin(["PRESENT","ABSENT"])]
    if len(z)==0: return {k:np.nan for k in ["n","agreement","agreement_ci_low","agreement_ci_high","kappa",
        "precision","precision_ci_low","precision_ci_high","recall","recall_ci_low","recall_ci_high","f1","tp","fp","fn","tn"]}
    ap=z.a.eq("PRESENT"); bp=z.b.eq("PRESENT"); tp=int((ap&bp).sum()); fp=int((~ap&bp).sum()); fn=int((ap&~bp).sum()); tn=int((~ap&~bp).sum())
    agree=(tp+tn)/len(z); pa=ap.mean(); pb=bp.mean(); pe=pa*pb+(1-pa)*(1-pb); k=(agree-pe)/(1-pe) if pe!=1 else np.nan
    precision=tp/(tp+fp) if tp+fp else np.nan; recall=tp/(tp+fn) if tp+fn else np.nan
    f1=2*precision*recall/(precision+recall) if precision+recall and not(np.isnan(precision) or np.isnan(recall)) else np.nan
    def wilson(x,n):
        if not n: return (np.nan,np.nan)
        zc=1.959963984540054; p=x/n; den=1+zc*zc/n; center=(p+zc*zc/(2*n))/den; half=zc*math.sqrt(p*(1-p)/n+zc*zc/(4*n*n))/den
        return center-half,center+half
    alo,ahi=wilson(tp+tn,len(z)); plo,phi=wilson(tp,tp+fp); rlo,rhi=wilson(tp,tp+fn)
    return {"n":len(z),"agreement":agree,"agreement_ci_low":alo,"agreement_ci_high":ahi,"kappa":k,
            "precision":precision,"precision_ci_low":plo,"precision_ci_high":phi,"recall":recall,
            "recall_ci_low":rlo,"recall_ci_high":rhi,"f1":f1,"tp":tp,"fp":fp,"fn":fn,"tn":tn}


def score(paths: list[Path], key_path: Path, out: Path) -> None:
    key=pd.read_csv(key_path,dtype=str); reviews=[load_assessment(p,f"validator_{i+1}") for i,p in enumerate(paths)]
    # Stable ID join only; row order is intentionally irrelevant.
    for d in reviews:
        unknown=set(d.sample_id)-set(key.sample_id)
        if unknown: raise ValueError(f"unknown sample IDs: {sorted(unknown)[:5]}")
    rows=[]
    if len(reviews)>=2:
        a,b=reviews[:2]
        j=a.merge(b,on="sample_id",suffixes=("_a","_b"),validate="one_to_one")
        codes=sorted({c.removeprefix("human_code__") for c in a if c.startswith("human_code__")} & {c.removeprefix("human_code__") for c in b if c.startswith("human_code__")})
        for code in codes: rows.append({"comparison":"human-human","code":code,**binary_metrics(j[f"human_code__{code}_a"],j[f"human_code__{code}_b"])})
        cbcol="codebook_id_a" if "codebook_id_a" in j else None
        groups=j.groupby(cbcol) if cbcol else [("unknown",j)]
        for cb,g in groups:
            relevant=[c for c in codes if g[f"human_code__{c}_a"].notna().any()]
            complete=[]
            for _,r in g.iterrows():
                sa={c for c in relevant if r[f"human_code__{c}_a"]=="PRESENT"}
                sb={c for c in relevant if r[f"human_code__{c}_b"]=="PRESENT"}
                reviewed=all(r[f"human_code__{c}_a"] in ("PRESENT","ABSENT") and r[f"human_code__{c}_b"] in ("PRESENT","ABSENT") for c in relevant)
                if reviewed: complete.append(sa==sb)
            rows.append({"comparison":"human-human-supplementary","codebook_id":cb,"code":"__EXACT_MULTI_LABEL_SET__","n":len(complete),
                         "agreement":np.mean(complete) if complete else np.nan})
    # Human-LLM comparisons stay separate, and diagnostic/probability components stay separate.
    for d in reviews:
        j=d.merge(key,on="sample_id",validate="one_to_one",suffixes=("_human","_key"))
        for c in [x for x in d if x.startswith("human_code__")]:
            code=c.removeprefix("human_code__")
            predcol=f"prediction_code__{code}"
            if predcol is None: continue
            if predcol not in j: continue
            pred=np.where(pd.to_numeric(j[predcol],errors="coerce").fillna(0).eq(1),"PRESENT","ABSENT")
            for component,gidx in j.groupby("sampling_component").groups.items():
                rows.append({"comparison":"human-LLM","validator":d.validator_id.iloc[0],"component":component,"code":code,
                             **binary_metrics(j.loc[gidx,c],pd.Series(pred,index=j.index).loc[gidx])})
    out.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(out,index=False)


def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--key",required=True); ap.add_argument("--out",required=True); ap.add_argument("assessments",nargs="+")
    a=ap.parse_args(); score([Path(x) for x in a.assessments],Path(a.key),Path(a.out)); return 0


if __name__=="__main__": raise SystemExit(main())
