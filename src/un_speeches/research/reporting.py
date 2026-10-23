"""Rebuild concise provisional reports from the current scoped outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def write_provisional_report(cfg: dict, data: dict[str, pd.DataFrame], root: Path) -> None:
    from .galtung_research import STATUS, population

    output = root / cfg["source"]["output"]
    reports = output / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    scope = pd.read_csv(output / "audit" / "meeting_scope.csv")
    main = population(data["matrix"], cfg)
    main = main[main.forum.isin(cfg["population"]["rq1_forums"])]
    sc = population(data["matrix"], cfg, cfg["population"]["rq2_rq3_forum"])
    prev = pd.read_csv(output / "rq1_provisional" / "prevalence.csv")
    model = pd.read_csv(output / "rq2_provisional" / "model_diagnostics.csv")
    sensitivity = pd.read_csv(output / "rq2_provisional" / "sensitivity_model_diagnostics.csv")
    diag = pd.read_csv(output / "rq3_provisional" / "cluster_k_diagnostics.csv")
    pc = pd.read_csv(output / "rq3_provisional" / "pca_explained_variance.csv")
    stability_path = output / "rq3_provisional" / "light_stability_status.json"
    stability = json.loads(stability_path.read_text()) if stability_path.exists() else None
    eligible = int(scope.eligible_by_rule.sum())
    annotated = int(scope.in_scope.sum())
    converged = int((model.status.eq("ok") & model.converged.astype(str).str.lower().eq("true")).sum())
    sensitivity_converged = int((sensitivity.status.eq("ok") &
                                 sensitivity.converged.astype(str).str.lower().eq("true")).sum())
    best = diag.loc[diag.silhouette.idxmax()]
    top = prev.sort_values("paragraph_prevalence", ascending=False).head(5)
    top_text = ", ".join(f"{r.code} ({100*r.paragraph_prevalence:.1f}%)" for r in top.itertuples())
    feature_count = pd.read_csv(output / "rq3_provisional" / "adjusted_profile_matrix.csv").shape[1] - 1
    country_count = pd.read_csv(output / "rq3_provisional" / "adjusted_profile_matrix.csv").shape[0]
    population_label = "UN member states and Palestine" if "Palestine" in cfg["population"].get("included_observers", []) else "UN member states"
    lines = [
        "# Provisional Galtung analysis report", "",
        "**PROVISIONAL / UNVALIDATED. Human assessment is pending.**", "",
        "## Scope and measurement", "",
        f"The rule is {cfg['population']['start_date']} through {cfg['population']['end_date']}, with a meeting agenda that names Palestine. "
        f"Of {eligible} eligible collected meetings, {annotated} have the selected annotation runs; "
        f"{eligible-annotated} are eligible but unannotated. The latest annotated eligible meeting is "
        f"{main.date.max().date()}. These results describe the annotated subset.", "",
        f"RQ1 uses {len(main):,} substantive representative paragraphs from {population_label} in {main.speech_id.nunique():,} speeches. "
        f"RQ2/RQ3 use the Security Council subset ({len(sc):,} paragraphs) subject to country thresholds. "
        "The source database and its annotations were read without alteration. Unobserved codes are zero only "
        "where the selected annotation job establishes complete evaluation; otherwise they are missing.", "",
        "## RQ1 — shared repertoire", "",
        f"The five highest conditional paragraph prevalences are {top_text}. "
        "Multi-label shares can overlap. The accompanying co-occurrence table and selected original passages "
        "support later qualitative interpretation; that interpretation is not yet complete.", "",
        "## RQ2 — country variation", "",
        f"{converged} of {len(model)} core code models converged under the scoped sample. "
        "Country tests, intervals, and planned contrasts are approximate outputs of mean-field variational Bayes. "
        "Country probabilities are standardized over a common observed time/length distribution and do not "
        "represent timeless national beliefs. See model diagnostics for skipped or non-converged codes.", "",
        f"{sensitivity_converged} of {len(sensitivity)} planned sensitivity refits converged. "
        "Non-converged variants are omitted from sensitivity predictions and remain visible in diagnostics.", "",
        "## RQ3 — exploratory profiles", "",
        f"The adjusted matrix has {country_count} countries and {feature_count} converged core features. "
        f"PC1 explains {100*pc.iloc[0].explained_variance_ratio:.1f}% of variance. "
        f"The largest silhouette among k=2–5 is {best.silhouette:.3f} at k={int(best.k)}; "
        f"the smallest resulting cluster has {int(best.min_cluster_size)} countries. "
        "These are exploratory geometry diagnostics, not a claim of substantive political groups.", "",
    ]
    if stability:
        lines += ["The separate light checks use "
                  f"{stability['model_draws']} fitted-model probability draws and "
                  f"{stability['speech_resamples']} speech-resampled descriptive profiles. "
                  "They measure different sources of variation and do not replace a full model-refit bootstrap. ", ""]
    lines += ["## Remaining work", "",
              "Human reviewers must return their completed assessments, after which code-level reliability, "
              "disagreements, and any effect on the feature set can be evaluated. Representative passages "
              "require close reading before the paper assigns substantive labels or reaches final conclusions.", ""]
    (reports / "provisional_report.md").write_text("\n".join(lines), encoding="utf-8")
    status = {"validation_status": "pending", "analysis_status": STATUS,
              "scope": {"start": str(cfg["population"]["start_date"]), "end": str(cfg["population"]["end_date"]),
                        "meeting_rule": cfg["population"]["meeting_rule"],
                        "eligible_collected_meetings": eligible, "annotated_eligible_meetings": annotated,
                        "latest_annotated_eligible_meeting": str(main.date.max().date())},
              "completed": ["scope audit", "RQ1 descriptive and co-occurrence analysis",
                            "RQ2 primary country models and diagnosed sensitivity refits", "RQ3 exploratory profiles",
                            "two separate light RQ3 stability sensitivities" if stability else "RQ3 profiles without stability checks",
                            "blinded reviewer pack"],
              "not_completed": ["real human validation and adjudication", "final validated reporting",
                                "full mixed-model refit bootstrap"]}
    (reports / "execution_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
