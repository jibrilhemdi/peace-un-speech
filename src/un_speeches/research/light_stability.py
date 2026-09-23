"""Fast, clearly separated stability checks for provisional country profiles.

Model draws reflect the fitted mean-field fixed-effect approximation. Speech
resampling reflects changes in the observed corpus composition using descriptive
profiles. Neither check is the archived full mixed-model refit bootstrap.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from .research_outputs import _best_match_jaccard


def _labels(profile: pd.DataFrame, blocks: dict[str, str], ks: list[int]) -> dict[int, np.ndarray]:
    if profile.isna().any().any():
        raise ValueError("stability profile contains missing values")
    z = pd.DataFrame(StandardScaler().fit_transform(profile), index=profile.index,
                     columns=profile.columns)
    for codebook in blocks:
        cols = [c for c in z if c.startswith(f"Y__{codebook}__")]
        if cols:
            z[cols] /= math.sqrt(len(cols))
    tree = linkage(z, method="ward", metric="euclidean")
    return {k: fcluster(tree, k, criterion="maxclust") for k in ks}


def _summarize(country_names: list[str], baseline: dict[int, np.ndarray],
               sampled: list[dict[int, np.ndarray]], method: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_rows, summary_rows = [], []
    for k, base in baseline.items():
        labels = [run[k] for run in sampled]
        co = np.mean([(x[:, None] == x[None, :]).astype(float) for x in labels], axis=0)
        for i, a in enumerate(country_names):
            for j in range(i + 1, len(country_names)):
                pair_rows.append({"method": method, "k": k, "country_a": a,
                                  "country_b": country_names[j], "co_cluster_rate": float(co[i, j]),
                                  "replicates": len(labels)})
        aris = [adjusted_rand_score(base, x) for x in labels]
        cluster_jaccards = np.array([[v for _, v in _best_match_jaccard(base, x)] for x in labels])
        summary_rows.append({"method": method, "k": k, "replicates": len(labels),
                             "mean_ari": float(np.mean(aris)),
                             "mean_cluster_jaccard": float(np.mean(cluster_jaccards)),
                             "minimum_cluster_mean_jaccard": float(cluster_jaccards.mean(axis=0).min()),
                             "interpretation": "provisional sensitivity; not full model-refit bootstrap"})
    return pd.DataFrame(pair_rows), pd.DataFrame(summary_rows)


def run_light_stability(cfg: dict, data: dict[str, pd.DataFrame], output: Path) -> dict:
    """Write model-uncertainty and speech-composition checks without refitting."""
    from .galtung_research import population, write_csv

    rdir = output / "rq3_provisional"
    qdir = output / "rq2_provisional"
    adjusted = pd.read_csv(rdir / "adjusted_profile_matrix.csv").set_index("country")
    countries, features = adjusted.index.tolist(), adjusted.columns.tolist()
    ks = [k for k in cfg["analysis"]["profile_k"] if k < len(countries)]
    blocks = cfg["core_profile_dimensions"]
    draws = pd.read_parquet(qdir / "adjusted_probability_draws.parquet")
    draws = draws[draws.country.isin(countries) & draws.feature.isin(features)]
    sampled_adjusted = []
    for _, group in draws.groupby("draw_id", sort=True):
        mat = group.pivot(index="country", columns="feature", values="probability")
        mat = mat.reindex(index=countries, columns=features)
        sampled_adjusted.append(_labels(mat, blocks, ks))
    if not sampled_adjusted:
        raise ValueError("no complete adjusted-probability draws")
    base_adjusted = _labels(adjusted, blocks, ks)
    model_pairs, model_summary = _summarize(countries, base_adjusted, sampled_adjusted,
                                            "mean_field_model_draws")

    frame = population(data["matrix"], cfg, cfg["population"]["rq2_rq3_forum"])
    frame = frame[frame.country.isin(countries)]
    speech_groups: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for country, group in frame.groupby("country"):
        by_speech = group.groupby("speech_id")[features]
        sums = by_speech.sum().to_numpy(dtype=float)
        counts = by_speech.count().to_numpy(dtype=float)
        speech_groups[country] = sums, counts
    if set(speech_groups) != set(countries):
        raise ValueError("descriptive resampling countries differ from adjusted profile")
    descriptive = pd.read_csv(qdir / "country_prevalence.csv")
    descriptive = descriptive[descriptive.country.isin(countries) & descriptive.feature.isin(features)]
    base_desc = descriptive.pivot(index="country", columns="feature", values="paragraph_weighted")
    base_desc = base_desc.reindex(index=countries, columns=features)
    base_desc_labels = _labels(base_desc, blocks, ks)
    rng = np.random.default_rng(cfg["analysis"]["seed"] + 8101)
    sampled_descriptive = []
    for _ in range(int(cfg["analysis"]["stability_draws"])):
        values = []
        for country in countries:
            sums, counts = speech_groups[country]
            index = rng.integers(0, len(sums), size=len(sums))
            denom = counts[index].sum(axis=0)
            if np.any(denom == 0):
                raise ValueError(f"speech resample has no evaluated units for {country}")
            values.append(sums[index].sum(axis=0) / denom)
        mat = pd.DataFrame(values, index=countries, columns=features)
        sampled_descriptive.append(_labels(mat, blocks, ks))
    desc_pairs, desc_summary = _summarize(countries, base_desc_labels, sampled_descriptive,
                                          "speech_resampled_descriptive")
    write_csv(model_pairs, rdir / "model_draw_coclustering.csv")
    write_csv(desc_pairs, rdir / "speech_resample_coclustering.csv")
    write_csv(pd.concat([model_summary, desc_summary], ignore_index=True),
              rdir / "light_stability_summary.csv")
    status = {"analysis_status": "PROVISIONAL / UNVALIDATED",
              "model_draws": len(sampled_adjusted),
              "speech_resamples": len(sampled_descriptive),
              "full_model_refit_bootstrap": "not run",
              "model_draw_limit": "mean-field fixed-effect posterior approximation; no speech refitting",
              "speech_resample_limit": "descriptive profiles; not adjusted-model uncertainty"}
    (rdir / "light_stability_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status
