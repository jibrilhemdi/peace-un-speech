"""Figures, stability summaries, and Overleaf-ready tables for the Galtung pipeline."""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import adjusted_rand_score, silhouette_samples
from sklearn.neighbors import NearestNeighbors


STATUS = "PROVISIONAL / UNVALIDATED"


def _plot_setup(cache: Path) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache / "fontconfig"))
    (cache / "matplotlib").mkdir(parents=True, exist_ok=True)
    (cache / "fontconfig").mkdir(parents=True, exist_ok=True)


def cluster_diagnostic_figure(rdir: Path, cache: Path) -> Path:
    """Silhouette curve, cluster balance, and per-country widths at selected k."""
    _plot_setup(cache)
    import matplotlib.pyplot as plt

    diag = pd.read_csv(rdir / "cluster_k_diagnostics.csv")
    z = pd.read_csv(rdir / "standardized_block_scaled_matrix.csv").set_index("country")
    assign = pd.read_csv(rdir / "cluster_assignments.csv")
    best_k = int(diag.loc[diag.silhouette.idxmax(), "k"])
    labels = (assign[assign.k.eq(best_k)].set_index("country")
              .reindex(z.index).profile_id.str.removeprefix("P").astype(int).to_numpy())
    widths = silhouette_samples(z.to_numpy(), labels, metric="euclidean")

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))
    ax = axes[0]
    ax.plot(diag.k, diag.silhouette, "o-", color="#3569a8", lw=1.7)
    ax.axvline(best_k, color="#c8453c", ls="--", lw=1.2)
    ax.annotate(f"silhouette maximum: k={best_k}",
                (best_k, float(diag.loc[diag.k.eq(best_k), "silhouette"].iloc[0])),
                xytext=(7, -16), textcoords="offset points", fontsize=8)
    ax.set_xticks(diag.k); ax.set_xlabel("Number of clusters $k$")
    ax.set_ylabel("Mean silhouette width"); ax.set_title("(a) Separation across $k$")

    ax = axes[1]
    ax.plot(diag.k, diag.min_cluster_size, "s-", color="#5a8f5a", label="smallest cluster")
    if "largest_cluster_share" in diag:
        ax2 = ax.twinx()
        ax2.plot(diag.k, diag.largest_cluster_share, "o--", color="#9b6bb3", label="largest share")
        ax2.set_ylabel("Largest-cluster share", color="#9b6bb3"); ax2.set_ylim(0, 1)
    ax.axvline(best_k, color="#c8453c", ls="--", lw=1.2)
    ax.set_xticks(diag.k); ax.set_xlabel("Number of clusters $k$")
    ax.set_ylabel("Countries in smallest cluster"); ax.set_title("(b) Cluster balance")

    ax = axes[2]; y = 0
    colors = plt.get_cmap("tab10")
    for i, group in enumerate(sorted(np.unique(labels))):
        values = np.sort(widths[labels == group])
        ax.barh(np.arange(y, y + len(values)), values, height=0.9,
                color=colors(i), edgecolor="none", label=f"P{group} (n={len(values)})")
        y += len(values) + 2
    ax.axvline(widths.mean(), color="black", ls="--", lw=1,
               label=f"mean={widths.mean():.3f}")
    ax.axvline(0, color="#777777", lw=0.7)
    ax.set_yticks([]); ax.set_xlabel("Country silhouette width")
    ax.set_title(f"(c) Country widths at $k={best_k}$"); ax.legend(fontsize=7)

    fig.suptitle(f"Ward cluster diagnostics — {STATUS}", fontsize=12)
    fig.tight_layout(); path = rdir / "fig_cluster_diagnostics.png"
    fig.savefig(path, dpi=220, bbox_inches="tight"); plt.close(fig)
    legacy=rdir/"cluster_diagnostics.png"
    if legacy.exists(): legacy.unlink()
    return path


def knn_similarity_graph(rdir: Path, cache: Path, neighbors: int = 3) -> tuple[Path, Path, Path]:
    """Descriptive local-similarity network on the exact adjusted RQ3 feature space."""
    _plot_setup(cache)
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "3")
    import matplotlib.pyplot as plt
    import networkx as nx

    z = pd.read_csv(rdir / "standardized_block_scaled_matrix.csv").set_index("country")
    scores = pd.read_csv(rdir / "pca_scores.csv").set_index("country")
    diag = pd.read_csv(rdir / "cluster_k_diagnostics.csv")
    assign = pd.read_csv(rdir / "cluster_assignments.csv")
    best_k = int(diag.loc[diag.silhouette.idxmax(), "k"])
    profiles = assign[assign.k.eq(best_k)].set_index("country").profile_id.reindex(z.index)

    k = min(max(1, int(neighbors)), len(z) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(z)
    distances, indices = nn.kneighbors(z)
    edge_map: dict[tuple[str, str], float] = {}
    names = list(z.index)
    for i, name in enumerate(names):
        for dist, j in zip(distances[i, 1:], indices[i, 1:]):
            pair = tuple(sorted((name, names[int(j)])))
            edge_map[pair] = min(edge_map.get(pair, float("inf")), float(dist))
    scale = float(np.median(list(edge_map.values()))) or 1.0
    edges = pd.DataFrame([
        {"country_a": a, "country_b": b, "euclidean_distance": d,
         "similarity": math.exp(-d / scale), "knn_k": k,
         "feature_space": "standardized and equal-codebook block-scaled adjusted probabilities"}
        for (a, b), d in sorted(edge_map.items())
    ])
    nodes = pd.DataFrame({"country": z.index, "PC1": scores.PC1.reindex(z.index),
                          "PC2": scores.PC2.reindex(z.index), "ward_profile": profiles,
                          "degree": 0, "status": STATUS})
    graph = nx.Graph(); graph.add_nodes_from(names)
    for r in edges.itertuples(): graph.add_edge(r.country_a, r.country_b, weight=r.similarity)
    nodes["degree"] = nodes.country.map(dict(graph.degree())).astype(int)

    fig, ax = plt.subplots(figsize=(11.5, 8.0))
    palette = {p: plt.get_cmap("tab10")(i) for i, p in enumerate(sorted(profiles.unique()))}
    pos = {c: (float(scores.loc[c, "PC1"]), float(scores.loc[c, "PC2"])) for c in names}
    widths = [0.35 + 2.3 * graph[a][b]["weight"] for a, b in graph.edges]
    alphas = [0.18 + 0.62 * graph[a][b]["weight"] for a, b in graph.edges]
    for (edge, width, alpha) in zip(graph.edges, widths, alphas):
        nx.draw_networkx_edges(graph, pos, edgelist=[edge], width=width,
                               alpha=alpha, edge_color="#66717f", ax=ax)
    nx.draw_networkx_nodes(graph, pos, node_size=85,
                           node_color=[palette[profiles.loc[c]] for c in names],
                           edgecolors="white", linewidths=0.6, ax=ax)
    for c, (x, y) in pos.items(): ax.annotate(c, (x, y), xytext=(3, 3),
                                              textcoords="offset points", fontsize=6.4)
    ax.axhline(0, color="#dddddd", lw=0.6); ax.axvline(0, color="#dddddd", lw=0.6)
    ax.set_xlabel("PC1 score (layout only)"); ax.set_ylabel("PC2 score (layout only)")
    ax.set_title(f"{k}-nearest-neighbor country similarity network\n"
                 f"Edges use full adjusted profile distance; positions use PC1–PC2 — {STATUS}")
    ax.text(0.01, 0.01, "Descriptive local-neighborhood view; not a clustering or significance test.",
            transform=ax.transAxes, fontsize=7, color="#555555")
    fig.tight_layout(); fig_path = rdir / "fig_knn_similarity_network.png"
    fig.savefig(fig_path, dpi=220, bbox_inches="tight"); plt.close(fig)
    legacy=rdir/"knn_similarity_network.png"
    if legacy.exists(): legacy.unlink()
    edge_path, node_path = rdir / "knn_edges.csv", rdir / "knn_nodes.csv"
    edges.to_csv(edge_path, index=False, encoding="utf-8-sig")
    nodes.to_csv(node_path, index=False, encoding="utf-8-sig")
    return fig_path, edge_path, node_path


def _best_match_jaccard(base: np.ndarray, boot: np.ndarray) -> list[tuple[int, float]]:
    rows=[]
    for group in np.unique(base):
        source=set(np.flatnonzero(base == group)); best=0.0
        for other in np.unique(boot):
            target=set(np.flatnonzero(boot == other)); union=len(source | target)
            if union: best=max(best, len(source & target) / union)
        rows.append((int(group), best))
    return rows


def summarize_bootstrap(rdir: Path, cfg: dict[str, Any], cache: Path) -> dict[str, Any]:
    """Calculate label-invariant stability metrics and graphs from completed checkpoints."""
    bdir = rdir / "bootstrap_checkpoints"
    files = sorted(bdir.glob("replicate_*.csv"))
    failed = sorted(bdir.glob("replicate_*.failed.txt"))
    target = int(cfg["analysis"]["bootstrap_target"])
    usable = len(files); run_label = "FULL STABILITY RUN" if usable >= target else "SMOKE / INCOMPLETE"
    status = {"target": target, "usable_replicates": usable, "failed_replicates": len(failed),
              "status": run_label, "stability_evidence_complete": usable >= target}
    (rdir / "bootstrap_status.json").write_text(
        __import__("json").dumps(status, indent=2), encoding="utf-8")
    if not files:
        return status

    boot = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    base_assign = pd.read_csv(rdir / "cluster_assignments.csv")
    countries = sorted(set(base_assign.country) & set(boot.country))
    ari_rows=[]; jac_rows=[]; co_by_k={}
    for k in sorted(boot.k.unique()):
        base=(base_assign[base_assign.k.eq(k)].set_index("country")
              .reindex(countries).profile_id.str.removeprefix("P").astype(int).to_numpy())
        co=np.zeros((len(countries),len(countries)),dtype=float); used=0
        for rep,g in boot[boot.k.eq(k)].groupby("replicate"):
            labels=g.set_index("country").reindex(countries).profile_id
            if labels.isna().any(): continue
            lab=labels.astype(int).to_numpy(); used += 1
            ari_rows.append({"replicate":int(rep),"k":int(k),
                             "adjusted_rand_index":adjusted_rand_score(base,lab),"run_status":run_label})
            for group,jaccard in _best_match_jaccard(base,lab):
                jac_rows.append({"replicate":int(rep),"k":int(k),"base_profile_id":f"P{group}",
                                 "best_match_jaccard":jaccard,"run_status":run_label})
            co += (lab[:,None] == lab[None,:])
        if used:
            co /= used; co_by_k[int(k)] = co
            pd.DataFrame(co,index=countries,columns=countries).reset_index(names="country").to_csv(
                rdir / f"coclustering_k{int(k)}.csv",index=False,encoding="utf-8-sig")
    ari=pd.DataFrame(ari_rows); jac=pd.DataFrame(jac_rows)
    ari.to_csv(rdir/"bootstrap_ari.csv",index=False,encoding="utf-8-sig")
    jac.to_csv(rdir/"bootstrap_cluster_jaccard.csv",index=False,encoding="utf-8-sig")
    summary=(ari.groupby("k").adjusted_rand_index.agg(["count","mean","median","std","min","max"])
             .reset_index().rename(columns={"count":"usable_replicates","mean":"mean_ari","median":"median_ari","std":"sd_ari","min":"min_ari","max":"max_ari"}))
    jsummary=(jac.groupby("k").best_match_jaccard.agg(["mean","min"])
              .reset_index().rename(columns={"mean":"mean_cluster_jaccard","min":"minimum_replicate_cluster_jaccard"}))
    cluster_means=(jac.groupby(["k","base_profile_id"]).best_match_jaccard.mean().reset_index()
                   .groupby("k").best_match_jaccard.min().rename("minimum_mean_cluster_jaccard").reset_index())
    summary=summary.merge(jsummary,on="k",how="left").merge(cluster_means,on="k",how="left")
    summary["all_clusters_mean_jaccard_ge_0_60"]=summary.minimum_mean_cluster_jaccard.ge(.60)
    summary["run_status"]=run_label; summary.to_csv(rdir/"bootstrap_stability_summary.csv",index=False,encoding="utf-8-sig")

    _plot_setup(cache); import matplotlib.pyplot as plt
    ks=sorted(ari.k.unique()); fig,axes=plt.subplots(1,2,figsize=(10.5,4.0))
    axes[0].boxplot([ari.loc[ari.k.eq(k),"adjusted_rand_index"] for k in ks],tick_labels=ks,showmeans=True)
    axes[0].set_ylim(-.05,1.02)
    axes[0].set_xlabel("Number of clusters $k$"); axes[0].set_ylabel("Adjusted Rand index")
    axes[0].set_title("(a) Whole-partition recovery")
    for profile,g in jac.groupby("base_profile_id"):
        means=g.groupby("k").best_match_jaccard.mean().reindex(ks)
        axes[1].plot(ks,means,"o-",label=profile)
    axes[1].axhline(.60,color="#777777",ls=":",lw=1,label="0.60 reference")
    axes[1].set_ylim(0,1.02); axes[1].set_xticks(ks); axes[1].set_xlabel("Number of clusters $k$")
    axes[1].set_ylabel("Mean best-match Jaccard"); axes[1].set_title("(b) Base-profile recovery")
    axes[1].legend(fontsize=6,ncol=2)
    fig.suptitle(f"Speech-within-country bootstrap stability ({usable}/{target} usable) — {run_label}")
    fig.tight_layout(); fig.savefig(rdir/"fig_cluster_stability.png",dpi=220,bbox_inches="tight"); plt.close(fig)

    diag=pd.read_csv(rdir/"cluster_k_diagnostics.csv"); chosen=int(diag.loc[diag.silhouette.idxmax(),"k"])
    if chosen in co_by_k:
        co=co_by_k[chosen]; base=(base_assign[base_assign.k.eq(chosen)].set_index("country")
                                 .reindex(countries).profile_id.str.removeprefix("P").astype(int).to_numpy())
        order=np.lexsort((-co.sum(axis=1),base)); mat=co[np.ix_(order,order)]; ordered=[countries[i] for i in order]
        fig,ax=plt.subplots(figsize=(9.0,8.0)); im=ax.imshow(mat,cmap="cividis",vmin=0,vmax=1)
        ax.set_xticks(range(len(ordered)),ordered,rotation=90,fontsize=5.5); ax.set_yticks(range(len(ordered)),ordered,fontsize=5.5)
        fig.colorbar(im,ax=ax,label="Co-clustering probability")
        ax.set_title(f"Co-clustering at silhouette-selected k={chosen}\n{usable}/{target} usable refits — {run_label}")
        fig.tight_layout(); fig.savefig(rdir/"fig_coclustering_heatmap.png",dpi=220,bbox_inches="tight"); plt.close(fig)
    return status


TABLE_SPECS: dict[str, tuple[list[str] | None, str]] = {
    "audit/selected_runs.csv": (None, "Selected annotation runs and source provenance"),
    "audit/quality_issues.csv": (None, "Data quality checks"),
    "audit/country_eligibility.csv": (["country","unique_paragraphs","speeches","included_primary","exclusion_reasons_primary","included_sensitivity"], "Country eligibility under primary and sensitivity thresholds"),
    "audit/coverage_overall.csv": (None, "Evaluation coverage by codebook and dimension"),
    "audit/joint_coverage.csv": (None, "Joint dimension-evaluation coverage"),
    "audit/source_reconciliation.csv": (None, "Source reconciliation"),
    "audit/dependency_blockers.csv": (None, "Dependency and blocker report"),
    "rq1_provisional/prevalence.csv": (["codebook_id","dimension","code","positive_units","evaluable_units","missing_units","paragraph_prevalence","cluster_boot_ci_low","cluster_boot_ci_high","speech_weighted_mean"], "Provisional Galtung-code prevalence"),
    "rq1_provisional/cooccurrence.csv": (["feature_a","feature_b","joint_n","joint_positive","joint_prevalence","phi","lift","sparse"], "Provisional pairwise co-occurrence statistics"),
    "rq2_provisional/model_diagnostics.csv": (["feature","n_units","n_speeches","n_meetings","n_countries","positives","status","converged","country_p","country_p_bh"], "Mixed-model diagnostics and approximate omnibus tests"),
    "rq2_provisional/planned_contrasts.csv": (["feature","contrast","difference","ci_low","ci_high","p_approx","p_approx_bh"], "Prespecified provisional probability contrasts"),
    "rq2_provisional/sensitivity_model_diagnostics.csv": (["feature","variant","status","converged","n_units","country_p"], "RQ2 model-specification sensitivities"),
    "rq3_provisional/cluster_k_diagnostics.csv": (None, "Ward clustering diagnostics across k"),
    "rq3_provisional/cluster_assignments.csv": (None, "Neutral profile assignments across k"),
    "rq3_provisional/pca_explained_variance.csv": (None, "PCA explained variance"),
    "rq3_provisional/profile_exclusions.csv": (None, "Features or countries excluded from adjusted profiles"),
    "rq3_provisional/representative_deviant_cases.csv": (None, "Representative and deviant countries by centroid distance"),
    "rq3_provisional/knn_edges.csv": (["country_a","country_b","euclidean_distance","similarity","knn_k"], "Supplementary k-nearest-neighbor similarity edges"),
    "rq3_provisional/light_stability_summary.csv": (None, "Model-draw and descriptive speech-resampling sensitivity summaries"),
}


def export_latex(root: Path, output: Path, decimals: int = 3) -> list[Path]:
    """Export curated analysis tables, excluding raw/text-bearing datasets."""
    output.mkdir(parents=True, exist_ok=True); written=[]; inputs=[]
    for rel,(columns,caption) in TABLE_SPECS.items():
        src=root/rel
        if not src.exists() or src.stat().st_size==0: continue
        df=pd.read_csv(src)
        if columns is not None: df=df[[c for c in columns if c in df.columns]]
        # Repeated bootstrap rows are useful as CSV, but unwieldy in a paper; summarize them.
        if rel.endswith("bootstrap_cluster_jaccard.csv") and len(df):
            df=(df.groupby(["k","base_profile_id","run_status"]).best_match_jaccard
                .agg(["count","mean","std","min","max"]).reset_index())
        target=output/Path(rel).with_suffix(".tex"); target.parent.mkdir(parents=True,exist_ok=True)
        label="tab:"+Path(rel).with_suffix("").as_posix().replace("/","-").replace("_","-")
        latex=df.to_latex(index=False,longtable=len(df)>45,escape=True,
                          float_format=lambda x:f"{x:.{decimals}f}",caption=f"{caption} ({STATUS}).",label=label,
                          na_rep="--")
        target.write_text("% Generated; do not hand-edit. Requires booktabs and longtable.\n"+latex,encoding="utf-8")
        written.append(target); inputs.append(target.relative_to(output).as_posix())
    master=output/"research_tables.tex"
    master.write_text("% Add \\usepackage{booktabs,longtable} to the preamble.\n"+
                      "\n".join(f"\\input{{{p}}}" for p in inputs)+"\n",encoding="utf-8")
    compact_dir = output / "paper"
    compact_dir.mkdir(parents=True, exist_ok=True)
    compact: list[tuple[str, pd.DataFrame, str]] = []
    if (root / "rq1_provisional/prevalence.csv").exists():
        d = pd.read_csv(root / "rq1_provisional/prevalence.csv")
        compact.append(("rq1_prevalence", pd.DataFrame({
            "Code": d.code,
            "Share": d.paragraph_prevalence.map(lambda x: f"{100*x:.1f}\\%"),
            "95\\% CI": [f"{100*a:.1f}--{100*b:.1f}\\%" for a,b in zip(d.cluster_boot_ci_low,d.cluster_boot_ci_high)],
            "Evaluated": d.evaluable_units.map(lambda x: f"{int(x):,}"),
        }), "Galtung code prevalence among eligible annotated paragraphs"))
    if (root / "rq2_provisional/model_diagnostics.csv").exists():
        d = pd.read_csv(root / "rq2_provisional/model_diagnostics.csv")
        compact.append(("rq2_models", pd.DataFrame({
            "Code": d.feature.str.split("__").str[-1],
            "Countries": d.n_countries.map(lambda x: "--" if pd.isna(x) else str(int(x))),
            "Paragraphs": d.n_units.map(lambda x: "--" if pd.isna(x) else f"{int(x):,}"),
            "Converged": d.converged.astype(str).str.lower().eq("true").map({True:"yes",False:"no"}),
        }), "Scoped country-model diagnostics"))
    if (root / "rq3_provisional/cluster_k_diagnostics.csv").exists():
        d = pd.read_csv(root / "rq3_provisional/cluster_k_diagnostics.csv")
        small = pd.DataFrame({
            "$k$": d.k.astype(int), "Silhouette": d.silhouette.map(lambda x:f"{x:.3f}"),
        })
        if "min_cluster_size" in d:
            small["Smallest group"] = d.min_cluster_size.astype(int)
        if "largest_cluster_share" in d:
            small["Largest share"] = d.largest_cluster_share.map(lambda x:f"{100*x:.1f}\\%")
        compact.append(("rq3_clusters", small, "Ward profile diagnostics"))
    if (root / "rq3_provisional/light_stability_summary.csv").exists():
        d = pd.read_csv(root / "rq3_provisional/light_stability_summary.csv")
        compact.append(("rq3_light_stability", pd.DataFrame({
            "Check": d.method.map({"mean_field_model_draws":"Model draws",
                                    "speech_resampled_descriptive":"Speech sample"}),
            "$k$": d.k.astype(int), "Runs": d.replicates.astype(int),
            "Mean ARI": d.mean_ari.map(lambda x:f"{x:.3f}"),
            "Min. mean Jaccard": d.minimum_cluster_mean_jaccard.map(lambda x:f"{x:.3f}"),
        }), "Two distinct profile sensitivity checks"))
    for stem, frame, caption in compact:
        target = compact_dir / f"{stem}.tex"
        latex = frame.to_latex(index=False, escape=False, caption=f"{caption} ({STATUS}).",
                               label=f"tab:{stem.replace('_','-')}")
        # These compact summaries fit a sigconf column at the class font size.
        latex = latex.replace("\\begin{table}", "\\begin{table}[!ht]\n\\centering", 1)
        target.write_text("% Generated compact two-column ACM table.\n" + latex, encoding="utf-8")
        written.append(target)
    figures=[]
    for name,caption,label in [
        ("fig_cluster_diagnostics.png","Silhouette, balance, and country-level cluster diagnostics.","fig:cluster-diagnostics"),
        ("dendrogram.png","Ward dendrogram on adjusted Galtung profiles.","fig:dendrogram"),
        ("fig_knn_similarity_network.png","Supplementary local country-similarity network.","fig:knn-network"),
    ]:
        if (root/"rq3_provisional"/name).exists():
            figures.append("\\begin{figure}[htbp]\n\\centering\n"
                f"\\includegraphics[width=0.98\\linewidth]{{rq3_provisional/{name}}}\n"
                f"\\caption{{{caption} {STATUS}.}}\n\\label{{{label}}}\n\\end{{figure}}")
    (output/"research_figures.tex").write_text(
        "% Add \\usepackage{graphicx} to the preamble. Paths assume research_outputs is the Overleaf project root.\n\n"
        +"\n\n".join(figures)+"\n",encoding="utf-8")
    readme=output/"README.md"
    readme.write_text("Copy this directory into Overleaf and add `\\input{research_tables.tex}`. "
                      "Raw paragraph/evidence files are intentionally not converted to LaTeX.\n",encoding="utf-8")
    return [*written,master,output/"research_figures.tex",readme]
