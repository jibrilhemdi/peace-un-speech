"""Step 3: Ward clustering of CLR fingerprints, k selection, bootstrap stability."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.metrics import silhouette_samples, silhouette_score

from .config import CONFIG
from .fingerprints import clr, unit_code_matrix
from .io_utils import LOG, cluster_colors, iso3, save_fig, set_style, write_data, write_table


def ward_labels(X: pd.DataFrame, k: int) -> tuple[np.ndarray, np.ndarray]:
    Z = linkage(X.to_numpy(), method="ward", metric="euclidean")
    return fcluster(Z, t=k, criterion="maxclust"), Z


def choose_k(X: pd.DataFrame, k_range: list[int], units=None, ann=None,
             dim_codes=None, b_grid: int = 0,
             seed: int = 0) -> tuple[int, pd.DataFrame, str]:
    """Select k on silhouette, restricted to solutions that survive the bootstrap.

    Silhouette alone almost always crowns the coarsest split, and it is blind to
    whether the finer partitions it rejects are merely less separated or actually
    irreproducible. Screening candidates on cluster-wise Jaccard first, then
    maximising silhouette among the survivors, answers both questions.
    """
    Z = linkage(X.to_numpy(), method="ward", metric="euclidean")
    rows = []
    for k in k_range:
        lab = fcluster(Z, t=k, criterion="maxclust")
        sizes = np.bincount(lab)[1:]
        row = {"k": k,
               "silhouette": silhouette_score(X.to_numpy(), lab, metric="euclidean"),
               "min_cluster_size": int(sizes.min()),
               "n_singletons": int((sizes == 1).sum()),
               "largest_share": sizes.max() / sizes.sum()}
        if b_grid:
            st, _ = bootstrap_stability(units, ann, dim_codes, list(X.index), k, lab,
                                        b_grid, seed)
            row["min_jaccard"] = float(st.mean_jaccard.min())
            row["mean_jaccard"] = float(st.mean_jaccard.mean())
            row["n_unstable"] = int((st.mean_jaccard < 0.6).sum())
        rows.append(row)
    diag = pd.DataFrame(rows)

    if b_grid:
        ok = diag[(diag.n_unstable == 0) & (diag.n_singletons == 0)]
        if len(ok):
            best = int(ok.loc[ok.silhouette.idxmax(), "k"])
            rejected = diag[diag.n_unstable > 0]
            rule = (f"k={best} maximises the mean silhouette ({ok.silhouette.max():.3f}) "
                    f"among the {len(ok)} solution(s) in which every cluster survives the "
                    f"bootstrap (all cluster-wise mean Jaccard >= 0.60, {b_grid} resamples)")
            if len(rejected):
                rule += ("; rejected " + ", ".join(
                    f"k={int(r.k)} (silhouette {r.silhouette:.3f}, "
                    f"{r.n_unstable} unstable cluster(s), min Jaccard {r.min_jaccard:.2f})"
                    for r in rejected.itertuples()))
        else:
            best = int(diag.loc[diag.silhouette.idxmax(), "k"])
            rule = (f"no k had a fully stable partition; fell back to the silhouette "
                    f"maximum at k={best}")
    else:
        best = int(diag.loc[diag.silhouette.idxmax(), "k"])
        rule = f"silhouette maximum at k={best}"
    return best, diag, rule


# ------------------------------------------------------------- bootstrap ----

def _fast_counts(ucm_np: np.ndarray, rows_by_country: dict, rng) -> np.ndarray:
    out = np.empty((len(rows_by_country), ucm_np.shape[1]))
    for i, idx in enumerate(rows_by_country.values()):
        pick = rng.integers(0, len(idx), size=len(idx))
        out[i] = ucm_np[idx[pick]].sum(axis=0)
    return out


def bootstrap_stability(units, ann, dim_codes, countries, k, base_labels, B, seed):
    """Hennig-style cluster-wise Jaccard plus a co-clustering matrix.

    Paragraphs are resampled with replacement *within* each country, so the
    country set is fixed and only fingerprint noise is perturbed.
    """
    ucm = unit_code_matrix(units, ann, dim_codes)
    cols = list(ucm.columns)
    ucm_np = ucm.to_numpy(dtype=float)
    pos = {u: i for i, u in enumerate(ucm.index)}
    rows_by_country = {
        c: np.array([pos[u] for u in g.unit_id], dtype=int)
        for c, g in units[units.country.isin(countries)].groupby("country")
    }
    rows_by_country = {c: rows_by_country[c] for c in countries}

    rng = np.random.default_rng(seed)
    n = len(countries)
    co = np.zeros((n, n))
    base_sets = {g: set(np.where(base_labels == g)[0]) for g in np.unique(base_labels)}
    jac = {g: [] for g in base_sets}

    for _ in range(B):
        cnt = pd.DataFrame(_fast_counts(ucm_np, rows_by_country, rng),
                           index=list(countries), columns=cols)
        Xb = clr(cnt, dim_codes)
        lab, _ = ward_labels(Xb, k)
        co += (lab[:, None] == lab[None, :])
        for g, s in base_sets.items():
            best = 0.0
            for h in np.unique(lab):
                t = set(np.where(lab == h)[0])
                inter = len(s & t)
                if inter:
                    best = max(best, inter / len(s | t))
            jac[g].append(best)

    co /= B
    stab = pd.DataFrame({
        "cluster": list(base_sets),
        "n_countries": [len(base_sets[g]) for g in base_sets],
        "mean_jaccard": [float(np.mean(jac[g])) for g in base_sets],
        "sd_jaccard": [float(np.std(jac[g])) for g in base_sets],
        "p_recovered_0.75": [float(np.mean(np.array(jac[g]) >= 0.75)) for g in base_sets],
    })
    stab["stable"] = stab.mean_jaccard >= 0.6
    return stab, pd.DataFrame(co, index=list(countries), columns=list(countries))


# --------------------------------------------------------------- figures ----

def fig_dendrogram(X, labels, k, colors, path="fig_dendrogram.png"):
    set_style()
    Z = linkage(X.to_numpy(), method="ward", metric="euclidean")
    order_thresh = Z[-(k - 1), 2] if k > 1 else 0
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    hierarchy.set_link_color_palette(list(colors))
    dendrogram(Z, labels=[iso3(c) for c in X.index], ax=ax, leaf_font_size=6,
               color_threshold=order_thresh, above_threshold_color=CONFIG["GREY"])
    ax.axhline(order_thresh, color="black", lw=0.7, ls="--", alpha=0.7)
    ax.text(0.995, order_thresh, f" cut at k={k}", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=6.5)
    ax.set_ylabel("Ward linkage distance (Euclidean on CLR)")
    ax.set_title("Member states clustered on Galtung-code fingerprints")
    ax.grid(False)
    hierarchy.set_link_color_palette(None)
    return save_fig(fig, path)


def fig_diagnostics(diag, X, labels, k, colors, path="fig_cluster_diagnostics.png"):
    set_style()
    ncol = 3 if "min_jaccard" in diag.columns else 2
    fig, axes = plt.subplots(1, ncol, figsize=(2.55 * ncol + 1.0, 3.0))
    ax = axes[0]
    ax.plot(diag.k, diag.silhouette, "o-", color=CONFIG["PALETTE"][0], ms=4)
    ax.axvline(k, color=CONFIG["PALETTE"][1], ls="--", lw=1)
    ax.annotate(f"chosen k={k}", (k, diag.silhouette.max()), xytext=(4, -8),
                textcoords="offset points", fontsize=7, color=CONFIG["PALETTE"][1])
    ax.set_xlabel("number of clusters $k$"); ax.set_ylabel("mean silhouette width")
    ax.set_title("(a) Silhouette across $k$")

    if ncol == 3:
        ax = axes[1]
        ax.plot(diag.k, diag.min_jaccard, "o-", color=CONFIG["PALETTE"][2], ms=4,
                label="weakest cluster")
        ax.plot(diag.k, diag.mean_jaccard, "s--", color=CONFIG["PALETTE"][4], ms=3.5,
                label="mean over clusters")
        ax.axhline(0.6, color=CONFIG["GREY"], ls=":", lw=1)
        ax.text(diag.k.max(), 0.61, "stability floor 0.60", ha="right", va="bottom",
                fontsize=6, color=CONFIG["GREY"])
        ax.axvline(k, color=CONFIG["PALETTE"][1], ls="--", lw=1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("number of clusters $k$"); ax.set_ylabel("bootstrap Jaccard")
        ax.set_title("(b) Cluster-wise stability")
        ax.legend(loc="lower left", fontsize=6)

    ax = axes[ncol - 1]
    sil = silhouette_samples(X.to_numpy(), labels, metric="euclidean")
    y = 0
    for i, g in enumerate(np.unique(labels)):
        v = np.sort(sil[labels == g])
        ax.barh(np.arange(y, y + len(v)), v, height=1.0, color=colors[i],
                edgecolor="none", label=f"C{g} (n={len(v)})")
        y += len(v) + 3
    ax.axvline(sil.mean(), color="black", ls="--", lw=0.8)
    ax.set_xlabel("silhouette width"); ax.set_yticks([])
    ax.set_title(f"({'c' if ncol == 3 else 'b'}) Per-country silhouette at $k$={k}")
    ax.legend(loc="lower right", fontsize=6)
    fig.tight_layout()
    return save_fig(fig, path)


def fig_coclustering(co, labels, colors, path="fig_coclustering_heatmap.png"):
    set_style()
    order = np.lexsort((-co.to_numpy().sum(axis=1), labels))
    M = co.to_numpy()[np.ix_(order, order)]
    names = [iso3(co.index[i]) for i in order]
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    im = ax.imshow(M, cmap="cividis", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=90, fontsize=4.4)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=4.4)
    b = 0
    for i, g in enumerate(np.unique(labels)):
        s = int((labels == g).sum())
        ax.add_patch(plt.Rectangle((b - .5, b - .5), s, s, fill=False,
                                   edgecolor=colors[i], lw=1.6))
        b += s
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label(f"co-clustering frequency over {CONFIG['BOOTSTRAP_B']} bootstraps",
                 fontsize=7)
    ax.set_title("Bootstrap co-clustering of member states\n"
                 "(paragraphs resampled within country; boxes = main solution)")
    return save_fig(fig, path)


# ------------------------------------------------------------------- run ----

def run(corpus, fp) -> dict:
    print("\n[Step 3] clustering")
    X = fp["clr"]
    k, diag, rule = choose_k(
        X, CONFIG["K_RANGE"], units=corpus.member_states, ann=corpus.ann,
        dim_codes=corpus.dim_codes, b_grid=max(25, CONFIG["BOOTSTRAP_B"] // 4),
        seed=CONFIG["SEED"])
    labels, Z = ward_labels(X, k)
    colors = cluster_colors(k)
    LOG.log("Step 3", f"Chose k={k} clusters: {rule}.",
            "Silhouette and stability curves in fig_cluster_diagnostics.png; the full "
            "grid is in table_k_selection.tex.")

    assign = pd.DataFrame({
        "country": X.index, "iso3": [iso3(c) for c in X.index],
        "cluster": labels, "n_paragraphs": fp["n"].reindex(X.index).to_numpy(),
    }).sort_values(["cluster", "n_paragraphs"], ascending=[True, False])

    stab, co = bootstrap_stability(
        corpus.member_states, corpus.ann, corpus.dim_codes, list(X.index), k, labels,
        CONFIG["BOOTSTRAP_B"], CONFIG["SEED"])
    unstable = stab[~stab.stable]
    LOG.log("Step 3",
            "Bootstrap stability (" + f"{CONFIG['BOOTSTRAP_B']} resamples): mean Jaccard "
            + ", ".join(f"C{r.cluster}={r.mean_jaccard:.2f}" for r in stab.itertuples())
            + (f". UNSTABLE (<0.60): {', '.join('C'+str(c) for c in unstable.cluster)}."
               if len(unstable) else ". All clusters at or above 0.60."))

    fig_dendrogram(X, labels, k, colors)
    fig_diagnostics(diag, X, labels, k, colors)
    fig_coclustering(co, labels, colors)

    write_data(assign, "cluster_assignments.csv")
    write_data(co.reset_index().rename(columns={"index": "country"}),
               "coclustering_matrix.csv")

    memb = (assign.groupby("cluster")
            .agg(**{"Countries": ("iso3", "size"),
                    "Paragraphs": ("n_paragraphs", "sum"),
                    "Members (ISO3)": ("iso3", lambda s: ", ".join(sorted(s)))})
            .reset_index().rename(columns={"cluster": "Cluster"}))
    memb["Cluster"] = "C" + memb.Cluster.astype(str)
    write_table(memb, "table_cluster_membership.tex",
                caption=f"Cluster membership at $k$={k}. Ward linkage on Euclidean "
                        "distances between CLR-transformed code fingerprints.",
                label="tab:cluster_membership",
                notes="Population: member states in national capacity clearing the pooled "
                      f"floor of {CONFIG['MIN_PARAGRAPHS_POOLED']} paragraphs. Israel and "
                      "the State of Palestine are computed as reference points and are "
                      "never clustered.",
                align="l" + "r" * 2 + "p{7.2cm}", float_fmt="{:.0f}")

    # Panel A: the chosen solution, cluster by cluster. Panel B: the k grid.
    st = stab.copy()
    st.insert(0, "Cluster", "C" + st.cluster.astype(str))
    st = st.drop(columns=["cluster"])
    st.columns = ["Cluster", "Countries", "Mean Jaccard", "SD", "P(J >= 0.75)", "Stable"]
    st["Stable"] = np.where(st["Stable"], "yes", "NO")

    dg = diag.copy()
    dg["Selected"] = np.where(dg.k == k, "yes", "")
    keep = ["k", "silhouette", "min_cluster_size", "min_jaccard", "mean_jaccard",
            "n_unstable", "Selected"]
    dg = dg[[c for c in keep if c in dg.columns]]
    dg.columns = ["k", "Silhouette", "Min size", "Min Jaccard", "Mean Jaccard",
                  "Unstable clusters", "Selected"][:len(dg.columns)]

    write_table(st, "table_cluster_stability.tex",
                caption=f"Bootstrap stability of the pooled clustering at $k$={k}. "
                        "Paragraphs are resampled with replacement within each country "
                        f"({CONFIG['BOOTSTRAP_B']} resamples); fingerprints, the CLR "
                        "transform and the Ward solution are recomputed each time.",
                label="tab:cluster_stability",
                notes="Mean Jaccard is the average best-match overlap between the original "
                      "cluster and its closest bootstrap counterpart. Below 0.60 marks a "
                      "cluster as unstable; 0.60--0.75 indicates a pattern present but not "
                      "well separated. Selection grid in Table~\\ref{tab:k_selection}.")
    write_table(dg, "table_k_selection.tex",
                caption="Choice of $k$. Silhouette measures separation; the bootstrap "
                        "columns measure reproducibility. $k$ is chosen as the "
                        "silhouette maximum among partitions in which every cluster is "
                        "reproducible.",
                label="tab:k_selection",
                notes=f"Stability columns use {max(25, CONFIG['BOOTSTRAP_B'] // 4)} "
                      "resamples for the grid and "
                      f"{CONFIG['BOOTSTRAP_B']} for the selected solution. A cluster counts "
                      "as unstable when its mean Jaccard falls below 0.60.")

    return {"k": k, "labels": labels, "diag": diag, "assign": assign, "stability": stab,
            "coclustering": co, "colors": colors, "linkage": Z, "rule": rule}
