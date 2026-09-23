"""Step 5: per-phase clustering, label alignment, movement, and trend figures."""

from __future__ import annotations

import textwrap

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from .clustering import choose_k, ward_labels
from .config import CONFIG
from .fingerprints import build_fingerprints, clr, outcome_flags
from .io_utils import LOG, iso3, save_fig, set_style, write_data, write_table


def phase_clusterings(corpus) -> dict:
    """Re-run Steps 1-3 inside each phase, at the per-phase country floor."""
    out = {}
    ms = corpus.member_states
    for name in corpus.phases:
        sub = ms[ms.phase == name]
        shares, counts, n = build_fingerprints(
            sub, corpus.ann, corpus.dim_codes, CONFIG["MIN_PARAGRAPHS_PHASE"])
        if len(n) < max(CONFIG["K_RANGE"]) + 1:
            LOG.log("Step 5", f"Phase `{name}`: only {len(n)} countries clear the phase "
                              "floor; phase clustering skipped.")
            continue
        X = clr(counts, corpus.dim_codes)
        krange = [k for k in CONFIG["K_RANGE"] if k < len(n)]
        k, diag, rule = choose_k(X, krange, units=sub, ann=corpus.ann,
                                 dim_codes=corpus.dim_codes,
                                 b_grid=max(25, CONFIG["BOOTSTRAP_B"] // 4),
                                 seed=CONFIG["SEED"])
        lab, _ = ward_labels(X, k)
        out[name] = {"X": X, "counts": counts, "shares": shares, "n": n,
                     "k": k, "labels": lab, "diag": diag, "rule": rule}
        LOG.log("Step 5",
                f"Phase `{name}`: {len(n)} countries above the floor of "
                f"{CONFIG['MIN_PARAGRAPHS_PHASE']} paragraphs, k={k} ({rule.split(';')[0]}).")
    return out


def align_to_pooled(phase_res: dict, pooled_X: pd.DataFrame,
                    pooled_labels: np.ndarray) -> dict:
    """Map each phase cluster onto the pooled cluster with the nearest centroid.

    Hungarian assignment gives the one-to-one backbone; any surplus phase cluster
    (k_phase > k_pooled) then falls to its nearest pooled centroid, so every phase
    cluster carries a pooled label and trajectories stay readable.
    """
    pooled_cent = pd.DataFrame(
        {g: pooled_X[pooled_labels == g].mean(axis=0) for g in np.unique(pooled_labels)}).T
    log_rows = []
    for name, r in phase_res.items():
        cent = pd.DataFrame(
            {g: r["X"][r["labels"] == g].mean(axis=0) for g in np.unique(r["labels"])}).T
        D = np.linalg.norm(cent.to_numpy()[:, None, :] - pooled_cent.to_numpy()[None], axis=2)
        rows, cols = linear_sum_assignment(D)
        mapping = {int(cent.index[i]): int(pooled_cent.index[j])
                   for i, j in zip(rows, cols)}
        for i, g in enumerate(cent.index):
            if int(g) not in mapping:
                j = int(np.argmin(D[i]))
                mapping[int(g)] = int(pooled_cent.index[j])
                log_rows.append(f"`{name}` cluster {g} unmatched by Hungarian assignment; "
                                f"fell back to nearest pooled centroid C{mapping[int(g)]}")
        r["mapping"] = mapping
        r["aligned"] = np.array([mapping[int(g)] for g in r["labels"]])
        dist = ", ".join(f"{g}->C{mapping[g]} (d={D[i, list(pooled_cent.index).index(mapping[g])]:.2f})"
                         for i, g in enumerate(cent.index))
        log_rows.append(f"`{name}`: {dist}")
    LOG.log("Step 5", "Phase-to-pooled centroid matching: " + " | ".join(log_rows),
            "Cost is the Euclidean distance between centroids in the shared CLR feature "
            "space, so phase and pooled centroids are directly comparable.")
    return phase_res


def trajectories(corpus, phase_res: dict, pooled_assign: pd.DataFrame) -> pd.DataFrame:
    order = list(corpus.phases)
    rows = []
    for c in sorted({c for r in phase_res.values() for c in r["n"].index}):
        row = {"country": c, "iso3": iso3(c)}
        pooled = pooled_assign.set_index("country").cluster
        row["pooled_cluster"] = int(pooled[c]) if c in pooled.index else np.nan
        for name in order:
            r = phase_res.get(name)
            if r is not None and c in r["n"].index:
                i = list(r["n"].index).index(c)
                row[name] = int(r["aligned"][i])
                row[f"{name}__n"] = int(r["n"][c])
            else:
                row[name] = np.nan
                row[f"{name}__n"] = 0
        present = [row[p] for p in order if pd.notna(row[p])]
        row["n_phases_present"] = len(present)
        row["is_mover"] = len(set(present)) > 1
        rows.append(row)
    return pd.DataFrame(rows)


def consecutive_ari(corpus, phase_res: dict) -> pd.DataFrame:
    order = [p for p in corpus.phases if p in phase_res]
    rows = []
    for a, b in zip(order, order[1:]):
        ra, rb = phase_res[a], phase_res[b]
        both = [c for c in ra["n"].index if c in set(rb["n"].index)]
        if len(both) < 3:
            continue
        la = [ra["aligned"][list(ra["n"].index).index(c)] for c in both]
        lb = [rb["aligned"][list(rb["n"].index).index(c)] for c in both]
        rows.append({"Phase A": a.replace("_", " "), "Phase B": b.replace("_", " "),
                     "Countries in both": len(both),
                     "ARI": adjusted_rand_score(la, lb),
                     "Same cluster": float(np.mean(np.array(la) == np.array(lb)))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------- figures ----

def fig_alluvial(corpus, traj, phase_res, colors, path="fig_alluvial_membership.png"):
    """Pure-matplotlib alluvial: phases left to right, clusters as blocks."""
    set_style()
    order = [p for p in corpus.phases if p in phase_res]
    if len(order) < 2:
        return None
    clusters = sorted({int(v) for p in order for v in traj[p].dropna().unique()})
    gap, block_w = 0.06, 0.16

    # Vertical layout: one stack of cluster blocks per phase, sized by membership.
    ypos, heights = {}, {}
    for p in order:
        counts = traj[p].value_counts()
        tot = counts.sum()
        y = 0.0
        for g in clusters:
            h = counts.get(g, 0) / tot * (1 - gap * (len(clusters) - 1)) if tot else 0
            ypos[(p, g)] = y
            heights[(p, g)] = h
            y += h + gap

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    xs = np.linspace(0, 1, len(order))
    for i, p in enumerate(order):
        for g in clusters:
            h = heights[(p, g)]
            if h <= 0:
                continue
            ax.add_patch(plt.Rectangle((xs[i] - block_w / 2, ypos[(p, g)]), block_w, h,
                                       facecolor=colors[clusters.index(g)], alpha=0.85,
                                       edgecolor="white", lw=0.8, zorder=3))
            ax.text(xs[i], ypos[(p, g)] + h / 2, f"C{g}\n{int(traj[p].eq(g).sum())}",
                    ha="center", va="center", fontsize=6.5, color="white",
                    fontweight="bold", zorder=4)

    # Ribbons: one cubic band per (source cluster -> target cluster) flow.
    for i in range(len(order) - 1):
        a, b = order[i], order[i + 1]
        sub = traj.dropna(subset=[a, b])
        off_a = {g: 0.0 for g in clusters}
        off_b = {g: 0.0 for g in clusters}
        flows = (sub.groupby([a, b]).size()
                 .reset_index(name="n").sort_values([a, "n"], ascending=[True, False]))
        tot_a = sub[a].value_counts()
        tot_b = sub[b].value_counts()
        # Index by column name: itertuples only renames invalid identifiers, so
        # positional access breaks whenever a phase name happens to be valid.
        for _, r in flows.iterrows():
            ga, gb, nflow = int(r[a]), int(r[b]), int(r["n"])
            ha = heights[(a, ga)] * nflow / tot_a[ga]
            hb = heights[(b, gb)] * nflow / tot_b[gb]
            y0, y1 = ypos[(a, ga)] + off_a[ga], ypos[(b, gb)] + off_b[gb]
            off_a[ga] += ha
            off_b[gb] += hb
            x0, x1 = xs[i] + block_w / 2, xs[i + 1] - block_w / 2
            t = np.linspace(0, 1, 60)
            s = 3 * t ** 2 - 2 * t ** 3
            X = x0 + (x1 - x0) * t
            ax.fill_between(X, y0 + (y1 - y0) * s, y0 + ha + (y1 + hb - y0 - ha) * s,
                            color=colors[clusters.index(ga)],
                            alpha=0.42 if ga == gb else 0.72, lw=0,
                            zorder=2 if ga == gb else 3,
                            hatch=None if ga == gb else "///")

    ax.set_xticks(xs)
    ax.set_xticklabels([textwrap.fill(p.replace("_", " "), 18) for p in order],
                       fontsize=6.5)
    ax.set_yticks([])
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(-0.12, 1.12)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Cluster membership across phases\n"
                 "block height = countries in that cluster; hatched ribbons = movers")
    h = [plt.Rectangle((0, 0), 1, 1, fc=colors[clusters.index(g)], alpha=.85)
         for g in clusters]
    ax.legend(h, [f"pooled type C{g}" for g in clusters], loc="upper center",
              bbox_to_anchor=(0.5, -0.11), ncol=len(clusters), fontsize=7)
    ax.text(0.5, -0.22, "Countries below the per-phase floor are absent from a phase, so "
            "ribbon totals need not equal block heights.", transform=ax.transAxes,
            ha="center", fontsize=6, color=CONFIG["GREY"])
    return save_fig(fig, path)


def fig_code_trends(corpus, assign, colors, path="fig_code_trends.png"):
    """Monthly outcome shares, overall and by pooled cluster."""
    set_style()
    ms = corpus.member_states.copy()
    cl = assign.set_index("country").cluster
    ms["cluster"] = ms.country.map(cl)
    span = corpus.model_span
    outcomes = CONFIG["OUTCOMES"]

    fig, axes = plt.subplots(len(outcomes), 1, figsize=(7.2, 2.35 * len(outcomes)),
                             sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (oname, spec) in zip(axes, outcomes.items()):
        restrict = spec["dim"] == CONFIG["MODEL_SPLIT_DIM"] and span["split"]
        d = ms.copy()
        fl = outcome_flags(d, corpus.ann, spec)
        d = d.merge(fl, on="unit_id")
        d = d[d.in_denom]
        if restrict:
            d = d[d.in_model_span]
        d["month"] = d.date.values.astype("datetime64[M]")

        overall = d.groupby("month").y.agg(["mean", "size"])
        overall = overall[overall["size"] >= 20]
        ax.plot(overall.index, overall["mean"], color="black", lw=1.6, zorder=4,
                label="all member states")
        for i, g in enumerate(sorted(cl.unique())):
            s = d[d.cluster == g].groupby("month").y.agg(["mean", "size"])
            s = s[s["size"] >= 15]
            ax.plot(s.index, s["mean"], color=colors[i], lw=1.2, alpha=0.9,
                    marker="o", ms=2.4, label=f"cluster C{g}")

        for name, (a, b) in corpus.phases.items():
            ax.axvline(a, color=CONFIG["GREY"], ls=":", lw=0.9, zorder=1)
        for ev, dt in CONFIG["EVENTS"].items():
            ax.axvline(pd.Timestamp(dt), color=CONFIG["PALETTE"][1], ls="--", lw=1.0,
                       zorder=2)
            ax.text(pd.Timestamp(dt), 1.005, ev.split("_")[0], transform=
                    ax.get_xaxis_transform(), fontsize=6, color=CONFIG["PALETTE"][1],
                    ha="center", va="bottom")

        if restrict:
            lo, hi = ms.date.min(), ms.date.max()
            for a, b in [(lo, span["start"]), (span["end"], hi)]:
                if a < b:
                    ax.axvspan(a, b, color=CONFIG["GREY"], alpha=0.20, lw=0, zorder=0)
            ax.text(0.995, 0.05, f"shaded = outside the {span['model'].split('/')[-1]} span",
                    transform=ax.transAxes, ha="right", fontsize=5.8,
                    color=CONFIG["GREY"])
        codes = "+".join(spec["codes"])
        ax.set_ylabel(f"{oname}\n({codes})", fontsize=7)
        ax.set_ylim(0, None)
        ax.margins(x=0.01)
    axes[0].legend(loc="upper left", ncol=3, fontsize=6.5)
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(axes[-1].get_xticklabels(), rotation=45, ha="right")
    axes[0].set_title("Monthly outcome shares, overall and by pooled cluster\n"
                      "dotted = phase boundaries, dashed orange = events")
    fig.tight_layout()
    return save_fig(fig, path)


# ------------------------------------------------------------------- run ----

def run(corpus, fp, cl) -> dict:
    print("\n[Step 5] phases and movement")
    pres = phase_clusterings(corpus)
    pres = align_to_pooled(pres, fp["clr"], cl["labels"])
    traj = trajectories(corpus, pres, cl["assign"])
    write_data(traj, "phase_trajectories.csv")

    ari = consecutive_ari(corpus, pres)
    if len(ari):
        write_table(ari, "table_phase_ari.tex",
                    caption="Adjusted Rand index between consecutive phase clusterings, "
                            "computed on the countries present in both phases after "
                            "alignment to the pooled typology.",
                    label="tab:phase_ari",
                    notes="ARI of 1 means identical partitions; 0 means the agreement "
                          "expected by chance. 'Same cluster' is the raw share of shared "
                          "countries keeping the same pooled label.")
        LOG.log("Step 5", "Consecutive-phase ARI: " + ", ".join(
            f"{r['Phase A']} -> {r['Phase B']}: {r['ARI']:.3f} "
            f"({r['Same cluster']:.0%} unchanged)"
            for _, r in ari.iterrows()))

    order = [p for p in corpus.phases if p in pres]
    movers = traj[traj.is_mover & (traj.n_phases_present >= 2)].copy()
    if len(movers):
        mt = pd.DataFrame({
            "Country": movers.country, "ISO3": movers.iso3,
            "Pooled": "C" + movers.pooled_cluster.astype("Int64").astype(str),
            **{p.replace("_", " "): movers[p].map(
                lambda v: "" if pd.isna(v) else f"C{int(v)}") for p in order},
            "Paragraphs": movers[[f"{p}__n" for p in order]].sum(axis=1).astype(int),
        }).sort_values("Paragraphs", ascending=False)
        write_table(mt, "table_phase_membership_movers.tex",
                    caption="Member states that change cluster between phases. Countries "
                            "holding one cluster throughout are omitted; full trajectories "
                            "are in outputs/data/phase\\_trajectories.csv.",
                    label="tab:phase_movers",
                    notes="An empty cell means the country fell below the per-phase floor "
                          f"of {CONFIG['MIN_PARAGRAPHS_PHASE']} paragraphs and is "
                          "unassigned for that phase. Phase labels are aligned to the "
                          "pooled typology by centroid matching.",
                    float_fmt="{:.0f}")
    n_present = int((traj.n_phases_present >= 2).sum())
    LOG.log("Step 5",
            f"{len(movers)} of {n_present} member states present in 2+ phases change "
            f"cluster at least once ({len(movers) / max(n_present, 1):.0%}).")

    fig_alluvial(corpus, traj, pres, cl["colors"])
    fig_code_trends(corpus, cl["assign"], cl["colors"])
    return {"phase_res": pres, "trajectories": traj, "ari": ari, "movers": movers}
