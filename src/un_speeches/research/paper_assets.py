"""Question-led, reproducible figures and tables for the ACM draft.

This module reads saved scoped analysis files. It does not fit models or annotate text.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from .research_outputs import _plot_setup


BLOCKS = ["positive_negative", "violence_type", "coupling_type", "relational", "transcend"]
BLOCK_NAMES = {"positive_negative": "Peace conception", "violence_type": "Violence",
               "coupling_type": "Coupling", "relational": "Relational",
               "transcend": "Conflict transformation"}
SHORT = {"PN0": "Ceasefire only", "PN1": "Ceasefire + protection",
         "PN2": "Transitional steps", "PN3": "Structural justice",
         "PN4": "Full cooperation", "VD": "Direct violence",
         "VS": "Structural violence", "VC": "Cultural violence",
         "AB": "No relationship", "ANTI": "Adversarial relation",
         "SYM": "Cooperative relation", "I0": "Full asymmetry",
         "I1": "Mostly asymmetric", "I2": "Negotiated balance",
         "I3": "Mostly equal", "I4": "Full equality",
         "T0": "Zero-sum framing", "T1": "Leaning zero-sum",
         "T2": "Compromise", "T3": "Leaning mutual gain",
         "T4": "Mutual gain"}
BLUE, ORANGE, TEAL, INK = "#aec4e3", "#ebbeaa", "#add3c6", "#394453"
BLOCK_COLORS = [BLUE, ORANGE, TEAL, "#cbbde0", "#e4cf9f"]
TRIPLETS = [("PN3", "SYM", "T3"), ("PN0", "VD", "T0"), ("VS", "I0", "T0")]
BLOCK_CODES = {
    "positive_negative": ["PN0", "PN1", "PN2", "PN3", "PN4"],
    "violence_type": ["VD", "VS", "VC"],
    "coupling_type": ["AB", "ANTI", "SYM"],
    "relational": ["I0", "I1", "I2", "I3", "I4"],
    "transcend": ["T0", "T1", "T2", "T3", "T4"],
}
EXCERPTS = [
    ("PN3+SYM+T3", "2a475af8ad6ad538",
     "A negotiated two-State solution is the best way to ensure long-term security for both Israelis and Palestinians"),
    ("PN3+VS+T3", "57d6db017ce9c959",
     "end Israel’s unlawful occupation and realize the Palestinian people’s right to self-determination through statehood that delivers true sovereignty on equal terms with Israel"),
    ("PN3+VS+SYM", "b65cb06caf93e464",
     "Israel and Palestine coexist side by side in peace and stability. We strongly oppose the expansion of settlements and any annexation in the West Bank"),
    ("PN3+SYM+I3", "23521dba6ffedbe2",
     "work towards a peaceful, definitive and just solution for the parties, with the existence of two States — Palestine and Israel — on the basis of the 1967 borders"),
    ("PN3+SYM+T4", "5a4b22b6327704be",
     "Israel and Palestine, living side by side in peace and security, as full members of the United Nations"),
    ("PN3+I3+T3", "b13686a97e811ef9",
     "a just and long-lasting peace in the Middle East can be achieved only with the establishment of an independent, sovereign State of Palestine, based on pre-1967 borders"),
    ("PN0+VD+T0", "1b4a61d5db42946d",
     "There is one entity that could stop violence for everyone in a moment and that is Hamas"),
    ("VS+SYM+T3", "9c75ef5367addfd8",
     "The occupation of Palestine and the apartheid system of injustice that serves it cannot continue if there is ever to be a lasting peace between Israelis and Palestinians"),
    ("SYM+I2+T3", "57bc93fe664fb125",
     "Diplomacy, dialogue and negotiation, with a central role for the United Nations, are the essential channels for reaching a solution that guarantees the rights and interests of the parties"),
    ("PN3+SYM+I2", "0e2bdd59234cdc37",
     "restore a political horizon and work towards building a State for the Palestinians and establishing security guarantees for Israel. The two-State solution is the only way to build a just and lasting peace"),
]
COUNTRY_ABBR = {
    "Algeria": "ALG", "Brazil": "BRA", "China": "CHN", "Denmark": "DEN",
    "Ecuador": "ECU", "Egypt": "EGY", "France": "FRA", "Greece": "GRE",
    "Guyana": "GUY", "Israel": "ISR", "Palestine": "PAL", "Japan": "JPN", "Jordan": "JOR",
    "Malta": "MAL", "Mozambique": "MOZ", "Pakistan": "PAK", "Panama": "PAN",
    "Republic of Korea": "KOR", "Russian Federation": "RUS",
    "Sierra Leone": "SLE", "Slovenia": "SVN", "Somalia": "SOM",
    "Switzerland": "SWI", "Türkiye": "TUR", "United Arab Emirates": "UAE",
    "United Kingdom": "UK", "United States": "US",
}


def _feature(code: str, columns) -> str:
    matches = [c for c in columns if c.startswith("Y__") and c.endswith("__" + code)]
    if len(matches) != 1:
        raise ValueError(f"Expected one feature for {code}, got {matches}")
    return matches[0]


def _tex(value: object) -> str:
    return (str(value).replace("\\", r"\textbackslash{}")
            .replace("&", r"\&").replace("%", r"\%").replace("_", r"\_"))


def _table(path: Path, label: str, caption: str, spec: str, headers: list[str],
           rows: list[list[str]], *, width: str = r"\textwidth") -> None:
    body = [r"\begin{table*}[!tbp]", r"\centering", r"\small",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}", rf"\begin{{tabular*}}{{{width}}}{{@{{\extracolsep{{\fill}}}}{spec}}}",
            r"\toprule", " & ".join(headers) + " \\\\", r"\midrule"]
    body += [" & ".join(row) + " \\\\" for row in rows]
    body += [r"\bottomrule", r"\end{tabular*}", r"\end{table*}", ""]
    path.write_text("% Generated from saved scoped outputs; do not hand-edit.\n" + "\n".join(body), encoding="utf-8")


def _main_population(matrix: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    from .galtung_research import population, load_config
    return population(matrix, cfg or load_config())


def _rq1(root: Path, paper: Path, plt) -> tuple[list[Path], dict]:
    prev = pd.read_csv(root / "rq1_provisional/prevalence.csv")
    core = prev[prev.dimension.isin(BLOCKS)].copy()
    assert len(core) == 21 and core.code.nunique() == 21
    # A single-column chart pairs with the ranked table in the two-column paper.
    # Group headers and printed values keep the display readable without color.
    fig, ax = plt.subplots(figsize=(3.45, 4.15))
    position = 0
    for i, block in enumerate(BLOCKS):
        rows = core[core.dimension.eq(block)].set_index("code").loc[BLOCK_CODES[block]].reset_index()
        ax.text(-.94, position, BLOCK_NAMES[block], transform=ax.get_yaxis_transform(),
                fontsize=8, fontweight="bold", color=INK, va="center")
        position += 1
        y = np.arange(position, position + len(rows))
        vals = 100 * rows.paragraph_prevalence.to_numpy()
        lo = 100 * rows.cluster_boot_ci_low.to_numpy()
        hi = 100 * rows.cluster_boot_ci_high.to_numpy()
        ax.hlines(y, 0, vals, color=BLOCK_COLORS[i], linewidth=3.5, zorder=2)
        ax.errorbar(vals, y, xerr=[vals-lo, hi-vals], fmt="o", color=BLOCK_COLORS[i],
                    markeredgecolor=INK, markeredgewidth=.45, markersize=4.4,
                    ecolor=INK, elinewidth=.75, capsize=1.8, zorder=3)
        for yy, code in zip(y, rows.code):
            ax.text(-.035, yy, SHORT[code], transform=ax.get_yaxis_transform(),
                    fontsize=7.2, color=INK, ha="right", va="center")
        for yy, v in zip(y, vals):
            ax.text(1.15, yy, f"{v:.1f}", transform=ax.get_yaxis_transform(),
                    va="center", ha="right", fontsize=7, color=INK)
        position += len(rows) + .75
    ax.set_ylim(position-.45, -1)
    ax.set_xlim(0, 32); ax.set_xticks([0, 10, 20, 30])
    ax.set_yticks([]); ax.set_xlabel("Paragraphs expressing component (%)", fontsize=7.5)
    ax.text(1.15, -1, "%", transform=ax.get_yaxis_transform(), ha="right", fontsize=7)
    ax.grid(axis="x", color="#e9ebef", linewidth=.55); ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    fig.subplots_adjust(left=.50, right=.88, top=.975, bottom=.08)
    p1 = paper / "rq1_core_prevalence.pdf"; fig.savefig(p1); plt.close(fig)

    pairs = pd.read_csv(root / "rq1_provisional/cooccurrence.csv")
    pairs = pairs[pairs.joint_positive.ge(100) & pairs.phi.notna()].copy()
    pairs["code_a"] = pairs.feature_a.str.split("__").str[-1]
    pairs["code_b"] = pairs.feature_b.str.split("__").str[-1]
    pairs["block_a"] = pairs.feature_a.str.split("__").str[2]
    pairs["block_b"] = pairs.feature_b.str.split("__").str[2]
    pairs = pairs[pairs.block_a.isin(BLOCKS) & pairs.block_b.isin(BLOCKS)
                  & pairs.block_a.ne(pairs.block_b)]
    selected = []
    for a, b in combinations(BLOCKS, 2):
        q = pairs[((pairs.block_a.eq(a)) & (pairs.block_b.eq(b))) |
                  ((pairs.block_a.eq(b)) & (pairs.block_b.eq(a)))].copy()
        if q.empty:
            raise ValueError(f"No supported association for {a}, {b}")
        q = q.sort_values(["phi", "joint_positive", "code_a", "code_b"],
                          ascending=[False, False, True, True], kind="stable")
        selected.append(q.iloc[0])
    assoc = pd.DataFrame(selected)
    assoc.to_csv(paper / "rq1_selected_associations.csv", index=False)
    labels = []
    for _, r in assoc.iterrows():
        labels.append(f"{BLOCK_NAMES[r.block_a]} × {BLOCK_NAMES[r.block_b]}\n"
                      f"{SHORT[r.code_a]} + {SHORT[r.code_b]}")
    fig, ax = plt.subplots(figsize=(7.25, 5.1))
    y = np.arange(len(assoc))[::-1]
    ax.hlines(y, 0, assoc.phi, color="#b9cbd8", lw=2)
    ax.scatter(assoc.phi, y, s=30, color=BLUE, zorder=3)
    ax.set_yticks(y, labels, fontsize=7)
    ax.set_xlabel("Phi correlation on jointly evaluable paragraphs", fontsize=8)
    ax.set_xlim(0, max(.82, assoc.phi.max()+.20))
    ax.grid(axis="x", color="#e5e9ec", lw=.7); ax.set_axisbelow(True)
    for yy, (_, r) in zip(y, assoc.iterrows()):
        ax.text(r.phi+.015, yy, f"{r.phi:.2f}  (n={int(r.joint_positive):,})",
                fontsize=6.8, va="center")
    fig.subplots_adjust(left=.54, right=.98, top=.98, bottom=.09)
    p2 = paper / "rq1_associations.pdf"; fig.savefig(p2); plt.close(fig)

    matrix = pd.read_parquet(root / "prepared_data/analysis_matrix.parquet")
    snapshot = json.loads((root / "audit/snapshot_manifest.json").read_text())
    d = _main_population(matrix, {"population": snapshot["population"]})
    from .paper_ranked import generate_ranked_rq1
    ranked_paths, ranked_summary = generate_ranked_rq1(
        root, paper, d, blocks=BLOCKS, block_codes=BLOCK_CODES,
        short=SHORT, triplets=TRIPLETS, excerpts=EXCERPTS,
        feature=_feature, tex=_tex,
    )
    return [p1, p2, paper / "rq1_selected_associations.csv", *ranked_paths], {
        "association_rows": len(assoc), **ranked_summary,
    }


def _rq2(root: Path, paper: Path, plt) -> tuple[list[Path], dict]:
    d = pd.read_csv(root / "rq2_provisional/adjusted_probabilities.csv")
    diagnostics = pd.read_csv(root / "rq2_provisional/model_diagnostics.csv")
    retained = diagnostics.status.eq("ok") & diagnostics.converged.astype(str).str.lower().eq("true")
    d = d[d.feature.isin(diagnostics.loc[retained, "feature"])].copy()
    assert d.variant.eq("primary").all()
    assert not d[["adjusted_probability_re0", "ci_low", "ci_high"]].isna().any().any()
    d["code"] = d.feature.str.split("__").str[-1]
    code_order = [c for c in SHORT if c in set(d.code)]
    scores = pd.read_csv(root / "rq3_provisional/pca_scores.csv").sort_values("PC1")
    countries = scores.country.tolist()
    n_countries, n_codes = len(countries), len(code_order)
    pivot = d.pivot(index="country", columns="code", values="adjusted_probability_re0").loc[countries, code_order]
    z = (pivot-pivot.mean(axis=0))/pivot.std(axis=0, ddof=0)
    assert np.isfinite(z.to_numpy()).all()
    fig, ax = plt.subplots(figsize=(7.25, 4.65))
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list("pastel_emphasis", [ORANGE, "#fbfaf8", "#9bb4d9"])
    im = ax.imshow(z, aspect="auto", cmap=cmap, interpolation="nearest",
                   norm=TwoSlopeNorm(vmin=-2.5, vcenter=0, vmax=2.5))
    ax.set_yticks(np.arange(n_countries), countries, fontsize=7.1)
    ax.set_xticks(np.arange(n_codes), [SHORT[c] for c in code_order], rotation=58,
                  ha="right", fontsize=6.7)
    ax.tick_params(axis="both", length=0, pad=4)
    ax.set_yticks(np.arange(n_countries)-.5, minor=True)
    ax.grid(which="minor", axis="y", color="white", lw=.35)
    ax.tick_params(which="minor", length=0)
    block_sizes = [sum(c in code_order for c in BLOCK_CODES[b]) for b in BLOCKS]
    boundaries = np.cumsum([0, *block_sizes])
    for split in boundaries[1:-1]:
        ax.axvline(split-.5, color="white", lw=2)
    ax.set_xlim(-.5, n_codes-.5)
    from matplotlib.patches import Rectangle
    for i, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        ax.add_patch(Rectangle((start-.5, 1.02), stop-start, .065,
                              transform=ax.get_xaxis_transform(), facecolor=BLOCK_COLORS[i],
                              edgecolor="white", clip_on=False))
        ax.text((start+stop-1)/2, 1.05,
                ["Peace", "Violence", "Coupling", "Relational", "Transformation"][i],
                transform=ax.get_xaxis_transform(), ha="center", va="center", fontsize=7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=.025, pad=.025, shrink=.68,
                        extend="both", ticks=[-2.5, 0, 2.5])
    cbar.set_label("Relative emphasis (SD)", fontsize=7)
    cbar.ax.tick_params(labelsize=7, length=0); cbar.outline.set_visible(False)
    fig.subplots_adjust(left=.20, right=.91, top=.91, bottom=.24)
    p = paper / "rq2_adjusted_heatmap.pdf"; fig.savefig(p); plt.close(fig)
    anchors = ["PN3", "VS", "SYM", "I1", "T3"]
    rows, extrema = [], []
    for code in anchors:
        q = d[d.code.eq(code)].sort_values(["adjusted_probability_re0", "country"])
        if q.empty:
            rows.append([_tex(SHORT[code]), "Not estimable", "--", "--", "--", "--"])
            extrema.append({"code": code, "low_country": "Not estimable", "low_probability": np.nan,
                            "low_ci_low": np.nan, "low_ci_high": np.nan, "high_country": "Not estimable",
                            "high_probability": np.nan, "high_ci_low": np.nan, "high_ci_high": np.nan,
                            "gap_percentage_points": np.nan, "estimation_status": "skipped or non-converged"})
            continue
        low, high = q.iloc[0], q.iloc[-1]
        gap = 100*(high.adjusted_probability_re0-low.adjusted_probability_re0)
        rows.append([_tex(SHORT[code]), _tex(low.country),
                     f"{100*low.adjusted_probability_re0:.1f} [{100*low.ci_low:.1f}, {100*low.ci_high:.1f}]",
                     _tex(high.country),
                     f"{100*high.adjusted_probability_re0:.1f} [{100*high.ci_low:.1f}, {100*high.ci_high:.1f}]",
                     f"{gap:.1f}"])
        extrema.append({"code": code, "low_country": low.country, "low_probability": low.adjusted_probability_re0,
                        "low_ci_low": low.ci_low, "low_ci_high": low.ci_high,
                        "high_country": high.country, "high_probability": high.adjusted_probability_re0,
                        "high_ci_low": high.ci_low, "high_ci_high": high.ci_high,
                        "gap_percentage_points": gap})
    t = paper / "rq2_anchor_extrema.tex"
    _table(t, "tab:rq2-anchor-extrema",
           "Descriptive low and high adjusted probabilities for one illustrative theory component per block. Failed convergence or support screens are shown as not estimable. Brackets are approximate model intervals; gaps are percentage points, not pairwise tests.",
           "lllllr", ["Theory component", "Low country", "Low \\% [95\\%]", "High country", "High \\% [95\\%]", "Gap"], rows)
    ep = paper / "rq2_anchor_extrema.csv"; pd.DataFrame(extrema).to_csv(ep, index=False)
    serializable_extrema = [{key: (None if isinstance(value, (float, np.floating)) and pd.isna(value) else value)
                            for key, value in row.items()} for row in extrema]
    return [p, t, ep], {"countries": countries, "codes": code_order, "extrema": serializable_extrema}


def _rq3(root: Path, paper: Path, plt) -> tuple[list[Path], dict]:
    scores = pd.read_csv(root / "rq3_provisional/pca_scores.csv")
    variances = pd.read_csv(root / "rq3_provisional/pca_explained_variance.csv")
    xvar, yvar = 100*variances.explained_variance_ratio.iloc[:2].to_numpy()
    assignments = pd.read_csv(root / "rq3_provisional/cluster_assignments.csv")
    k2 = assignments[assignments.k.eq(2)].set_index("country").profile_id
    scores["group"] = scores.country.map(k2)
    assert scores.group.notna().all()
    n_countries = len(scores)
    detail = scores[scores.PC1.lt(3)]
    def limits(values):
        lo, hi = float(values.min()), float(values.max())
        pad = max(.25, (hi-lo)*.12)
        return lo-pad, hi+pad
    detail_x, detail_y = limits(detail.PC1), limits(detail.PC2)
    group_styles = {"P1": (ORANGE, "D"), "P2": (BLUE, "o")}
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 4.25),
                             gridspec_kw={"width_ratios": [1.05, 1.6]})
    fig.subplots_adjust(left=.07, right=.98, top=.92, bottom=.36, wspace=.28)
    for ax, zoom in [(axes[0], False), (axes[1], True)]:
        q = scores if not zoom else scores[scores.PC1.lt(3)]
        for _, r in q.iterrows():
            color, marker = group_styles[r.group]
            ax.scatter(r.PC1, r.PC2, color=color, marker=marker,
                       edgecolor=INK, linewidth=.45, s=30, zorder=3)
        ax.axhline(0, color="#ccd4d9", lw=.7); ax.axvline(0, color="#ccd4d9", lw=.7)
        ax.set_ylabel(f"PC2 ({yvar:.1f}%)", fontsize=8)
        ax.set_title(f"a  All {n_countries} countries" if not zoom else f"b  Detail: {len(q)} countries", loc="left", fontsize=9,
                     fontweight="bold")
        ax.set_xlabel(f"PC1 ({xvar:.1f}%)", fontsize=8)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(color="#ecf0f2", lw=.5); ax.set_axisbelow(True)
        ax.tick_params(labelsize=7)
        if not zoom:
            from matplotlib.patches import Rectangle
            ax.add_patch(Rectangle((detail_x[0], detail_y[0]), detail_x[1]-detail_x[0], detail_y[1]-detail_y[0], fill=False,
                                   linestyle="--", edgecolor="#9098a4", linewidth=.7))
            ax.set_xlim(*limits(scores.PC1)); ax.set_ylim(*limits(scores.PC2))
            for _, r in q.iterrows():
                if r.PC1 >= 3 or r.country == "United States":
                    ax.annotate(r.country, (r.PC1, r.PC2), xytext=(3, 3),
                                textcoords="offset points", fontsize=7)
        else:
            ax.set_xlim(*detail_x); ax.set_ylim(*detail_y)
            # Place each label in display coordinates, avoiding labels and markers.
            # Fixed candidate offsets make every regeneration deterministic.
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            bounds = ax.get_window_extent(renderer)
            from matplotlib.transforms import Bbox
            coords = ax.transData.transform(q[["PC1", "PC2"]].to_numpy())
            marker_boxes = [Bbox.from_bounds(x-4, y-4, 8, 8) for x, y in coords]
            distances = np.linalg.norm(coords[:, None]-coords[None, :], axis=2)
            density = (distances < 45).sum(axis=1)
            offsets = [(0, 9), (0, -10), (11, 0), (-11, 0),
                       (12, 10), (-12, 10), (12, -10), (-12, -10),
                       (0, 20), (0, -21), (23, 0), (-23, 0),
                       (22, 18), (-22, 18), (22, -18), (-22, -18),
                       (0, 30), (0, -31)]
            placed = []
            for index in np.argsort(-density, kind="stable"):
                r = q.iloc[index]
                artist = ax.annotate(COUNTRY_ABBR[r.country], (r.PC1, r.PC2),
                                     xytext=offsets[0], textcoords="offset points",
                                     fontsize=7, ha="center", va="center", color=INK)
                best, best_score = offsets[0], float("inf")
                for candidate in offsets:
                    artist.set_position(candidate)
                    box = artist.get_window_extent(renderer).expanded(1.12, 1.15)
                    inside = bounds.contains(box.x0, box.y0) and bounds.contains(box.x1, box.y1)
                    overlap = sum(box.overlaps(other) for other in placed + marker_boxes)
                    score = overlap*1000 + (0 if inside else 10000) + np.hypot(*candidate)
                    if score < best_score:
                        best, best_score = candidate, score
                    if overlap == 0 and inside:
                        break
                artist.set_position(best)
                placed.append(artist.get_window_extent(renderer).expanded(1.12, 1.15))
                if np.hypot(*best) > 10:
                    target = ax.transData.inverted().transform(
                        ax.transData.transform((r.PC1, r.PC2)) + np.array(best)*fig.dpi/72)
                    ax.plot([r.PC1, target[0]], [r.PC2, target[1]],
                            color="#a6b5bf", lw=.45, zorder=1)
    # Membership comes from the full profile space, not the displayed PCA axes.
    from matplotlib.lines import Line2D
    legend_labels = {}
    for group in group_styles:
        members = sorted(k2.index[k2.eq(group)])
        label = " + ".join(members) if len(members) <= 3 else "Eligible countries"
        legend_labels[group] = f"{group}: {label} (n={len(members)})"
    handles = [Line2D([], [], linestyle="none", marker=marker,
                      markerfacecolor=color, markeredgecolor=INK,
                      markeredgewidth=.45, markersize=5, label=legend_labels[group])
               for group, (color, marker) in group_styles.items()]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .28),
               ncol=2, frameon=False, fontsize=7, handletextpad=.4, columnspacing=2)
    # Print the key in the figure itself, so a reviewer need not open a CSV.
    for index, country in enumerate(sorted(scores.country)):
        column, row = divmod(index, 7)
        fig.text(.035 + column*.245, .20-row*.029,
                 f"{COUNTRY_ABBR[country]}  {country} [{k2.loc[country]}]", fontsize=6.5, color=INK)
    p1 = paper / "rq3_pca.pdf"; fig.savefig(p1); plt.close(fig)
    key = paper / "rq3_pca_country_key.csv"
    pd.DataFrame([{"plot_label": COUNTRY_ABBR[c], "country": c,
                   "profile_id_k2": k2.loc[c]}
                  for c in sorted(scores.country)]).to_csv(key, index=False)

    countries = scores.sort_values("PC1").country.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 4.25), sharex=True, sharey=True)
    for ax, method, title in zip(axes,
            ["model_draw_coclustering", "speech_resample_coclustering"],
            ["Fitted-model draws", "Speech-resampled descriptive"]):
        d = pd.read_csv(root / f"rq3_provisional/{method}.csv")
        d = d[d.k.eq(2)]
        assert len(d) == n_countries*(n_countries-1)//2
        ix = {c:i for i,c in enumerate(countries)}
        matrix = np.eye(n_countries)
        for _, r in d.iterrows():
            i, j = ix[r.country_a], ix[r.country_b]
            matrix[i, j] = matrix[j, i] = r.co_cluster_rate
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("pastel_co_cluster", ["#fbfaf8", BLUE, "#8b9ac1"])
        im = ax.imshow(matrix, vmin=0, vmax=1, cmap=cmap, interpolation="nearest")
        ax.set_xticks(range(n_countries), [COUNTRY_ABBR[c] for c in countries], rotation=90, fontsize=5.2)
        ax.set_yticks(range(n_countries), countries, fontsize=5.2)
        ax.set_title(title, fontsize=8.5, fontweight="bold")
    fig.colorbar(im, ax=axes, fraction=.025, pad=.045, shrink=.85,
                 label="Co-clustering rate")
    fig.subplots_adjust(left=.16, right=.86, top=.92, bottom=.17, wspace=.12)
    p2 = paper / "rq3_coclustering.pdf"; fig.savefig(p2); plt.close(fig)

    diag = pd.read_csv(root / "rq3_provisional/cluster_k_diagnostics.csv")
    stab = pd.read_csv(root / "rq3_provisional/light_stability_summary.csv")
    rows = []
    for _, r in diag.iterrows():
        m = stab[(stab.k.eq(r.k)) & stab.method.eq("mean_field_model_draws")].iloc[0]
        s = stab[(stab.k.eq(r.k)) & stab.method.eq("speech_resampled_descriptive")].iloc[0]
        rows.append([str(int(r.k)), f"{r.silhouette:.3f}", str(int(r.min_cluster_size)),
                     f"{m.mean_ari:.3f}", f"{m.minimum_cluster_mean_jaccard:.3f}",
                     f"{s.mean_ari:.3f}", f"{s.minimum_cluster_mean_jaccard:.3f}"])
    t = paper / "rq3_profile_checks.tex"
    _table(t, "tab:rq3-profile-checks",
           "Ward solutions and two distinct 400-draw checks. ARI compares partitions; minimum Jaccard is the weakest group's mean best match. Model draws use the adjusted baseline; speech resampling uses its own descriptive baseline. Neither check refits the models.",
           "rrrrrrr", ["$k$", "Silhouette", "Min. group", "Model ARI", "Model Jacc.",
                        "Speech ARI", "Speech Jacc."], rows)
    return [p1, p2, t, key], {"k2_sizes": k2.value_counts().to_dict(), "pca_explained": [xvar, yvar]}


def _paper_values(root: Path, paper: Path) -> Path:
    """Keep manuscript sample sizes and numerical claims tied to the saved run."""
    snapshot = json.loads((root / "audit/snapshot_manifest.json").read_text())
    matrix = pd.read_parquet(root / "prepared_data/analysis_matrix.parquet")
    from .galtung_research import population
    cfg = {"population": snapshot["population"]}
    main = population(matrix, cfg)
    sc = population(matrix, cfg, snapshot["population"]["rq2_rq3_forum"])
    eligibility = pd.read_csv(root / "audit/country_eligibility.csv")
    primary = sc[sc.country.isin(eligibility.loc[eligibility.included_primary, "country"])]
    prev = pd.read_csv(root / "rq1_provisional/prevalence.csv").set_index("code")
    ranked = pd.read_csv(paper / "rq1_triplet_rankings.csv").iloc[0]
    diagnostics = pd.read_csv(root / "rq2_provisional/model_diagnostics.csv")
    converged = diagnostics.status.eq("ok") & diagnostics.converged.astype(str).str.lower().eq("true")
    variance = pd.read_csv(root / "rq3_provisional/pca_explained_variance.csv").explained_variance_ratio
    scores = pd.read_csv(root / "rq3_provisional/pca_scores.csv")
    assignments = pd.read_csv(root / "rq3_provisional/cluster_assignments.csv")
    k2 = assignments[assignments.k.eq(2)]
    diag = pd.read_csv(root / "rq3_provisional/cluster_k_diagnostics.csv")
    best = diag.loc[diag.silhouette.idxmax()]
    stability = pd.read_csv(root / "rq3_provisional/light_stability_summary.csv")
    model = stability[stability.k.eq(best.k) & stability.method.eq("mean_field_model_draws")].iloc[0]
    speech = stability[stability.k.eq(best.k) & stability.method.eq("speech_resampled_descriptive")].iloc[0]
    new_manifest = root / "validation_member_states_palestine_v1/coordinator_only/sampling_manifest.json"
    review = json.loads(new_manifest.read_text()) if new_manifest.exists() else {}
    values = {"MainParagraphs": f"{len(main):,}", "MainSpeeches": f"{main.speech_id.nunique():,}",
              "SCParagraphs": f"{len(sc):,}", "EligibleCountries": str(primary.country.nunique()),
              "PrimaryParagraphs": f"{len(primary):,}", "PrimarySpeeches": f"{primary.speech_id.nunique():,}",
              "PrimaryMeetings": str(primary.meeting_id.nunique()), "ModelConverged": str(int(converged.sum())),
              "ModelTotal": str(len(diagnostics)), "ProfileCodes": str(int(converged.sum())),
              "ExcludedCodes": _tex(", ".join(SHORT[code].lower() for code in
                  diagnostics.loc[~converged, "feature"].str.split("__").str[-1])),
              "LeadingCount": f"{ranked.paragraphs:,}", "LeadingShare": f"{100*ranked.share:.2f}",
              "LeadingMeetings": str(int(ranked.meetings)), "LeadingDescription": _tex(ranked.constructs),
              "PCOne": f"{100*variance.iloc[0]:.1f}", "PCTwo": f"{100*variance.iloc[1]:.1f}",
              "PCShown": f"{100*variance.iloc[:2].sum():.1f}", "PCHidden": f"{100*(1-variance.iloc[:2].sum()):.1f}",
              "ZoomCountries": str(int(scores.PC1.lt(3).sum())), "BestK": str(int(best.k)),
              "BestSilhouette": f"{best.silhouette:.3f}",
              "BestSizes": " and ".join(str(int(n)) for n in assignments[assignments.k.eq(best.k)].profile_id.value_counts().sort_values()),
              "KTwoOneSize": str(int(k2.profile_id.eq("P1").sum())),
              "KTwoTwoSize": str(int(k2.profile_id.eq("P2").sum())),
              "KTwoOneMembers": _tex(" and ".join(sorted(k2.loc[k2.profile_id.eq("P1"), "country"]))),
              "ModelARI": f"{model.mean_ari:.3f}", "SpeechARI": f"{speech.mean_ari:.3f}",
              "ModelJaccard": f"{model.minimum_cluster_mean_jaccard:.3f}",
              "SpeechJaccard": f"{speech.minimum_cluster_mean_jaccard:.3f}",
              "ReviewUniqueParagraphs": str(review.get("unique_paragraphs", "historical sample"))}
    outside = sorted(scores.loc[scores.PC1.ge(3), "country"])
    values["OutsideDetailDescription"] = ((_tex(" and ".join(outside)) +
        (" lies" if len(outside) == 1 else " lie") + " outside the detail region.") if outside
        else "All countries remain in the detail region.")
    group_sizes = k2.profile_id.value_counts()
    smaller = group_sizes.idxmin(); larger = group_sizes.idxmax()
    values["KTwoSmallGroup"] = smaller
    values["KTwoLargeGroup"] = larger
    values["KTwoSmallSize"] = str(int(group_sizes.loc[smaller]))
    values["KTwoLargeSize"] = str(int(group_sizes.loc[larger]))
    members = sorted(k2.loc[k2.profile_id.eq(smaller), "country"])
    values["KTwoSmallMembers"] = _tex(", ".join(members[:-1]) + (", and " + members[-1] if len(members) > 1 else members[0]))
    for code in ("VD", "VS"):
        values["Prev"+code] = f"{100*prev.loc[code, 'paragraph_prevalence']:.1f}"
    values["PrevPNThree"] = f"{100*prev.loc['PN3', 'paragraph_prevalence']:.1f}"
    extrema = pd.read_csv(paper / "rq2_anchor_extrema.csv").set_index("code")
    for code in ("VS", "SYM"):
        row = extrema.loc[code]
        values[code+"Low"] = f"{100*row.low_probability:.1f}" if pd.notna(row.low_probability) else "--"
        values[code+"High"] = f"{100*row.high_probability:.1f}" if pd.notna(row.high_probability) else "--"
        values[code+"LowCountry"] = _tex(row.low_country)
        values[code+"HighCountry"] = _tex(row.high_country)
        label = "structural violence" if code == "VS" else "cooperative relations"
        if pd.notna(row.low_probability) and pd.notna(row.high_probability):
            values[code+"RangeSentence"] = ("Adjusted probabilities for " + label + f" range from {100*row.low_probability:.1f}\\% for "
                + _tex(row.low_country) + f" to {100*row.high_probability:.1f}\\% for " + _tex(row.high_country) + ".")
        else:
            values[code+"RangeSentence"] = f"The adjusted model for {label} is not estimable under the declared screens; no range is reported."
    path = paper / "paper_values.tex"
    path.write_text("% Generated numerical manuscript values; do not hand-edit.\n" +
                    "\n".join("\\newcommand{\\" + key + "}{" + value + "}" for key, value in values.items()) + "\n")
    return path


def generate_paper_assets(root: Path, output: Path) -> list[Path]:
    """Generate four TeX tables and five vector figures from existing outputs."""
    needed = ["rq1_provisional/prevalence.csv", "rq1_provisional/cooccurrence.csv",
              "prepared_data/analysis_matrix.parquet", "rq2_provisional/adjusted_probabilities.csv",
              "rq3_provisional/pca_scores.csv", "rq3_provisional/model_draw_coclustering.csv",
              "rq3_provisional/speech_resample_coclustering.csv"]
    missing = [s for s in needed if not (root / s).exists()]
    if missing:
        raise FileNotFoundError("Paper assets need saved analysis files: " + ", ".join(missing))
    paper = output / "paper"; paper.mkdir(parents=True, exist_ok=True)
    _plot_setup(root / "audit" / "figure_cache")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK,
                         "axes.edgecolor": "#cdd2da", "axes.linewidth": .6,
                         "xtick.major.size": 2, "ytick.major.size": 0,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})
    a, asummary = _rq1(root, paper, plt)
    b, bsummary = _rq2(root, paper, plt)
    c, csummary = _rq3(root, paper, plt)
    values = _paper_values(root, paper)
    manifest = paper / "paper_asset_manifest.json"
    manifest.write_text(json.dumps({"source": "saved scoped analysis outputs",
                                    "status": "provisional_unvalidated",
                                    "rq1": asummary, "rq2": bsummary, "rq3": csummary,
                                    "tables": [p.name for p in [*a, *b, *c] if p.suffix == ".tex"],
                                    "vector_figures": [p.name for p in [*a, *b, *c] if p.suffix == ".pdf"]},
                                   indent=2, default=str)+"\n", encoding="utf-8")
    return [*a, *b, *c, values, manifest]
