"""Step 2: correspondence analysis map and the CLR cosine-similarity network."""

from __future__ import annotations

import warnings

import community as community_louvain
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import prince
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity

from .config import CONFIG
from .fingerprints import ABSENT, code_label
from .io_utils import LOG, iso3, place_labels, save_fig, set_style, write_data


def fit_ca(counts: pd.DataFrame, n_components: int = 4, seed: int = 42):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ca = prince.CA(n_components=n_components, n_iter=25, copy=True,
                       check_input=True, engine="sklearn", random_state=seed)
        ca = ca.fit(counts)
    rows = ca.row_coordinates(counts)
    cols = ca.column_coordinates(counts)
    inertia = np.asarray(ca.percentage_of_variance_) / 100.0
    return ca, rows, cols, inertia


def code_contributions(counts: pd.DataFrame, cols: pd.DataFrame,
                       eigenvalues: np.ndarray) -> pd.DataFrame:
    """Column contributions to the first two axes (mass * coord^2 / eigenvalue)."""
    P = counts.to_numpy() / counts.to_numpy().sum()
    cmass = P.sum(axis=0)
    out = {}
    for ax in (0, 1):
        out[f"contrib_{ax + 1}"] = cmass * cols.iloc[:, ax].to_numpy() ** 2 / eigenvalues[ax]
    d = pd.DataFrame(out, index=cols.index)
    d["mass"] = cmass
    d["contrib_12"] = d.contrib_1 + d.contrib_2
    return d


def _short(col: str) -> str:
    dim, code = col.split("__", 1)
    return f"{dim.split('_')[0]}:none" if code == ABSENT else code


def _axis_label(cols, contrib, ax, inertia, n_side: int = 2) -> str:
    """Name an axis by its highest-contribution codes at each pole."""
    top = contrib[f"contrib_{ax + 1}"].sort_values(ascending=False).head(6).index
    co = cols.iloc[:, ax].reindex(top).sort_values()
    neg = [t for t in co.index if co[t] < 0][:n_side]
    pos = [t for t in co.index if co[t] > 0][-n_side:][::-1]
    parts = []
    if neg:
        parts.append("<- " + ", ".join(_short(t) for t in neg))
    if pos:
        parts.append(", ".join(_short(t) for t in pos) + " ->")
    return (f"Axis {ax + 1} ({inertia[ax]:.1%} of inertia)\n"
            + "     ".join(parts))


def fig_ca_map(counts, party_counts, assign, colors, labels, n_codes=14,
               path="fig_ca_map.png"):
    """Countries sized by volume and coloured by cluster; top codes overlaid."""
    set_style()
    # Parties are projected as supplementary points: they must not shape the axes.
    ca, rows, cols, inertia = fit_ca(counts, n_components=4, seed=CONFIG["SEED"])
    ev = np.asarray(ca.eigenvalues_)
    contrib = code_contributions(counts, cols, ev)
    prows = ca.row_coordinates(party_counts.reindex(columns=counts.columns).fillna(0))

    keep = contrib.contrib_12.sort_values(ascending=False).head(n_codes).index
    a = assign.set_index("country")

    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    ax.axhline(0, color=CONFIG["GREY"], lw=0.6, zorder=0)
    ax.axvline(0, color=CONFIG["GREY"], lw=0.6, zorder=0)

    sizes = a.n_paragraphs.reindex(rows.index).to_numpy(dtype=float)
    smin, smax = sizes.min(), sizes.max()
    pt = 14 + 170 * (sizes - smin) / max(smax - smin, 1)
    for i, g in enumerate(sorted(a.cluster.unique())):
        m = (a.cluster.reindex(rows.index) == g).to_numpy()
        ax.scatter(rows.iloc[m, 0], rows.iloc[m, 1], s=pt[m], c=colors[i],
                   alpha=0.72, edgecolors="white", linewidths=0.5, zorder=3,
                   label=f"Cluster C{g} (n={int(m.sum())})")
    for t in keep:
        x, y = cols.loc[t].iat[0], cols.loc[t].iat[1]
        ax.scatter([x], [y], marker="^", s=46, c="black", zorder=5)

    ax.margins(0.09)
    # Codes first: they anchor the reading of the map, so they claim slots before
    # the country labels have to fit around them.
    taken = place_labels(
        ax, [(cols.loc[t].iat[0], cols.loc[t].iat[1]) for t in keep],
        [_short(t) for t in keep], fontsize=6.8, zorder=7,
        fontweight="bold", leader_from="black", avoid_center=True,
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    place_labels(ax, [(rows.loc[c].iat[0], rows.loc[c].iat[1]) for c in rows.index],
                 [iso3(c) for c in rows.index],
                 priority=list(a.n_paragraphs.reindex(rows.index).to_numpy()),
                 fontsize=5.4, zorder=4, leader_from=CONFIG["GREY"], occupied=taken)

    for p in prows.index:
        ax.scatter([prows.loc[p].iat[0]], [prows.loc[p].iat[1]], marker="o", s=150,
                   facecolors="none", edgecolors=CONFIG["PALETTE"][3], linewidths=1.8,
                   zorder=7)
        ax.annotate(f"{iso3(p)} (party)", (prows.loc[p].iat[0], prows.loc[p].iat[1]),
                    textcoords="offset points", xytext=(0, -14), fontsize=6.5,
                    ha="center", color=CONFIG["PALETTE"][3], fontweight="bold", zorder=9,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                              alpha=0.85))

    ax.set_xlabel(_axis_label(cols, contrib, 0, inertia))
    ax.set_ylabel(_axis_label(cols, contrib, 1, inertia))
    ax.set_title("Correspondence analysis of member-state code profiles\n"
                 f"first two axes carry {inertia[0] + inertia[1]:.1%} of total inertia")
    h, lg = ax.get_legend_handles_labels()
    h.append(plt.Line2D([], [], marker="^", ls="none", color="black", ms=6))
    lg.append(f"{n_codes} highest-contribution codes")
    h.append(plt.Line2D([], [], marker="o", ls="none", mfc="none",
                        mec=CONFIG["PALETTE"][3], ms=9, mew=1.8))
    lg.append("Parties (supplementary, not clustered)")
    ax.legend(h, lg, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2,
              fontsize=6.8)
    save_fig(fig, path)
    return {"ca": ca, "rows": rows, "cols": cols, "inertia": inertia,
            "contrib": contrib, "party_rows": prows}


def similarity_network(X: pd.DataFrame, k: int = 5, seed: int = 42):
    """kNN graph on cosine similarity of CLR vectors, Louvain communities."""
    S = cosine_similarity(X.to_numpy())
    np.fill_diagonal(S, -np.inf)
    G = nx.Graph()
    G.add_nodes_from(X.index)
    for i, c in enumerate(X.index):
        for j in np.argsort(S[i])[::-1][:k]:
            w = float(S[i, j])
            if w > 0 and not G.has_edge(c, X.index[j]):
                G.add_edge(c, X.index[j], weight=w)
    part = community_louvain.best_partition(G, weight="weight", random_state=seed)
    return G, part


def fig_network(G, part, assign, colors, path="fig_similarity_network.png"):
    set_style()
    a = assign.set_index("country")
    pos = nx.spring_layout(G, seed=CONFIG["SEED"], weight="weight", k=0.55, iterations=350)
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.28, width=0.7,
                           edge_color=CONFIG["GREY"])
    nodes = list(G.nodes)
    sizes = a.n_paragraphs.reindex(nodes).to_numpy(dtype=float)
    pt = 40 + 260 * (sizes - sizes.min()) / max(sizes.max() - sizes.min(), 1)
    ward_keys = sorted(a.cluster.unique())
    node_c = [colors[ward_keys.index(a.cluster[n])] for n in nodes]

    # Fill = Ward cluster; ring = Louvain community. A constant-width ring keeps
    # high-volume states from acquiring a huge outline that reads as emphasis.
    comms = sorted(set(part.values()))
    lou_pal = [CONFIG["PALETTE"][(i + 2) % len(CONFIG["PALETTE"])]
               for i in range(len(comms))]
    edge_c = [lou_pal[comms.index(part[n])] for n in nodes]
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=nodes, node_size=pt,
                           node_color=node_c, edgecolors=edge_c, linewidths=1.6,
                           alpha=0.95)
    ax.set_axis_off()
    ax.margins(0.06)
    place_labels(ax, [pos[n] for n in nodes], [iso3(n) for n in nodes],
                 priority=list(sizes), fontsize=5.4, leader_from=CONFIG["GREY"])

    ward_h = [plt.Line2D([], [], marker="o", ls="none", ms=7, mfc=colors[i],
                         mec="none") for i in range(len(ward_keys))]
    lou_h = [plt.Line2D([], [], marker="o", ls="none", ms=7, mfc="white",
                        mec=lou_pal[i], mew=1.8) for i in range(len(comms))]
    l1 = ax.legend(ward_h, [f"Ward cluster C{g} (fill)" for g in ward_keys],
                   loc="upper left", fontsize=6.5, title="Step 3 clustering",
                   title_fontsize=6.5)
    l1.get_title().set_fontweight("bold")
    ax.add_artist(l1)
    ax.legend(lou_h, [f"community {c} (n={sum(1 for n in nodes if part[n] == c)})"
                      for c in comms],
              loc="lower left", fontsize=6.5, title="Louvain (ring)",
              title_fontsize=6.5)
    ax.set_title("Cosine-similarity kNN graph of CLR fingerprints (k=5)\n"
                 "node fill = Ward cluster, ring colour = Louvain community, "
                 "size = paragraph volume")
    return save_fig(fig, path)


def run(corpus, fp, cl) -> dict:
    print("\n[Step 2] the map")
    counts, X = fp["counts"], fp["clr"]
    res = fig_ca_map(counts, fp["party_counts"], cl["assign"], cl["colors"],
                     corpus.code_labels)
    inertia = res["inertia"]
    LOG.log("Step 2",
            f"Correspondence analysis on the {counts.shape[0]}x{counts.shape[1]} "
            f"country-by-code count table; axes 1-2 carry "
            f"{inertia[0]:.1%} and {inertia[1]:.1%} of inertia "
            f"({inertia[0] + inertia[1]:.1%} together).",
            "The synthetic per-dimension none categories are kept as columns, so axis 1 "
            "partly reflects overall coding density as well as code mix. Israel and "
            "Palestine are projected as supplementary rows after the axes are fitted on "
            "member states alone, so they cannot shape the solution.")

    top = res["contrib"].contrib_12.sort_values(ascending=False).head(15)
    write_data(pd.DataFrame({
        "code": top.index,
        "label": [code_label(corpus.code_labels, t) for t in top.index],
        "mass": res["contrib"].mass.reindex(top.index).to_numpy(),
        "contrib_axis1": res["contrib"].contrib_1.reindex(top.index).to_numpy(),
        "contrib_axis2": res["contrib"].contrib_2.reindex(top.index).to_numpy(),
        "coord_axis1": res["cols"].iloc[:, 0].reindex(top.index).to_numpy(),
        "coord_axis2": res["cols"].iloc[:, 1].reindex(top.index).to_numpy(),
    }), "ca_code_contributions.csv")
    write_data(res["rows"].reset_index().rename(
        columns={"index": "country", 0: "dim1", 1: "dim2", 2: "dim3", 3: "dim4"}),
        "ca_country_coordinates.csv")

    G, part = similarity_network(X, k=5, seed=CONFIG["SEED"])
    comm = np.array([part[c] for c in X.index])
    ari = adjusted_rand_score(cl["labels"], comm)
    fig_network(G, part, cl["assign"], cl["colors"])

    n_comm = len(set(part.values()))
    agree = "agree closely" if ari >= 0.7 else ("agree only partially" if ari >= 0.4
                                                else "DISAGREE")
    # A low ARI can mean two things: the partitions cut across each other, or one
    # simply refines the other. Distinguish them, because only the first is a
    # disagreement worth reporting as such.
    nest = []
    for cm in sorted(set(part.values())):
        members = [i for i, c in enumerate(X.index) if part[c] == cm]
        wards = {int(cl["labels"][i]) for i in members}
        nest.append((cm, len(members), sorted(wards)))
    n_split = sum(1 for _, _, w in nest if len(w) > 1)
    nested = n_split == 0
    LOG.log("Step 2",
            f"Louvain on the k=5 cosine kNN graph finds {n_comm} communities; "
            f"adjusted Rand index against the Ward clustering = {ari:.3f} "
            f"— the two partitions {agree}. "
            + (f"All {n_comm} communities nest entirely inside a single Ward cluster "
               f"({'; '.join(f'community {c} (n={n}) subset of C{w[0]}' for c, n, w in nest)}), "
               "so the disagreement is one of resolution, not of structure."
               if nested else
               f"{n_split} of {n_comm} communities straddle Ward clusters, so the two "
               "partitions genuinely cut across each other."),
            "Louvain optimises modularity on a sparse neighbour graph while Ward minimises "
            "within-cluster variance in the full CLR space, so a lower ARI reflects the "
            "different objectives as much as genuine instability."
            if ari < 0.7 else "")

    net = pd.DataFrame({"country": X.index, "iso3": [iso3(c) for c in X.index],
                        "ward_cluster": cl["labels"], "louvain_community": comm})
    write_data(net, "network_communities.csv")
    return {**res, "graph": G, "louvain": part, "ari_ward_louvain": float(ari),
            "n_communities": n_comm, "nested": nested, "nesting": nest}
