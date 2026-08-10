"""Event-anchored regressions around ICJ orders and the first ceasefire."""

from __future__ import annotations

import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from .config import CONFIG
from .fingerprints import outcome_flags
from .io_utils import LOG, save_fig, set_style, write_data, write_table

CAVEAT = (
    "Estimates describe discourse shifts around the event under a stable-composition "
    "assumption; the set and mix of speakers changes across meetings. Country fixed "
    "effects and the restriction to countries observed both before and after mitigate "
    "but do not eliminate this. No causal claim is made."
)


def event_window(corpus, event_date: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Pre/post bounds: EVENT_WINDOW_DAYS each side, post truncated at the next
    boundary of the *original* CONFIG phases.

    The original boundaries are used deliberately: the auto-merge in Phase 0 is a
    sample-size fix for clustering, whereas the truncation exists to stop a post
    window bleeding into a substantively different period.
    """
    w = pd.Timedelta(days=CONFIG["EVENT_WINDOW_DAYS"])
    bounds = sorted({pd.Timestamp(s) for s, _ in CONFIG["PHASES"].values()}
                    | {pd.Timestamp(e) for _, e in CONFIG["PHASES"].values()})
    later = [b for b in bounds if b > event_date]
    post_end = min(event_date + w, later[0]) if later else event_date + w
    return event_date - w, post_end


def build_panel(corpus, event_date, oname, spec) -> tuple[pd.DataFrame, dict]:
    """Paragraph-level panel for one event x outcome cell."""
    lo, hi = event_window(corpus, event_date)
    ms = corpus.member_states
    d = ms[(ms.date >= lo) & (ms.date <= hi)].copy()
    info = {"window_start": lo, "window_end": hi, "skipped": None}

    restrict = spec["dim"] == CONFIG["MODEL_SPLIT_DIM"] and corpus.model_span["split"]
    if restrict:
        span = corpus.model_span
        # Test the window clipped to the observed date range, not the nominal
        # window: a window reaching back before the corpus starts crosses no
        # model boundary, it simply has no data there.
        obs_lo = max(lo, corpus.units.date.min())
        obs_hi = min(hi, corpus.units.date.max())
        if not (span["start"] <= obs_lo and obs_hi <= span["end"]):
            info["skipped"] = (
                f"observed window {obs_lo.date()}..{obs_hi.date()} crosses the boundary "
                f"of the single-model span {span['start'].date()}.."
                f"{span['end'].date()} ({span['model'].split('/')[-1]})")
            return pd.DataFrame(), info
        d = d[d.in_model_span]

    fl = outcome_flags(d, corpus.ann, spec)
    d = d.merge(fl, on="unit_id")
    n_all = len(d)
    d = d[d.in_denom].copy()
    info["dropped_denominator"] = n_all - len(d)

    d["post"] = (d.date >= event_date).astype(int)
    d["months_since_event"] = (d.date - event_date).dt.days / 30.4375

    both = (d.groupby("country").post.nunique() == 2)
    keep = set(both[both].index)
    info["countries_all"] = d.country.nunique()
    info["countries_dropped"] = info["countries_all"] - len(keep)
    info["paragraphs_all"] = len(d)
    d = d[d.country.isin(keep)].copy()
    info["paragraphs_dropped"] = info["paragraphs_all"] - len(d)
    return d, info


def fit_models(d: pd.DataFrame) -> dict:
    """LPM (main) plus logit and country-meeting WLS robustness fits."""
    out = {}
    f = "y ~ post + months_since_event + post:months_since_event + C(country)"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = smf.ols(f, data=d).fit(cov_type="cluster",
                                   cov_kwds={"groups": d.country})
        out["lpm"] = m
        try:
            out["logit"] = smf.logit(f, data=d).fit(
                disp=0, method="bfgs", maxiter=300,
                cov_type="cluster", cov_kwds={"groups": d.country})
        except Exception as exc:                                # pragma: no cover
            out["logit"] = None
            out["logit_error"] = str(exc)

        agg = (d.groupby(["country", "meeting_id"])
               .agg(y=("y", "mean"), n=("y", "size"), post=("post", "max"),
                    months_since_event=("months_since_event", "mean"))
               .reset_index())
        try:
            w = smf.wls(f, data=agg, weights=agg.n).fit(
                cov_type="cluster", cov_kwds={"groups": agg.country})
            out["wls"] = w
        except Exception as exc:                                # pragma: no cover
            out["wls"] = None
            out["wls_error"] = str(exc)
        out["agg"] = agg
    return out


def _coef(m, name: str = "post") -> dict | None:
    """Pull one coefficient with its clustered SE, 95% CI and p-value."""
    if m is None or name not in getattr(m.params, "index", []):
        return None
    ci = m.conf_int()
    return {"b": float(m.params[name]), "se": float(m.bse[name]),
            "lo": float(ci.loc[name, 0]), "hi": float(ci.loc[name, 1]),
            "p": float(m.pvalues[name])}


def fig_event(corpus, event_name, event_date, panels, path):
    """Monthly outcome share with a CI band and the event line."""
    set_style()
    live = [(o, p) for o, p in panels.items() if len(p["data"])]
    if not live:
        return None
    fig, axes = plt.subplots(len(live), 1, figsize=(6.6, 2.2 * len(live)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (oname, p) in zip(axes, live):
        d = p["data"].copy()
        d["month"] = d.date.values.astype("datetime64[M]")
        g = d.groupby("month").y.agg(["mean", "size"])
        g = g[g["size"] >= 15]
        se = np.sqrt(g["mean"] * (1 - g["mean"]) / g["size"])
        ax.fill_between(g.index, g["mean"] - 1.96 * se, g["mean"] + 1.96 * se,
                        color=CONFIG["PALETTE"][0], alpha=0.20, lw=0)
        ax.plot(g.index, g["mean"], color=CONFIG["PALETTE"][0], marker="o", ms=3)
        ax.axvline(event_date, color=CONFIG["PALETTE"][1], ls="--", lw=1.3)
        ax.text(event_date, 1.01, event_name, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=6.5, color=CONFIG["PALETTE"][1])
        c = p["coef"]
        if c:
            ax.text(0.015, 0.06,
                    f"post = {c['b']:+.3f}  (SE {c['se']:.3f}, "
                    f"95% CI [{c['lo']:+.3f}, {c['hi']:+.3f}])",
                    transform=ax.transAxes, fontsize=6.4,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=CONFIG["GREY"],
                              lw=0.5, alpha=0.9))
        ax.set_ylabel(f"{oname}\n({'+'.join(p['codes'])})", fontsize=7)
        ax.set_ylim(0, None)
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(axes[-1].get_xticklabels(), rotation=45, ha="right")
    axes[0].set_title(f"Monthly outcome share around {event_name} "
                      f"({event_date.date()})\nband = 95% CI on the monthly share; "
                      "member states in national capacity only")
    fig.tight_layout()
    return save_fig(fig, path)


def shift_by_cluster(corpus, assign, event_date, oname, spec) -> pd.DataFrame:
    d, info = build_panel(corpus, event_date, oname, spec)
    if not len(d):
        return pd.DataFrame()
    cl = assign.set_index("country").cluster
    d = d.assign(cluster=d.country.map(cl)).dropna(subset=["cluster"])
    g = (d.groupby(["cluster", "post"]).y.agg(["mean", "size"]).unstack("post"))
    rows = []
    for c in g.index:
        pre, post = g[("mean", 0)].get(c), g[("mean", 1)].get(c)
        npre, npost = g[("size", 0)].get(c, 0), g[("size", 1)].get(c, 0)
        rows.append({"Cluster": f"C{int(c)}", "Outcome": oname,
                     "Pre": pre, "Post": post, "Change": post - pre,
                     "N pre": int(npre), "N post": int(npost)})
    return pd.DataFrame(rows)


def run(corpus, cl) -> dict:
    print("\n[Event studies]")
    results, rows, skipped = {}, [], []
    shift_rows = []

    # The contested transformative_framing definition is run both ways: as CONFIG
    # specifies it and in the stricter T3+T4 form the codebook's theory implies.
    variants = {v["name"]: v for v in CONFIG.get("OUTCOME_VARIANTS", {}).values()}
    all_outcomes = {**CONFIG["OUTCOMES"], **variants}

    for ev_name, ev_str in CONFIG["EVENTS"].items():
        ev = pd.Timestamp(ev_str)
        panels = {}
        for oname, spec in all_outcomes.items():
            d, info = build_panel(corpus, ev, oname, spec)
            if info["skipped"]:
                skipped.append(f"{ev_name} x {oname}: {info['skipped']}")
                LOG.log("Event study",
                        f"SKIPPED {ev_name} x {oname} — {info['skipped']}.",
                        "Policy: a transcend outcome may not be compared across an "
                        "annotation-model boundary.")
                panels[oname] = {"data": d, "coef": None, "info": info,
                                 "codes": spec["codes"]}
                rows.append({"Event": ev_name.replace("_", " "), "Outcome": oname,
                             "post coef": np.nan, "Clustered SE": np.nan,
                             "95% CI": "skipped (model boundary)", "N paragraphs": 0,
                             "N countries": 0, "Logit post": np.nan,
                             "WLS post": np.nan, "WLS SE": np.nan})
                continue

            fits = fit_models(d)
            c = _coef(fits["lpm"])
            cl_lg = _coef(fits.get("logit"))
            cl_wl = _coef(fits.get("wls"))
            panels[oname] = {"data": d, "coef": c, "info": info, "fits": fits,
                             "codes": spec["codes"]}
            rows.append({
                "Event": ev_name.replace("_", " "), "Outcome": oname,
                "post coef": c["b"], "Clustered SE": c["se"],
                "95% CI": f"[{c['lo']:+.3f}, {c['hi']:+.3f}]",
                "N paragraphs": len(d), "N countries": d.country.nunique(),
                "Logit post": cl_lg["b"] if cl_lg else np.nan,
                "WLS post": cl_wl["b"] if cl_wl else np.nan,
                "WLS SE": cl_wl["se"] if cl_wl else np.nan,
            })
            LOG.log("Event study",
                    f"{ev_name} x {oname}: window {info['window_start'].date()}.."
                    f"{info['window_end'].date()}, post={c['b']:+.4f} "
                    f"(SE {c['se']:.4f}, p={c['p']:.3g}), N={len(d)} paragraphs / "
                    f"{d.country.nunique()} countries; dropped "
                    f"{info['countries_dropped']} countries and "
                    f"{info['paragraphs_dropped']} paragraphs for not appearing on both "
                    "sides of the event.")
            sh = shift_by_cluster(corpus, cl["assign"], ev, oname, spec)
            if len(sh):
                sh.insert(0, "Event", ev_name.replace("_", " "))
                shift_rows.append(sh)

        results[ev_name] = panels
        fig_event(corpus, ev_name.split("_")[0], ev, panels,
                  f"fig_event_study_{ev_name.split('_')[0]}.png")

    tab = pd.DataFrame(rows)
    note = (CAVEAT + " Main model: linear probability at paragraph level, "
            "y ~ post + months_since_event + post:months_since_event + C(country), "
            "standard errors clustered by country. 'Logit post' and 'WLS post' are the "
            "corresponding coefficients from the logit and the country-meeting "
            "aggregated WLS weighted by paragraph counts; the logit coefficient is on "
            "the log-odds scale and is not comparable in magnitude to the LPM. Every "
            "regressor is constant within a country-meeting cell, so the aggregated WLS "
            "point estimate is algebraically identical to the paragraph-level LPM by "
            "construction; it checks the standard-error construction, not the "
            "coefficient. "
            "Denominators: attribution_israel among paragraphs carrying any attribution "
            "code, structural_violence among paragraphs carrying any violence_type code, "
            "transformative_framing among all paragraphs. "
            "transformative_framing_strict is the same outcome restricted to T3+T4, "
            "reported because Galtung's TRANSCEND method treats compromise (T2) as "
            "distinct from transcendence; see report.md section 8.")
    if skipped:
        note += " Skipped cells: " + "; ".join(skipped) + "."
    write_table(tab, "table_event_study.tex",
                caption="Event studies. One panel per event; the post coefficient is the "
                        "level shift in the outcome after the event, net of a linear "
                        "time trend and country fixed effects.",
                label="tab:event_study", notes=note, panel_col="Event",
                align="l" + "r" * 2 + "l" + "r" * 5)

    if shift_rows:
        sh = pd.concat(shift_rows, ignore_index=True)
        write_table(sh, "table_event_shift_by_cluster.tex",
                    caption="Pre/post change in each outcome by pooled cluster, within "
                            "the event windows used in Table~\\ref{tab:event_study}.",
                    label="tab:event_shift_by_cluster",
                    notes="Raw within-window means, not regression-adjusted. " + CAVEAT,
                    panel_col="Event", align="ll" + "r" * 5)
        write_data(sh, "event_shift_by_cluster.csv")
    write_data(tab, "event_study_coefficients.csv")
    LOG.log("Event study", "CAVEAT printed under the regression table: " + CAVEAT)
    return {"results": results, "table": tab, "skipped": skipped}
