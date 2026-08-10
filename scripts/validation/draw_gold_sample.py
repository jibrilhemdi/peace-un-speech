#!/usr/bin/env python3
"""Draw a probability sample of units for human coding, with exact inclusion probabilities.

    python scripts/validation/draw_gold_sample.py --codebook violence -n 300
    python scripts/validation/draw_gold_sample.py --codebook violence -n 300 \
        --stratify-on violence_type --oversample VC=4 --oversample VS=2
    python scripts/validation/draw_gold_sample.py --codebook violence -n 300 --allocation neyman

STATUS: sampling frame and pi bookkeeping are complete and correct. The human-coding UI is
deliberately out of scope — this writes the sample and stops.

Why pi matters. A later measurement-error correction (Hausman-style, or a simple
inverse-probability-weighted recalibration of the model's labels against gold) needs the
inclusion probability of every sampled unit. Those probabilities must be KNOWN BY DESIGN, not
reconstructed afterwards from what happened to be drawn. Reconstructed probabilities are
wrong whenever the strata were built from model output — which is exactly what oversampling
rare codes does. So pi is computed here, at draw time, from the realised stratum sizes, and
written into the output.

Weights are 1/pi. Under stratified sampling without replacement, a unit in stratum h with
N_h units of which n_h are drawn has pi = n_h / N_h exactly.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib import codebook as cbmod  # noqa: E402
from lib.config import Config  # noqa: E402
from lib.db import Store  # noqa: E402


def parse_oversample(pairs: list[str] | None) -> dict[str, float]:
    out: dict[str, float] = {}
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit(f"--oversample expects CODE=FACTOR, got {p!r}")
        code, factor = p.split("=", 1)
        try:
            out[code.strip()] = float(factor)
        except ValueError:
            raise SystemExit(f"--oversample factor must be a number, got {factor!r}") from None
    return out


def allocate(
    sizes: dict[str, int],
    n: int,
    weights: dict[str, float],
    scheme: str,
) -> dict[str, int]:
    """Allocate n draws across strata, never asking for more than a stratum holds.

    proportional  n_h proportional to N_h * w_h
    neyman        n_h proportional to N_h * sqrt(p_h(1-p_h)) — here approximated by N_h*w_h,
                  since the response variance is unknown before coding; documented so the
                  choice is visible rather than silently equal to proportional
    equal         every stratum gets the same count, capped by its size
    """
    strata = sorted(sizes)
    if scheme == "equal":
        base = {h: n / len(strata) for h in strata}
    else:
        raw = {h: sizes[h] * weights.get(h, 1.0) for h in strata}
        total = sum(raw.values()) or 1.0
        base = {h: n * raw[h] / total for h in strata}

    alloc = {h: min(sizes[h], int(base[h])) for h in strata}
    # Largest-remainder, respecting capacity, so a small oversampled stratum is not rounded away.
    remainder = sorted(strata, key=lambda h: -(base[h] - int(base[h])))
    left = n - sum(alloc.values())
    i = 0
    while left > 0 and any(alloc[h] < sizes[h] for h in strata):
        h = remainder[i % len(remainder)]
        if alloc[h] < sizes[h]:
            alloc[h] += 1
            left -= 1
        i += 1
        if i > len(strata) * (max(sizes.values()) + 1):
            break
    return alloc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--codebook", required=True)
    ap.add_argument("-n", "--size", type=int, required=True, help="total units to draw")
    ap.add_argument("--run-id", default="main")
    ap.add_argument("--seed", type=int, default=None, help="default: provider.seed from config")
    ap.add_argument("--stratify-on", action="append", default=None,
                    help="dimension name, or a unit column (country, bloc). Repeatable; "
                         "strata are the cross-product.")
    ap.add_argument("--oversample", action="append", default=None, metavar="CODE=FACTOR",
                    help="relative weight for strata containing CODE, e.g. VC=4")
    ap.add_argument("--allocation", choices=["proportional", "neyman", "equal"],
                    default="proportional")
    ap.add_argument("--out", default=None, help="default: data/gold/gold_sample.parquet")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    store = Store(cfg.path_for("db"))
    cb = cbmod.require_codebook(cfg.path_for("codebooks"), args.codebook)
    seed = args.seed if args.seed is not None else int(cfg.get("provider.seed") or 0)

    version = store.conn.execute(
        """SELECT MAX(codebook_version) FROM jobs
           WHERE codebook_id=? AND run_id=? AND status='ok'""",
        (cb.id, args.run_id),
    ).fetchone()[0]
    if version is None:
        print(f"no completed annotations for {cb.id} run_id={args.run_id} — "
              "the frame is the set of annotated units, so there is nothing to sample yet.",
              file=sys.stderr)
        store.close()
        return 1

    # -- frame: every successfully annotated, non-procedural unit -----------------------
    frame = pd.read_sql_query(
        """SELECT u.unit_id, u.speech_id, u.para_index, u.country, u.bloc, u.is_party,
                  u.meeting_date, u.n_words, u.text
           FROM units u JOIN jobs j ON j.unit_id = u.unit_id
           WHERE j.codebook_id=? AND j.codebook_version=? AND j.run_id=? AND j.status='ok'
             AND u.para_is_procedural = 0
           ORDER BY u.unit_id""",
        store.conn, params=(cb.id, version, args.run_id),
    )
    if frame.empty:
        print("frame is empty", file=sys.stderr)
        store.close()
        return 1

    ann = pd.read_sql_query(
        """SELECT unit_id, dimension, value FROM annotations
           WHERE codebook_id=? AND codebook_version=? AND run_id=?""",
        store.conn, params=(cb.id, version, args.run_id),
    )

    # Predicted label per dimension, as a sorted code set — the model's own output is a legal
    # stratifier precisely because pi is recorded.
    for dim in cb.dimensions:
        vals = (
            ann[ann.dimension == dim.name]
            .groupby("unit_id").value.apply(lambda s: "+".join(sorted(set(s))))
        )
        frame[f"pred__{dim.name}"] = frame.unit_id.map(vals).fillna("NONE")

    # -- strata -------------------------------------------------------------------------
    keys = args.stratify_on or []
    stratum_cols: list[str] = []
    for k in keys:
        if cb.dimension(k) is not None:
            stratum_cols.append(f"pred__{k}")
        elif k in frame.columns:
            stratum_cols.append(k)
        else:
            print(f"--stratify-on '{k}' is neither a dimension of {cb.id} nor a unit column "
                  f"({', '.join(frame.columns)})", file=sys.stderr)
            store.close()
            return 1

    if stratum_cols:
        # Nulls (a briefer with no country, an unmapped bloc) become an explicit stratum
        # rather than crashing or silently vanishing — they still need a known pi.
        parts = [frame[c].where(frame[c].notna(), "NA").astype(str) for c in stratum_cols]
        frame["_stratum"] = parts[0]
        for extra in parts[1:]:
            frame["_stratum"] = frame["_stratum"] + " | " + extra
    else:
        frame["_stratum"] = "ALL"

    sizes = frame.groupby("_stratum").size().to_dict()
    if args.size > len(frame):
        print(f"requested {args.size} but the frame holds {len(frame)}", file=sys.stderr)
        store.close()
        return 1

    over = parse_oversample(args.oversample)
    weights = {
        h: max((f for code, f in over.items() if code in h.split(" | ")), default=1.0)
        for h in sizes
    }
    alloc = allocate(sizes, args.size, weights, args.allocation)

    # -- draw ---------------------------------------------------------------------------
    rng = np.random.default_rng(seed)
    picked: list[pd.DataFrame] = []
    for h in sorted(sizes):
        n_h, N_h = alloc[h], sizes[h]
        if n_h <= 0:
            continue
        pool = frame[frame._stratum == h].sort_values("unit_id")
        idx = rng.choice(len(pool), size=n_h, replace=False)
        take = pool.iloc[np.sort(idx)].copy()
        # Exact by design: SRS without replacement inside the stratum.
        take["pi"] = n_h / N_h
        take["weight"] = N_h / n_h
        take["stratum_size"] = N_h
        take["stratum_drawn"] = n_h
        picked.append(take)

    sample = pd.concat(picked, ignore_index=True) if picked else frame.iloc[0:0].copy()
    sample = sample.rename(columns={"_stratum": "stratum"})
    sample["codebook_id"] = cb.id
    sample["codebook_version"] = version
    sample["run_id"] = args.run_id
    sample["sample_seed"] = seed
    sample["drawn_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    # Blank columns for the human coder. Deliberately not pre-filled with the model's guess:
    # showing the prediction while coding is what makes gold data agree with the model.
    for dim in cb.dimensions:
        sample[f"gold__{dim.name}"] = pd.NA
    sample["gold__evidence"] = pd.NA
    sample["gold__notes"] = pd.NA

    out_path = Path(args.out) if args.out else cfg.path_for("gold") / "gold_sample.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(out_path, index=False)

    design = {
        "codebook_id": cb.id,
        "codebook_version": int(version),
        "run_id": args.run_id,
        "seed": seed,
        "allocation": args.allocation,
        "stratify_on": keys,
        "oversample": over,
        "frame_size": int(len(frame)),
        "sample_size": int(len(sample)),
        "drawn_at": sample.drawn_at.iloc[0] if len(sample) else None,
        "strata": [
            {"stratum": h, "N_h": int(sizes[h]), "n_h": int(alloc[h]),
             "pi": (alloc[h] / sizes[h]) if alloc[h] else 0.0,
             "weight_factor": weights[h]}
            for h in sorted(sizes)
        ],
    }
    design_path = out_path.with_suffix(".design.json")
    design_path.write_text(json.dumps(design, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"frame            {len(frame):,} annotated units ({cb.id} v{version}, run={args.run_id})")
    print(f"strata           {len(sizes)} on {stratum_cols or ['(none)']}")
    print(f"allocation       {args.allocation}" + (f", oversample {over}" if over else ""))
    print(f"drawn            {len(sample):,} units, seed {seed}")
    print()
    print(f"  {'stratum':<40} {'N_h':>7} {'n_h':>6} {'pi':>8} {'weight':>9}")
    for s in design["strata"]:
        if s["n_h"]:
            print(f"  {s['stratum'][:40]:<40} {s['N_h']:>7,} {s['n_h']:>6,} "
                  f"{s['pi']:>8.4f} {1 / s['pi']:>9.2f}")
    uncovered = [s for s in design["strata"] if not s["n_h"]]
    if uncovered:
        print(f"  ({len(uncovered)} stratum/strata received zero draws — they have pi = 0 and "
              "are NOT representable in a weighted estimate)")
    print()
    print(f"wrote {out_path}")
    print(f"wrote {design_path}")
    print()
    print("Columns gold__* are empty for human coding. `pi` is the exact inclusion")
    print("probability and `weight` = 1/pi; keep them with the data — they are the design,")
    print("and they cannot be reconstructed later once the strata came from model output.")

    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
