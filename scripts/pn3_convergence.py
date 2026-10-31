"""Reproduce PN3's stop and assess longer fits across fixed initializations.

Writes diagnostics only; never activates a fit or changes source annotations.
Run with OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/pn3_convergence.py.
"""
import copy
import argparse
import json

import numpy as np
import pandas as pd

from un_speeches.research import galtung_research as gr

FEATURE = "Y__positive_negative__positive_negative__PN3"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["primary", "quarter_time"], default="primary")
    variant = parser.parse_args().variant
    cfg = gr.load_config()
    output = gr.outdir(cfg, "audit", "pn3_convergence")
    if variant != "primary":
        output = output / variant
        output.mkdir(exist_ok=True)
    data = gr.load_prepared(cfg)
    frame = gr.population(data["matrix"], cfg, cfg["population"]["rq2_rq3_forum"])
    eligibility = pd.read_csv(gr.outdir(cfg, "audit") / "country_eligibility.csv")
    frame = frame[frame.country.isin(eligibility.loc[eligibility.included_primary.astype(bool), "country"])]
    summary, predictions = [], {}
    for name, iterations, offset in [("reproduce_500", 500, 0),
                                      ("long_0", 2000, 0),
                                      ("long_1", 2000, 1),
                                      ("long_2", 2000, 2)]:
        trial = copy.deepcopy(cfg)
        # Always reproduce the pre-repair initialization, even after activation.
        trial["analysis"].pop("optimizer_initialization_offsets", None)
        trial["analysis"]["seed"] += offset
        trial["analysis"]["optimizer_feature_options"] = {FEATURE: {"maxiter": iterations}}
        pred, meta, draws = gr.fit_mixed_model(frame, FEATURE, trial, variant)
        meta.update(trial=name, initialization_seed_offset=offset)
        summary.append(meta)
        predictions[name] = pred.set_index("country").adjusted_probability_re0
        gr.write_csv(pred, output / f"{name}_probabilities.csv")
        draws.to_parquet(output / f"{name}_draws.parquet", index=False)
        (output / "diagnostics.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(meta), flush=True)
    comparisons = []
    for a, x in predictions.items():
        for b, y in predictions.items():
            if a >= b:
                continue
            comparisons.append({"trial_a": a, "trial_b": b,
                                "max_absolute_probability_difference": float((x-y).abs().max()),
                                "rms_probability_difference": float(np.sqrt(((x-y)**2).mean()))})
    if variant == "primary":
        old = pd.read_csv(output / "prechange/research_outputs/rq2_provisional/adjusted_probabilities.csv")
        old = old[old.feature.eq(FEATURE)].set_index("country").adjusted_probability_re0
        comparisons.append({"trial_a": "saved_failed_fit", "trial_b": "reproduce_500",
                            "max_absolute_probability_difference": float((old-predictions["reproduce_500"]).abs().max())})
    (output / "comparisons.json").write_text(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
