"""Entry point: `python -m analysis.pipeline` runs everything, seeded and idempotent."""

from __future__ import annotations

import random
import shutil
import sys
import time

import numpy as np

from . import clustering, discovery, event_study, fingerprints, mapping
from . import phases, sensitivity, signatures
from .config import CONFIG, OUT, OUT_DATA, OUT_FIGURES, OUT_TABLES
from .io_utils import LOG, ensure_dirs
from .report import write_report


def reset_outputs() -> None:
    """Overwrite cleanly: generated artefacts are removed before each run."""
    for d in (OUT_TABLES, OUT_FIGURES, OUT_DATA):
        if d.exists():
            shutil.rmtree(d)
    for f in (OUT / "report.md", OUT / "data_report.md"):
        if f.exists():
            f.unlink()
    ensure_dirs()


def main() -> int:
    t0 = time.time()
    random.seed(CONFIG["SEED"])
    np.random.seed(CONFIG["SEED"])
    print("=" * 78)
    print("UNSC Palestine-question debates — Galtung code analysis")
    print(f"seed={CONFIG['SEED']}  bootstrap={CONFIG['BOOTSTRAP_B']}  "
          f"k_range={CONFIG['K_RANGE'][0]}..{CONFIG['K_RANGE'][-1]}")
    print("=" * 78)

    reset_outputs()

    corpus = discovery.build_corpus()
    fp = fingerprints.run(corpus)
    cl = clustering.run(corpus, fp)
    mp = mapping.run(corpus, fp, cl)
    sg = signatures.run(corpus, fp, cl)
    ph = phases.run(corpus, fp, cl)
    es = event_study.run(corpus, cl)
    sn = sensitivity.run(corpus, fp, cl)

    write_report(corpus, fp, cl, mp, sg, ph, es, sn)

    n_tab = len(list(OUT_TABLES.glob("*.tex")))
    n_fig = len(list(OUT_FIGURES.glob("*.png")))
    n_dat = len(list(OUT_DATA.glob("*.csv")))
    print("\n" + "=" * 78)
    print(f"done in {time.time() - t0:.0f}s — {n_tab} tables, {n_fig} figures, "
          f"{n_dat} data files, {len(LOG.entries)} logged decisions")
    print("read outputs/report.md and outputs/data_report.md")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
