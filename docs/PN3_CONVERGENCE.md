# PN3 convergence follow-up (2026-10-23)

The initial expanded-population analysis excluded PN3 automatically because
SciPy reported unsuccessful optimization. The source annotations, population,
model formula, priors and estimation method have not changed in this follow-up.

## Numerical diagnosis

The original primary fit is reproduced to within 8.4e-17 in country probabilities.
It stopped after 216 iterations with status 2, “Desired error not necessarily
achieved due to precision loss.” Its maximum absolute gradient was 7.48e-5,
above the unchanged 1e-5 tolerance. Increasing the cap from 500 to 2,000 iterations
produced exactly the same stop and estimates: the cap was not the cause.

Two alternative initialization seeds converged after 236 and 238 iterations.
Their maximum absolute gradients were 8.56e-6 and 5.69e-6. Their objective values
agreed within 4e-11, and their country probabilities within 3.74e-8 (less than
0.000004 percentage points). Both also closely matched the failed fit.

We retain the first successful initialization, an offset of +1 for primary PN3.
The iteration cap remains 500 and the tolerance remains 1e-5. Posterior
draw seeds retain their original values. Selection was based on convergence and
agreement of estimates, before inspecting the resulting grouping.

Every primary model was rerun to retain full optimizer diagnostics. All other
primary point estimates are exactly unchanged. Twenty of 21 core models now
converge; AB remains excluded by the original frequency screen.

The quarter-time PN3 sensitivity fit also stopped on precision loss (249
iterations, maximum gradient 1.79e-5); raising its cap did not help. Two
alternative starts converged in 255 and 235 iterations, agreeing within
1.29e-8 in country probabilities. We likewise retain offset +1 for this
variant, with the original cap and tolerance. All four planned sensitivity
fits now converge. The no-length PN3 and both VD variants are unchanged.

## Consequences for the paper

The adjusted profile now contains 27 countries and 20 features. PN3 remains the
Table 2 anchor: its low/high estimates are Israel 0.5% and Pakistan 23.8%, a
23.3 percentage-point descriptive gap.

At k=2 the groups remain 19 and 8 countries, with silhouette 0.282613.
At k=3 the groups remain 19, 7 and 1 (Israel), with silhouette 0.245523.
Both 400-replicate light stability checks, PCA, figures, tables and generated
manuscript values are recalculated with PN3 included. These light checks do not
refit the mixed models and do not establish enduring political groups.

The earlier ablation removed PN3 from the **historical fitted profiles**, turning
the US–Israel pair into seven countries. It does not imply that restoring PN3
to the **expanded-population fitted profiles** will recover that pair; it does
not. The historical/current contrast therefore cannot be attributed solely to
PN3 exclusion or solely to adding Palestine.

On 2026-10-31, a direct audit of the current 20-feature profiles reproduced
the saved k=2 membership exactly. Substituting either successful PN3 fit also
gave identical membership. Removing Palestine from the current profile matrix,
without refitting any model, restored the US–Israel pair (silhouette 0.445575
after rescaling). Holding the 27-country scaling fixed also restored the pair
(0.445853). Conversely, including Palestine while using the 26-country scaling
still produced the eight-country group (0.288701). Thus the current Ward split
is sensitive to the country set even when scaling is held fixed. This is a
clustering sensitivity result, not a full no-Palestine model refit, and limits
substantive interpretation of the groups. The four controlled cases are saved
in `research_outputs/audit/pn3_convergence/country_set_sensitivity.json`.

## Reproduction and retained evidence

From the repository root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/pn3_convergence.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/pn3_convergence.py --variant quarter_time
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m un_speeches.research.galtung_research rq2 --workers 2
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m un_speeches.research.galtung_research rq2_sensitivity --workers 2
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m un_speeches.research.galtung_research rq3
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m un_speeches.research.galtung_research light_stability
.venv/bin/python -m un_speeches.research.galtung_research latex
.venv/bin/python -m un_speeches.research.galtung_research report
latexmk -pdf -cd -interaction=nonstopmode -halt-on-error paper/main.tex
```

Trial diagnostics, probabilities, draws and comparisons are saved under
`research_outputs/audit/pn3_convergence/`. The `prechange/` directory retains
the prior RQ2/RQ3 outputs, paper assets and manuscript. The migration guide's
initial activation statistics remain a historical record, rather than current
results. Human validation and a full model-refit bootstrap remain outstanding.
