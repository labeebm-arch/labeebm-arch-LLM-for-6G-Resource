# Manuscript alignment

This repository accompanies the manuscript:

> **Language-Model Priors for Resource Allocation in 6G Mobile Networks**

The submitted manuscript file is intentionally not distributed in the code
repository. Its publication and reuse are governed separately by the journal's
terms. The numerical regression tests in `tests/` encode the reported values so
that code and stored results can be checked without bundling the manuscript.

## Output mapping

| Manuscript item | Canonical repository source |
|---|---|
| Table 3 | `results/model_campaigns/*/hard_comparison_results.json` |
| Table 4 | `results/seeded/seeded30_results.json` |
| Table 5 | Host-ablation histories in `results/seeded/seeded30_results.json` |
| Table 6 | Seed-ablation histories in `results/seeded/seeded30_results.json` |
| Table 7 | `results/ns3/ns3_validation.json`, `ns3_oneshot.json`, and `calibration_summary.json` |
| Appendix A | Cold-baseline summaries in the 30-run model campaigns |
| Appendix B | Per-baseline matching evaluations in the 30-run model campaigns |
| Appendix C | `results/model_campaigns/best_configurations_table_c.json` |
| Figure 1 | `results/prior_ns3/figure1_histories.json` |
| Figures 3--5 | `scripts/reproduce_outputs.py` and the stored result files (2 x 4 layout) |
| Figures 3 and 5 as published | `figures/manuscript/source/extract_fig_data.py` and `plot_fig3_fig5.py` (4 x 2 layout, same stored results) |
| Figure 2 | `figures/manuscript/source/fig2_methodology.drawio` |

## Two distinct model-result collections

The manuscript uses two intentionally different collections:

1. `results/model_campaigns/` contains **30 model repetitions** per scenario and
   supports Table 3.
2. `results/seed_bank/` contains the exact **10 historical model
   configurations** per scenario that are cycled in the 30-repetition seeded
   experiments for Tables 4--6. Repetition `r` uses index `r mod 10`.

Using the first ten configurations from the 30-run collection is not equivalent
to using the historical seed bank. The regression tests explicitly prevent that
provenance error.
