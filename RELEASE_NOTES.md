# Release 1.0.5

Alignment with the revised Journal of Engineering manuscript (revision of
October 2026).

- Manuscript title updated to "Language-Model Priors for Resource Allocation in 6G Mobile Networks" in README, CITATION.cff, and
  docs/MANUSCRIPT_ALIGNMENT.md.
- Figures 2, 3, and 5 in `figures/manuscript/` replaced by the versions in the
  revised manuscript: Figure 2 re-exported at 300 dpi with the arrow label
  "one-evaluation prior"; Figures 3 and 5 redrawn as 4 x 2 full-page grids from
  the same stored results. The scripts, CSV data, and draw.io source are in
  `figures/manuscript/source/`.
- Package, runtime, citation, and test version synchronized at 1.0.5.
- No change to code, stored results, or tables.

# Release 1.0.4

- Python 3.10 dropped from the CI matrix (tomllib requires 3.11+).
- Seed-bank regression tests compare floating-point histories within a
  relative tolerance of 1e-9 instead of exact equality, so that the checks pass
  across operating systems and BLAS builds. The stored reference values are
  unchanged.

# Release 1.0.3

This release is the final deeply audited GitHub package aligned with the
supplied Journal of Engineering manuscript.

## Scientific reproducibility correction

- Added the exact historical ten-configuration Llama and Qwen seed banks used
  for the 30-repetition seeded experiments.
- Separated the 30-run model campaigns used for Table 3 from the 10-run seed
  banks used for Tables 4--6.
- Changed the seeded-study defaults to the historical seed banks.
- Added immutable provenance hashes and explicit `r mod 10` mapping.
- Added regression tests that exactly reproduce all 960 PSO seed-ablation
  histories, all 240 GA ensemble histories, all 240 DE ensemble histories, all
  480 stored model-configuration scores, and the first GP acquisition step for
  every scenario.

## Release engineering and licensing

- Fixed the clean-install/CI contradiction by checking the release tree before
  editable installation and permitting installation-generated `.egg-info`
  metadata during post-install verification.
- Replaced the misleading `results/mock/` layout with
  `results/model_campaigns/`, `results/seed_bank/`, and `results/seeded/`.
- Applied MIT SPDX headers to Python files and GPL-2.0-only to the 5G-LENA C++
  integration, with CC BY 4.0 for original research artifacts.
- Removed the manuscript DOCX from the public code package; numerical alignment
  is documented and enforced through regression tests.
- Updated the package, runtime, and citation version to 1.0.3.

## Remaining external limitations

The Python implementation and stored proxy histories are deeply tested. The C++
program has not been compiled inside this packaging environment against the
exact author-side ns-3/5G-LENA checkout. Exact regeneration of the original LLM
text responses also requires the original Ollama version and model digests,
which were not preserved; the parsed configurations actually used in the paper
are included and cryptographically identified.
