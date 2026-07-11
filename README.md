# Large Language Model Priors for 6G Resource Allocation

Reproducibility code and stored results for the manuscript:

> **Engineer-Free Priors from Large Language Models for 6G Resource Allocation**

The work uses large language models as sources of configuration priors rather
than replacement optimizers. It evaluates zero-shot and cross-scenario few-shot
configurations on an eight-scenario, twelve-variable proxy benchmark, then
injects the model and heuristic priors into matched-budget optimizers. A reduced
packet-level validation is provided for ns-3 with 5G-LENA.

## Release status

Version **1.0.3** is the deeply audited public-code package. It contains:

- the eight proxy scenarios and twelve-parameter objective;
- GA, PSO, DE, Gaussian-process Bayesian optimization, and random search;
- deterministic configuration parsing, validation, encoding, and repair;
- the 30-run Llama-3-8B and Qwen2.5-7B campaigns used for Table 3;
- the distinct, exact 10-configuration historical seed banks used for Tables
  4--6;
- the manuscript's explicit `r mod 10` prior-cycling protocol;
- host and seed ablations over thirty paired repetitions;
- stored packet-level results for dense urban eMBB, highway URLLC, and rural
  mMTC;
- scripts that regenerate Tables 3--7, Appendices A--C, and Figures 1 and 3--5;
- deep regression tests that rerun the deterministic seeded proxy histories.

## Quick verification without Ollama or ns-3

```bash
python scripts/check_release_tree.py
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
python scripts/verify_package.py
```

The pre-install tree check is deliberately run before `pip install -e` because
an editable installation creates package metadata under `src/`. The main
verifier then checks versions, licensing headers, the checksum manifest,
structured result files, Python compilation, the complete test suite, and
output regeneration.

The verification path does not contact Ollama and does not launch ns-3. It uses
the immutable stored results and exact historical parsed configurations.

## Two model-result collections must not be mixed

The manuscript uses two different collections for different claims:

```text
results/model_campaigns/   30 model repetitions per scenario; supports Table 3
results/seed_bank/         exact 10 historical priors cycled in Tables 4--6
```

The first ten entries of the 30-run campaign are **not** the historical seed
bank. The historical seed-bank files were recovered from the original study
archive and are cryptographically recorded in
`results/seed_bank/PROVENANCE.json`.

For seeded repetition `r`:

```text
model-prior index = r mod 10
optimizer seed     = 20260610 + r
```

The deep tests recompute exactly:

- 480 stored model-configuration scores;
- 960 PSO histories for Llama-only, Qwen-only, heuristic-only, and ensemble
  seeding;
- 240 GA ensemble histories;
- 240 DE ensemble histories;
- Bayesian initialization and the first Gaussian-process acquisition step for
  all eight scenarios.

A complete 240-run Bayesian rerun remains available through the main seeded
script but is not placed in routine CI because repeatedly refitting the Gaussian
process is slow.

## Regenerate the manuscript tables and figures

```bash
python scripts/reproduce_outputs.py
```

This writes nine CSV tables under `tables/` and four data-driven figures under
`figures/generated/`. It does not overwrite any reference JSON file.

## Exact seeded-study rerun from the historical priors

The defaults point to the recovered ten-configuration seed banks:

```bash
python scripts/run_seeded_study.py
```

A reduced smoke run is:

```bash
python scripts/run_seeded_study.py \
  --reps 2 --budget 8 --scenarios 1 \
  --out results/reproduced/seeded_smoke.json
```

The full default run executes all cold baselines, four seeded hosts, and four
PSO seed sets for eight scenarios and thirty repetitions. The Bayesian host is
computationally expensive and can take hours depending on the machine.

## Fresh LLM campaigns

Ollama must be running locally. The manuscript used these model tags:

```text
llama3:8b
qwen2.5:7b-instruct-q5_K_M
```

Run new campaigns as follows:

```bash
python scripts/run_proxy_study.py \
  --config configs/proxy_study.yaml \
  --model llama3:8b

python scripts/run_proxy_study.py \
  --config configs/proxy_study.yaml \
  --model qwen2.5:7b-instruct-q5_K_M
```

New outputs are written under `results/reproduced/`. The immutable reference
files are not overwritten. Invalid or incomplete model replies stop the run;
the implementation does not silently substitute random configurations.

To run a new seeded experiment from newly generated campaigns, pass those files
explicitly:

```bash
python scripts/run_seeded_study.py \
  --llama-results results/reproduced/llama3_8b/hard_comparison_results.json \
  --qwen-results results/reproduced/qwen2.5_7b-instruct-q5_K_M/hard_comparison_results.json \
  --out results/reproduced/seeded30_results.json
```

That creates a new study using the first ten configurations in each supplied
file. It is not a claim that a fresh LLM invocation must reproduce the original
language-model outputs byte-for-byte. The original Ollama version and model
image digests were not preserved; the exact parsed configurations actually used
for Tables 4--6 are preserved in `results/seed_bank/`.

## Packet-level validation with ns-3/5G-LENA

The integration program is `ns3/cttc-nr-llm6g.cc`. Build and execution details
are in [`docs/README_NS3.md`](docs/README_NS3.md).

After building the appropriate ns-3/5G-LENA environment:

```bash
export NS3_DIR=/path/to/ns-3-dev
python scripts/run_ns3_validation.py \
  --scenarios 1,2,4 --reps 3 --budget 60 --jobs 4
```

New packet-level files are written under `results/reproduced/`. Launch failures,
timeouts, and unparseable outputs are fatal by default.

### Scope of the 5G-LENA mapping

The supplied program actively maps scheduler, admitted users and load,
transmit power, bandwidth, packet size, numerology, and antenna count. The four
remaining proxy variables (`mimoLayers`, `harqProcesses`, `targetBler`, and
`bandSplit`) are accepted and logged but are not all actively mapped in the
supplied 5G-LENA v4.x program because relevant attributes differ across releases.
This limitation is stated explicitly rather than presenting the packet-level
program as a complete twelve-variable implementation.

The historical campaign retained the reported Spearman coefficients but not all
fifteen raw calibration pairs. New calibration runs save every pair.

## Manuscript mapping

See [`docs/MANUSCRIPT_ALIGNMENT.md`](docs/MANUSCRIPT_ALIGNMENT.md). The article
DOCX is intentionally not bundled because publication rights are governed
separately by the journal. Numerical alignment is enforced by the test suite.

## Repository layout

```text
src/llm6g/                 Core benchmark, model interface, seeded optimizers
scripts/                   Verification, reproduction, and rerun entry points
configs/                   Proxy-study configuration
results/model_campaigns/   Thirty-run model campaigns used for Table 3
results/seed_bank/         Exact ten-configuration priors used for Tables 4--6
results/seeded/            Reference thirty-repetition seeded histories
results/ns3/               Packet-level reference results
results/prior_ns3/         Historical curves used for Figure 1
results/reproduced/        Default destination for new runs; gitignored
ns3/                       5G-LENA C++ integration
figures/manuscript/        Figure renderings corresponding to the manuscript
figures/generated/         Figures regenerated from reference data
tables/                     CSV tables regenerated from reference data
tests/                      Unit, manuscript-consistency, and deep provenance tests
docs/                       NS-3 and manuscript-alignment documentation
LICENSES/                   Full texts for the repository's mixed licenses
```

## Reproducibility limits

- The deterministic Python benchmark and stored seeded histories are deeply
  regression-tested.
- The C++ program was not compiled in this packaging environment against the
  exact author-side ns-3/5G-LENA checkout. Record the exact ns-3 and 5G-LENA
  commits before creating a permanent archival DOI.
- Exact regeneration of the original LLM text responses requires the original
  Ollama version and model digests, which were not preserved. The parsed
  configurations used for the reported seeded study are included and hashed.
- `requirements-lock.txt` records the complete Python 3.13.5 clean-verification
  environment. To recreate it, install that file and then run
  `pip install -e . --no-deps`. For other supported Python versions, use the
  bounded dependencies in `pyproject.toml` and rely on the tests rather than
  forcing an incompatible lock file.

## Licensing

This is a mixed-license repository:

- Python source, scripts, tests, and configurations: **MIT**;
- `ns3/cttc-nr-llm6g.cc`: **GPL-2.0-only**;
- original results, tables, documentation, and figures: **CC BY 4.0**;
- third-party software: its respective licence;
- the submitted or published article: governed separately by the journal.

See the root [`LICENSE`](LICENSE) and the complete texts in [`LICENSES/`](LICENSES/).
Citation metadata is provided in `CITATION.cff`; replace the anonymous author
entry after the public author list can be disclosed.
