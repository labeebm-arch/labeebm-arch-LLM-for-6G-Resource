# Deep final audit — release 1.0.3

**Audit date:** 2026-07-11  
**Verification environment:** Python 3.13.5 on Linux x86_64

## Verdict

The Python repository, stored proxy results, exact seeded-study provenance,
table regeneration, and figure regeneration are internally consistent and
ready for public GitHub upload. Packet-level recompilation and original LLM text
regeneration remain external-environment limitations described below.

## Scientific provenance checks

The recovered historical files in `results/seed_bank/` are distinct from the
30-run campaigns in `results/model_campaigns/`. The seed-bank hashes are fixed
in `results/seed_bank/PROVENANCE.json` and linked from the metadata of
`results/seeded/seeded30_results.json`.

The following deterministic checks were executed against the stored reference
histories:

| Check | Result |
|---|---:|
| Stored LLM zero-shot, few-shot, and search configurations re-evaluated | 480/480 exact |
| PSO Llama-only, Qwen-only, heuristic-only, and ensemble histories | 960/960 exact |
| GA ensemble histories | 240/240 exact |
| DE ensemble histories | 240/240 exact |
| Full 60-evaluation Bayesian histories, selected across all eight scenarios | 8/8 exact |
| Bayesian initialization plus first GP acquisition step in routine tests | 8/8 exact |

The exact seeded mapping is:

```text
prior index for repetition r = r mod 10
optimizer seed for repetition r = 20260610 + r
```

## Automated verification

The final test suite contains 31 tests and passed in the audit environment. It
covers:

- the eight scenarios and twelve-variable benchmark;
- deterministic validation and encoding;
- the reported random-search difficulty statistic;
- Table 3, Tables 4--7, and Appendices A--C;
- host and seed ablations;
- exact seed-bank provenance and hashes;
- parser failure behaviour;
- safe output paths and non-destructive reruns;
- NS-3 command construction and capacity-file validation;
- version, licensing, and release-tree consistency.

All Python entry points responded to `--help`. Reduced mock-model and seeded
smoke runs completed successfully. `scripts/reproduce_outputs.py` regenerated
all nine CSV tables and the four data-driven figures without modifying the
reference result files.

## Release-engineering checks

- clean pre-install tree check;
- editable-install-safe post-install verification;
- package/runtime/citation version synchronized at 1.0.3;
- mixed licences and SPDX headers verified;
- JSON, YAML/CFF, CSV, and Python syntax validation;
- complete SHA-256 manifest coverage for immutable repository files;
- no encrypted files, path traversal entries, symbolic-link entries, compiled
  caches, or build outputs in the release archive;
- no obvious passwords, API keys, email addresses, personal home paths, or
  OneDrive paths detected.

## External limitations

1. The C++ program was not compiled in this packaging environment against the
   authors' exact ns-3/5G-LENA checkout. The exact ns-3 and 5G-LENA commit IDs
   should be recorded before a permanent DOI release.
2. The original Ollama version and model image digests were not preserved.
   Therefore, the package preserves and exactly reuses the parsed configurations
   that generated the reported seeded study, but it cannot guarantee identical
   natural-language responses from a fresh model installation.
3. The historical packet-level calibration retained the reported Spearman
   coefficients but not every original raw calibration pair. New runs save the
   raw pairs.
4. Four proxy variables are logged but not all actively mapped in the supplied
   5G-LENA v4.x integration; this scope is disclosed in the README and source.

## Licensing scope

- Python software and configurations: MIT;
- `ns3/cttc-nr-llm6g.cc`: GPL-2.0-only;
- original results, tables, documentation, and figures: CC BY 4.0;
- article: not bundled and governed separately by journal terms.

The anonymous citation author should be replaced once the public author list may
be disclosed.
