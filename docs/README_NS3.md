# Packet-level validation with ns-3 and 5G-LENA

This directory documents the packet-level validation reported in Section 5.7
of the manuscript. The Python optimization code is unchanged; only the metric
source is replaced by the 5G-LENA program in `ns3/cttc-nr-llm6g.cc`.

## Prerequisites

- A built ns-3 tree with the 5G-LENA NR module.
- Python dependencies installed with `pip install -e .` from the repository root.
- Enough CPU time for repeated packet-level runs.

The supplied source targets the 5G-LENA v4.x API on ns-3.42 or newer. API names
may require small changes on other releases.

## Build

From the repository root:

```bash
cp ns3/cttc-nr-llm6g.cc /path/to/ns-3-dev/scratch/
cd /path/to/ns-3-dev
./ns3 configure -d optimized --enable-examples
./ns3 build
```

Confirm that a single run emits a line beginning with `RESULT_JSON=`:

```bash
./ns3 run "cttc-nr-llm6g --numUes=20 --bandwidthMHz=100 \
  --useCase=eMBB --interferenceDb=15 --mobilityMps=1 \
  --simTime=1.5 --rngRun=1"
```

## Environment variables

```bash
export NS3_DIR=/path/to/ns-3-dev
export NS3_PROGRAM=cttc-nr-llm6g
export NS3_SIM_TIME=1.5
export NS3_RNG_RUN=1
export NS3_RUN_TIMEOUT=700
```

Optional variables:

- `NS3_STOCHASTIC=1`: make the ns-3 run number follow the evaluation seed.
- `NS3_VERBOSE=1`: print commands and process errors.
- `NS3_MMTC_RADIUS=1000`: rural mMTC cell radius in metres.
- `NS3_CELL_RADIUS=150`: default eMBB/URLLC radius.
- `NS3_ALLOW_FAILURE_SCORE=1`: convert failed runs to a zero-quality score.
  This is intended only for debugging; reported experiments should fail loudly.

## Reproduce calibration and comparison

From the repository root:

```bash
python scripts/run_ns3_validation.py \
  --scenarios 1,2,4 \
  --calib-only \
  --capacity-out results/reproduced/ns3_capacity.json \
  --out results/reproduced/ns3_calibration.json

python scripts/run_ns3_validation.py \
  --scenarios 1,2,4 \
  --reps 3 \
  --budget 60 \
  --jobs 4 \
  --no-calib \
  --capacity-file results/reproduced/ns3_capacity.json \
  --out results/reproduced/ns3_validation.json
```

The first command estimates per-scenario throughput normalizers and writes them
to a non-destructive path. The second command validates that all requested
scenarios are present in that file before the comparison starts. The immutable
reference file under `results/ns3/` is never overwritten.

To evaluate the stored one-shot and few-shot priors directly in 5G-LENA:

```bash
JOBS=4 python scripts/run_ns3_oneshot.py
```

This writes `results/reproduced/ns3_oneshot.json`.

## Configuration mapping

The supplied C++ build actively maps:

- scheduler;
- admitted UE count;
- transmit power;
- bandwidth-part width;
- packet size;
- per-user traffic rate;
- numerology;
- gNB antenna elements;
- scenario mobility and interference proxy.

`mimoLayers`, `harqProcesses`, `targetBler`, and `bandSplit` are accepted on the
command line and logged, but are not all applied in the supplied v4.x build.
Their 5G-LENA attribute names and multi-BWP APIs are release-dependent. This is
why the manuscript describes native mappings only for the eight active
variables rather than claiming a complete twelve-variable packet-level mapping.

## Stored results

The release contains:

- `ns3_capacity.json`: throughput normalization values;
- `ns3_oneshot.json`: heuristic, model-prior, and random one-call scores;
- `ns3_validation.json`: three-repetition cold and prior-guided comparison;
- `calibration_summary.json`: the three Spearman values reported in the paper.

The historical packet-level file retained the final comparison but not the raw
fifteen calibration pairs. Therefore, the historical correlations can be
reconstructed only by rerunning calibration on the target 5G-LENA installation.
New calibration output stores every configuration-score pair and its raw
packet-level metrics for later audit.
