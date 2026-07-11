# SPDX-License-Identifier: MIT
"""NS-3 (5G-LENA) evaluation backend for the 6G resource-allocation study.

This module is a drop-in replacement for the evaluation parts of
``hard_problem.py``. It keeps the exact parameter schema, scenario family,
use-case weights, scoring equations, vector encode/decode, and the expert
heuristic from ``hard_problem``, and only replaces the per-phase metric source:
instead of the physically motivated mock, it runs the 5G-LENA program
``cttc-nr-llm6g`` and reads real packet-level metrics.

Because it exposes the same public names that the optimizers depend on
(``DIM``, ``decode_vector``, ``encode_params``, ``score_vector``, ``evaluate``,
``validate``, ``knowledgeable_config``, ``SCENARIOS``, ``SCENARIO_BY_ID``), the
existing baselines in ``hard_baselines.py`` run unchanged. The runner switches
backends with one line::

    import hard_baselines as hb, ns3_problem as n3
    hb.hp = n3                 # baselines now evaluate with NS-3

Design notes
------------
* The objective is deterministic by default: a fixed ns-3 RngRun is used, so a
  given configuration always yields the same metrics. This matches the mock,
  where ``evaluate(seed=0)`` is deterministic and the variation across
  repetitions comes from the optimizer search, not the objective. It also lets
  the backend cache results, so identical configurations are simulated once.
* The throughput score normalizes by a per-scenario capacity. The mock capacity
  is in its own units and will not match absolute NS-3 Mbps, so ``NS3_CAPACITY``
  overrides it. ``run_ns3_validation.py`` estimates these values during
  calibration and writes them to an explicit non-destructive output path; this
  module loads the immutable reference file by default.
* No NS-3 build exists in the authoring sandbox, so this file is verified for
  Python correctness and interface match only. Run it against a built ns-3 +
  5G-LENA tree (see README_NS3.md).
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from typing import Dict, List

# Reuse every structural and scoring component so the score is identical to the
# mock except for the raw metrics. Importing _phase_score guarantees the exact
# per-phase formula (Eq. 2 and 3) and the same use-case weights.
from . import hard_problem as hp
from .hard_problem import (  # noqa: F401  (re-exported for the optimizers)
    PARAM_SCHEMA,
    PARAM_ORDER,
    SCENARIOS,
    SCENARIO_BY_ID,
    USE_CASE_WEIGHTS,
    ROBUSTNESS_LAMBDA,
    DIM,
    validate,
    decode_vector,
    encode_params,
    knowledgeable_config,
    sample_random,
    sc_lookup,
    _phase_score,
    _INTERF_DB,
)

# ----------------------------------------------------------------------------
# Configuration. Override through environment variables or by editing here.
# ----------------------------------------------------------------------------
NS3_DIR = os.environ.get("NS3_DIR", os.path.expanduser("~/ns-3-dev"))
NS3_PROGRAM = os.environ.get("NS3_PROGRAM", "cttc-nr-llm6g")
# Seconds of traffic simulated per load phase. Short is enough for stable
# FlowMonitor statistics on a single cell and keeps each run fast.
SIM_TIME = float(os.environ.get("NS3_SIM_TIME", "0.5"))
# Fixed ns-3 RngRun for a deterministic objective. Set NS3_STOCHASTIC=1 to make
# the RngRun follow the evaluation seed instead.
RNG_RUN = int(os.environ.get("NS3_RNG_RUN", "1"))
STOCHASTIC = os.environ.get("NS3_STOCHASTIC", "0") == "1"
RUN_TIMEOUT = int(os.environ.get("NS3_RUN_TIMEOUT", "350"))  # seconds per run
VERBOSE = os.environ.get("NS3_VERBOSE", "0") == "1"
ALLOW_FAILURE_SCORE = os.environ.get("NS3_ALLOW_FAILURE_SCORE", "0") == "1"
# Cell radius (UE max distance from the gNB, metres). The mMTC scenario is a
# rural wide-area cell, so its UEs sit far out and the link budget binds,
# which is the capacity axis the mock penalizes. eMBB and URLLC keep the
# dense 150 m radius that already calibrated well.
MMTC_RADIUS = float(os.environ.get("NS3_MMTC_RADIUS", "1000"))
DEFAULT_RADIUS = float(os.environ.get("NS3_CELL_RADIUS", "150"))

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_CAP_FILE = os.environ.get(
    "LLM6G_NS3_CAPACITY_FILE",
    os.path.join(_REPO_ROOT, "results", "ns3", "ns3_capacity.json"),
)


def _resolve_binary():
    """Locate the compiled scratch binary so each run executes it directly,
    bypassing the ./ns3 Python wrapper. The wrapper re-checks the build through
    cmake/ninja on every call, which collapses under many parallel workers."""
    import glob
    found = []
    for c in glob.glob(os.path.join(NS3_DIR, "build", "scratch", "**", "*" + NS3_PROGRAM + "*"), recursive=True):
        if os.path.isfile(c) and os.access(c, os.X_OK):
            found.append(c)
    found.sort(key=lambda c: (0 if "opt" in os.path.basename(c) else 1, len(c)))
    return found[0] if found else None


NS3_BINARY = _resolve_binary()


# Optional cross-worker progress counter, set by the runner's pool initializer.
# Each actual ns-3 simulation (cache miss) ticks it; every Nth tick prints a line
# so a long parallel run shows continuous progress in the log.
_PROGRESS = None
_PROGRESS_EVERY = int(os.environ.get("NS3_PROGRESS_EVERY", "20"))


def set_progress(counter):
    global _PROGRESS
    _PROGRESS = counter


def _tick_progress():
    if _PROGRESS is None:
        return
    with _PROGRESS.get_lock():
        _PROGRESS.value += 1
        n = _PROGRESS.value
    if _PROGRESS_EVERY > 0 and n % _PROGRESS_EVERY == 0:
        print("[progress] %d simulations completed" % n, flush=True)


def _load_ns3_capacity() -> Dict[int, float]:
    if os.path.exists(_CAP_FILE):
        try:
            raw = json.load(open(_CAP_FILE))
            return {int(k): float(v) for k, v in raw.items()}
        except Exception:
            return {}
    return {}


NS3_CAPACITY: Dict[int, float] = _load_ns3_capacity()


def _capacity(sc: Dict) -> float:
    """Throughput-normalization capacity. Uses the calibrated NS-3 value when
    available, otherwise falls back to the mock capacity."""
    return float(NS3_CAPACITY.get(sc["id"], sc["capacity"]))


# ----------------------------------------------------------------------------
# NS-3 invocation and metric parsing, with caching.
# ----------------------------------------------------------------------------
_CACHE: Dict[str, Dict[str, float]] = {}


def _cache_key(p: Dict, sc: Dict, load_mult: float, rng_run: int) -> str:
    parts = [f"{k}={p[k]}" for k in PARAM_ORDER]
    parts += [f"sid={sc['id']}", f"lm={round(load_mult, 3)}", f"rr={rng_run}"]
    return "|".join(parts)


def _build_args(p: Dict, sc: Dict, load_mult: float, rng_run: int) -> List[str]:
    """Command-line arguments for cttc-nr-llm6g, one per knob plus scenario."""
    mob_mps = round(sc["mobility"] * 1000.0 / 3600.0, 3)  # km/h to m/s
    radius = MMTC_RADIUS if sc["use_case"] == "mMTC" else DEFAULT_RADIUS
    a = {
        "scheduler": p["scheduler"],
        "numUes": p["numUes"],
        "txPower": p["txPower"],
        "bandwidthMHz": p["bandwidth"],
        "packetSize": p["packetSize"],
        "trafficRateMbps": round(p["trafficRate"] * load_mult, 4),
        "numerology": p["numerology"],
        "mimoLayers": p["mimoLayers"],
        "harqProcesses": p["harqProcesses"],
        "targetBler": p["targetBler"],
        "bandSplit": p["bandSplit"],
        "txAntennas": p["txAntennas"],
        "useCase": sc["use_case"],
        "mobilityMps": mob_mps,
        "interferenceDb": _INTERF_DB[sc["interference"]],
        "cellRadiusM": radius,
        "simTime": SIM_TIME,
        "rngRun": rng_run,
    }
    return [f"--{k}={v}" for k, v in a.items()]


def _parse_metrics(stdout: str) -> Dict[str, float]:
    """Read the RESULT_JSON line emitted by the simulation program."""
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("RESULT_JSON="):
            payload = line[len("RESULT_JSON="):]
            d = json.loads(payload)
            return {
                "throughput": float(d["throughput_mbps"]),
                "pdr": float(d["pdr"]),
                "delay": float(d["delay_ms"]),
                "fairness": float(d["fairness"]),
            }
    raise RuntimeError("no RESULT_JSON line in ns-3 output")


def ensure_available() -> None:
    """Fail early when the configured ns-3 installation is unavailable."""
    if not os.path.isdir(NS3_DIR):
        raise FileNotFoundError(
            f"NS3_DIR does not exist: {NS3_DIR}. Set NS3_DIR to a built ns-3 tree."
        )
    wrapper = os.path.join(NS3_DIR, "ns3")
    if NS3_BINARY is None and not os.path.isfile(wrapper):
        raise FileNotFoundError(
            f"Neither a compiled {NS3_PROGRAM} binary nor the ns3 wrapper was found in {NS3_DIR}."
        )


def _failed_metrics(message: str) -> Dict[str, float]:
    if ALLOW_FAILURE_SCORE:
        print("WARNING:", message, "; scoring as failure", flush=True)
        return {"throughput": 0.0, "pdr": 0.0, "delay": 1e3, "fairness": 0.0}
    raise RuntimeError(message)


def _run_ns3(p: Dict, sc: Dict, load_mult: float, rng_run: int) -> Dict[str, float]:
    args = _build_args(p, sc, load_mult, rng_run)
    _tick_progress()
    run_env = None
    if NS3_BINARY:
        cmd = [NS3_BINARY] + args
        run_env = dict(os.environ)
        libdir = os.path.join(NS3_DIR, "build", "lib")
        run_env["LD_LIBRARY_PATH"] = libdir + os.pathsep + run_env.get("LD_LIBRARY_PATH", "")
    else:
        cmd = ["./ns3", "run", "--no-build", NS3_PROGRAM + " " + " ".join(args)]
    if VERBOSE:
        print("NS3 run:", NS3_PROGRAM, " ".join(args))
    try:
        proc = subprocess.run(
            cmd, cwd=NS3_DIR, capture_output=True, text=True, timeout=RUN_TIMEOUT, env=run_env
        )
    except subprocess.TimeoutExpired as exc:
        return _failed_metrics(f"ns-3 run timed out after {RUN_TIMEOUT}s: {exc}")
    except OSError as exc:
        return _failed_metrics(f"ns-3 process could not be started: {exc}")
    if proc.returncode != 0:
        detail = proc.stderr[-2000:] if proc.stderr else "no stderr"
        return _failed_metrics(
            f"ns-3 run failed with return code {proc.returncode}: {detail}"
        )
    try:
        return _parse_metrics(proc.stdout)
    except Exception as exc:
        detail = proc.stdout[-2000:] if proc.stdout else "no stdout"
        return _failed_metrics(f"could not parse ns-3 metrics: {exc}; output: {detail}")


def _phase_metrics_ns3(p: Dict, sc: Dict, load_mult: float, seed: int) -> Dict[str, float]:
    rng_run = (RNG_RUN + seed) if STOCHASTIC else RNG_RUN
    key = _cache_key(p, sc, load_mult, rng_run)
    if key in _CACHE:
        return _CACHE[key]
    m = _run_ns3(p, sc, load_mult, rng_run)
    _CACHE[key] = m
    return m


# ----------------------------------------------------------------------------
# Public evaluation interface, mirroring hard_problem.evaluate exactly.
# Only the metric source differs; the scoring is identical.
# ----------------------------------------------------------------------------
def evaluate(params: Dict, scenario, seed: int = 0) -> Dict:
    sc = SCENARIO_BY_ID[scenario] if isinstance(scenario, int) else scenario
    p = validate(params)

    # Score with the calibrated NS-3 capacity. _phase_score reads sc["capacity"],
    # so evaluate against a shallow copy that carries the calibrated value.
    sc_scored = dict(sc)
    sc_scored["capacity"] = _capacity(sc)

    phase_scores = []
    agg_metrics = {"throughput": 0.0, "pdr": 0.0, "delay": 0.0, "fairness": 0.0}
    for lm in sc["phases"]:
        m = _phase_metrics_ns3(p, sc, lm, seed)
        phase_scores.append(_phase_score(m, sc_scored))
        for k in agg_metrics:
            agg_metrics[k] += m[k] / len(sc["phases"])

    mean_s = sum(phase_scores) / len(phase_scores)
    var = sum((s - mean_s) ** 2 for s in phase_scores) / len(phase_scores)
    std_s = math.sqrt(var)
    aggregate = mean_s - ROBUSTNESS_LAMBDA * std_s

    return {
        "score": aggregate,
        "mean_score": mean_s,
        "phase_std": std_s,
        "phase_scores": phase_scores,
        "metrics": agg_metrics,
    }


def score_vector(x: List[float], scenario, seed: int = 0) -> float:
    """Decode a unit-cube vector and return the NS-3 score. Same signature as
    hard_problem.score_vector, so the baselines work unchanged."""
    return evaluate(decode_vector(x), scenario, seed=seed)["score"]


def cache_size() -> int:
    return len(_CACHE)


def save_cache(path: str) -> None:
    json.dump(_CACHE, open(path, "w"), indent=1)


def set_ns3_capacity(cap: Dict[int, float], persist: bool = False) -> None:
    """Set NS-3 throughput normalizers.

    Persistence is opt-in so calibration cannot overwrite the immutable
    reference capacity file accidentally.
    """
    global NS3_CAPACITY
    NS3_CAPACITY = {int(k): float(v) for k, v in cap.items()}
    if persist:
        os.makedirs(os.path.dirname(_CAP_FILE), exist_ok=True)
        with open(_CAP_FILE, "w", encoding="utf-8") as handle:
            json.dump(NS3_CAPACITY, handle, indent=1)
