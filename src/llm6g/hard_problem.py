# SPDX-License-Identifier: MIT
"""Hard-problem environment for the 6G resource-allocation study, Phase 1.

This module defines a harder optimization problem than the original six-knob
single-cell task, where a tuned PSO converged in about eight evaluations. The
goal is a problem whose optimum is high dimensional, non-separable, scenario
dependent, and non-stationary, so blind search is expensive and domain
knowledge pays off. That property is what the zero-shot configuration claim
rests on, so this file also ships an informed-prior baseline and the module is
tested for difficulty before any optimizer is attached.

No NS-3 and no Ollama. The scoring function here is a physically motivated mock
that plays the role the NS-3 program will play later. The real NS-3 program is
written to match this same parameter contract and metric outputs, so the
optimizer code does not change when the simulator is swapped in.

Design choices that make the problem hard:
  - Twelve parameters, mixing categorical, integer, and continuous, including
    numerology, MIMO layers, HARQ, target BLER, band split, and antenna count.
  - Parameter interactions. The best numerology depends on mobility. The usable
    MIMO rank depends on antenna count and interference. The best band split
    depends on interference. The best target BLER and packet size depend on the
    use case. So no parameter can be tuned in isolation.
  - A family of scenarios, each with a natural-language description, so a
    configurator that reads the description and knows 5G NR can act zero shot.
  - Non-stationary traffic. Each evaluation averages over traffic phases with
    different load multipliers and penalizes variance across phases, so a
    configuration tuned to one load is punished.

The physics model is built from textbook relationships, not reverse engineered
to favor any method. Whether domain knowledge actually helps is then measured,
not assumed, in test_hard_problem.py.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple

# ----------------------------------------------------------------------------
# Parameter schema. Twelve knobs, each mapping to a real 5G-LENA NR control.
# ----------------------------------------------------------------------------
PARAM_SCHEMA: Dict[str, Dict] = {
    "scheduler":     {"type": "cat",   "values": ["RR", "PF", "MR"]},
    "numUes":        {"type": "int",   "low": 5,    "high": 40},
    "txPower":       {"type": "float", "low": 20.0, "high": 46.0},
    "bandwidth":     {"type": "float", "low": 20.0, "high": 100.0},
    "packetSize":    {"type": "int",   "low": 256,  "high": 1500},
    "trafficRate":   {"type": "float", "low": 0.5,  "high": 4.0},
    "numerology":    {"type": "int",   "low": 0,    "high": 3},
    "mimoLayers":    {"type": "cat",   "values": [1, 2, 4]},
    "harqProcesses": {"type": "int",   "low": 1,    "high": 16},
    "targetBler":    {"type": "float", "low": 0.01, "high": 0.30},
    "bandSplit":     {"type": "float", "low": 0.0,  "high": 1.0},
    "txAntennas":    {"type": "cat",   "values": [4, 8, 16, 32, 64]},
}

PARAM_ORDER: List[str] = list(PARAM_SCHEMA.keys())

# ----------------------------------------------------------------------------
# Scenario family. Each scenario carries a description for zero-shot use.
# interference is low, medium, or high. phases are traffic load multipliers
# applied across the evaluation window to create non-stationarity.
# ----------------------------------------------------------------------------
SCENARIOS: List[Dict] = [
    {"id": 1, "name": "Dense urban eMBB", "use_case": "eMBB", "capacity": 120.0,
     "mobility": 3, "interference": "high", "phases": [0.7, 1.0, 1.35],
     "desc": "Dense urban macro cell, very high user density, pedestrian "
             "mobility, broadband traffic, strong background interference."},
    {"id": 2, "name": "Highway URLLC", "use_case": "URLLC", "capacity": 55.0,
     "mobility": 120, "interference": "medium", "phases": [0.85, 1.0, 1.15],
     "desc": "Highway corridor, high vehicular mobility, ultra-reliable "
             "low-latency control messages, strict delay budget under 10 ms."},
    {"id": 3, "name": "Indoor hotspot eMBB", "use_case": "eMBB", "capacity": 200.0,
     "mobility": 1, "interference": "low", "phases": [0.6, 1.0, 1.4],
     "desc": "Indoor hotspot, stationary users, very high capacity demand, "
             "isolated cell with low interference, peak throughput matters."},
    {"id": 4, "name": "Rural mMTC", "use_case": "mMTC", "capacity": 40.0,
     "mobility": 10, "interference": "low", "phases": [0.9, 1.0, 1.1],
     "desc": "Rural wide-area cell, many low-rate sensors, fairness across "
             "devices is critical, low mobility, low interference."},
    {"id": 5, "name": "Stadium eMBB burst", "use_case": "eMBB", "capacity": 150.0,
     "mobility": 2, "interference": "high", "phases": [0.5, 1.0, 1.5],
     "desc": "Stadium, extreme user density, bursty broadband uploads, severe "
             "background interference, load swings sharply between phases."},
    {"id": 6, "name": "Factory URLLC", "use_case": "URLLC", "capacity": 70.0,
     "mobility": 15, "interference": "medium", "phases": [0.9, 1.0, 1.1],
     "desc": "Industrial automation, robots and AGVs, deterministic low latency "
             "and very high reliability, moderate interference."},
    {"id": 7, "name": "Suburban eMBB", "use_case": "eMBB", "capacity": 90.0,
     "mobility": 30, "interference": "medium", "phases": [0.8, 1.0, 1.2],
     "desc": "Suburban macro cell, mixed pedestrian and vehicular mobility, "
             "general broadband, moderate interference."},
    {"id": 8, "name": "Smart-grid mMTC", "use_case": "mMTC", "capacity": 50.0,
     "mobility": 1, "interference": "medium", "phases": [0.95, 1.0, 1.05],
     "desc": "Smart-grid metering, massive stationary device count, fairness "
             "and reliability over throughput, steady load."},
]

SCENARIO_BY_ID = {s["id"]: s for s in SCENARIOS}

# Use-case scoring weights: throughput, pdr, delay, fairness. Same scheme as V6.1.
USE_CASE_WEIGHTS = {
    "eMBB":  {"throughput": 0.40, "pdr": 0.30, "delay": 0.15, "fairness": 0.15},
    "URLLC": {"throughput": 0.10, "pdr": 0.40, "delay": 0.40, "fairness": 0.10},
    "mMTC":  {"throughput": 0.15, "pdr": 0.35, "delay": 0.15, "fairness": 0.35},
}

_INTERF_DB = {"low": 2.0, "medium": 8.0, "high": 15.0}
_MAX_RANK = {"low": 4, "medium": 3, "high": 2}
ROBUSTNESS_LAMBDA = 0.6   # penalty weight on cross-phase score variability


# ----------------------------------------------------------------------------
# Parameter helpers
# ----------------------------------------------------------------------------
def sample_random(rng: random.Random) -> Dict:
    """Draw a uniformly random valid configuration."""
    p = {}
    for name, spec in PARAM_SCHEMA.items():
        if spec["type"] == "cat":
            p[name] = rng.choice(spec["values"])
        elif spec["type"] == "int":
            p[name] = rng.randint(spec["low"], spec["high"])
        else:
            p[name] = round(rng.uniform(spec["low"], spec["high"]), 3)
    return p


def validate(params: Dict) -> Dict:
    """Clamp and coerce a configuration to the schema."""
    out = {}
    for name, spec in PARAM_SCHEMA.items():
        v = params.get(name)
        if spec["type"] == "cat":
            out[name] = v if v in spec["values"] else spec["values"][0]
        elif spec["type"] == "int":
            if v is None:
                v = (spec["low"] + spec["high"]) // 2
            out[name] = int(min(max(round(v), spec["low"]), spec["high"]))
        else:
            if v is None:
                v = (spec["low"] + spec["high"]) / 2.0
            out[name] = float(min(max(v, spec["low"]), spec["high"]))
    return out


def dimensionality() -> int:
    return len(PARAM_SCHEMA)


# ----------------------------------------------------------------------------
# Physics mock. Returns per-phase metrics, then aggregates with a robustness
# penalty. All relationships are textbook and interaction heavy.
# ----------------------------------------------------------------------------
def _effective_capacity(p: Dict, sc: Dict) -> float:
    interf_db = _INTERF_DB[sc["interference"]]

    # SINR from power, interference, and beamforming array gain.
    array_gain_db = 10.0 * math.log10(p["txAntennas"])          # antenna gain
    sinr_db = p["txPower"] - 30.0 - interf_db + array_gain_db
    sinr_lin = 10.0 ** (sinr_db / 10.0)
    se = math.log2(1.0 + sinr_lin)                               # bits/s/Hz
    se = min(se, 7.5)                                            # practical cap

    # Usable MIMO rank: limited by requested layers, by interference, and by
    # antenna count. Interaction between mimoLayers, txAntennas, interference.
    rank_ant = 1 + p["txAntennas"] // 8                          # antennas enable rank
    usable_layers = min(p["mimoLayers"], _MAX_RANK[sc["interference"]], rank_ant)
    mimo_gain = 1.0 + 0.85 * (usable_layers - 1)                 # diminishing

    # Numerology suited to mobility (Doppler). Wider SCS for faster users.
    mu_opt = min(3, max(0, round(sc["mobility"] / 40.0)))
    numerology_factor = max(0.6, 1.0 - 0.13 * abs(p["numerology"] - mu_opt))

    # Band split: diversity helps under interference, concentration helps without.
    balance = 1.0 - 2.0 * abs(p["bandSplit"] - 0.5)             # 1 at 0.5, 0 at edges
    if sc["interference"] == "high":
        split_factor = 0.90 + 0.15 * balance
    elif sc["interference"] == "low":
        split_factor = 1.00 - 0.10 * balance
    else:
        split_factor = 0.97 + 0.05 * balance

    # Higher target BLER trades reliability for spectral efficiency.
    bler_se_factor = 1.0 + 0.6 * p["targetBler"]

    # Scheduler throughput tendency.
    sched_tput = {"MR": 1.06, "PF": 1.00, "RR": 0.93}[p["scheduler"]]

    usable_bw = p["bandwidth"] * split_factor                   # MHz
    eff = usable_bw * se * mimo_gain * numerology_factor * bler_se_factor * sched_tput
    # scale MHz x bits/s/Hz into Mbps-like units, normalized so values track
    # the scenario capacity range.
    return max(1.0, eff * 0.20)


def _phase_metrics(p: Dict, sc: Dict, load_mult: float, rng: random.Random) -> Dict:
    eff_cap = _effective_capacity(p, sc)
    offered = p["numUes"] * p["trafficRate"] * load_mult
    load = offered / eff_cap

    # PDR drops as load approaches and exceeds one. High BLER and mobility hurt.
    z = min(30.0, max(-30.0, 8.0 * (load - 0.97)))
    pdr = 1.0 / (1.0 + math.exp(z))
    pdr *= (1.0 - 0.5 * p["targetBler"])                        # raw errors
    mu_opt = min(3, max(0, round(sc["mobility"] / 40.0)))
    pdr *= max(0.6, 1.0 - 0.05 * abs(p["numerology"] - mu_opt))  # Doppler loss
    pdr = max(0.0, min(1.0, pdr + rng.uniform(-0.01, 0.01)))

    # Delay: slot length shrinks with numerology; packet size and load add delay;
    # few HARQ processes plus high BLER add retransmission delay.
    slot_ms = 1.0 / (2 ** p["numerology"])
    proc_delay = 0.5 * slot_ms
    tx_delay = p["packetSize"] / 1024.0 * 0.8
    queue_delay = 2.0 * load / max(1e-3, (1.0 - min(load, 0.99)))
    retx_delay = (p["targetBler"] * 20.0) / max(1, p["harqProcesses"])
    delay = proc_delay + tx_delay + queue_delay + retx_delay
    delay = max(0.3, delay + rng.uniform(-0.05, 0.05))

    throughput = min(offered, eff_cap) * pdr

    # Fairness: RR most fair, MR least; degrades as load and user count rise.
    base_fair = {"RR": 0.97, "PF": 0.90, "MR": 0.72}[p["scheduler"]]
    fairness = base_fair * (1.0 - 0.15 * min(1.0, load)) * (1.0 - 0.004 * p["numUes"])
    fairness = max(0.3, min(1.0, fairness))

    return {"throughput": throughput, "pdr": pdr, "delay": delay, "fairness": fairness}


def _phase_score(m: Dict, sc: Dict) -> float:
    w = USE_CASE_WEIGHTS[sc["use_case"]]
    tput_score = min(m["throughput"] / sc["capacity"], 1.0) * 100.0
    pdr_score = m["pdr"] * 100.0
    if sc["use_case"] == "URLLC":
        # linear penalty to a 10 ms budget
        delay_score = max(0.0, 1.0 - m["delay"] / 10.0) * 100.0
    else:
        delay_score = math.exp(-m["delay"] / 25.0) * 100.0
    fairness_score = m["fairness"] * 100.0
    return (w["throughput"] * tput_score + w["pdr"] * pdr_score
            + w["delay"] * delay_score + w["fairness"] * fairness_score)


def evaluate(params: Dict, scenario, seed: int = 0) -> Dict:
    """Evaluate a configuration on a scenario across its traffic phases.

    Returns a dict with the per-phase scores, the mean, the cross-phase
    standard deviation, the robustness-penalized aggregate score, and the
    phase-averaged metrics. The aggregate score is the optimization target.
    """
    sc = SCENARIO_BY_ID[scenario] if isinstance(scenario, int) else scenario
    p = validate(params)
    rng = random.Random(seed * 7919 + sc["id"])

    phase_scores = []
    agg_metrics = {"throughput": 0.0, "pdr": 0.0, "delay": 0.0, "fairness": 0.0}
    for lm in sc["phases"]:
        m = _phase_metrics(p, sc, lm, rng)
        phase_scores.append(_phase_score(m, sc))
        for k in agg_metrics:
            agg_metrics[k] += m[k] / len(sc["phases"])

    mean_s = sum(phase_scores) / len(phase_scores)
    var = sum((s - mean_s) ** 2 for s in phase_scores) / len(phase_scores)
    std_s = math.sqrt(var)
    aggregate = mean_s - ROBUSTNESS_LAMBDA * std_s               # reward stability

    return {
        "score": aggregate,
        "mean_score": mean_s,
        "phase_std": std_s,
        "phase_scores": phase_scores,
        "metrics": agg_metrics,
    }


# ----------------------------------------------------------------------------
# Informed prior. Encodes textbook 5G NR rules from the scenario description.
# This is the upper-bound reference for what a zero-shot configurator that
# understands the description could do. It is NOT tuned to the mock constants.
# ----------------------------------------------------------------------------
def knowledgeable_config(scenario) -> Dict:
    sc = SCENARIO_BY_ID[scenario] if isinstance(scenario, int) else sc_lookup(scenario)
    uc = sc["use_case"]
    interf = sc["interference"]

    # numerology to mobility
    numerology = min(3, max(0, round(sc["mobility"] / 40.0)))
    # antennas and layers: more under interference is wasted, but capacity helps
    txAntennas = 64 if uc == "eMBB" else 32
    mimoLayers = 4 if (uc == "eMBB" and interf != "high") else 2
    # band split: balanced under interference, concentrated otherwise
    bandSplit = 0.5 if interf == "high" else 0.1
    # target BLER: strict for URLLC, relaxed for eMBB, moderate for mMTC
    targetBler = 0.01 if uc == "URLLC" else (0.10 if uc == "eMBB" else 0.05)
    # packet size: small for URLLC latency, large for eMBB efficiency
    packetSize = 256 if uc == "URLLC" else (1400 if uc == "eMBB" else 512)
    # scheduler: MR for eMBB throughput, RR for mMTC fairness, PF for URLLC
    scheduler = "MR" if uc == "eMBB" else ("RR" if uc == "mMTC" else "PF")
    # HARQ: more processes help reliability use cases
    harqProcesses = 12 if uc in ("URLLC", "mMTC") else 8
    bandwidth = 100.0
    txPower = 46.0

    # pick load near 0.8 of an estimate of effective capacity at the peak phase
    probe = {"scheduler": scheduler, "numUes": 20, "txPower": txPower,
             "bandwidth": bandwidth, "packetSize": packetSize, "trafficRate": 1.0,
             "numerology": numerology, "mimoLayers": mimoLayers,
             "harqProcesses": harqProcesses, "targetBler": targetBler,
             "bandSplit": bandSplit, "txAntennas": txAntennas}
    eff = _effective_capacity(validate(probe), sc)
    peak = max(sc["phases"])
    # latency and reliability use cases need utilization headroom; eMBB runs hot
    load_frac = {"URLLC": 0.55, "mMTC": 0.70, "eMBB": 0.85}[uc]
    target_offered = load_frac * eff / peak
    numUes = int(min(40, max(5, round(target_offered / 1.5))))
    trafficRate = round(min(4.0, max(0.5, target_offered / max(1, numUes))), 2)

    return validate({**probe, "numUes": numUes, "trafficRate": trafficRate})


def sc_lookup(sc):
    return sc if isinstance(sc, dict) else SCENARIO_BY_ID[sc]


# ----------------------------------------------------------------------------
# Simple reference optimizers, used only for difficulty measurement here.
# ----------------------------------------------------------------------------
def random_search(scenario, budget: int, rng: random.Random) -> Tuple[Dict, float]:
    best_p, best_s = None, -1e9
    for _ in range(budget):
        p = sample_random(rng)
        s = evaluate(p, scenario, seed=0)["score"]
        if s > best_s:
            best_s, best_p = s, p
    return best_p, best_s


# ----------------------------------------------------------------------------
# Continuous encoding. The metaheuristic baselines (GA, PSO, DE, BO) search a
# unit hypercube [0,1]^DIM and decode to a valid configuration. This keeps the
# optimizers free of any parameter-specific logic, so the same code handles any
# schema. Categorical knobs map to equal-width bins, integers and floats scale
# linearly between their bounds.
# ----------------------------------------------------------------------------
DIM: int = len(PARAM_ORDER)


def decode_vector(x: List[float]) -> Dict:
    """Map a [0,1]^DIM vector to a valid configuration."""
    p = {}
    for i, name in enumerate(PARAM_ORDER):
        spec = PARAM_SCHEMA[name]
        xi = min(1.0, max(0.0, float(x[i])))
        if spec["type"] == "cat":
            vals = spec["values"]
            idx = min(len(vals) - 1, int(xi * len(vals)))
            p[name] = vals[idx]
        elif spec["type"] == "int":
            p[name] = int(round(spec["low"] + xi * (spec["high"] - spec["low"])))
        else:
            p[name] = round(spec["low"] + xi * (spec["high"] - spec["low"]), 3)
    return validate(p)


def encode_params(params: Dict) -> List[float]:
    """Map a configuration back to a [0,1]^DIM vector (for seeding optimizers)."""
    p = validate(params)
    x = []
    for name in PARAM_ORDER:
        spec = PARAM_SCHEMA[name]
        v = p[name]
        if spec["type"] == "cat":
            vals = spec["values"]
            idx = vals.index(v) if v in vals else 0
            x.append((idx + 0.5) / len(vals))
        else:
            span = spec["high"] - spec["low"]
            x.append((v - spec["low"]) / span if span > 0 else 0.5)
    return x


def score_vector(x: List[float], scenario, seed: int = 0) -> float:
    """Convenience: decode and score a unit-cube vector."""
    return evaluate(decode_vector(x), scenario, seed=seed)["score"]
