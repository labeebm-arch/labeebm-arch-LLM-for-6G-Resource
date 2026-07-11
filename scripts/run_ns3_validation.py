# SPDX-License-Identifier: MIT
"""Packet-level (NS-3 / 5G-LENA) validation runner for the 6G study.

Reuses the existing optimizers and the LLM prior code, swapping the mock
objective for the NS-3 backend by reassigning the baselines' ``hp`` reference.

Stage 1, calibration: evaluate a spread of configurations in both the mock and
NS-3, estimate a per-scenario NS-3 throughput capacity (so the throughput score
discriminates), and report the Spearman rank correlation between mock and NS-3
scores. Stage 2, reduced comparison: cold PSO and Bayesian optimization, the
expert heuristic, and prior-guided PSO at the matched budget over a few
repetitions.

The comparison runs the independent (method, repetition) optimizer runs in
parallel with ``--jobs N``. Each worker drives one ns-3 simulation at a time,
so N workers use N cores. Within a single optimizer run the evaluations are
sequential, because the optimizer needs each result before the next.

Set NS3_DIR (and optionally NS3_PROGRAM, NS3_SIM_TIME) in the environment. See
README_NS3.md.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import statistics as st
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from llm6g import hard_problem as hp
from llm6g import hard_baselines as hb
from llm6g import hard_llm as hl
from llm6g import ns3_problem as n3
from llm6g.paths import RESULTS_DIR
from llm6g.stored_results import best_campaign_config

LLAMA_DIR = "llama3_8b"
QWEN_DIR = "qwen2.5_7b-instruct-q5_K_M"


def activate_ns3_backend() -> None:
    """Route baseline evaluations to NS-3 without mutating state at import time."""
    hb.hp = n3



# ----------------------------------------------------------------------------
# Priors
# ----------------------------------------------------------------------------
def pick_examples(sid: int, k: int = 3) -> List[Dict]:
    others = [s for s in hp.SCENARIOS if s["id"] != sid]
    picked, chosen = [], set()
    for uc in ["eMBB", "URLLC", "mMTC"]:
        for s in others:
            if s["use_case"] == uc and s["id"] not in chosen:
                picked.append(s)
                chosen.add(s["id"])
                break
    for s in others:
        if len(picked) >= k:
            break
        if s["id"] not in chosen:
            picked.append(s)
            chosen.add(s["id"])
    return picked[:k]


def stored_fewshot(sid: int, model_dir: str) -> Dict:
    """Load the stored few-shot configuration used by the manuscript."""
    return best_campaign_config(model_dir, sid, "LLM-fewshot")


def get_priors(sid: int, use_ollama: bool, url: str, llama_model: str,
               qwen_model: str, seed: int) -> Dict[str, Dict]:
    heur = hp.knowledgeable_config(sid)
    if use_ollama:
        examples = pick_examples(sid)
        llama = hl.llm_fewshot(sid, examples, url, llama_model, mock=False, seed=seed)
        qwen = hl.llm_fewshot(sid, examples, url, qwen_model, mock=False, seed=seed)
    else:
        llama = stored_fewshot(sid, LLAMA_DIR)
        qwen = stored_fewshot(sid, QWEN_DIR)
    return {"heuristic": hp.validate(heur),
            "llama": hp.validate(llama),
            "qwen": hp.validate(qwen)}


# ----------------------------------------------------------------------------
# Prior-guided PSO: hard_baselines.pso with the priors injected as the first
# particles of the initial swarm. Seeds consume the budget like any point.
# ----------------------------------------------------------------------------
def pso_seeded(scenario, budget: int, seed: int, seed_configs: List[Dict]) -> Dict:
    activate_ns3_backend()
    rng = np.random.default_rng(seed)
    t = hb._Tracker(scenario, budget)            # uses hb.hp == n3
    D, n = n3.DIM, 10
    w, c1, c2 = 0.7, 1.5, 1.5
    X = rng.random((n, D))
    for i, cfg in enumerate(seed_configs[:n]):
        X[i] = np.clip(np.array(n3.encode_params(cfg), dtype=float), 0.0, 1.0)
    V = rng.normal(0, 0.1, (n, D))
    f = np.array([t(x) for x in X])
    pbest, pbest_f = X.copy(), f.copy()
    g = int(np.argmax(pbest_f))
    gbest, gbest_f = pbest[g].copy(), pbest_f[g]
    while t.n < budget:
        for i in range(n):
            if t.n >= budget:
                break
            r1, r2 = rng.random(D), rng.random(D)
            V[i] = w * V[i] + c1 * r1 * (pbest[i] - X[i]) + c2 * r2 * (gbest - X[i])
            V[i] = np.clip(V[i], -0.5, 0.5)
            X[i] = np.clip(X[i] + V[i], 0.0, 1.0)
            fi = t(X[i])
            if fi > pbest_f[i]:
                pbest[i], pbest_f[i] = X[i].copy(), fi
                if fi > gbest_f:
                    gbest, gbest_f = X[i].copy(), fi
    return t.result("PriorGuidedPSO")


# ----------------------------------------------------------------------------
# One optimizer run, the unit of parallel work. Module-level so it is picklable.
# ----------------------------------------------------------------------------
def _pool_init(counter):
    activate_ns3_backend()
    n3.set_progress(counter)


def _worker(task):
    activate_ns3_backend()
    kind, sid, budget, seed, seed_cfgs = task
    if kind == "PSO":
        r = hb.pso(sid, budget, seed=seed)
    elif kind == "Bayesian":
        r = hb.bayesian(sid, budget, seed=seed)
    else:  # PriorGuided
        r = pso_seeded(sid, budget, seed=seed, seed_configs=seed_cfgs)
    return (sid, kind, seed, r["history"], r["best_score"])


# ----------------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------------
def _rank(v: List[float]) -> List[float]:
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    for pos, i in enumerate(order):
        r[i] = pos
    return r


def spearman(a: List[float], b: List[float]) -> float:
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(a, b).correlation)
    except Exception:
        ra, rb = _rank(a), _rank(b)
        n = len(a)
        if n < 2:
            return float("nan")
        d2 = sum((ra[i] - rb[i]) ** 2 for i in range(n))
        return 1.0 - 6.0 * d2 / (n * (n * n - 1))


def calibrate(scenarios: List[int], n_random: int, seed: int) -> Dict:
    """Calibrate throughput normalizers and retain all correlation pairs.

    The shipped reference capacity file is never modified. Newly estimated
    capacities are held in memory and are written by ``main`` to the explicit
    ``--capacity-out`` path.
    """
    rng = np.random.default_rng(seed)
    cap: Dict[int, float] = {}
    pairs_all = []
    per_scenario = {}
    for sid in scenarios:
        configs = [hp.knowledgeable_config(sid)]
        labels = ["heuristic"]
        for model_dir, label in ((LLAMA_DIR, "llama"), (QWEN_DIR, "qwen")):
            config = stored_fewshot(sid, model_dir)
            if config:
                configs.append(hp.validate(config))
                labels.append(label)
        for index in range(n_random):
            configs.append(hp.decode_vector(rng.random(n3.DIM)))
            labels.append(f"random_{index + 1}")

        packet_metrics = [n3.evaluate(config, sid)["metrics"] for config in configs]
        throughputs = [metrics["throughput"] for metrics in packet_metrics]
        cap[sid] = max(1e-6, max(throughputs) * 1.05)
        n3.set_ns3_capacity({**n3.NS3_CAPACITY, sid: cap[sid]}, persist=False)

        mock_scores = [hp.evaluate(config, sid)["score"] for config in configs]
        ns3_scores = [n3.evaluate(config, sid)["score"] for config in configs]
        rho = spearman(mock_scores, ns3_scores)
        records = [
            {
                "label": label,
                "config": hp.validate(config),
                "mock_score": mock_score,
                "ns3_score": ns3_score,
                "ns3_metrics": metrics,
            }
            for label, config, mock_score, ns3_score, metrics in zip(
                labels, configs, mock_scores, ns3_scores, packet_metrics
            )
        ]
        per_scenario[str(sid)] = {
            "rho": rho,
            "ns3_capacity": cap[sid],
            "n_configs": len(configs),
            "pairs": records,
        }
        pairs_all.extend(zip(mock_scores, ns3_scores))
        print(
            f"  scenario {sid} ({hp.SCENARIO_BY_ID[sid]['name']}): "
            f"Spearman rho = {rho:.3f}, NS-3 capacity = {cap[sid]:.1f} Mbps, "
            f"n = {len(configs)}"
        )
    overall = spearman([pair[0] for pair in pairs_all], [pair[1] for pair in pairs_all])
    print(f"  overall Spearman rho = {overall:.3f}")
    return {
        "overall_rho": overall,
        "per_scenario": per_scenario,
        "capacity": {str(key): value for key, value in cap.items()},
    }


def load_capacity_file(path: str, scenarios: List[int]) -> Dict[int, float]:
    capacity_path = Path(path).expanduser()
    if not capacity_path.is_file():
        raise FileNotFoundError(
            f"Capacity file does not exist: {capacity_path}. Run calibration first "
            "or provide --capacity-file."
        )
    with capacity_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    capacity = {int(key): float(value) for key, value in raw.items()}
    missing = [sid for sid in scenarios if sid not in capacity]
    if missing:
        raise ValueError(
            f"Capacity file {capacity_path} is missing scenarios: {missing}"
        )
    return capacity


def atomic_write_json(path: str | Path, payload: Dict) -> Path:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(output_path)
    return output_path


# ----------------------------------------------------------------------------
# Reduced comparison
# ----------------------------------------------------------------------------
def mean_curve(histories: List[List[float]], budget: int) -> List[float]:
    padded = []
    for h in histories:
        hh = (list(h) + [h[-1]] * (budget - len(h))) if h else [0.0] * budget
        padded.append(hh[:budget])
    return [st.mean(col) for col in zip(*padded)] if padded else []


def run_comparison(scenarios: List[int], reps: int, budget: int, jobs: int,
                   use_ollama: bool, url: str, llama_model: str,
                   qwen_model: str) -> Dict:
    # Build all independent tasks across scenarios for balanced parallelism.
    heur = {}
    priors = {}
    tasks = []
    # Seed ablation only where prior-guided wins (URLLC, mMTC); eMBB favours cold.
    SEED_ABLATION = {2, 4}
    for sid in scenarios:
        heur[sid] = n3.evaluate(hp.knowledgeable_config(sid), sid)["score"]
        p = get_priors(sid, use_ollama, url, llama_model, qwen_model, seed=0)
        seed_cfgs = [p["qwen"], p["llama"], p["heuristic"]]
        priors[sid] = seed_cfgs
        for r in range(reps):
            tasks.append(("PSO", sid, budget, r, None))
            tasks.append(("Bayesian", sid, budget, r, None))
            tasks.append(("PriorGuided", sid, budget, r, seed_cfgs))
            if sid in SEED_ABLATION:
                tasks.append(("PG_heuristic", sid, budget, r, [p["heuristic"]]))
                tasks.append(("PG_llama", sid, budget, r, [p["llama"]]))
                tasks.append(("PG_qwen", sid, budget, r, [p["qwen"]]))

    print(f"  running {len(tasks)} optimizer runs on {jobs} worker(s) ...", flush=True)
    counter = mp.Value("i", 0)
    t0 = time.time()
    outs = []
    if jobs > 1:
        with mp.Pool(jobs, initializer=_pool_init, initargs=(counter,)) as pool:
            for i, res in enumerate(pool.imap_unordered(_worker, tasks), 1):
                el = (time.time() - t0) / 60.0
                eta = el * (len(tasks) - i) / max(1, i)
                print("  [task %d/%d] sid=%d %s seed=%d best=%.1f  (%.0f min elapsed, ~%.0f min left)"
                      % (i, len(tasks), res[0], res[1], res[2], res[4], el, eta), flush=True)
                outs.append(res)
    else:
        n3.set_progress(counter)
        for i, t in enumerate(tasks, 1):
            outs.append(_worker(t))

    # Group histories per scenario and method (arbitrary kinds, incl. seed ablation).
    from collections import defaultdict
    by = {sid: defaultdict(list) for sid in scenarios}
    for (sid, kind, seed, hist, best) in outs:
        by[sid][kind].append(hist)

    results = {}
    for sid in scenarios:
        name = hp.SCENARIO_BY_ID[sid]["name"]
        pso_mean = mean_curve(by[sid]["PSO"], budget)
        bo_mean = mean_curve(by[sid]["Bayesian"], budget)
        pg_mean = mean_curve(by[sid]["PriorGuided"], budget)
        pso_final = pso_mean[-1] if pso_mean else 0.0
        bo_final = bo_mean[-1] if bo_mean else 0.0
        best_cold_final = max(pso_final, bo_final)
        pg_finals = [h[-1] for h in by[sid]["PriorGuided"]]
        pg_final = st.mean(pg_finals) if pg_finals else 0.0
        pso_finals = [h[-1] for h in by[sid]["PSO"]]
        bo_finals = [h[-1] for h in by[sid]["Bayesian"]]

        cold_env = [max(a, b) for a, b in zip(pso_mean, bo_mean)] if pso_mean else []
        anytime = st.mean([1.0 if pg_mean[i] >= cold_env[i] - 1e-9 else 0.0
                           for i in range(len(pg_mean))]) if pg_mean and cold_env else 0.0
        passes = [i + 1 for i in range(len(pg_mean)) if pg_mean[i] >= best_cold_final]
        pass_eval = passes[0] if passes else budget + 1

        results[sid] = {
            "name": name, "heuristic": heur[sid],
            "cold_PSO_final": pso_final, "cold_Bayesian_final": bo_final,
            "best_cold_final": best_cold_final, "prior_guided_final": pg_final,
            "prior_guided_finals": pg_finals,
            "cold_PSO_finals": pso_finals, "cold_Bayesian_finals": bo_finals,
            "pso_curve": pso_mean, "bayesian_curve": bo_mean, "prior_guided_curve": pg_mean,
            "anytime_vs_cold_envelope": anytime, "pass_eval": pass_eval,
            "beats_best_cold": pg_final > best_cold_final,
            "beats_heuristic": pg_final > heur[sid],
        }
        print(f"  {name}: heuristic={heur[sid]:.1f}  best cold={best_cold_final:.1f}  "
              f"prior-guided={pg_final:.1f}  pass@{pass_eval}  anytime={anytime*100:.0f}%  "
              f"> cold: {results[sid]['beats_best_cold']}  > heur: {results[sid]['beats_heuristic']}")
        if sid in SEED_ABLATION:
            def _seed_final(kind):
                hs = by[sid].get(kind, [])
                return st.mean([h[-1] for h in hs]) if hs else None
            sa = {
                "best_cold": best_cold_final,
                "heuristic_only": _seed_final("PG_heuristic"),
                "llama_only": _seed_final("PG_llama"),
                "qwen_only": _seed_final("PG_qwen"),
                "ensemble": pg_final,
                "heuristic_only_finals": [h[-1] for h in by[sid].get("PG_heuristic", [])],
                "llama_only_finals": [h[-1] for h in by[sid].get("PG_llama", [])],
                "qwen_only_finals": [h[-1] for h in by[sid].get("PG_qwen", [])],
            }
            results[sid]["seed_ablation"] = sa
            print("    seed ablation: heuristic-only=%.1f  llama-only=%.1f  qwen-only=%.1f  "
                  "ensemble=%.1f  (best cold=%.1f)" % (
                      sa["heuristic_only"] or 0.0, sa["llama_only"] or 0.0,
                      sa["qwen_only"] or 0.0, sa["ensemble"], sa["best_cold"]))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="1,2,4")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--budget", type=int, default=60)
    ap.add_argument("--jobs", type=int, default=1, help="parallel optimizer runs")
    ap.add_argument("--calib-random", type=int, default=12)
    ap.add_argument("--calib-only", action="store_true")
    ap.add_argument("--no-calib", action="store_true",
                    help="skip calibration and load the explicit --capacity-file")
    ap.add_argument("--use-ollama", action="store_true")
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--llama-model", default="llama3:8b")
    ap.add_argument("--qwen-model", default="qwen2.5:7b-instruct-q5_K_M")
    ap.add_argument(
        "--capacity-file",
        default=str(RESULTS_DIR / "ns3" / "ns3_capacity.json"),
        help="Capacity JSON to load with --no-calib (reference file by default).",
    )
    ap.add_argument(
        "--capacity-out",
        default=str(RESULTS_DIR / "reproduced" / "ns3_capacity.json"),
        help="Non-destructive output path for newly calibrated capacities.",
    )
    ap.add_argument(
        "--out",
        default=str(RESULTS_DIR / "reproduced" / "ns3_validation.json"),
    )
    args = ap.parse_args()

    scenarios = [int(s) for s in args.scenarios.split(",") if s.strip()]
    print("=" * 70)
    print("NS-3 / 5G-LENA VALIDATION")
    print(f"scenarios={scenarios}  reps={args.reps}  budget={args.budget}  "
          f"jobs={args.jobs}  ns3_dir={n3.NS3_DIR}")
    print("=" * 70)
    activate_ns3_backend()
    n3.ensure_available()

    if args.no_calib:
        capacity = load_capacity_file(args.capacity_file, scenarios)
        n3.set_ns3_capacity(capacity, persist=False)
        print(
            "\n[1] Calibration skipped, reusing capacity file:",
            args.capacity_file,
            {key: round(value, 1) for key, value in capacity.items()},
        )
        calib = {
            "reused_capacity": {str(key): value for key, value in capacity.items()},
            "capacity_file": str(Path(args.capacity_file).expanduser()),
        }
    else:
        print("\n[1] Calibration (mock vs NS-3)")
        calib = calibrate(scenarios, args.calib_random, seed=20260610)
        capacity_path = atomic_write_json(args.capacity_out, calib["capacity"])
        print(f"  saved calibrated capacities to {capacity_path}")

    out = {"calibration": calib, "config": vars(args)}
    if not args.calib_only:
        print("\n[2] Reduced prior-guided comparison")
        out["comparison"] = run_comparison(
            scenarios, args.reps, args.budget, args.jobs, args.use_ollama,
            args.ollama_url, args.llama_model, args.qwen_model)

    output_path = atomic_write_json(args.out, out)
    print(
        f"\nSaved {output_path}  "
        f"(NS-3 cache: {n3.cache_size()} unique runs in main process)"
    )


if __name__ == "__main__":
    main()
