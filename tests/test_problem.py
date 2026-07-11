# SPDX-License-Identifier: MIT
from __future__ import annotations

import random
import statistics

from llm6g import hard_problem as hp


def test_bounds_and_validate():
    rng = random.Random(1)
    for _ in range(500):
        params = hp.sample_random(rng)
        assert hp.validate(params) == params
    bad = {
        "numUes": 999,
        "txPower": -5,
        "scheduler": "ZZ",
        "mimoLayers": 3,
        "bandSplit": 5.0,
        "txAntennas": 7,
    }
    valid = hp.validate(bad)
    assert valid["numUes"] == 40
    assert valid["txPower"] == 20.0
    assert valid["scheduler"] == "RR"
    assert valid["mimoLayers"] == 1
    assert valid["bandSplit"] == 1.0
    assert valid["txAntennas"] == 4


def test_eight_scenarios_and_twelve_parameters():
    assert len(hp.SCENARIOS) == 8
    assert hp.DIM == 12
    assert set(hp.SCENARIO_BY_ID) == set(range(1, 9))


def test_determinism_and_encoding_round_trip():
    rng = random.Random(7)
    params = hp.sample_random(rng)
    first = hp.evaluate(params, 1, seed=3)["score"]
    second = hp.evaluate(params, 1, seed=3)["score"]
    assert first == second
    decoded = hp.decode_vector(hp.encode_params(params))
    assert hp.validate(decoded) == hp.validate(params)


def test_non_stationarity_penalty():
    rng = random.Random(5)
    for _ in range(200):
        result = hp.evaluate(hp.sample_random(rng), 5, seed=0)
        if result["phase_std"] > 5.0:
            assert result["score"] < result["mean_score"]
            assert abs(
                result["mean_score"]
                - result["score"]
                - hp.ROBUSTNESS_LAMBDA * result["phase_std"]
            ) < 1e-6
            return
    raise AssertionError("No high-variance configuration was found.")


def test_reported_random_search_difficulty():
    budget = 400
    restarts = 15
    evaluations_needed = []
    for scenario in hp.SCENARIOS:
        informed = hp.evaluate(
            hp.knowledgeable_config(scenario["id"]), scenario["id"], seed=0
        )["score"]
        per_seed = []
        for repetition in range(restarts):
            rng = random.Random(5000 + repetition)
            crossing = budget
            best = -1e9
            for evaluation in range(1, budget + 1):
                best = max(
                    best,
                    hp.evaluate(
                        hp.sample_random(rng), scenario["id"], seed=0
                    )["score"],
                )
                if best >= informed:
                    crossing = evaluation
                    break
            per_seed.append(crossing)
        evaluations_needed.append(statistics.median(per_seed))
    assert statistics.median(evaluations_needed) == 252
