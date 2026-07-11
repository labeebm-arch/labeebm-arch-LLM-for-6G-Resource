# SPDX-License-Identifier: MIT
"""Seeded optimizer implementations used in manuscript Tables 4--6."""
from __future__ import annotations

import statistics as st
from typing import Dict, Iterable, List

import numpy as np

from . import hard_baselines as hb
from . import hard_problem as hp


def _seeded_population(
    rng: np.random.Generator,
    seed_configs: Iterable[Dict],
    population_size: int,
    dim: int,
) -> np.ndarray:
    population = rng.random((population_size, dim))
    for index, config in enumerate(list(seed_configs)[:population_size]):
        population[index] = np.clip(
            np.asarray(hp.encode_params(config), dtype=float), 0.0, 1.0
        )
    return population


class Tracker:
    def __init__(self, scenario: int, budget: int):
        self.scenario = scenario
        self.budget = budget
        self.n = 0
        self.best = -1e9
        self.history: List[float] = []

    def __call__(self, vector) -> float:
        if self.n >= self.budget:
            return self.best
        score = hp.score_vector(vector, self.scenario, seed=0)
        self.n += 1
        self.best = max(self.best, score)
        self.history.append(self.best)
        return score


def pso_seeded(scenario: int, budget: int, seed: int, seed_configs: List[Dict]) -> List[float]:
    rng = np.random.default_rng(seed)
    tracker = Tracker(scenario, budget)
    dim, swarm_size = hp.DIM, 10
    inertia, cognitive, social = 0.7, 1.5, 1.5
    positions = _seeded_population(rng, seed_configs, swarm_size, dim)
    velocities = rng.normal(0, 0.1, (swarm_size, dim))
    fitness = np.array([tracker(position) for position in positions])
    personal_best = positions.copy()
    personal_fitness = fitness.copy()
    global_index = int(np.argmax(personal_fitness))
    global_best = personal_best[global_index].copy()
    global_fitness = personal_fitness[global_index]

    while tracker.n < budget:
        for index in range(swarm_size):
            if tracker.n >= budget:
                break
            r1, r2 = rng.random(dim), rng.random(dim)
            velocities[index] = (
                inertia * velocities[index]
                + cognitive * r1 * (personal_best[index] - positions[index])
                + social * r2 * (global_best - positions[index])
            )
            velocities[index] = np.clip(velocities[index], -0.5, 0.5)
            positions[index] = np.clip(positions[index] + velocities[index], 0.0, 1.0)
            candidate_fitness = tracker(positions[index])
            if candidate_fitness > personal_fitness[index]:
                personal_best[index] = positions[index].copy()
                personal_fitness[index] = candidate_fitness
                if candidate_fitness > global_fitness:
                    global_best = positions[index].copy()
                    global_fitness = candidate_fitness
    return tracker.history


def ga_seeded(scenario: int, budget: int, seed: int, seed_configs: List[Dict]) -> List[float]:
    rng = np.random.default_rng(seed)
    tracker = Tracker(scenario, budget)
    dim, population_size = hp.DIM, 10
    population = _seeded_population(rng, seed_configs, population_size, dim)
    fitness = np.array([tracker(individual) for individual in population])

    while tracker.n < budget:
        children = []
        while len(children) < population_size and tracker.n < budget:
            i, j = rng.integers(0, population_size, 2)
            parent1 = population[i] if fitness[i] >= fitness[j] else population[j]
            i, j = rng.integers(0, population_size, 2)
            parent2 = population[i] if fitness[i] >= fitness[j] else population[j]
            blend = rng.random(dim)
            child = blend * parent1 + (1.0 - blend) * parent2
            mutation_mask = rng.random(dim) < 0.2
            child = np.clip(
                child + mutation_mask * rng.normal(0, 0.15, dim), 0.0, 1.0
            )
            children.append(child)
        child_array = np.array(children)
        child_fitness = np.array([tracker(child) for child in child_array])
        candidates = np.vstack([population, child_array])
        candidate_fitness = np.concatenate([fitness, child_fitness])
        keep = np.argsort(candidate_fitness)[-population_size:]
        population, fitness = candidates[keep], candidate_fitness[keep]
    return tracker.history


def de_seeded(scenario: int, budget: int, seed: int, seed_configs: List[Dict]) -> List[float]:
    rng = np.random.default_rng(seed)
    tracker = Tracker(scenario, budget)
    dim, population_size = hp.DIM, 12
    mutation_factor, crossover_rate = 0.6, 0.9
    population = _seeded_population(rng, seed_configs, population_size, dim)
    fitness = np.array([tracker(individual) for individual in population])

    while tracker.n < budget:
        for index in range(population_size):
            if tracker.n >= budget:
                break
            choices = [candidate for candidate in range(population_size) if candidate != index]
            a, b, c = population[rng.choice(choices, 3, replace=False)]
            mutant = np.clip(a + mutation_factor * (b - c), 0.0, 1.0)
            crossover = rng.random(dim) < crossover_rate
            if not crossover.any():
                crossover[rng.integers(0, dim)] = True
            trial = np.where(crossover, mutant, population[index])
            trial_fitness = tracker(trial)
            if trial_fitness >= fitness[index]:
                population[index], fitness[index] = trial, trial_fitness
    return tracker.history


def bayesian_seeded(
    scenario: int, budget: int, seed: int, seed_configs: List[Dict]
) -> List[float]:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern

    rng = np.random.default_rng(seed)
    tracker = Tracker(scenario, budget)
    dim = hp.DIM
    initial_size = min(8, budget)
    x_values = _seeded_population(rng, seed_configs, initial_size, dim)
    y_values = np.array([tracker(vector) for vector in x_values])
    kernel = ConstantKernel(1.0) * Matern(length_scale=0.3, nu=2.5)

    while tracker.n < budget:
        mean, std = y_values.mean(), y_values.std() + 1e-9
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=False,
            n_restarts_optimizer=0,
            random_state=int(rng.integers(0, 1 << 30)),
        )
        gp.fit(x_values, (y_values - mean) / std)
        candidates = rng.random((512, dim))
        mu, sigma = gp.predict(candidates, return_std=True)
        acquisition = hb._expected_improvement(
            mu, sigma, (y_values.max() - mean) / std
        )
        next_x = candidates[int(np.argmax(acquisition))]
        next_y = tracker(next_x)
        x_values = np.vstack([x_values, next_x])
        y_values = np.append(y_values, next_y)
    return tracker.history


SEED_HOSTS = {
    "GA+ENS": ga_seeded,
    "PSO+ENS": pso_seeded,
    "DE+ENS": de_seeded,
    "Bayesian+ENS": bayesian_seeded,
}


def mean_curve(histories: List[List[float]], budget: int) -> List[float]:
    padded = []
    for history in histories:
        if not history:
            raise ValueError("An optimizer history is empty.")
        padded.append((list(history) + [history[-1]] * budget)[:budget])
    return [st.mean(column) for column in zip(*padded)]


def cliffs_delta(a: List[float], b: List[float]) -> float:
    greater = sum(1 for x in a for y in b if x > y)
    lower = sum(1 for x in a for y in b if x < y)
    return (greater - lower) / (len(a) * len(b))


def holm_bonferroni(p_values: Dict[str, float]) -> Dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    result: Dict[str, float] = {}
    previous = 0.0
    total = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        adjusted = min(1.0, max(previous, value * (total - rank)))
        result[name] = adjusted
        previous = adjusted
    return result
