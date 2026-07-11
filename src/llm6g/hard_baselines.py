# SPDX-License-Identifier: MIT
"""Search baselines for the hard problem.

Each baseline searches the unit hypercube [0,1]^DIM defined by hard_problem and
decodes to a valid configuration. None of them know anything about the specific
parameters, so the same code handles any schema. Every baseline records the
best score reached after each evaluation, so methods can be compared at a fixed
simulation budget and the budget needed to match a target can be measured.

The Bayesian baseline uses a Gaussian process with expected improvement when
scikit-learn is available, and falls back to random-restart local search
otherwise, with the fallback clearly labeled in the returned name.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from . import hard_problem as hp


class _Tracker:
    """Counts evaluations and records the running best score."""

    def __init__(self, scenario, budget: int, obj_seed: int = 0):
        self.scenario = scenario
        self.budget = budget
        self.obj_seed = obj_seed
        self.n = 0
        self.best = -1e9
        self.best_x: List[float] = None
        self.history: List[float] = []

    def __call__(self, x) -> float:
        if self.n >= self.budget:        # hard cap: no method exceeds budget
            return self.best
        s = hp.score_vector(x, self.scenario, seed=self.obj_seed)
        self.n += 1
        if s > self.best:
            self.best = s
            self.best_x = [float(v) for v in x]
        self.history.append(self.best)
        return s

    def result(self, name: str) -> Dict:
        return {
            "name": name,
            "best_score": self.best,
            "best_params": hp.decode_vector(self.best_x),
            "history": self.history,
        }


def random_baseline(scenario, budget: int, seed: int) -> Dict:
    rng = np.random.default_rng(seed)
    t = _Tracker(scenario, budget)
    while t.n < budget:
        t(rng.random(hp.DIM))
    return t.result("Random")


def ga(scenario, budget: int, seed: int) -> Dict:
    """Real-coded (mu + lambda) genetic algorithm."""
    rng = np.random.default_rng(seed)
    t = _Tracker(scenario, budget)
    D, mu = hp.DIM, 10
    pop = rng.random((mu, D))
    fit = np.array([t(ind) for ind in pop])

    while t.n < budget:
        children = []
        while len(children) < mu and t.n < budget:
            # tournament selection of two parents
            i, j = rng.integers(0, mu, 2)
            p1 = pop[i] if fit[i] >= fit[j] else pop[j]
            i, j = rng.integers(0, mu, 2)
            p2 = pop[i] if fit[i] >= fit[j] else pop[j]
            # blend crossover then gaussian mutation
            a = rng.random(D)
            child = a * p1 + (1.0 - a) * p2
            m = rng.random(D) < 0.2
            child = np.clip(child + m * rng.normal(0, 0.15, D), 0.0, 1.0)
            children.append(child)
        ch = np.array(children)
        ch_fit = np.array([t(c) for c in ch])
        # mu + lambda survival
        allp = np.vstack([pop, ch])
        allf = np.concatenate([fit, ch_fit])
        keep = np.argsort(allf)[-mu:]
        pop, fit = allp[keep], allf[keep]
    return t.result("GA")


def pso(scenario, budget: int, seed: int) -> Dict:
    """Standard particle swarm optimization."""
    rng = np.random.default_rng(seed)
    t = _Tracker(scenario, budget)
    D, n = hp.DIM, 10
    w, c1, c2 = 0.7, 1.5, 1.5
    X = rng.random((n, D))
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
    return t.result("PSO")


def de(scenario, budget: int, seed: int) -> Dict:
    """Differential evolution, rand/1/bin."""
    rng = np.random.default_rng(seed)
    t = _Tracker(scenario, budget)
    D, NP = hp.DIM, 12
    F, CR = 0.6, 0.9
    pop = rng.random((NP, D))
    fit = np.array([t(x) for x in pop])

    while t.n < budget:
        for i in range(NP):
            if t.n >= budget:
                break
            idxs = [k for k in range(NP) if k != i]
            a, b, c = pop[rng.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + F * (b - c), 0.0, 1.0)
            cross = rng.random(D) < CR
            if not cross.any():
                cross[rng.integers(0, D)] = True
            trial = np.where(cross, mutant, pop[i])
            ft = t(trial)
            if ft >= fit[i]:
                pop[i], fit[i] = trial, ft
    return t.result("DE")


def _expected_improvement(mu, std, best, xi=0.01):
    from scipy.stats import norm
    std = np.maximum(std, 1e-9)
    imp = mu - best - xi
    z = imp / std
    return imp * norm.cdf(z) + std * norm.pdf(z)


def bayesian(scenario, budget: int, seed: int) -> Dict:
    """Gaussian-process Bayesian optimization with expected improvement.

    Falls back to random-restart local search if scikit-learn is unavailable.
    """
    rng = np.random.default_rng(seed)
    t = _Tracker(scenario, budget)
    D = hp.DIM
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern, ConstantKernel
    except Exception:
        return _bo_fallback(scenario, budget, seed, t)

    n_init = min(8, budget)
    X = rng.random((n_init, D))
    y = np.array([t(x) for x in X])

    kernel = ConstantKernel(1.0) * Matern(length_scale=0.3, nu=2.5)
    while t.n < budget:
        ym, ys = y.mean(), y.std() + 1e-9
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6,
                                      normalize_y=False, n_restarts_optimizer=0,
                                      random_state=int(rng.integers(0, 1 << 30)))
        gp.fit(X, (y - ym) / ys)
        cand = rng.random((512, D))
        mu, std = gp.predict(cand, return_std=True)
        ei = _expected_improvement(mu, std, (y.max() - ym) / ys)
        x_next = cand[int(np.argmax(ei))]
        y_next = t(x_next)
        X = np.vstack([X, x_next])
        y = np.append(y, y_next)
    return t.result("Bayesian")


def _bo_fallback(scenario, budget, seed, t) -> Dict:
    rng = np.random.default_rng(seed)
    D = hp.DIM
    x = rng.random(D)
    fx = t(x)
    step = 0.2
    while t.n < budget:
        cand = np.clip(x + rng.normal(0, step, D), 0.0, 1.0)
        fc = t(cand)
        if fc >= fx:
            x, fx = cand, fc
            step = min(0.3, step * 1.1)
        else:
            step = max(0.05, step * 0.9)
        if rng.random() < 0.1:                  # occasional restart
            x = rng.random(D)
            fx = t(x)
    return t.result("Bayesian(fallback)")


ALL_BASELINES = {
    "Random": random_baseline,
    "GA": ga,
    "PSO": pso,
    "DE": de,
    "Bayesian": bayesian,
}
