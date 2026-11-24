"""Surrogate-assisted evolutionary optimisation (SAEA) for expensive black-box tuning, vs a plain genetic
algorithm and random search under the same evaluation budget.

SAEA here = Gaussian-process surrogate (Matern-5/2 ARD + white noise) on [0,1]^d, an expected-improvement
infill criterion, and an inner evolutionary search over the *surrogate* to choose the next point(s) to evaluate
on the true objective (Jin 2011; Jones et al. 1998). Mixed categorical/continuous spaces are encoded in [0,1]^d
and decoded by binning, with duplicate decoded configurations skipped.
"""
from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


@dataclass
class Trace:
    method: str
    seed: int
    X: list = field(default_factory=list)
    y: list = field(default_factory=list)
    wall_s: float = 0.0

    @property
    def best_so_far(self):
        return np.maximum.accumulate(np.array(self.y)).tolist()

    def best(self):
        i = int(np.argmax(self.y))
        return self.X[i], self.y[i]


def random_search(f, dim, budget, seed):
    rng = np.random.default_rng(seed)
    tr = Trace("Random search", seed)
    t0 = time.time()
    for _ in range(budget):
        u = rng.random(dim)
        tr.X.append(u.tolist()); tr.y.append(f(u))
    tr.wall_s = time.time() - t0
    return tr


def _offspring(pop, fit, rng, n, sigma=0.15, pc=0.9):
    """Binary tournament selection, uniform crossover, Gaussian mutation, clipped to [0,1]^d."""
    pop, fit = np.asarray(pop), np.asarray(fit)
    d = pop.shape[1]
    kids = []
    for _ in range(n):
        a, b = rng.integers(len(pop), size=2), rng.integers(len(pop), size=2)
        p1, p2 = pop[a[np.argmax(fit[a])]], pop[b[np.argmax(fit[b])]]
        child = np.where(rng.random(d) < 0.5, p1, p2) if rng.random() < pc else p1.copy()
        mut = rng.random(d) < (1.0 / d + 0.1)
        kids.append(np.clip(child + mut * rng.normal(0, sigma, d), 0, 1))
    return np.array(kids)


def genetic_algorithm(f, dim, budget, seed, pop_size=10):
    rng = np.random.default_rng(seed)
    tr = Trace("Genetic algorithm", seed)
    t0 = time.time()
    pop = rng.random((pop_size, dim))
    fit = []
    for u in pop:
        tr.X.append(u.tolist()); tr.y.append(f(u)); fit.append(tr.y[-1])
    fit = np.array(fit)
    while len(tr.y) < budget:
        kids = _offspring(pop, fit, rng, min(pop_size, budget - len(tr.y)))
        kfit = []
        for u in kids:
            tr.X.append(u.tolist()); tr.y.append(f(u)); kfit.append(tr.y[-1])
        allp, allf = np.vstack([pop, kids]), np.concatenate([fit, kfit])
        keep = np.argsort(-allf)[:pop_size]
        pop, fit = allp[keep], allf[keep]
    tr.wall_s = time.time() - t0
    return tr


def fit_gp(X, y, seed):
    d = np.asarray(X).shape[1]
    k = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=np.full(d, 0.3), length_scale_bounds=(1e-2, 10.0),
                                                  nu=2.5) + WhiteKernel(1e-4, (1e-8, 1e-1))
    gp = GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=2, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp.fit(np.asarray(X), np.asarray(y))
    return gp


def expected_improvement(gp, X, best, xi=0.002):
    mu, sd = gp.predict(np.atleast_2d(X), return_std=True)
    sd = np.maximum(sd, 1e-9)
    z = (mu - best - xi) / sd
    return (mu - best - xi) * norm.cdf(z) + sd * norm.pdf(z)


def saea(f, dim, budget, seed, key=None, n_init=12, pop_size=24, inner_gens=30):
    """GP + EI, with the EI maximised by an evolutionary search on the surrogate. ``key(u)`` maps an encoded
    point to a hashable decoded configuration so that duplicates are never re-evaluated."""
    key = key or (lambda u: tuple(np.round(u, 3)))
    rng = np.random.default_rng(seed)
    tr = Trace("SAEA (GP + EI, evolutionary infill)", seed)
    t0 = time.time()
    init = (rng.permuted(np.tile(np.arange(n_init), (dim, 1)), axis=1).T + rng.random((n_init, dim))) / n_init
    for u in init:  # Latin hypercube design
        tr.X.append(u.tolist()); tr.y.append(f(u))
    while len(tr.y) < budget:
        X, y = np.array(tr.X), np.array(tr.y)
        gp = fit_gp(X, y, seed)
        best = y.max()
        elite = X[np.argsort(-y)[: pop_size // 2]]
        pop = np.vstack([elite, rng.random((pop_size - len(elite), dim))])
        acq = expected_improvement(gp, pop, best)
        for _ in range(inner_gens):
            kids = _offspring(pop, acq, rng, pop_size, sigma=0.1)
            kacq = expected_improvement(gp, kids, best)
            allp, alla = np.vstack([pop, kids]), np.concatenate([acq, kacq])
            keep = np.argsort(-alla)[:pop_size]
            pop, acq = allp[keep], alla[keep]
        seen = {json.dumps(key(x), sort_keys=True, default=str) for x in X}
        chosen = False
        for c in pop[np.argsort(-acq)]:
            if json.dumps(key(c), sort_keys=True, default=str) in seen:
                continue
            tr.X.append(c.tolist()); tr.y.append(f(c)); chosen = True
            break
        if not chosen:
            u = rng.random(dim)
            tr.X.append(u.tolist()); tr.y.append(f(u))
    tr.wall_s = time.time() - t0
    return tr
