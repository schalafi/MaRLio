"""Novelty search for NEAT: reward runs that are different, not only runs that go far.

Plain NEAT breeds the genomes that get furthest right. On levels where the
way forward first looks worse (wait for a fire bar, go back to find a
lift), the whole population then crowds into the same dead end. Novelty
search also rewards a genome for *doing something new*: each run is
described by Mario's path through the level (``RunResult.behavior``), and a
run whose path is far from the paths seen before is novel.

Every generation, ``select_scores()`` turns the runs into the number NEAT
breeds by (``genome.fitness``); the run's real score stays in the log. Two
ways to combine novelty with how good a run was:

``nslc``, novelty search with local competition (Lehman & Stanley, 2011)
    *novelty* = mean distance to the k nearest paths (this generation and
    the archive of past novel paths); *local competition* = how many of
    those k neighbours it beats on fitness. A genome is rewarded for
    being different and for being good among genomes that play like it.

``dns``, Dominated Novelty Search (Bahlous-Boldi et al., 2025)
    One number for both: the mean distance to the k nearest genomes that
    are *better* than it. The best genome of a niche has nobody better
    nearby and scores high; a copy of a better genome scores near zero.
    The overall best gets the top score.

Both are blended with plain fitness, by rank, so the search keeps moving
right while it explores::

    score = (1 - w) * rank(fitness) + w * rank(novelty part)

with w = ``novelty_weight`` (0.5 by default). Ranks run from 1 (worst) to
the population size, so scores are always positive (NEAT reads 0 as "not
played yet").
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

SELECTIONS = ("fitness", "nslc", "dns")


def _distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Euclidean distance between every row of ``a`` and every row of ``b``."""
    sq = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * a @ b.T
    return np.sqrt(np.maximum(sq, 0.0))


def _ranks(values: np.ndarray) -> np.ndarray:
    """1 for the smallest value up to len(values), ties sharing the mean rank."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values))
    ranks[order] = np.arange(1, len(values) + 1)
    for v in np.unique(values):
        same = values == v
        if same.sum() > 1:
            ranks[same] = ranks[same].mean()
    return ranks


class NoveltyArchive:
    """Paths of past novel runs, so novelty is measured against history too.

    Each generation the ``add_per_generation`` most novel runs are added,
    plus one random run; the oldest are dropped past ``max_size``.
    """

    def __init__(self, k: int = 15, add_per_generation: int = 5, max_size: int = 5000,
                 seed: int = 0):
        self.k = k
        self.add_per_generation = add_per_generation
        self.max_size = max_size
        self.behaviors: list[list[float]] = []
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.behaviors)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "k": self.k, "add_per_generation": self.add_per_generation,
            "max_size": self.max_size, "behaviors": self.behaviors}))

    @classmethod
    def load(cls, path: str | Path) -> NoveltyArchive:
        data = json.loads(Path(path).read_text())
        archive = cls(data["k"], data["add_per_generation"], data["max_size"])
        archive.behaviors = data["behaviors"]
        return archive

    def select_scores(self, behaviors: list[list[float]], fitnesses: list[float],
                      method: str, novelty_weight: float = 0.5) -> dict[str, np.ndarray]:
        """Breeding scores for one generation; also grows the archive.

        Returns ``score`` (what NEAT breeds by), ``novelty`` and, for
        ``nslc``, ``local_competition`` / for ``dns``, ``dominated_novelty``,
        one value per run, for logging.
        """
        if method not in SELECTIONS:
            raise ValueError(f"selection must be one of {SELECTIONS}")
        pop = np.asarray(behaviors, dtype=float)
        fit = np.asarray(fitnesses, dtype=float)
        n = len(pop)
        # Distances from every run to every other run of this generation...
        d_pop = _distances(pop, pop)
        np.fill_diagonal(d_pop, np.inf)
        # ...and to the archive.
        if self.behaviors:
            arc = np.asarray(self.behaviors, dtype=float)
            d_arc = _distances(pop, arc)
        else:
            d_arc = np.empty((n, 0))
        d_all = np.concatenate([d_pop, d_arc], axis=1)
        k = min(self.k, d_all.shape[1] - 1) if d_all.shape[1] > 1 else 1
        nearest = np.argsort(d_all, axis=1)[:, :k]
        novelty = np.take_along_axis(d_all, nearest, axis=1).mean(axis=1)

        out = {"novelty": novelty}
        if method == "fitness":
            part = None
        elif method == "nslc":
            # Of the k nearest runs of this generation, how many it beats.
            kp = min(self.k, n - 1)
            near_pop = np.argsort(d_pop, axis=1)[:, :kp]
            local = (fit[:, None] > fit[near_pop]).sum(axis=1).astype(float)
            out["local_competition"] = local
            part = (_ranks(novelty) + _ranks(local)) / 2
        else:  # dns
            dominated = np.empty(n)
            for i in range(n):
                better = np.flatnonzero(fit > fit[i])
                if len(better) == 0:
                    dominated[i] = np.inf
                else:
                    d = np.sort(d_pop[i, better])[: self.k]
                    dominated[i] = d.mean()
            finite = dominated[np.isfinite(dominated)]
            top = (finite.max() if len(finite) else 0.0) + 1.0
            dominated[~np.isfinite(dominated)] = top
            out["dominated_novelty"] = dominated
            part = _ranks(dominated)

        if part is None:
            score = fit
        else:
            score = (1 - novelty_weight) * _ranks(fit) + novelty_weight * part
        out["score"] = score

        # Remember the most novel runs, and one at random.
        chosen = list(np.argsort(-novelty)[: self.add_per_generation])
        chosen.append(self.rng.randrange(n))
        for i in dict.fromkeys(chosen):
            self.behaviors.append(list(map(float, pop[i])))
        del self.behaviors[: max(0, len(self.behaviors) - self.max_size)]
        return out
