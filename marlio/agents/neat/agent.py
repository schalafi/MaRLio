"""A NEAT network playing Mario, and MarI/O's way of scoring a run."""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from marlio.agents.base import Agent
from marlio.agents.neat.genome import Genome
from marlio.agents.neat.network import Network
from marlio.controller import BUTTONS
from marlio.env import DEFAULT_LEVEL, make_marios_env
from marlio.inputs import GRID_SIZE

# 169 grid cells + 1 bias input; one output per button.
N_INPUTS = GRID_SIZE * GRID_SIZE + 1
N_OUTPUTS = len(BUTTONS)

# MarI/O asks the network for new buttons every 5 frames and holds them
# in between (it's faster and the game doesn't need finer control).
THINK_EVERY = 5
# Frames Mario may go without reaching a new rightmost point before the run
# ends (MarI/O's TimeoutConstant). Runs get a bonus of frames / 4 on top.
TIMEOUT = 20
# MarI/O's x-position for "reached the end of World 1-1".
LEVEL_END_X = 3186


class NEATAgent(Agent):
    def __init__(self, genome: Genome):
        self.network = Network(genome, N_INPUTS, N_OUTPUTS)
        self.frame = 0
        self.buttons = [0] * N_OUTPUTS

    def reset(self) -> None:
        self.network.reset()
        self.frame = 0
        self.buttons = [0] * N_OUTPUTS

    def act(self, observation: np.ndarray) -> list[int]:
        if self.frame % THINK_EVERY == 0:
            self.buttons = self.network.evaluate(observation.ravel())
        self.frame += 1
        return self.buttons


@dataclass
class RunResult:
    fitness: float
    rightmost: int
    frames: int
    cleared: bool


def run_genome(env: gym.Env, genome: Genome, render: bool = False) -> RunResult:
    """Play one run and score it like MarI/O.

    fitness = how far right Mario got - frames used / 2, plus 1000 for
    clearing the level. The run ends when Mario stops making progress
    (timeout), dies, or reaches the flag.
    """
    agent = NEATAgent(genome)
    grid, info = env.reset()
    rightmost, timeout, frame = 0, TIMEOUT, 0
    cleared = False
    while True:
        grid, _, terminated, truncated, info = env.step(agent.act(grid))
        if render:
            env.render()
        frame += 1
        if info["x_pos"] > rightmost:
            rightmost = info["x_pos"]
            timeout = TIMEOUT
        timeout -= 1
        cleared = cleared or info["flag_get"]
        # MarI/O only stops on the timeout; stopping on death or the flag
        # as well gives the same scores, just sooner.
        if (timeout + frame / 4 <= 0 or terminated or truncated or cleared
                or info["is_dying"] or info["is_dead"]):
            break
    fitness = rightmost - frame / 2
    if cleared or rightmost > LEVEL_END_X:
        fitness += 1000
    if fitness == 0:
        fitness = -1  # 0 means "not measured yet"
    return RunResult(fitness, rightmost, frame, cleared)


# --- Running many genomes in parallel ----------------------------------------

_worker_env: gym.Env | None = None


def _init_worker(level: str) -> None:
    global _worker_env
    _worker_env = make_marios_env(level)


def _run_in_worker(genome: Genome) -> RunResult:
    return run_genome(_worker_env, genome)


class Evaluator:
    """Plays genomes on ``workers`` CPU cores, one game per core."""

    def __init__(self, level: str = DEFAULT_LEVEL, workers: int | None = None):
        self.level = level
        self.workers = workers or mp.cpu_count()
        self._pool = None
        self._env = None
        if self.workers > 1:
            self._pool = mp.get_context("spawn").Pool(self.workers, _init_worker, (level,))
        else:
            self._env = make_marios_env(level)

    def run(self, genomes: list[Genome]) -> list[RunResult]:
        if self._pool is not None:
            return self._pool.map(_run_in_worker, genomes, chunksize=1)
        return [run_genome(self._env, genome) for genome in genomes]

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
        if self._env is not None:
            self._env.close()
