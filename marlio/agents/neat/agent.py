"""A NEAT network playing Mario, and running genomes to score them."""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from marlio.agents.base import Agent
from marlio.agents.neat.genome import Genome
from marlio.agents.neat.network import Network
from marlio.controller import BUTTONS
from marlio.env import DEFAULT_LEVEL, make_marios_env
from marlio.fitness import RunStats, fitness
from marlio.inputs import GRID_SIZE, input_grid

# 169 grid cells + 1 bias input; one output per button.
N_INPUTS = GRID_SIZE * GRID_SIZE + 1
N_OUTPUTS = len(BUTTONS)

# MarI/O asks the network for new buttons every 5 frames and holds them
# in between (it's faster and the game doesn't need finer control).
THINK_EVERY = 5
# Frames Mario may go without reaching a new rightmost point before the run
# ends (MarI/O's TimeoutConstant). Runs get a bonus of frames / 4 on top.
TIMEOUT = 20
# MarI/O's x-position for "reached the end of World 1-1". Only used for
# drawing progress charts now; the fitness uses the game's flag_get.
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
    died: bool = False
    stats: RunStats | None = None
    # The run from the practice start, when training uses one (see Evaluator).
    practice: RunResult | None = None


def run_genome(
    env: gym.Env,
    genome: Genome,
    render: bool = False,
    on_frame: Callable[[NEATAgent, dict, RunStats], None] | None = None,
    start_from=None,
    start_frame: int = 0,
) -> RunResult:
    """Play one run and score it with ``marlio.fitness.fitness()``.

    The run ends like in MarI/O: when Mario stops making progress
    (timeout), dies, or reaches the flag.

    ``on_frame(agent, info, stats)`` is called after every frame, for
    example to draw the network over the game or record a video; ``stats``
    is the run so far (``fitness(stats)`` is its score at that point).

    ``start_from`` is a nes-py snapshot (``env.unwrapped.dump_state()``) to
    start the run from instead of the start of the level; x positions, and
    so the fitness, still count from the start of the level. ``start_frame``
    is how many frames into the level that state is, so the timeout allows
    the same waiting as it would there in a full run.
    """
    agent = NEATAgent(genome)
    grid, info = env.reset()
    if start_from is not None:
        env.unwrapped.load_state(start_from)
        env.unwrapped._did_reset()
        grid = input_grid(env.unwrapped.ram)
        info = env.unwrapped._get_info()
    stats = RunStats(frames=start_frame)
    stats.start(info, env.unwrapped.ram)
    timeout = TIMEOUT
    while True:
        grid, _, terminated, truncated, info = env.step(agent.act(grid))
        if render:
            env.render()
        rightmost = stats.rightmost
        stats.update(info, env.unwrapped.ram)
        if on_frame is not None:
            on_frame(agent, info, stats)
        if stats.rightmost > rightmost:
            timeout = TIMEOUT
        timeout -= 1
        # MarI/O only stops on the timeout; stopping on death or the flag
        # as well gives the same scores, just sooner.
        if (timeout + stats.frames / 4 <= 0 or terminated or truncated
                or stats.cleared or stats.died):
            break
    score = fitness(stats)
    if score == 0:
        score = -1  # 0 means "not measured yet"
    return RunResult(score, stats.rightmost, stats.frames, stats.cleared, stats.died, stats)


# --- Running many genomes in parallel ----------------------------------------

_worker_env: gym.Env | None = None
_worker_practice = None


def practice_state(env: gym.Env, buttons: list[list[int]]):
    """The game state after pressing ``buttons`` (one entry per frame) from the start.

    The emulator is deterministic, so replaying the same buttons always
    gives the same state. (nes-py's snapshots can't be sent to other
    processes, so each worker builds its own this way.)
    """
    env.reset()
    for pressed in buttons:
        env.step(pressed)
    return env.unwrapped.dump_state(), len(buttons)


def play(env: gym.Env, genome: Genome, practice=None) -> RunResult:
    """One run from the start of the level and, with a practice state, one from there.

    The fitness is the sum of both, so a network that learns to get past
    the hard part in practice scores higher, even before it gets that far
    from the start.
    """
    result = run_genome(env, genome)
    if practice is None:
        return result
    # The practice run's fitness counts frames from the start of the level
    # too, so both runs are scored on the same scale.
    state, frames_in = practice
    extra = run_genome(env, genome, start_from=state, start_frame=frames_in)
    return RunResult(result.fitness + extra.fitness, result.rightmost, result.frames,
                     result.cleared, result.died, result.stats, practice=extra)


def _init_worker(level: str, practice_buttons) -> None:
    global _worker_env, _worker_practice
    _worker_env = make_marios_env(level)
    if practice_buttons:
        _worker_practice = practice_state(_worker_env, practice_buttons)


def _run_in_worker(genome: Genome) -> RunResult:
    return play(_worker_env, genome, _worker_practice)


class Evaluator:
    """Plays genomes on ``workers`` CPU cores, one game per core.

    ``practice_buttons`` (one list of 6 buttons per frame) leads from the
    start of the level to a practice start: each genome then also plays a
    run from there (see ``play``).
    """

    def __init__(self, level: str = DEFAULT_LEVEL, workers: int | None = None,
                 practice_buttons: list[list[int]] | None = None):
        self.level = level
        self.workers = workers or mp.cpu_count()
        self._pool = None
        self._env = None
        self._practice = None
        if self.workers > 1:
            self._pool = mp.get_context("spawn").Pool(
                self.workers, _init_worker, (level, practice_buttons))
        else:
            self._env = make_marios_env(level)
            if practice_buttons:
                self._practice = practice_state(self._env, practice_buttons)

    def run(self, genomes: list[Genome]) -> list[RunResult]:
        if self._pool is not None:
            return self._pool.map(_run_in_worker, genomes, chunksize=1)
        return [play(self._env, genome, self._practice) for genome in genomes]

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
        if self._env is not None:
            self._env.close()
