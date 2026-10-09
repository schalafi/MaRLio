"""Creating Super Mario Bros. environments.

Every algorithm in this repo starts from the same gym-super-mario-bros
environment. What changes between algorithms is how the agent sees the game
(observation) and how it presses buttons (actions), so both are wrappers:

* ``make_env``: the game as most RL libraries expect it. Observations are
  screen pixels (240x256x3) and actions are a short menu of button combos.
* ``make_marios_env``: the game as MarI/O sees it. Observations are the
  13x13 input grid and actions are 6 independent buttons.
"""

from collections.abc import Sequence

import gym_super_mario_bros  # noqa: F401  (registers the SuperMarioBros-* ids)
import gymnasium as gym
import numpy as np
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from marlio.controller import ButtonController
from marlio.inputs import GRID_SIZE, input_grid

# World 1, stage 1, the level MarI/O learned to beat.
DEFAULT_LEVEL = "SuperMarioBros-1-1-v0"


class InputGridObservation(gym.ObservationWrapper):
    """Replace the screen pixels with MarI/O's 13x13 input grid.

    The screen is still drawn when ``render_mode="human"``, so you can watch
    the game while the agent only receives the grid.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(GRID_SIZE, GRID_SIZE), dtype=np.int8
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return input_grid(self.unwrapped.ram)


def make_env(
    level: str = DEFAULT_LEVEL,
    render_mode: str | None = None,
    actions: Sequence[Sequence[str]] = SIMPLE_MOVEMENT,
) -> gym.Env:
    """Make a Mario env with pixel observations and a menu of button combos."""
    env = gym.make(level, render_mode=render_mode)
    return JoypadSpace(env, actions)


def make_marios_env(level: str = DEFAULT_LEVEL, render_mode: str | None = None) -> gym.Env:
    """Make a Mario env that sees and acts like MarI/O."""
    env = gym.make(level, render_mode=render_mode)
    return InputGridObservation(ButtonController(env))
