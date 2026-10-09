"""An agent that presses random buttons: the baseline every algorithm must beat."""

from typing import Any

import gymnasium as gym

from marrlio.agents.base import Agent


class RandomAgent(Agent):
    def __init__(self, action_space: gym.Space):
        self.action_space = action_space

    def act(self, observation: Any) -> Any:
        return self.action_space.sample()
