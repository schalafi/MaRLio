"""The interface every agent in this repo implements.

An agent only has to answer one question: given what it sees now, which
action does it take? Keeping this tiny lets scripts run, record and compare
any algorithm (random, NEAT, PPO, ...) the same way.
"""

from abc import ABC, abstractmethod
from typing import Any


class Agent(ABC):
    @abstractmethod
    def act(self, observation: Any) -> Any:
        """Return the action to take for this observation."""

    def reset(self) -> None:
        """Called at the start of every episode. Override if the agent has state."""
