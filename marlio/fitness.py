"""How good a run of Mario was, on any level.

The score counts what a good player does on every level: go right, reach
the flag, pick up coins and power-ups, stomp enemies, and don't get hurt or
die. Every term is in "pixels of progress", so the weights read as "a coin
is worth going 20 px further right".

``RunStats`` follows a run frame by frame (from the env's ``info`` and the
NES RAM), and ``fitness()`` turns it into one number. It doesn't depend on
any one algorithm, so NEAT uses it as its fitness and an RL agent could use
the change per frame as its reward.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Weights, in pixels of progress.
TIME_PENALTY = 1 / 10   # per frame: waiting 200 frames costs 20
DEATH_PENALTY = 100     # dying is worth giving up 100 px of progress
FLAG_BONUS = 1000       # the game's flag_get, so it works on every level
COIN_BONUS = 20
KILL_BONUS = 50         # stomped, or hit by a fireball or a shell
POWERUP_BONUS = 200     # mushroom (small -> big) or fire flower (big -> fire)
DAMAGE_PENALTY = 100    # hit by an enemy and shrinking, without dying

# The game's 5 enemy slots: Enemy_State in RAM 0x1E-0x22. Bit 2 (0x04) is
# set when the enemy is stomped, bit 5 (0x20) when it is knocked out by a
# fireball, a shell or a star. A slot going from neither bit to one of them
# is one defeated enemy.
_ENEMY_STATE = slice(0x1E, 0x23)
_DEFEATED = 0x04 | 0x20
_POWER = {"small": 0, "tall": 1, "fireball": 2}


@dataclass
class RunStats:
    """What happened in one run, so far."""

    rightmost: int = 0
    frames: int = 0
    coins: int = 0
    kills: int = 0
    powerups: int = 0
    hits: int = 0
    died: bool = False
    cleared: bool = False

    def start(self, info: dict, ram: np.ndarray) -> None:
        """Remember the state the run starts from (call after reset)."""
        self._coins = info["coins"]
        self._power = _POWER.get(info["status"], 0)
        self._enemies = ram[_ENEMY_STATE].copy()

    def update(self, info: dict, ram: np.ndarray) -> None:
        """Add one frame (call after every env.step)."""
        self.frames += 1
        self.rightmost = max(self.rightmost, info["x_pos"])
        # The coin counter goes back to 0 at 100 (and gives a life).
        self.coins += (info["coins"] - self._coins) % 100
        self._coins = info["coins"]
        power = _POWER.get(info["status"], 0)
        if power > self._power:
            self.powerups += 1
        elif power < self._power and not (info["is_dying"] or info["is_dead"]):
            self.hits += 1
        self._power = power
        enemies = ram[_ENEMY_STATE]
        self.kills += int(np.count_nonzero(((self._enemies & _DEFEATED) == 0) & ((enemies & _DEFEATED) != 0)))
        self._enemies = enemies.copy()
        self.died = self.died or info["is_dying"] or info["is_dead"]
        self.cleared = self.cleared or info["flag_get"]


def fitness(stats: RunStats) -> float:
    """One number for a run: progress, plus the good things, minus the bad."""
    score = stats.rightmost - stats.frames * TIME_PENALTY
    score += stats.coins * COIN_BONUS
    score += stats.kills * KILL_BONUS
    score += stats.powerups * POWERUP_BONUS
    score -= stats.hits * DAMAGE_PENALTY
    if stats.died:
        score -= DEATH_PENALTY
    if stats.cleared:
        score += FLAG_BONUS
    return score
