"""MarI/O's controller: one network output per button.

gym-super-mario-bros usually gives agents a short menu of button combos
(``SIMPLE_MOVEMENT`` has 7, such as "right" or "right + A"). MarI/O instead
lets the network press any of 6 buttons independently: each output neuron
above 0 means "hold this button". This module turns those 6 on/off values
into the single byte the NES controller expects.
"""

from collections.abc import Sequence

import gymnasium as gym
import numpy as np

# MarI/O's button order for Super Mario Bros.
BUTTONS = ("A", "B", "up", "down", "left", "right")

# Bit of each button in the NES controller byte (as nes-py encodes it).
BUTTON_BITS = {
    "right": 0b10000000,
    "left": 0b01000000,
    "down": 0b00100000,
    "up": 0b00010000,
    "start": 0b00001000,
    "select": 0b00000100,
    "B": 0b00000010,
    "A": 0b00000001,
}


def buttons_to_byte(pressed: Sequence[bool | int]) -> int:
    """Convert 6 on/off values (in ``BUTTONS`` order) to a controller byte.

    Like MarI/O, pressing two opposite directions at once cancels both.
    """
    held = {name for name, on in zip(BUTTONS, pressed, strict=True) if on}
    if {"left", "right"} <= held:
        held -= {"left", "right"}
    if {"up", "down"} <= held:
        held -= {"up", "down"}
    byte = 0
    for name in held:
        byte |= BUTTON_BITS[name]
    return byte


class ButtonController(gym.ActionWrapper):
    """Let the agent press each of MarI/O's 6 buttons independently.

    The action space becomes ``MultiBinary(6)``: ``[A, B, up, down, left,
    right]``, 1 meaning held for this frame.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = gym.spaces.MultiBinary(len(BUTTONS))

    def action(self, action: np.ndarray) -> int:
        return buttons_to_byte(action)
