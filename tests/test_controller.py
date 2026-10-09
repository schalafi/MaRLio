import numpy as np

from marrlio.controller import BUTTON_BITS, buttons_to_byte


def test_single_buttons():
    assert buttons_to_byte([1, 0, 0, 0, 0, 0]) == BUTTON_BITS["A"]
    assert buttons_to_byte([0, 0, 0, 0, 0, 1]) == BUTTON_BITS["right"]


def test_combination():
    assert buttons_to_byte([1, 1, 0, 0, 0, 1]) == BUTTON_BITS["A"] | BUTTON_BITS["B"] | BUTTON_BITS["right"]


def test_opposite_directions_cancel():
    assert buttons_to_byte([1, 0, 1, 1, 1, 1]) == BUTTON_BITS["A"]


def test_accepts_numpy_actions():
    assert buttons_to_byte(np.array([0, 1, 0, 0, 1, 0], dtype=np.int8)) == BUTTON_BITS["B"] | BUTTON_BITS["left"]


def test_marios_env_runs():
    from marrlio.env import make_marios_env

    env = make_marios_env()
    env.reset(seed=0)
    for _ in range(120):
        grid, reward, terminated, truncated, info = env.step([0, 0, 0, 0, 0, 1])
    env.close()
    assert info["x_pos"] > 40  # holding right moved Mario forward
