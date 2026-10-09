"""Watch Mario press random buttons.

    python scripts/random_agent.py

The simplest possible agent: every frame it picks one of the 7 button combos
in SIMPLE_MOVEMENT at random. It rarely gets far, which is the point: it is
the baseline any learning algorithm has to beat.
"""

import argparse

import gym_super_mario_bros  # noqa: F401  (registers the SuperMarioBros-* ids)
import gymnasium as gym
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--level", default="SuperMarioBros-v0")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    env = gym.make(args.level, render_mode="human")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    done = True
    for step in range(args.steps):
        if done:
            state, info = env.reset(seed=args.seed)
        state, reward, terminated, truncated, info = env.step(env.action_space.sample())
        done = terminated or truncated
        env.render()

    env.close()


if __name__ == "__main__":
    main()
