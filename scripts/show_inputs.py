"""Watch the game and, side by side in the terminal, what MarI/O's network sees.

    python scripts/show_inputs.py

Mario runs right and jumps at random so the grid has something to show.
Every ``--every`` frames the 13x13 input grid is printed:
``#`` solid tile, ``E`` enemy, ``.`` empty, ``M`` Mario (for reference only;
Mario is not part of the input).

Use ``--no-render`` to only print the grid (works without a display).
"""

import argparse
import random

from marlio.env import DEFAULT_LEVEL, make_marios_env
from marlio.inputs import grid_to_text

#          A  B  up down left right
RUN_RIGHT = [0, 1, 0, 0, 0, 1]
RUN_RIGHT_AND_JUMP = [1, 1, 0, 0, 0, 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--level", default=DEFAULT_LEVEL)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--every", type=int, default=30, help="print the grid every N frames")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    env = make_marios_env(args.level, render_mode=None if args.no_render else "human")
    grid, info = env.reset(seed=0)
    jumping = 0
    for frame in range(args.steps):
        # Hold jump for a random stretch now and then, otherwise run right.
        if jumping == 0 and random.random() < 0.05:
            jumping = random.randint(5, 30)
        action = RUN_RIGHT_AND_JUMP if jumping else RUN_RIGHT
        jumping = max(0, jumping - 1)

        grid, reward, terminated, truncated, info = env.step(action)
        env.render()
        if frame % args.every == 0:
            print(f"\nframe {frame}  x={info['x_pos']}\n{grid_to_text(grid)}")
        if terminated or truncated:
            grid, info = env.reset()
    env.close()


if __name__ == "__main__":
    main()
