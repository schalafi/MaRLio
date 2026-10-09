"""Draw the game screen next to MarI/O's input grid (the figure in docs/).

    python scripts/make_input_figure.py --out docs/images/input-grid.png

Runs Mario right until an enemy enters the grid, then saves a picture of the
screen (with the area the grid covers outlined) beside the grid itself.
Needs matplotlib (``pip install -e ".[dev]"``); no display required.
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from marlio.env import make_marios_env  # noqa: E402
from marlio.inputs import BOX_RADIUS, ENEMY, TILE_SIZE, mario_position  # noqa: E402

#          A  B  up down left right
RUN_RIGHT = [0, 0, 0, 0, 0, 1]


def grid_outline_on_screen(ram):
    """Return (left, top, size) in screen pixels of the area the grid covers."""
    mario_x, mario_y = mario_position(ram)
    camera_x = int(ram[0x071A]) * 0x100 + int(ram[0x071C])
    # Same block arithmetic as inputs.is_solid, for the top-left cell.
    left_block = (mario_x - BOX_RADIUS * TILE_SIZE + 8) // TILE_SIZE
    top_row = (mario_y - BOX_RADIUS * TILE_SIZE - 16 - 32) // TILE_SIZE
    left = left_block * TILE_SIZE - camera_x
    top = top_row * TILE_SIZE + 32
    return left, top, (2 * BOX_RADIUS + 1) * TILE_SIZE


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="docs/images/input-grid.png")
    args = parser.parse_args()

    env = make_marios_env(render_mode="rgb_array")
    grid, info = env.reset(seed=0)
    for _ in range(2000):
        grid, reward, terminated, truncated, info = env.step(RUN_RIGHT)
        if (grid == ENEMY).any() and info["y_pos"] <= 79:  # enemy visible, Mario on the ground
            break
    screen = env.render()
    ram = env.unwrapped.ram
    env.close()

    fig, (left_ax, right_ax) = plt.subplots(1, 2, figsize=(11, 5), width_ratios=[256, 208])
    left_ax.imshow(screen)
    x, y, size = grid_outline_on_screen(ram)
    left_ax.add_patch(Rectangle((x, y), size, size, fill=False, edgecolor="yellow", linewidth=2))
    left_ax.set_title("What you see: 256 x 240 pixels x 3 colours")
    left_ax.axis("off")

    # -1 enemy (red), 0 empty (white), 1 solid (dark grey)
    right_ax.imshow(grid, cmap=ListedColormap(["#d62728", "#ffffff", "#444444"]), vmin=-1, vmax=1)
    right_ax.add_patch(
        Rectangle((BOX_RADIUS - 0.5, BOX_RADIUS - 0.5), 1, 1, fill=False, edgecolor="#1f77b4", linewidth=3)
    )
    right_ax.set_xticks([i - 0.5 for i in range(grid.shape[1] + 1)], labels=[])
    right_ax.set_yticks([i - 0.5 for i in range(grid.shape[0] + 1)], labels=[])
    right_ax.tick_params(length=0)
    right_ax.grid(color="#bbbbbb", linewidth=0.5)
    right_ax.set_title("What the network sees: 13 x 13 = 169 inputs\n"
                       "grey = solid (1), red = enemy (-1), blue = grid centre")

    fig.tight_layout()
    fig.savefig(args.out, dpi=100)
    print(f"saved {args.out} (Mario at x={info['x_pos']})")


if __name__ == "__main__":
    main()
