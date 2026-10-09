"""Plot how far the best network got in each generation (from train_neat.py's log.csv).

    python scripts/plot_neat_progress.py runs/neat/log.csv --out docs/images/neat-progress.png
"""

import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from marlio.agents.neat.agent import LEVEL_END_X  # noqa: E402

SERIES = "#2a78d6"
TEXT = "#0b0b0b"
MUTED = "#52514e"
SURFACE = "#fcfcfb"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("log", help="log.csv written by train_neat.py")
    parser.add_argument("--out", default="docs/images/neat-progress.png")
    args = parser.parse_args()

    with open(args.log) as f:
        rows = list(csv.DictReader(f))
    generations = [int(r["generation"]) for r in rows]
    best_x = [int(r["best_x"]) for r in rows]

    fig, ax = plt.subplots(figsize=(8, 4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axhline(LEVEL_END_X, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(generations[0], LEVEL_END_X + 60, "end of World 1-1", color=MUTED, fontsize=9)
    ax.step(generations, best_x, where="post", color=SERIES, linewidth=2)
    cleared = [r for r in rows if r["cleared"] == "True"]
    if cleared:
        g = int(cleared[0]["generation"])
        ax.plot([g], [int(cleared[0]["best_x"])], "o", color=SERIES, markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(f"cleared in generation {g}", (g, int(cleared[0]["best_x"])),
                    xytext=(-10, -22), textcoords="offset points", ha="right", color=TEXT, fontsize=9)
    ax.set_title("Furthest point reached by the best network", loc="left", color=TEXT, fontsize=12)
    ax.set_xlabel("generation", color=MUTED)
    ax.set_ylabel("x position (pixels)", color=MUTED)
    ax.set_ylim(0, LEVEL_END_X * 1.12)
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c9c8c2")
    ax.tick_params(colors=MUTED, length=0)
    fig.tight_layout()
    fig.savefig(args.out, dpi=100)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
