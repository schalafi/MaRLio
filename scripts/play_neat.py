"""Watch a trained NEAT genome play.

    python scripts/play_neat.py runs/neat/best.json
"""

import argparse
import json
from pathlib import Path

from marlio.agents.neat import Genome, run_genome
from marlio.env import DEFAULT_LEVEL, make_marios_env


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("genome", help="a genome .json saved by train_neat.py")
    parser.add_argument("--level", default=DEFAULT_LEVEL)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    genome = Genome.from_dict(json.loads(Path(args.genome).read_text()))
    env = make_marios_env(args.level, render_mode=None if args.no_render else "human")
    for _ in range(args.runs):
        result = run_genome(env, genome, render=not args.no_render)
        print(f"fitness {result.fitness:.1f}, reached x={result.rightmost} in {result.frames} frames"
              + (", LEVEL CLEARED" if result.cleared else ""))
    env.close()


if __name__ == "__main__":
    main()
