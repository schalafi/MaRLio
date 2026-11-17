"""Evolve networks with NEAT until one clears World 1-1 (Python MarI/O).

    python scripts/train_neat.py                # train, saving to runs/neat/
    python scripts/train_neat.py --watch        # ...and watch each generation's best play
    python scripts/train_neat.py --resume runs/neat/population.json

Each generation every new genome plays one run (in parallel on all CPU
cores), gets a fitness (``marlio/fitness.py``), and the population breeds the next
generation. Saved in ``--out`` after every generation:

* ``population.json``  the whole population, to resume training
* ``best.json``        the best genome so far (watch it with play_neat.py)
* ``gen_NNN_best.json`` the best genome of each generation
* ``log.csv``          one row of statistics per generation
* ``practice.json``    the buttons that lead to the practice start, if any

``--selection nslc`` or ``--selection dns`` adds novelty search
(``marlio/agents/neat/novelty.md``): genomes are bred by a mix of how good
and how *new* their run was. ``novelty_archive.json`` is saved with the
population, so ``--resume`` keeps it.

``--max-minutes 60`` stops training after an hour of generations (counted
from ``log.csv``, so a resumed run only gets the time that is left) and
saves the population, so it can be resumed later.

Practice starts (``--practice-from best.json --practice-x 1700``): when
training stalls at a hard spot, every genome also plays a second run that
starts just before it, and its fitness is the sum of both runs. The practice
start is where ``--practice-from``'s genome first reaches x = ``--practice-x``.
The level still only counts as cleared from the real start.
"""

import argparse
import csv
import json
import time
from pathlib import Path

from marlio.agents.neat import N_INPUTS, N_OUTPUTS, Evaluator, Genome, NEATConfig, Population, run_genome
from marlio.agents.neat.agent import set_auto_release_a
from marlio.agents.neat.novelty import SELECTIONS, NoveltyArchive
from marlio.env import DEFAULT_LEVEL, make_marios_env


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--level", default=DEFAULT_LEVEL)
    parser.add_argument("--population", type=int, default=NEATConfig.population)
    parser.add_argument("--generations", type=int, default=200, help="stop after this generation")
    parser.add_argument("--workers", type=int, default=None, help="CPU cores to use (default: all)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="runs/neat")
    parser.add_argument("--resume", help="population.json to continue from")
    parser.add_argument("--watch", action="store_true", help="show each generation's best in a window")
    parser.add_argument("--keep-going", action="store_true", help="don't stop when the level is cleared")
    parser.add_argument("--practice-from", help="genome whose run sets the practice start")
    parser.add_argument("--practice-x", type=int, default=None, help="x position of the practice start")
    parser.add_argument("--auto-release-a", type=int, default=0, metavar="FRAMES",
                        help="let go of A for one frame after it is held FRAMES frames on the ground "
                             "(so a network holding A can jump again; default off)")
    parser.add_argument("--selection", choices=SELECTIONS, default="fitness",
                        help="what NEAT breeds by: fitness (MarI/O), nslc (novelty search with local "
                             "competition) or dns (dominated novelty search); see novelty.py")
    parser.add_argument("--novelty-weight", type=float, default=0.5,
                        help="share of novelty in the breeding score (0 = plain fitness)")
    parser.add_argument("--novelty-k", type=int, default=15, help="nearest neighbours for novelty")
    parser.add_argument("--max-minutes", type=float, default=None,
                        help="stop after this many minutes of training, counted over resumes")
    args = parser.parse_args()
    if args.auto_release_a:
        set_auto_release_a(args.auto_release_a)
    if bool(args.practice_from) != (args.practice_x is not None):
        parser.error("--practice-from and --practice-x go together")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.resume:
        population = Population.load(args.resume)
        print(f"Resumed at generation {population.generation}")
    else:
        population = Population(N_INPUTS, N_OUTPUTS, NEATConfig(population=args.population), seed=args.seed)

    practice = None
    if args.practice_from:
        practice = _buttons_to(args.level, args.practice_from, args.practice_x)
        (out / "practice.json").write_text(json.dumps(
            {"from": args.practice_from, "x": args.practice_x, "buttons": practice}))
        print(f"Practice start at x={args.practice_x} ({len(practice)} frames in)")
    if args.resume or practice:
        # Every genome is scored afresh, the same way (both runs with a
        # practice start; and a saved population may have been scored by an
        # older fitness function).
        for genome in population.genomes():
            genome.fitness = 0
        population.max_fitness = 0.0
        for species in population.species:
            species.top_fitness, species.staleness = 0.0, 0

    evaluator = Evaluator(args.level, args.workers, practice)
    watch_env = make_marios_env(args.level, render_mode="human") if args.watch else None
    log_path = out / "log.csv"
    new_log = not log_path.exists() or not args.resume
    log_file = open(log_path, "w" if new_log else "a", newline="")
    log = csv.writer(log_file)
    if new_log:
        log.writerow(["generation", "species", "genomes_run", "best_fitness", "best_x",
                      "mean_fitness", "max_fitness_ever", "cleared", "seconds"]
                     + (["novelty_max", "archive"] if args.selection != "fitness" else []))
    results = {}  # id(genome) -> (genome, RunResult), so champions keep their run
    best_ever = 0.0  # the best real fitness so far (genome.fitness may be a novelty score)
    archive = None
    if args.selection != "fitness":
        archive_path = out / "novelty_archive.json"
        if args.resume and archive_path.exists():
            archive = NoveltyArchive.load(archive_path)
            print(f"Novelty archive: {len(archive)} paths")
        else:
            archive = NoveltyArchive(k=args.novelty_k, seed=args.seed)
    # Time already spent in earlier sessions of this run (for --max-minutes).
    spent = 0.0 if new_log else _seconds_in_log(log_path)

    print(f"Training on {args.level} with {evaluator.workers} workers, "
          f"population {population.config.population}")
    try:
        while population.generation < args.generations:
            start = time.time()
            # Only genomes that haven't played yet (champions keep their run).
            todo = [g for g in population.genomes() if results.get(id(g), (None,))[0] is not g]
            for genome, result in zip(todo, evaluator.run(todo)):
                genome.fitness = result.fitness
                results[id(genome)] = (genome, result)

            genomes = population.genomes()
            results = {id(g): results[id(g)] for g in genomes}
            runs = [results[id(g)][1] for g in genomes]
            extra = []
            if archive is not None:
                # NEAT breeds by the novelty-aware score; the log keeps the real fitness.
                selected = archive.select_scores([r.behavior for r in runs], [r.fitness for r in runs],
                                                 args.selection, args.novelty_weight)
                for genome, score in zip(genomes, selected["score"]):
                    genome.fitness = float(score)
                extra = [round(float(selected["novelty"].max()), 1), len(archive)]
                archive.save(out / "novelty_archive.json")
            population.record_fitness()

            best, best_result = max(zip(genomes, runs), key=lambda pair: pair[1].fitness)
            best_ever = max(best_ever, best_result.fitness)
            cleared = any(r.cleared for r in runs)
            mean = sum(r.fitness for r in runs) / len(runs)
            seconds = time.time() - start
            print(f"gen {population.generation:3d} | species {len(population.species):3d} | "
                  f"best fitness {best_result.fitness:7.1f} (x={best_result.rightmost:4d}) | "
                  f"mean {mean:6.1f} | {len(todo)} runs in {seconds:5.1f}s"
                  + (f" | practice best x={max(r.practice.rightmost for r in runs)}" if practice else "")
                  + (f" | novelty max {extra[0]}, archive {extra[1]}" if extra else "")
                  + ("  << LEVEL CLEARED" if cleared else ""))
            log.writerow([population.generation, len(population.species), len(todo),
                          round(best_result.fitness, 1), best_result.rightmost, round(mean, 1),
                          round(best_ever, 1), cleared, round(seconds, 1), *extra])
            log_file.flush()

            best_path = out / f"gen_{population.generation:03d}_best.json"
            best_path.write_text(_genome_json(best))
            if best_result.fitness >= best_ever:
                (out / "best.json").write_text(_genome_json(best))

            if watch_env is not None:
                run_genome(watch_env, best, render=True)

            if cleared and not args.keep_going:
                print(f"World cleared in generation {population.generation}! "
                      f"Watch it: python scripts/play_neat.py {out / 'best.json'}")
                population.save(out / "population.json")
                break
            population.new_generation()
            population.save(out / "population.json")
            spent += time.time() - start
            if args.max_minutes is not None and spent >= args.max_minutes * 60:
                print(f"Stopping: {spent / 60:.1f} of {args.max_minutes:g} minutes used. "
                      f"Resume with --resume {out / 'population.json'}")
                break
    finally:
        evaluator.close()
        log_file.close()
        if watch_env is not None:
            watch_env.close()


def _buttons_to(level: str, genome_path: str, x: int) -> list[list[int]]:
    """The buttons the genome presses, frame by frame, until Mario first reaches ``x``."""
    genome = Genome.from_dict(json.loads(Path(genome_path).read_text()))
    env = make_marios_env(level)
    buttons, reached = [], []

    def on_frame(agent, info, stats):
        if not reached:
            buttons.append(list(agent.buttons))
            if info["x_pos"] >= x:
                reached.append(True)

    run_genome(env, genome, on_frame=on_frame)
    env.close()
    if not reached:
        raise SystemExit(f"{genome_path} never reaches x={x}")
    return buttons


def _seconds_in_log(path: Path) -> float:
    """Seconds of training already recorded in a run's log.csv."""
    with open(path) as f:
        return sum(float(row["seconds"]) for row in csv.DictReader(f) if row.get("seconds"))


def _genome_json(genome) -> str:
    return json.dumps(genome.to_dict())


if __name__ == "__main__":
    main()
