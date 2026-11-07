# NEAT on World 1-1, every generation

The best genome of every generation of two NEAT training runs on World 1-1,
kept so the training can be shown step by step (and so odd generations can be
found and replayed).

| Folder | Seed | Cleared at | Distinct best genomes | Milestones (new best x) |
|--------|------|-----------:|----------------------:|------------------------:|
| `seed0/` | 0 | gen 67 | 18 of 68 | 0, 1, 5, 8, 9, 11, 18, 19, 34, 41, 54, 59, 67 |
| `seed1/` | 1 | gen 86 | 17 of 87 | 0, 1, 4, 5, 8, 9, 12, 23, 27, 28, 49, 53, 76, 86 |

`seed0` is the original run behind `trained/neat-1-1.json`. Its per-generation
genomes were not kept the first time, so it was re-run on 2026-11-07 with the
same code (`train_neat.py --seed 0`, the fitness from step 2). NEAT training here
is deterministic, so the re-run matched the original `neat-1-1-log.csv` row for
row and ended with the same winning genome. `seed1` is a second run with the
same settings and `--seed 1`.

Each folder has:

* `log.csv`: the training log (fitness here is the step-2 MarI/O fitness)
* `genomes/gen_NNN_best.json`: the best genome of generation NNN, re-scored
  with the current fitness (`marlio/fitness.py`) so `play_neat.py` shows a
  consistent "Fitness a / b"
* `index.csv`: one row per generation: best x, run length in frames, whether
  it cleared, whether it was a new best, and `same_genome_as_gen` when the
  generation kept an earlier generation's best genome unchanged

Watch one: `python scripts/play_neat.py trained/neat-1-1-generations/seed0/genomes/gen_018_best.json`

Record a clip per generation plus a milestone montage:
`python scripts/record_generations.py trained/neat-1-1-generations/seed0/genomes footage/seed0 --label seed0`
(it also takes a `train_neat.py --out` folder directly).
