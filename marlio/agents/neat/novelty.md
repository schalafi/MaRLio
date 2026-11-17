# Novelty search for NEAT (NSLC and Dominated Novelty Search)

Plain NEAT (MarI/O) breeds the networks that get furthest right. That works
on 1-1, where going right is always good. It fails on levels where the way
forward first *looks* worse: waiting for a fire bar to swing past, not
running into a piranha plant, going down to come back up. The whole
population then piles up in the same dead end (on 1-2 we were down to one
species). Novelty search keeps rewarding networks that do something *new*,
so the population keeps exploring until one of the new ideas pays off.

The network, the 13x13 inputs, the controller outputs and the overlay don't
change at all. Only the number NEAT breeds by changes.

## How a run is described

Every run records Mario's position every 30 frames, in tiles (16 px), for up
to 80 samples (40 seconds): a path of 80 (x, y) points. A run that ends
early repeats its last point, so "died in the first pit" is one place in
this space and "jumped the pit, then died" is another. Two runs are as
different as the distance between their paths.

## How a run is scored

`marlio/agents/neat/novelty.py`, `NoveltyArchive.select_scores()`, once per
generation:

1. **Novelty** = the mean distance from a run's path to its 15 nearest
   neighbours, among this generation and an *archive* of past novel paths
   (each generation the 5 most novel paths, plus one at random, are kept, up
   to 5,000). A path nobody has walked before is far from everything.
2. Then one of two ways to also ask "how good was it?":
   * **NSLC**, novelty search with local competition (Lehman & Stanley,
     2011): *local competition* = how many of its 15 nearest neighbours in
     this generation it beats on fitness. The novelty part is the average
     of the novelty rank and the local-competition rank.
   * **DNS**, Dominated Novelty Search (Bahlous-Boldi et al., 2025): the
     mean distance to the 15 nearest runs that are *better* than it. The
     best run of each niche has nobody better nearby, so it scores high; a
     near copy of a better run scores close to zero. The overall best gets
     the top score. One number does both jobs, with no grid or threshold to
     tune.
3. The breeding score mixes that with plain fitness, by rank:
   `score = (1 - w) * rank(fitness) + w * rank(novelty part)`, with
   `w = --novelty-weight` (0.5). So the search still drifts right while it
   explores.

NEAT then does everything as before (species, culling, breeding) with
`genome.fitness` = this score. The real fitness is still what the log, the
console, `best.json` and "cleared" use.

## Using it

```
python scripts/train_neat.py --level SuperMarioBros-1-4-v0 --selection nslc --auto-release-a 30
python scripts/train_neat.py --level SuperMarioBros-1-4-v0 --selection dns --novelty-weight 0.5
```

On Modal (`docs/06-training-on-modal.md`):

```
uv run modal run --detach marlio/cloud/modal_app.py::launch --run novelty-1-4 \
    --level SuperMarioBros-1-4-v0 --selection nslc --minutes 30 --hard-minutes 72
```

The archive is saved as `novelty_archive.json` next to `population.json`,
so `--resume` continues with it. `log.csv` gets two more columns: the
largest novelty of the generation and the archive's size.

## Things to know

* `--selection fitness` (the default) is exactly MarI/O; a 3-generation run
  with seed 0 gives the same log as before this change.
* Species go stale when their best *score* stops rising. Scores are ranks,
  so a species only keeps "improving" while its best stays near the top;
  expect more turnover of species than with plain fitness.
* The path is the main design choice. If novelty wanders (e.g. exploring
  going left), lower `--novelty-weight`.
