# 6. Training on Modal

Training used to run on whatever machine we had: a 4-core cloud sandbox that
is thrown away when idle, or the 10-core Mac mini, one run at a time. This
step moves training to [Modal](https://modal.com), a cloud where a Python
function can ask for 16 cores, run for hours, and save its files to a disk
that outlives it. Nothing about NEAT changes: the same `scripts/train_neat.py`
runs inside the container.

## What runs where

```
your computer                      Modal
-------------                      -----
modal run ... ::launch   --->      container "train" (16 cores, 16 GiB)
  uploads the start population       scripts/train_neat.py --workers 32
  starts the run, then exits         writes /runs/<run>/... every generation
                                            |
modal run ... ::status   <---      Volume "marlio-runs" (the run's files)
modal volume get ...     <---
```

* **One container per run.** A genome plays for well under a second on
  average, so sending genomes one by one to other containers would spend
  more time on the network than on playing. Instead each run gets one big
  container and plays 32 games at a time in it, like the Mac plays 10.
* **Many runs at once.** Each `launch` starts another container, so an
  experiment with 9 runs takes as long as one run.
* **Nothing is lost.** Files go to the Volume `marlio-runs` and are committed
  every 30 seconds. If Modal stops the container (it can, to rebalance its
  machines), it is restarted and the run resumes from its last
  `population.json`.
* **A time limit.** `--max-minutes` (new in `train_neat.py`) stops training
  after that many minutes of generations and saves the population. Time used
  before a restart counts, so a run never trains longer than asked.

## Setting it up (once)

```
uv sync --extra cloud
uv run modal token new        # opens the browser; saves ~/.modal.toml
```

The token stays in your home folder, never in the repo.

## Using it

```
# Start a run: plain NEAT on 1-2, warm-started, at most 60 minutes
uv run modal run --detach marlio/cloud/modal_app.py::launch \
    --run neat-1-2-modal --level SuperMarioBros-1-2-v0 \
    --start runs/neat-1-3-release/population.json --minutes 60

# How every run is doing
uv run modal run marlio/cloud/modal_app.py::status

# Copy a run's files here
uv run modal volume get marlio-runs neat-1-2-modal runs/modal/

# Record its best genome (network overlay) as an .mp4, on Modal
uv run modal run marlio/cloud/modal_app.py::render --run neat-1-2-modal --level SuperMarioBros-1-2-v0
```

`--detach` keeps the run going after the command returns (and if the
computer sleeps). Live output is also on the run's page at modal.com.
`launch` uses `--auto-release-a 30` by default, the A-button fix that 1-3
needed; pass `--auto-release-a 0` for 1-1's champion.

## What it costs

Modal bills the cores and memory a container reserves, per second
([prices](https://modal.com/pricing), checked 2026-11-17):

| | price |
|---|---|
| CPU | $0.0000131 per physical core per second (about $0.047 per core-hour) |
| Memory | $0.00000222 per GiB per second |
| Starter plan | $30 of free usage per month |

A run's container (16 cores, 16 GiB) costs about **$0.88 per hour**. Each run
writes `modal-usage.json` with its wall time and this estimate; the real bill
is on modal.com under Usage.

## How fast

The NES emulator does about 580 frames per second on one core (measured in
the sandbox). Most of a generation's time is spent in it, so speed grows with
the number of game processes, until the generation has to wait for its
slowest genome (2 to 4 seconds for a long run). 32 processes should be about
2 to 3 times the Mac's 10.
