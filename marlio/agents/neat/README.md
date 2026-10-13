# NEAT (Python MarI/O)

A port of the NEAT part of SethBling's MarI/O. How it works is explained in
[docs/02-neat.md](../../../docs/02-neat.md); this page is the how-to.

Run every command from the repository root. In a Codespace, open the
**Desktop** tab first (Ports → 6080, password `vscode`) to see the game window.

## Train

```bash
python scripts/train_neat.py                 # train, saving to runs/neat/
python scripts/train_neat.py --watch         # also show each generation's best run
python scripts/train_neat.py --resume runs/neat/population.json   # continue a stopped run
```

Each generation prints one line:

```
gen   1 | species 227 | best fitness   360.5 (x= 594) | mean   48.4 | 150 runs in   2.7s
```

Training stops by itself when a network reaches the flag (World 1-1 took 67
generations, about 40 minutes on 4 cores). `Ctrl+C` is safe: the population is
saved after every generation, so `--resume` picks up where it stopped.

| option | default | what it does |
|--------|---------|--------------|
| `--watch` | off | play each generation's best in the game window |
| `--population` | 300 | genomes per generation |
| `--generations` | 200 | stop after this generation |
| `--workers` | all cores | games played in parallel |
| `--seed` | 0 | random seed (same seed, same run) |
| `--level` | `SuperMarioBros-1-1-v0` | which level to train on |
| `--out` | `runs/neat` | where to save |
| `--resume` | | `population.json` to continue from |
| `--keep-going` | off | keep training after the level is cleared |

### What gets saved in `runs/neat/`

| file | contents |
|------|----------|
| `population.json` | the whole population, for `--resume` |
| `best.json` | the best network so far |
| `gen_NNN_best.json` | the best network of generation NNN |
| `log.csv` | one row of statistics per generation |

## Play (test a trained network)

```bash
python scripts/play_neat.py trained/neat-1-1.json        # a network that already clears 1-1
python scripts/play_neat.py runs/neat/best.json          # the best from your own run
python scripts/play_neat.py runs/neat/gen_020_best.json  # any generation's best
```

It prints the result of each run, for example:

```
fitness 3467.5, reached x=3161 in 1387 frames, LEVEL CLEARED
```

| option | default | what it does |
|--------|---------|--------------|
| `--runs` | 1 | how many times to play |
| `--no-render` | off | no window, only print the result (and record, with `--record`) |
| `--level` | `SuperMarioBros-1-1-v0` | which level to play |
| `--record` | none | also save the run as an .mp4, e.g. `--record videos/neat-1-1.mp4` |
| `--fps` | 60 | video frame rate; 60 keeps every game frame, 30 every other one |
| `--no-overlay` | off | show the game without the network drawn on top |
| `--scale` | 3 | enlarge the 256 x 240 screen this many times |
| `--outro` | 3 | seconds to keep showing the game after the flag |

The window and the video show the network on top of the game, as MarI/O
does: the input grid, hidden neurons, connections lighting up and the
buttons being pressed. [docs/03-overlay-and-recording.md](../../../docs/03-overlay-and-recording.md)
explains how to read it.

## Chart a run

```bash
python scripts/plot_neat_progress.py runs/neat/log.csv --out progress.png
```

## Use it from Python

```python
import json
from marlio.agents.neat import Genome, NEATAgent
from marlio.env import make_marios_env

genome = Genome.from_dict(json.load(open("trained/neat-1-1.json")))
agent = NEATAgent(genome)
env = make_marios_env(render_mode="human")
grid, info = env.reset()
while True:
    grid, reward, terminated, truncated, info = env.step(agent.act(grid))
    env.render()
    if terminated or truncated or info["flag_get"] or info["is_dying"]:
        break
env.close()
```

## Files

| file | contents |
|------|----------|
| `genome.py` | genes, mutations, crossover, speciation distance, MarI/O's constants (`NEATConfig`) |
| `network.py` | builds and runs the network for a genome |
| `population.py` | species, selection, breeding, save/load |
| `agent.py` | `NEATAgent` (presses buttons every 5 frames), `run_genome` (MarI/O's fitness), `Evaluator` (parallel games) |

## Inside the trained network

[inside-the-network.md](inside-the-network.md) takes apart `trained/neat-1-1.json`:
its inputs, hidden neurons and where they came from, outputs, and edges.
