# MaRLio

Super Mario Bros. solved by learning algorithms, shown interactively.

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/schalafi/MaRLio)

The first milestone is a Python version of SethBling's
[MarI/O](https://gist.github.com/SethBling/598639f8d5e8afb5453a0b9519be51ff),
which uses NEAT ([Stanley & Miikkulainen, 2002](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf))
to evolve a neural network that beats World 1-1. The repo is laid out so
other algorithms (PPO, DQN, ...) can be added and compared on the same game.

## Run it in GitHub Codespaces

1. On GitHub, click **Code → Codespaces → Create codespace**. Setup uses
   [uv](https://docs.astral.sh/uv/) to create `.venv` with everything
   installed (takes a few minutes the first time). The terminal's `python`
   is that environment.
2. Open the **Ports** tab, find **Desktop (game window)** (port 6080) and
   click the globe icon to open it in a browser tab. Click **Connect**; the
   password is `vscode`.
3. In the Codespace terminal, run:

   ```bash
   python scripts/random_agent.py
   ```

   The game window appears in the desktop tab.

## Run it locally

Needs Python 3.13+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv sync                                  # creates .venv from uv.lock
uv run python scripts/random_agent.py
```

On Linux you may also need `sudo apt-get install libgl1 libglu1-mesa`.

## What's here

```
marlio/
  env.py          make_env() for pixel-based RL, make_marios_env() for MarI/O
  inputs.py       MarI/O's 13x13 input grid, read from the game's memory
  controller.py   MarI/O's 6-button controller
  agents/         one module per algorithm, all sharing the Agent interface
    base.py
    random_agent.py
    neat/           MarI/O's NEAT: genomes, species, network, fitness
scripts/
  random_agent.py       watch random play
  show_inputs.py        watch the game and the grid the network sees
  make_input_figure.py  regenerate the figure in docs/
  train_neat.py         evolve networks until one clears World 1-1
  play_neat.py          watch a trained network play
  plot_neat_progress.py chart a training run's progress
trained/
  neat-1-1.json         a network that clears World 1-1
docs/
  01-environment-and-inputs.md   step 1 explained
  02-neat.md                     step 2 explained
tests/
```

To add an algorithm, put it in `marlio/agents/` as a subclass of `Agent`
(one method: `act(observation) -> action`), and pick the environment that
matches what it needs to see.

## Docs

1. [The environment and what the agent sees](docs/01-environment-and-inputs.md)
2. [Evolving a network with NEAT](docs/02-neat.md)

## Train MarI/O

```bash
python scripts/train_neat.py --watch              # watch each generation's best in the desktop tab
python scripts/play_neat.py runs/neat/best.json   # replay the best network
```

A network that already clears World 1-1 (evolved in 67 generations) is in
`trained/`:

```bash
python scripts/play_neat.py trained/neat-1-1.json
```

## Tests

```bash
uv run pytest
```
