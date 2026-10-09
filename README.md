# MaRLio

Super Mario Bros. solved by learning algorithms, shown interactively.

The first milestone is a Python version of SethBling's
[MarI/O](https://gist.github.com/SethBling/598639f8d5e8afb5453a0b9519be51ff),
which uses NEAT ([Stanley & Miikkulainen, 2002](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf))
to evolve a neural network that beats World 1-1. The repo is laid out so
other algorithms (PPO, DQN, ...) can be added and compared on the same game.

## Run it in GitHub Codespaces

1. On GitHub, click **Code → Codespaces → Create codespace**. Setup installs
   everything (takes a few minutes the first time).
2. Open the **Ports** tab, find **Desktop (game window)** (port 6080) and
   click the globe icon to open it in a browser tab. Click **Connect**; the
   password is `vscode`.
3. In the Codespace terminal, run:

   ```bash
   python scripts/random_agent.py
   ```

   The game window appears in the desktop tab.

## Run it locally

Needs Python 3.13+.

```bash
pip install -e ".[dev]"
python scripts/random_agent.py
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
scripts/
  random_agent.py       watch random play
  show_inputs.py        watch the game and the grid the network sees
  make_input_figure.py  regenerate the figure in docs/
docs/
  01-environment-and-inputs.md   step 1 explained
tests/
```

To add an algorithm, put it in `marlio/agents/` as a subclass of `Agent`
(one method: `act(observation) -> action`), and pick the environment that
matches what it needs to see.

## Docs

1. [The environment and what the agent sees](docs/01-environment-and-inputs.md)

## Tests

```bash
pytest
```
