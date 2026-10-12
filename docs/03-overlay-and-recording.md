# Step 3: Watching the network think, and recording it

Step 2 produced a network that clears World 1-1, but watching it you only
see Mario. This step draws the network itself over the game, the way MarI/O
does, and saves the result as an .mp4 you can share.

```bash
python scripts/play_neat.py trained/neat-1-1.json                                # live window with the network
python scripts/play_neat.py trained/neat-1-1.json --record videos/neat-1-1.mp4   # ... and save it as a video
python scripts/play_neat.py trained/neat-1-1.json --record videos/neat-1-1.mp4 --no-render   # video only, no window
```

In a Codespace the window appears in the desktop tab (port 6080), as in the
earlier steps. `--no-render` works anywhere, even without a desktop, and
records about as fast as the game can run (about 20 s for the 26 s clip).

![The trained network playing World 1-1, with its network drawn on top](images/network-overlay.png)

## Reading the overlay

The layout is MarI/O's (`displayGenome` in the Lua script), in the same
screen positions, ported in `marlio/agents/neat/display.py`.

| what | where | meaning |
|------|-------|---------|
| **Input grid** | grey box, top left | The 13 x 13 cells the network sees ([step 1](01-environment-and-inputs.md)). White squares are solid blocks, black squares are enemies; empty cells are not drawn. The red mark in the middle is Mario. |
| **Bias** | the square under the grid | The input that is always 1, so the network can press buttons even when it sees nothing. |
| **Hidden neurons** | the squares in the middle | Neurons NEAT added during evolution. The shade is the neuron's value, from black (-1) through grey (0) to white (+1). |
| **Outputs** | right edge | One square per button. The name turns **blue** while the network holds that button. |
| **Connections** | lines | **Green** = positive weight (the source pushes the target up), **red** = negative (pushes it down). Darker = stronger. A line is bright while its source neuron is sending a signal and faint while the source is 0. |
| **Banner** | white strip at the top | The level, the genome's training fitness, and the fitness of this run as it builds up (x reached - frames / 2, +1000 at the flag), like MarI/O's banner. |

What to watch for: as a pipe edge (white squares) or an enemy (black
square) moves into the grid, the lines leaving those cells light up and the
button names change colour as the network reacts.

The network only thinks every 5 frames (MarI/O's choice, see step 2) and
holds its buttons in between, so the overlay changes in small steps.

### How hidden neurons get their place

Inputs and outputs have fixed places. Hidden neurons have no natural
position, so MarI/O starts them all at one spot and then, 4 times over,
pulls each one a quarter of the way towards the neurons it connects to,
pushing it 40 pixels left of anything it feeds and right of anything that
feeds it, and keeping it between x = 90 and 220. The result roughly reads
left to right, from the grid to the buttons. The layout only depends on the
genome, so it is computed once per run.

## Recording

`marlio/video.py` has two small classes that work on plain RGB frames, so
any future agent (PPO, DQN, ...) can reuse them:

* `VideoRecorder(path, fps=60)` writes an H.264 .mp4 (yuv420p, which plays
  in every browser and phone). It uses the ffmpeg binary that the
  `imageio-ffmpeg` package bundles, so nothing needs installing with apt.
* `LiveViewer()` shows frames in the same pyglet window gym-super-mario-bros
  uses, and waits between frames so the game runs at real speed rather
  than as fast as the computer allows.

### Frame rate

The NES draws 60 frames per second, and each `env.step` advances exactly one
frame. So:

* `--fps 60` (default) keeps every frame: real speed, smoothest motion.
* `--fps 30` keeps every other frame: still real speed, about 20% smaller.
* `--fps 20` and `--fps 15` work the same way. Rates that don't divide 60
  (such as 25) are refused, because they would need frames duplicated or
  dropped unevenly and the game would judder.

The 60 fps recording of the trained network is about 26 s (23 s of play
plus a 3 s outro) and about 9 MB at 768 x 720.

### Other options

| option | default | what it does |
|--------|---------|--------------|
| `--scale` | 3 | Enlarges the 256 x 240 NES screen (3 gives 768 x 720). The overlay is drawn at full size, so text and lines stay sharp. |
| `--no-overlay` | off | Shows or records the plain game. |
| `--outro` | 3 | Seconds to keep the game running after the flag (no buttons pressed), so the video ends with Mario sliding down the pole and walking to the castle instead of cutting off mid-jump. |
| `--runs` | 1 | Plays several times in a row. |

## How it fits in

* `run_genome` (step 2) gained an optional `on_frame(agent, info)` callback
  that's called after every frame. `play_neat.py` uses it to draw and
  record, so the replay runs exactly the same loop, and gets exactly the
  same score, as training did (3467.5 for the trained network).
* `NetworkDisplay(genome, network)` draws the network's *current* neuron
  values, which the `Network` from step 2 already keeps between frames.
* The network's picture is redrawn only when its values change (when it
  thinks, every 5 frames), and only the part of the screen it covers is
  blended, which keeps the live window at 60 fps.
