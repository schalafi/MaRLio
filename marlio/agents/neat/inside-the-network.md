# Inside the network that clears World 1-1

This page takes apart the network in `trained/neat-1-1.json`, the one you
see playing in the overlay video. It covers what goes in (inputs), what
happens in the middle (hidden neurons, and where they came from), what comes
out (buttons), and what connects them (edges). All numbers on this page were
measured from that file and from replaying it.

## The whole thing at a glance

```
  170 inputs                16 hidden neurons            6 outputs
  (13x13 grid + bias)       (grown by evolution)         (buttons)

  [grid cell] ----\
  [grid cell] -----+----> (hidden) ---+
  [grid cell] ----/          |         +----> [A]  [B]  [up]
     ...                  (hidden) ----+----> [down] [left] [right]
  [bias = 1] ------------------------------/
```

| part | count | in the overlay |
|------|-------|----------------|
| Inputs | 169 grid cells + 1 bias | squares in the grid on the left, the small circle at its corner |
| Hidden neurons | 16 | numbered circles in the middle |
| Outputs | 6 | the NES controller on the right |
| Connections (edges) | 122 genes: 116 switched on, 6 switched off | white and black lines |

Unlike a textbook neural network there are no layers. Any neuron can
connect to any other one, and the network's *shape* was found by evolution
together with its weights (see [step 2](../../../docs/02-neat.md)).

## 1. Inputs: what the network sees

Every 5 frames the network gets **170 numbers**:

* **169 grid cells.** A 13 x 13 picture of the level around Mario, read from
  the game's memory (RAM), not from the screen. Each cell is one 16 x 16
  pixel block of the level and holds:
  * `1`: something solid (ground, brick, pipe, ? block, stairs)
  * `-1`: an enemy (goomba, koopa)
  * `0`: empty
  
  Mario is always the centre cell. The grid reaches 6 blocks in every
  direction, about 100 pixels each way. How it's built is explained in
  [step 1](../../../docs/01-environment-and-inputs.md).
* **1 bias.** An input that is always `1`. It lets the network press a
  button even when it sees nothing, like a constant "default urge". In this
  network the bias feeds "right" with weight +1.01, which on its own is
  enough to keep Mario running right.

What the network does **not** get: pixels, colours, Mario's speed or
direction, his position in the level, the timer, coins, power-ups, or any
difference between a brick and a ? block. Each look at the grid is a single
snapshot; nothing tells it how things are moving. Unlike pixel-based RL,
which stacks several frames so it can work out motion, NEAT here works in a
low-information setting where everything it sees is already labelled.

### Which cells this network actually uses

Evolution only connected **58 of the 169 cells**; the other 111 are ignored
completely. `#` marks a cell with at least one connection, `M` is Mario:

```
. . . # # # . # # . . . #
. . . # . . . # . # # . .
. . . . # . # . . # # # #
. . . . . . # . . . # # .
# . . # . # . . # . # # #
# # . . # . # . # . . # #
# # . . . . M # . . . . .
. . . . . . # # # # # . .
. . # # . . . # . . . . .
. # . . . . . . . . . . #
# # # . . . . # # . . . .
. . . . . . . . . . . # .
# . . . # # # # . # . . .
```

The pattern isn't something a person would design. NEAT adds connections at
random and keeps whatever helps Mario get further, so the used cells are a
mix of genuinely useful ones (for example the row just under Mario's feet
and cells ahead of him) and harmless leftovers that evolution had no reason
to remove. Which cell does what can't be read off this map; the overlay
shows it better, by lighting up the connections that are active at each
moment.

## 2. Hidden neurons: the middle of the network

### What one hidden neuron does

Each hidden neuron adds up its incoming signals, each multiplied by its
connection's weight, and squashes the total into the range -1 to +1:

```
total = sum(weight * value of the neuron it comes from)
value = 2 / (1 + e^(-4.9 * total)) - 1      (MarI/O's sigmoid)
```

The curve is very steep around 0, so in practice most neurons sit at about
`-1` or `+1` (or exactly `0` while nothing feeding them is active): they act
like switches more than dials. That's why the overlay mostly shows `1`, `0`
and `-1` in the circles, with an occasional `.3` or `-.9`.

Example, hidden neuron 170: it listens to 8 grid cells and the bias (weights
from -1.93 to +1.01), and feeds the "B" button (-0.33) and another hidden
neuron (+1.06).

### Where they came from

The network started with **no hidden neurons at all**. NEAT begins minimal:
generation 1's networks are a few random direct connections from inputs to
buttons. Hidden neurons appear through one specific mutation, **add node**
(`node_mutate` in `marlio/agents/neat/genome.py`):

1. Pick an existing connection, say `cell -> A` with weight `w`.
2. Switch it off.
3. Add a new neuron `h` in its place, with two new connections:
   `cell -> h` (weight 1) and `h -> A` (weight `w`).

At first the behaviour barely changes (the signal now passes through `h`).
Later mutations then give `h` more connections, and if that helps Mario get
further, the change survives into the next generations. Each new neuron gets
the next free number in its genome, which is why this network's hidden
neurons are numbered between 170 and 210: 170 was the first one added in
this network's line of ancestors, and 210 one of the latest. 16 of them
survived to the winner of generation 67.

Some switched-off connections in the file are the "scars" of these splits:
for example `cell (2 below, 1 right of Mario) -> neuron 176` is switched off.
6 connections are off in total.

### A little memory

3 connections go "backwards" between hidden neurons (174 → 173, 193 → 176,
209 → 189). The network computes neurons in number order, so these read
the value the source had at the *previous* decision (5 frames earlier).
Neurons also keep their value between decisions. Together, that gives the
network a very short memory, its only one.

## 3. Outputs: the buttons

There are 6 output neurons, one per button: **A** (jump), **B** (run),
**up**, **down**, **left**, **right**. They work like hidden neurons, and
the rule is simple: **value above 0 means hold the button** until the next
decision 5 frames later.

How often this network held each button during its winning run (1,387
frames):

| button | held | what that means |
|--------|------|-----------------|
| right | 100% | always moving right (bias +1.01 keeps it on) |
| B | 99% | always running |
| A | 33% | jumping a third of the time: over pipes, gaps, enemies, stairs |
| up | 52% | does nothing in 1-1; evolution never had a reason to remove it |
| left | 7% | with right also held, they cancel each other (see below) |
| down | 4% | cancels with up too; otherwise crouching, harmless while running |

Two details:

* **Opposite directions cancel.** Like MarI/O, pressing left and right
  together counts as neither (same for up and down). So when "left" lights
  up, Mario stops pushing right for a moment and slows down a little.
* **Two outputs feed other outputs**: up → down (weight -0.78) and
  down → B (-1.35). Evolution may connect any neuron to any other, outputs
  included.

What each output listens to:

| button | incoming connections |
|--------|----------------------|
| A | 10 (7 grid cells, the bias, 2 hidden neurons) |
| B | 17 |
| up | 11 |
| down | 10 |
| left | 8 |
| right | 4 (3 grid cells and the bias) |

## 4. Edges: the connections

Each edge is one **gene** in the genome file:

```json
{"into": 97, "out": 1000001, "weight": -0.993, "enabled": true, "innovation": 425}
```

| field | meaning |
|-------|---------|
| `into` | the neuron the signal comes from (0-168 grid cells, 169 bias, 170+ hidden, 1000000+ buttons) |
| `out` | the neuron it goes to (1000000 = A, 1000001 = B, ... 1000005 = right) |
| `weight` | how strongly and which way: positive pushes the target towards +1, negative towards -1 |
| `enabled` | switched on or off (off edges are kept, because they might be switched back on) |
| `innovation` | a global id for when this connection first appeared anywhere in the population |

This network's weights range from -1.93 to +1.98 (63 positive, 53
negative, typical size about 0.9). In the overlay a positive weight is a
**white** line and a negative one a **black** line (white is +, black is -,
the same rule as the neurons), thicker for a stronger weight. A line is only
drawn while its source isn't 0, that is while it's actually carrying a
signal.

### Where edges come from

Edges come from mutations too (all in `genome.py`):

* **Add connection** (`link_mutate`): connect two neurons that weren't
  connected, with a random weight between -2 and 2.
* **Add bias connection**: the same, from the bias input.
* **Change weights** (`point_mutate`): nudge every weight a little, or now
  and then replace one with a new random value.
* **Enable / disable**: switch an existing edge on or off.
* **Add node** (above) also switches one edge off and adds two.

And from **crossover**: a child takes the edges of two parents, matched up
by innovation number. That's what the `innovation` field is for: it lets
NEAT tell that two genomes' edges are "the same connection" even though
the genomes evolved separately. This network's innovation numbers go from
425 to 60194: its oldest edge appeared early in training and its newest
near the end.

## 5. One decision, step by step

Every 5 frames:

1. Read the RAM and build the 13 x 13 grid; add the bias (1).
2. Put those 170 values into the input neurons.
3. Go through the hidden neurons in number order (170, 172, ... 210), then
   the 6 outputs. For each one, sum `weight * value` over its incoming
   edges and apply the sigmoid.
4. Every output above 0 is a held button. Hold them for the next 5 frames.

That's all the "thinking" there is: about 116 multiplications every 5
frames. The intelligence is entirely in which edges exist and their
weights, which is what 67 generations of evolution found.

## Where to find it in the code

| what | file |
|------|------|
| the input grid | `marlio/inputs.py` |
| buttons and the cancel rule | `marlio/controller.py` |
| genes, mutations, crossover | `marlio/agents/neat/genome.py` |
| evaluating the network | `marlio/agents/neat/network.py` |
| deciding every 5 frames, fitness | `marlio/agents/neat/agent.py` |
| drawing it over the game | `marlio/agents/neat/display.py` |
