# Step 2: Evolving a network with NEAT

Step 1 gave the network its eyes (the 13 x 13 grid) and hands (6 buttons).
This step gives it a brain that improves on its own, using the same method
as MarI/O: **NEAT**, NeuroEvolution of Augmenting Topologies
([Stanley & Miikkulainen, 2002](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)).

```bash
python scripts/train_neat.py --watch        # train, showing each generation's best
python scripts/play_neat.py runs/neat/best.json   # watch the best network so far
```

## The idea: breeding networks, not training them

There is no gradient descent and no backpropagation. NEAT works like
breeding animals:

1. Start with a **population** of 300 tiny random networks.
2. Let each one **play** World 1-1 once and give it a score (**fitness**).
3. Keep the better ones, let them **reproduce** (with small random changes),
   and drop the rest.
4. Repeat. Each round is a **generation**.

What makes NEAT special is that it evolves the network's *shape*, not just
its weights. Networks start with almost no connections and grow neurons and
connections only when that helps.

## The pieces (`marlio/agents/neat/`)

### Genome (`genome.py`)

A network is described by a list of **genes**. Each gene is one connection:

| field | meaning |
|-------|---------|
| `into`, `out` | which neuron feeds which |
| `weight` | how strongly (positive or negative) |
| `enabled` | switched on or off |
| `innovation` | a global id for this connection, given when it first appears |

Neurons are numbered: `0..168` are the grid cells, `169` is the **bias**
(always 1), `1000000..1000005` are the 6 buttons, and hidden neurons get
numbers in between as they are created.

### Mutations

Every child is mutated. Each genome carries its own mutation rates, which
themselves drift up or down 5% each generation, so evolution also tunes how
boldly it mutates. Starting rates are MarI/O's:

| mutation | what it does | starting rate |
|----------|--------------|---------------|
| point | nudge every weight a little (10% of the time: replace it) | 25% |
| link | connect two neurons that weren't connected | 2 tries per child |
| bias | connect the bias neuron to something | 40% |
| node | split a connection in two with a new neuron in the middle | 50% |
| enable / disable | switch a connection on / off | 20% / 40% |

The **node** mutation is how networks grow: the new neuron starts with
weights 1 and the old weight, so the network behaves the same until later
mutations change it.

### Crossover

Two parents make a child by lining up their genes by innovation number.
Matching genes are taken from either parent at random; genes only the
fitter parent has are kept. Without innovation numbers there would be no way
to tell which parts of two differently shaped networks correspond.

### Species (`population.py`)

A brand-new connection usually makes a network *worse* at first, before
evolution tunes its weight. To give new ideas a chance, NEAT groups similar
genomes into **species** and makes genomes compete mostly within their own
species. Two genomes are the same species when

```
2.0 x (share of genes without a partner) + 0.4 x (mean weight difference) < 1.0
```

Each generation (`Population.new_generation`):

1. Each species keeps its better half.
2. Species that haven't improved in 15 generations are removed (unless they
   hold the best genome ever).
3. Species get children in proportion to the average *rank* of their members,
   and species too weak to earn one child are removed.
4. Children are made (75% by crossover, 25% by copying), mutated, and sorted
   into species. Each species' champion survives unchanged.

### Network (`network.py`)

A genome becomes a network: each neuron sums its weighted inputs and squashes
them with `2 / (1 + e^(-4.9 x)) - 1` (between -1 and 1). A button is pressed
when its output is above 0. Like MarI/O, neurons keep their values between
decisions, so a connection looping back gives the network some memory.

### Playing and scoring (`agent.py`)

`NEATAgent` asks its network for buttons every **5 frames** and holds them in
between, as MarI/O does. `run_genome` plays one run and scores it with
MarI/O's fitness:

```
fitness = (rightmost x reached) - (frames used) / 2   (+ 1000 for clearing the level)
```

Going right is rewarded, wasting time is penalised. A run ends when Mario
hasn't reached a new rightmost point for 20 frames (plus a grace period of a
quarter of the frames so far), dies, or reaches the flag. MarI/O only uses the
timeout; ending at death or the flag gives the same scores, just sooner.

Genomes are played in parallel, one game per CPU core (`Evaluator`). Champions
that survive unchanged keep their score instead of playing again, because the
game is deterministic.

## Training (`scripts/train_neat.py`)

Each generation prints one line:

```
gen   1 | species 227 | best fitness   360.5 (x= 594) | mean   48.4 | 150 runs in   2.7s
```

and saves to `runs/neat/`: the whole population (`population.json`, use
`--resume` to continue), the best genome (`best.json`), each generation's
best (`gen_NNN_best.json`) and a `log.csv` of the numbers above. Training
stops when a genome clears the level.

## Where this differs from MarI/O

* Python, and it runs headless and in parallel instead of in an emulator's
  Lua console, so a generation takes seconds, not minutes.
* Runs also end at death or at the flag (same score, faster).
* When two genomes share no genes, MarI/O's speciation divides by zero; here
  that counts as "different species", which is what the NaN did in Lua.
