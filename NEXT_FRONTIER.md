# NEXT_FRONTIER: other ways to solve Super Mario Bros

*Written 2026-11-24. A map of the methods we could use after NEAT, how each one
would fit MaRLio, and which ones to compare. No training has been started for
anything in this document.*

So far MaRLio has used **NEAT** (the MarI/O method) and NEAT with **novelty
search**. They cleared 1-1, 1-3 and 1-4, and got stuck on 1-2 (x 2,278) and
2-1 (the piranha pipe at x ~1,950). The next main experiment is **AlphaZero**
(plan in `marlio-research/alphazero-plan.md`). This document places AlphaZero
among the other families of methods, so we can pick fair baselines to compare
it with.

---

## 1. The whole map on one page

| Family | Methods | Learns a player? | Needs a GPU? | Rough cost to clear 1-1 | Fit for MaRLio |
|---|---|---|---|---|---|
| **Search** | A\*, plain MCTS, Go-Explore | A\*/MCTS: no. Go-Explore: yes, in its second phase | No | **< $1** (A\*, MCTS), **$2-20** (Go-Explore) | Excellent: the emulator is an exact, deterministic model |
| **Value-based RL** | DQN, Double DQN, Rainbow, R2D2, Agent57, BBF | Yes | Yes (pixels) | **$5-20** (DDQN/Rainbow); **$100s-1,000s** (R2D2/Agent57) | Good, classic pixel route |
| **Policy gradient** | PPO, PPO + curiosity (ICM, RND) | Yes | Helpful | **$2-10** | Very good, the most common Mario method |
| **Model-based RL** | MuZero, EfficientZero, DreamerV3 | Yes | Yes | **$15-100+** | Possible, but we already have an exact model |
| **Imitation / offline** | Behavior cloning, Decision Transformer | Yes, from recorded runs | Optional | **< $2** once we have winning runs | Very good as a "distil the winners" step |
| **Evolution** (current) | NEAT, novelty search, MAP-Elites, CMA-ES, Deep GA | Yes | No | **$0-1** for 1-1 (measured with NEAT) | Already built |
| **AlphaZero** (next) | Gumbel AlphaZero with savestate search | Yes | No (grid version) | **$1.20-4.50** (planned) | Main experiment |

How to read the cost column: these are order-of-magnitude guesses for **one
level (1-1)**, using the prices and speeds below. They are not quotes. The
first experiment of each method gives the real number.

### Prices and speeds used throughout

- **Modal** (checked 2026-11-24): $0.0472 per CPU core-hour, $0.0080 per GiB-hour
  of memory, T4 GPU $0.59/h, L4 $0.80/h, A10 $1.10/h. Starter plan has $30/month
  of free credit. A 16-core container with 16 GiB costs about **$0.88/hour**.
- **Emulator speed**: nes-py about 1,000 frames/s per core; the stable-retro core
  about 4,000-5,000 frames/s per core (measured on the title screen; still to be
  checked mid-level).
- **Rule of thumb**: on Modal CPU, **100 million emulated frames cost about
  $1.50 with nes-py, or about $0.40 with stable-retro.** On the Mac mini
  (10 cores) they cost nothing, and take about 2 hours with nes-py.
- **For scale**: one pass through 1-1 is about 2,000-3,000 frames at full
  speed. NEAT cleared 1-1 at generation 67, in about 40 minutes on 4 cores.

---

## 2. Two choices that every method has to make

### 2.1 What the agent sees: the 13x13 grid or pixels

- **The MarI/O grid** (`marlio/inputs.py`): 169 numbers read from the game's
  memory (ground = 1, enemy = -1, empty = 0) around Mario. It is tiny, works on
  a CPU, is the same for every level, and matches our overlay. Its blind spots:
  all enemies look the same (a spiny looks like a goomba), and projectiles,
  hammers and moving platforms need extra channels.
- **Pixels** (usually 84x84 grayscale, 4 frames stacked): no hand-made features,
  so it is the "standard" RL setting and what most papers use. It needs a
  convolutional network, a GPU, and roughly 10x more compute.

Any method below can use either input. For a fair comparison, **all methods in
one race should see the same input.**

### 2.2 Whether the method may use savestates

Because the emulator is deterministic, we can save the game at any moment and
jump back to it. nes-py has only one save slot; stable-retro has unlimited
13 KB snapshots (see the AlphaZero plan, section 2.2). This splits the methods:

- **Need savestates**: A\*, MCTS, Go-Explore, AlphaZero. They plan by trying
  moves in a copy of the game.
- **Do not need them**: DQN, PPO, Dreamer, NEAT and the rest. They only play
  forward, like a human would. They can still *use* savestates for "practice
  starts" (start some episodes near the hard part), which we already do with
  NEAT's `--practice-from`.

This matters when we report results: a search method that uses savestates is
"allowed to try a move and take it back", which a human player can't do. It is
fair, but it should be stated.

---

## 3. Search: use the emulator as a perfect model

The game is deterministic, so we can try a move, look at what happens, and take
it back. Search methods exploit this directly. They are the strongest way to
**clear** a level, and the cheapest. The catch: plain search **learns nothing**.
It must search again every time it plays.

### 3.1 A\* / best-first search (Baumgarten)

**What it is.** A\* keeps a list of game states it could expand, and always
expands the one that looks most promising: "distance travelled so far" plus an
estimate of "time still needed to reach the end of the screen". To expand a
state, it tries each button combination for a few frames in the simulator.
Robin Baumgarten's A\* agent won the **2009 Mario AI Competition** and played
almost perfectly, dodging enemies by a pixel.

**How it fits MaRLio.** Very well. We need the savestate environment from the
AlphaZero plan (step 1), the same 8-action menu with frame skip 4, and a
heuristic "how many decisions to the right edge at top speed". Deaths and
damage get a huge cost. The plan per decision is short (half a second to a
second of game time), then the agent replans.

**Strengths.**
- Very likely to clear most of our levels, including 1-2 and 2-1, because it
  can find "wait for the plant, then jump" by trying it.
- Cheap and simple: a few hundred lines, no training.
- Gives a **ceiling**: "this level is solvable with this action menu".

**Weaknesses.**
- Learns nothing: no network, nothing to show in the overlay except the search.
- Myopic: it only looks a second ahead, so mazes and dead ends (4-4, 7-4, the
  1-2 ceiling route) can fool it.
- Baumgarten's version used a simulator that ran far faster than real time.
  With nes-py we expand maybe a few hundred states per decision, so it is
  slower than the original.

**Rough cost.** Expanding ~200-1,000 states per decision x ~4 frames x
~700 decisions ≈ 1-3 million frames per level. **Minutes on the Mac, a few
cents on Modal.**

**References.**
- Togelius, Karakovskiy & Baumgarten, *The 2009 Mario AI Competition* (IEEE CEC 2010).
- Karakovskiy & Togelius, *The Mario AI Benchmark and Competitions* (IEEE TCIAIG, 2012).

### 3.2 Plain MCTS / rollout search

**What it is.** Monte Carlo Tree Search builds a tree of moves from the current
state. Each "simulation" walks down the tree, tries a new move, and scores the
result (by a short random or "run right" rollout). After a fixed number of
simulations, it plays the move that was tried most. This is **AlphaZero with
the network removed**: the policy is uniform and the value comes from rollouts.

**How it fits MaRLio.** It is step 2 of the AlphaZero plan: the search-only
baseline. Same savestate env, same reward (change in `marlio/fitness.py` score
per decision), same action menu. Building it first tests the whole savestate
and tree machinery before any network is involved.

**Strengths.**
- Almost free once the AlphaZero code exists; shares most of it.
- Tells us exactly how much AlphaZero's network adds on top of search.
- Handles randomness and long rollouts better than A\*.

**Weaknesses.**
- Learns nothing (same as A\*).
- Rollouts are noisy and expensive: each one costs extra emulator frames.
- With a small budget (16-32 simulations) it may play worse than A\*.

**Rough cost.** 32-200 simulations x ~10-20 frames each x ~700 decisions
≈ 0.5-3 million frames per level. **Minutes, cents.**

**References.**
- Browne et al., *A Survey of Monte Carlo Tree Search Methods* (IEEE TCIAIG, 2012).
- Jacobsen, Greve & Togelius, *Monte Mario: Platforming with MCTS* (GECCO 2014).

### 3.3 Go-Explore

**What it is.** Go-Explore (Uber AI, 2019) has two phases.
1. **Explore**: keep an archive of "cells", coarse summaries of states such as
   *(x / 16, y / 16, level section)*. Repeatedly pick a promising cell (one that
   is new or rarely visited), **return** to it instantly by loading its
   savestate, then **explore** from it with random moves. Any new cell reached
   is added with its savestate. Stepping stones are never lost.
2. **Robustify**: train a network (with imitation or PPO, starting episodes
   from points along the best run, working backwards) so that it can repeat
   the winning run *without* savestates and with small random delays.

It was the first method to beat Montezuma's Revenge and Pitfall, Atari games
famous for dead ends.

**How it fits MaRLio.** Very well, and it is the natural cure for our NEAT
problems. The 2-1 piranha pipe and the 1-2 stall are exactly "dead ends where
the score says go right but you must wait or detour". Our NEAT practice starts
are a small version of the same idea. The archive can be drawn as a map of the
level, which would look great interactively. Phase 2 can produce a NEAT
network, a small CNN on the grid, or a PPO policy.

**Strengths.**
- Best-known method for deceptive, dead-end levels.
- Phase 1 is cheap and simple (no network at all).
- Phase 2 gives a real learned player, so it counts as "learning".

**Weaknesses.**
- Phase 1 relies on savestates (fine for us, but say so).
- Choosing the cell summary is the main design decision. Too coarse misses
  details, too fine floods the archive.
- Phase 2 (robustify) is where most of the compute and engineering goes.

**Rough cost.** Phase 1: 10-100 million frames per level (**$0.15-1.50** on
Modal, under an hour on the Mac). Phase 2: 50-500 million frames
(**$1-8**). So **about $2-10 per level**, maybe $20 for the hardest.

**References.**
- Ecoffet et al., *Go-Explore: a New Approach for Hard-Exploration Problems* ([arXiv 1901.10995](https://arxiv.org/abs/1901.10995), 2019).
- Ecoffet et al., *First return, then explore* (Nature, 2021; [arXiv 2004.12919](https://arxiv.org/abs/2004.12919)).
- Salimans & Chen, *Learning Montezuma's Revenge from a Single Demonstration* ([arXiv 1812.03381](https://arxiv.org/abs/1812.03381), 2018). The "start from the end and move backwards" trick used in phase 2.

---

## 4. Value-based RL: learn how good each move is

These methods learn a **Q-function**: for the current screen, a number for each
button combination saying "how much reward will I get if I press this now and
play well afterwards". The agent presses the button with the highest number,
sometimes picking a random one to explore. They learn from a **replay buffer**
of past moves, so every frame is reused many times.

### 4.1 DQN and Double DQN

**What it is.** DQN (DeepMind, 2015) learned 49 Atari games from pixels with
one convolutional network. Double DQN fixes DQN's habit of overestimating
values by using one network to pick the move and a second, slowly updated copy
to score it. The well-known **PyTorch Mario tutorial** trains a Double DQN on
1-1 from pixels.

**How it fits MaRLio.** Straightforward with gym-super-mario-bros (it is what
the tutorial uses): pixels resized to 84x84, 4 stacked frames, a small action
set (`SIMPLE_MOVEMENT` or our 8-action menu), frame skip 4. We could also feed
it the 13x13 grid, which would make it much faster and CPU-friendly.

**Strengths.**
- The textbook RL baseline; everyone knows it, so results are easy to compare.
- Reuses experience (sample-efficient compared with PPO and evolution).
- Very little code; many working Mario examples.

**Weaknesses.**
- Unstable and slow on pixels: the tutorial needs tens of thousands of
  episodes for a good 1-1 player.
- Struggles with sparse or deceptive spots (2-1 pipe, 1-2).
- Needs a GPU for the pixel version.

**Rough cost.** 5-20 million frames. Learning, not the emulator, is the
bottleneck: about **10-30 hours on a T4 ($6-18 on Modal)**, or 1-3 days on the
Mac (PyTorch MPS). With the 13x13 grid: a few hours on CPU, **about $1-3**.

**References.**
- Mnih et al., *Human-level control through deep reinforcement learning* ([Nature, 2015](https://www.nature.com/articles/nature14236)).
- van Hasselt, Guez & Silver, *Deep Reinforcement Learning with Double Q-learning* ([arXiv 1509.06461](https://arxiv.org/abs/1509.06461), 2015).
- PyTorch tutorial, *Train a Mario-playing RL Agent* ([pytorch.org](https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html)).

### 4.2 Rainbow

**What it is.** Rainbow (2017) combines six improvements to DQN:
- Double Q-learning (4.1).
- **Prioritized replay**: replay surprising moves more often.
- **Dueling heads**: separate "how good is this state" from "how much better is
  this button".
- **n-step returns**: learn from several real rewards before guessing.
- **Distributional values**: predict the spread of outcomes, not only the average.
- **Noisy nets**: explore by adding learned noise to the weights instead of
  pressing random buttons.

Together they are several times more sample-efficient than DQN.

**How it fits MaRLio.** Same setup as DQN. Libraries make it easy (CleanRL has
single-file versions of DQN and C51; Dopamine and others have full Rainbow).

**Strengths.** Much better results than DQN for the same frames; still one GPU.

**Weaknesses.** More moving parts and hyperparameters; still weak on deceptive
spots without an exploration bonus.

**Rough cost.** 2-10 million frames, **about $5-15** on a T4.

**References.**
- Hessel et al., *Rainbow: Combining Improvements in Deep Reinforcement Learning* ([arXiv 1710.02298](https://arxiv.org/abs/1710.02298), 2017).
- Schaul et al., *Prioritized Experience Replay* ([arXiv 1511.05952](https://arxiv.org/abs/1511.05952)).
- Wang et al., *Dueling Network Architectures* ([arXiv 1511.06581](https://arxiv.org/abs/1511.06581)).

### 4.3 R2D2, Agent57 and BBF

**What they are.** The strongest members of this family.
- **R2D2** (2019) adds memory (an LSTM) and many parallel actors feeding one
  learner.
- **Agent57** (2020) builds on R2D2 with curiosity bonuses and a controller that
  chooses how much to explore. It was the first agent to beat the human
  baseline on all 57 Atari games, but it used **tens of billions of frames**.
- **BBF** ("Bigger, Better, Faster", 2023) goes the other way: human-level Atari
  with only **100,000 agent steps** (about 2 hours of play), using a large
  network and many updates per step.

**How they fit MaRLio.** Mostly as **reference points**, not things to run.
R2D2/Agent57 are far beyond our budget. BBF is interesting if we ever want to
show "learns Mario from as little play as a human gets", but it is
GPU-heavy per frame.

**Rough cost.** R2D2/Agent57: hundreds to thousands of GPU-hours,
**$1,000s**. Not recommended. BBF: few frames but heavy training,
roughly **$10-50** per level on one GPU.

**References.**
- Kapturowski et al., *Recurrent Experience Replay in Distributed RL* (R2D2, [ICLR 2019](https://openreview.net/forum?id=r1lyTjAqYX)).
- Badia et al., *Agent57: Outperforming the Atari Human Benchmark* ([arXiv 2003.13350](https://arxiv.org/abs/2003.13350), 2020).
- Schwarzer et al., *Bigger, Better, Faster: Human-level Atari with human-level efficiency* ([arXiv 2305.19452](https://arxiv.org/abs/2305.19452), 2023).

---

## 5. Policy gradient: learn which move to press directly

Instead of scoring every button, these methods learn a **policy**: the
probability of pressing each button. Moves that led to more reward than
expected become more likely. They learn from fresh experience only (no replay
buffer), so they need more frames, but they are stable and run very well with
many games in parallel.

### 5.1 PPO

**What it is.** Proximal Policy Optimization (OpenAI, 2017) is the most-used RL
algorithm today. It collects a batch of play from, say, 16 parallel games,
then updates the policy, but clips the update so the policy never changes too
much at once. That clipping is what makes it stable. It also learns a value
(like AlphaZero's value head) to judge whether a move was better than expected.

**How it fits MaRLio.** It is the **most common choice for Mario**. The
open-source project *Super-mario-bros-PPO-pytorch* reports PPO agents that clear
most World 1-8 levels (31 of 32 stages, one model per stage) from pixels.
Stable-Baselines3 and CleanRL both have tested PPO implementations. It works
with the 13x13 grid too (a tiny network on CPU) or pixels (GPU).

**Strengths.**
- Proven on almost every Mario level; the standard RL baseline.
- Simple to scale: more parallel games = faster.
- Robust: few hyperparameters, rarely diverges.
- The learned policy plays without search, so it is fast at test time.

**Weaknesses.**
- Needs more frames than DQN/Rainbow, because it throws experience away.
- Still pushed around by deceptive reward (2-1 pipe). Needs curiosity (5.2) or
  practice starts there.
- One model per level is typical; transfer between levels is not automatic.

**Rough cost.** 5-30 million frames per level. With 16 parallel games and a T4,
about 3-10 hours, **$3-10 per level** on Modal. With the grid on CPU (Mac or a
16-core Modal container), **$1-4 per level**.

**References.**
- Schulman et al., *Proximal Policy Optimization Algorithms* ([arXiv 1707.06347](https://arxiv.org/abs/1707.06347), 2017).
- uvipen, *Super-mario-bros-PPO-pytorch* ([GitHub](https://github.com/uvipen/Super-mario-bros-PPO-pytorch)).
- Stable-Baselines3 ([GitHub](https://github.com/DLR-RM/stable-baselines3)) and CleanRL ([GitHub](https://github.com/vwxyzjn/cleanrl)).

### 5.2 PPO + curiosity (ICM, RND)

**What it is.** Add an **intrinsic reward** for reaching situations the agent
can't predict yet, so it explores on purpose.
- **ICM** (Intrinsic Curiosity Module, Pathak et al., 2017) predicts the
  effect of its own action. The prediction error is the bonus. The paper's
  showcase was **Super Mario Bros**: with curiosity alone (no reward for
  progress) the agent learned to move through a large part of 1-1, and the
  skills carried over to later levels.
- **RND** (Random Network Distillation, Burda et al., 2018) is simpler: the
  bonus is how badly a network predicts the output of a fixed random network
  on this screen. New screens are badly predicted, so they are rewarded.

**How it fits MaRLio.** An add-on to PPO (or DQN). It targets exactly our
problem spots: 1-2 and the 2-1 pipe, where moving right stops paying off. It is
the RL cousin of the novelty search we added to NEAT.

**Strengths.** Classic Mario result; helps where reward is sparse or deceptive.

**Weaknesses.**
- The "noisy TV" problem: anything unpredictable (flickering, random
  animations) can attract the agent. Less of an issue with the grid input.
- One more network to train, roughly +30-50% compute.

**Rough cost.** PPO cost plus about 50%: **$5-15 per level**.

**References.**
- Pathak et al., *Curiosity-driven Exploration by Self-supervised Prediction* ([arXiv 1705.05363](https://arxiv.org/abs/1705.05363), 2017).
- Burda et al., *Large-Scale Study of Curiosity-Driven Learning* ([arXiv 1808.04355](https://arxiv.org/abs/1808.04355), 2018). Includes Mario.
- Burda et al., *Exploration by Random Network Distillation* ([arXiv 1810.12894](https://arxiv.org/abs/1810.12894), 2018).

---

## 6. Model-based RL: learn a model of the game, then plan in it

These methods learn their *own* simulator of the game, then plan or imagine
inside it. That is valuable when the real game is slow, costly or can't be
rewound. For us the real emulator is fast, exact and rewindable, so the learned
model is mostly unnecessary. They are interesting for **showing** what the
agent imagines.

### 6.1 MuZero

**What it is.** AlphaZero with a learned model: the network learns to predict
the next hidden state, the reward and the value, and MCTS searches inside
these predictions instead of in the real game. It matched AlphaZero in Go and
chess and set records on Atari.

**How it fits MaRLio.** Only needed if the emulator were too slow to search,
and it is not. The AlphaZero plan (section 2.8) chose AlphaZero over MuZero
for this reason.

**Rough cost.** GPU-days for Atari-level results: **$100+ per level.**

**Reference.** Schrittwieser et al., *Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model* ([arXiv 1911.08265](https://arxiv.org/abs/1911.08265), 2019).

### 6.2 EfficientZero

**What it is.** MuZero tuned for very few real frames (the Atari 100k
benchmark), with extra self-supervised losses and value corrections.

**How it fits MaRLio.** It saves real frames, but our frames are cheap. Complex
to set up. Not recommended unless we want a "sample efficiency" show.

**Rough cost.** About one GPU-day per level, **$20-30**.

**References.**
- Ye et al., *Mastering Atari Games with Limited Data* ([arXiv 2111.00210](https://arxiv.org/abs/2111.00210), 2021).
- LightZero library, which has MuZero, EfficientZero and Gumbel MuZero ([GitHub](https://github.com/opendilab/LightZero)).

### 6.3 DreamerV3

**What it is.** Learns a "world model" that compresses the screen into a small
state and predicts how it changes. The policy is then trained entirely on
**imagined** play inside that model. One set of settings works across many
domains (Atari, DeepMind Control, Minecraft diamonds).

**How it fits MaRLio.** Not needed to clear levels, but it would make a striking
interactive demo: show the real game next to the agent's "dream" of the next
few seconds.

**Strengths.** Robust defaults; good sample efficiency; visual dreams.

**Weaknesses.** Large codebase (JAX), needs a GPU, slow wall-clock per frame.

**Rough cost.** 1-10 million frames, about 1 day on one GPU,
**$15-30 per level**.

**Reference.** Hafner et al., *Mastering Diverse Domains through World Models* ([arXiv 2301.04104](https://arxiv.org/abs/2301.04104), 2023).

---

## 7. Imitation and offline learning: copy the winners

Once any method has cleared a level, we have **recordings of winning runs**
(the screen or grid at each step and the buttons pressed). We can train a
network to copy them. This turns many separate winners (a NEAT network for
1-1, A\* for 2-1, ...) into **one network that plays many levels**.

### 7.1 Behavior cloning (BC)

**What it is.** Plain supervised learning: input = the grid (or screen),
target = the button pressed in the winning run. A known weakness is that small
mistakes take the agent to places the demos never visited ("compounding
errors"). **DAgger** fixes it: let the network play, ask the expert (for us,
A\* or the search) what it would have pressed in those new places, and add
that to the training data.

**How it fits MaRLio.** Very well, because we can generate as many expert
demonstrations as we want with search, and A\* is a perfect DAgger expert. We
already save winning genomes and can replay them to get demos.

**Strengths.** Cheapest way to a learned player; fast to train; one model for
all levels.

**Weaknesses.** Never better than the demos; needs DAgger or noise for
robustness; does not "discover" anything by itself.

**Rough cost.** Minutes on a CPU, **under $1**, plus the cost of making the demos.

**References.**
- Pomerleau, *ALVINN: An Autonomous Land Vehicle in a Neural Network* (NeurIPS 1988). The first behavior cloning.
- Ross, Gordon & Bagnell, *A Reduction of Imitation Learning to No-Regret Online Learning* (DAgger, [arXiv 1011.0686](https://arxiv.org/abs/1011.0686), 2011).

### 7.2 Decision Transformer

**What it is.** Treats play as a sequence, like text: *(desired return,
state, action, desired return, state, action, ...)*. A transformer learns to
predict the next action. At test time, you ask for a high return and it plays
like the best runs it saw. It can learn from a mix of good and bad runs (for
example, every NEAT generation's best).

**How it fits MaRLio.** We have thousands of logged NEAT runs of all qualities.
A Decision Transformer could learn from all of them and be asked "play like a
winner". It would be a nice bridge between the evolution work and the deep RL
work.

**Strengths.** Uses all our past data; one model across levels; a fun
"ask for a score" demo.

**Weaknesses.** Usually no better than the best runs in the data; transformer
training wants a GPU for larger datasets.

**Rough cost.** A few GPU-hours, **$1-5**.

**Reference.** Chen et al., *Decision Transformer: Reinforcement Learning via Sequence Modeling* ([arXiv 2106.01345](https://arxiv.org/abs/2106.01345), 2021).

---

## 8. Evolution: our current family

These methods keep a **population** of networks, play each one, and breed new
networks from the best. No gradients, so no GPU, and they parallelise across
CPU cores trivially. The emulator is about 95% of the time.

| Method | What it adds | Status in MaRLio | Rough cost per level |
|---|---|---|---|
| **NEAT** | Evolves the network's *shape* as well as its weights, starting tiny. Species protect new ideas. MarI/O's method. | Built. Cleared 1-1 (gen 67), 1-3 (gen 335, with auto-release-A), 1-4 (warm start, 4 gens). Stuck on 1-2 and 2-1. | **$0-1** (measured: 1-2 hour on Modal = $0.89) |
| **Novelty search / NSLC / Dominated Novelty Search** | Rewards *behaving differently*, not only going further, so dead ends do not trap the population. | Built (PR #8): `--selection nslc\|dns`. On 1-4, cleared at gen 35 (NSLC) and 23 (DNS) from scratch. | Same as NEAT |
| **MAP-Elites** | Keeps the best network for every *kind* of run (by how far, how high, how it died), so stepping stones are permanent. | Planned, then dropped for now (2026-11-17). | Same as NEAT, a bit more |
| **CMA-ES** | Evolves the weights of a *fixed* network by learning which directions in weight space work. Very strong for small networks (up to a few thousand weights). | Not built. Good fit for a fixed small network on the 13x13 grid. | **$1-5** |
| **Deep neuroevolution (Deep GA, OpenAI-ES)** | A very simple genetic algorithm (mutation only) or evolution strategies, evolving a big CNN with millions of weights on pixels. Competitive with DQN/A3C on Atari. | Not built. | **$10-100**: needs many CPU cores; the original runs used hundreds to thousands |

**Strengths of the family.** Simple, no GPU, deterministic and reproducible,
networks small enough to draw (the overlay). Novelty search directly targets
deception.

**Weaknesses.** Sample-hungry: every network plays a full episode for one
number. Weak at precise timing. Gets stuck when the fitness lies (2-1).

**References.**
- Stanley & Miikkulainen, *Evolving Neural Networks through Augmenting Topologies* ([Evolutionary Computation, 2002](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)).
- SethBling, *MarI/O* ([source gist](https://gist.github.com/SethBling/598639f8d5e8afb5453a0b9519be51ff)).
- Lehman & Stanley, *Abandoning Objectives: Evolution through the Search for Novelty Alone* (Evolutionary Computation, 2011).
- Bahlous-Boldi et al., *Dominated Novelty Search* ([arXiv 2502.00593](https://arxiv.org/abs/2502.00593), 2025).
- Mouret & Clune, *Illuminating search spaces by mapping elites* ([arXiv 1504.04909](https://arxiv.org/abs/1504.04909), 2015). Library: [pyribs](https://pyribs.org).
- Hansen, *The CMA Evolution Strategy: A Tutorial* ([arXiv 1604.00772](https://arxiv.org/abs/1604.00772)).
- Such et al., *Deep Neuroevolution: Genetic Algorithms Are a Competitive Alternative for Training Deep Neural Networks for RL* ([arXiv 1712.06567](https://arxiv.org/abs/1712.06567), 2017).
- Salimans et al., *Evolution Strategies as a Scalable Alternative to RL* ([arXiv 1703.03864](https://arxiv.org/abs/1703.03864), 2017).
- More detail: `marlio-research/neat-successors.md` in the project files.

---

## 9. AlphaZero (Gumbel): the main experiment

**What it is.** One network with two heads: a **policy** (which move looks
good) and a **value** (how well things will go from here). To choose each move
it runs an MCTS in a copy of the game, guided by the network. The search's
choices become the training target for the policy, and the real outcome
becomes the target for the value. Better network, better search, better
targets, better network. **Gumbel AlphaZero** (2022) makes the search work with
only 8-32 simulations per move, instead of the original 800, which keeps it
cheap.

**How it fits MaRLio.** It combines the best of two families: it plans with the
exact emulator like A\*/MCTS, and it learns a network like PPO. It also answers
the question the project cares about: "did it *learn* to play, or only
search?", because we can test the network alone with no search.

The plan (`marlio-research/alphazero-plan.md`):
- Savestate environment on the stable-retro core (unlimited snapshots), checked
  frame by frame against nes-py.
- 8-action menu, frame skip 4, reward = change in the `marlio/fitness.py` score.
- 13x13 grid stacked over the last 4 decisions, plus a few RAM values.
- Small CNN (~150k parameters), PyTorch on CPU; our own short MCTS in
  `marlio/agents/alphazero/`.
- Success on 1-1: clears it with search 9 of 10 times (random start delay), and
  the **network alone** clears it 5 of 10 times.

**Strengths.**
- Uses the perfect model we already have; no GPU needed for the grid version.
- Learns a player that can run without search.
- The search can be shown live (policy bars, chosen move), which suits the
  interactive goal.

**Weaknesses.**
- The most code to write of anything here (savestate env + MCTS + learner).
- Long levels and sparse reward: the tree only looks half a second ahead, so
  the value network must learn the rest.
- If search alone already clears levels, the learning part must be shown
  separately (the "network alone" test).

**Rough cost (from the plan).** 1-1: about 240 million frames,
**$1.20 (stable-retro) to $4.50 (nes-py)** on a 16-core Modal container, or
1-4.5 hours on the Mac for free. All 8 target levels: **about $12-47** on Modal,
or 12-48 hours on the Mac.

**References.**
- Silver et al., *Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm* (AlphaZero, [arXiv 1712.01815](https://arxiv.org/abs/1712.01815), 2017).
- Danihelka et al., *Policy improvement by planning with Gumbel* (Gumbel AlphaZero/MuZero, [ICLR 2022](https://openreview.net/forum?id=bERaNdoegnO)).
- DeepMind `mctx` library ([GitHub](https://github.com/google-deepmind/mctx)), for reference; we plan our own small MCTS because `mctx` expects the model in JAX.

---

## 10. Recommended comparison

Three methods, one from each of the strongest families, all using the **same
savestate environment, action menu, input (13x13 grid) and reward**, on the
same levels:

| Role | Method | Why this one | Cost for 1-1 | Cost for the 8 target levels |
|---|---|---|---|---|
| **Standard RL baseline** | **PPO** (add RND curiosity on 1-2 and 2-1) | The most-used method for Mario; proven on almost all levels; a fair "what normal RL gets" number | $1-4 (grid, CPU) | $10-40 |
| **Search baseline** | **A\*** first, then **Go-Explore** | A\* gives the ceiling cheaply ("is the level solvable with this menu?"). Go-Explore adds learning and handles dead ends. | A\*: cents. Go-Explore: $2-10 | A\*: < $1. Go-Explore: $20-80 |
| **Main experiment** | **Gumbel AlphaZero** | Plans with the exact emulator *and* learns a network; directly answers "learned or searched?" | $1.20-4.50 | $12-47 |
| *(Already have)* | NEAT + novelty search | The MarI/O reference point | $0-1 | $5-15 |

**What to measure for each method, on each level:**
1. Did it clear the level? In how many of 10 runs with a random 0-30 frame
   start delay?
2. Emulated frames used, wall time and dollars until the first clear.
3. Does the *learned* part play without search or savestates? (For A\*, no; for
   Go-Explore after phase 2, PPO, AlphaZero and NEAT, yes.)
4. A video of each winner, with sound, 10 seconds after the flag, side by side.

**Order of work.**
1. Savestate environment (shared by A\*, MCTS, Go-Explore and AlphaZero). This
   is step 1 of the AlphaZero plan anyway.
2. A\* on 1-1, 1-2 and 2-1: a cheap ceiling, and an expert for imitation later.
3. AlphaZero experiment on 1-1, including the plain-MCTS baseline.
4. PPO on the same env and levels (Stable-Baselines3 or CleanRL, grid input).
5. Go-Explore on the levels where PPO and AlphaZero stall.
6. Optional: distil all winners into one network with behavior cloning / DAgger,
   for a single "plays every level" demo.

**What I would not do now.** R2D2/Agent57 (far too expensive), MuZero and
EfficientZero (we already have an exact model), and pixel inputs (10x the cost;
keep as a later "same method, pixels vs grid" experiment).

---

## 11. Libraries and environments, for reference

- gym-super-mario-bros ([GitHub](https://github.com/Kautenja/gym-super-mario-bros)) on nes-py: our current environment.
- stable-retro ([GitHub](https://github.com/Farama-Foundation/stable-retro)): our `sound` extra; candidate emulator for search.
- Stable-Baselines3 ([GitHub](https://github.com/DLR-RM/stable-baselines3)): PPO, DQN.
- CleanRL ([GitHub](https://github.com/vwxyzjn/cleanrl)): one-file PPO, DQN, C51, RND; easiest to read and explain.
- LightZero ([GitHub](https://github.com/opendilab/LightZero)): MuZero family, Gumbel MuZero.
- pyribs ([pyribs.org](https://pyribs.org)): MAP-Elites and quality-diversity.
