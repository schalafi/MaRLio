# Step 4 (plan): training on every level, fast, on Modal

Goal: train an agent that clears each of the 32 Super Mario Bros. levels
(1-1 to 8-4), without waiting days, by running the training on
[Modal](https://modal.com) instead of a 4-core Codespace or sandbox.

This is a plan, not code yet. It says where the time goes today, what makes
it faster, how to spread the work on Modal, what it should cost, and whether
NEAT or PPO is the better tool for all 32 levels.

## 1. Where the time goes today

`scripts/train_neat.py` already plays genomes in parallel: `Evaluator`
(`marlio/agents/neat/agent.py`) keeps a `multiprocessing` pool with one
emulator per CPU core and sends it every genome that hasn't played yet.
World 1-1 was cleared in generation 67 after about 40 minutes on 4 cores.

A quick benchmark in the cloud sandbox (one core, no window):

| What | Speed |
| --- | --- |
| nes-py emulator, bare `_frame_advance` | ~1,070 frames/s |
| full env step through Gymnasium wrappers | ~1,040 frames/s |
| `make_marios_env` step (incl. 13x13 grid every frame) | ~985 frames/s |
| `input_grid()` alone | ~22,000 calls/s |

So **the NES emulator is ~95% of the cost**. The Python wrappers, the grid
and the network are small. That has two consequences:

* "Frame skip" doesn't help NEAT the way it helps pixel RL: every frame still
  has to be emulated. MarI/O already only asks the network every 5 frames.
* The only big levers are **more cores** and **fewer frames per run**.

One generation is about 300 runs × ~500 frames ≈ 150k frames, i.e. ~150
core-seconds. It parallelises almost perfectly because every genome is
independent.

## 2. Cheap speedups before spending money

These help locally and on Modal alike.

1. **No rendering during training** (already the default; `--watch` costs a
   lot, use it only to look).
2. **End stalled runs early** (already done: timeout on no progress, death,
   flag). For other levels, also stop when Mario falls below the screen
   (`y_pos` / `player_state`), which nes-py reports a little before
   `is_dying`.
3. **Compute the grid only on think frames** (every 5th frame). ~5% win, easy.
4. **Warm-start each level from the previous level's population**
   (`--resume` with a different `--level`). The 1-1 networks already know
   "run right and jump over things"; starting 1-2 from them instead of from
   empty networks should save many generations. This is likely the single
   biggest win for "all 32 levels".
5. **Use the right "end of level" signal.** `LEVEL_END_X = 3186` only fits
   1-1. gym-super-mario-bros' `flag_get` also covers castles (it is true when
   the stage or world is over), so other levels should rely on `flag_get`
   alone and drop the x bonus.

## 3. How to parallelise on Modal

Modal runs Python functions in containers you describe in code. The pieces
we need:

* an **Image** with our deps (`modal.Image.debian_slim().uv_sync()` or
  `.pip_install(...)` from `pyproject.toml`, plus `libgl1` only if we record
  video),
* a **Function** with `cpu=N` (N physical cores; Modal bills a physical core
  as 2 vCPUs),
* `Function.map(...)` to fan out many calls at once (results come back in
  order; up to 100 containers at once on the Starter plan, 5,000 on Team),
* a **Volume** to keep checkpoints between calls (`vol.commit()` after
  writing, `vol.reload()` before reading).

There are two levels of parallelism, and they combine.

### A. One level per container (recommended first)

Run today's `train_neat.py` loop unchanged inside one big container per
level, and fan the levels out:

```python
# modal_train.py (sketch)
import modal

app = modal.App("marlio")
image = modal.Image.debian_slim(python_version="3.13").pip_install_from_pyproject("pyproject.toml") \
    .add_local_python_source("marlio")
runs = modal.Volume.from_name("marlio-runs", create_if_missing=True)

@app.function(image=image, cpu=16, memory=8192, timeout=24 * 3600,
              volumes={"/runs": runs})
def train_level(level: str, resume_from: str | None = None, generations: int = 300):
    # same loop as scripts/train_neat.py, with Evaluator(level, workers=32)
    # (2 emulator processes per physical core, one per vCPU),
    # saving population.json to /runs/<level>/ and calling runs.commit()
    # every few generations so a crash or timeout resumes from there.
    ...

@app.local_entrypoint()
def main():
    levels = [f"SuperMarioBros-{w}-{s}-v0" for w in range(1, 9) for s in range(1, 5)]
    for level, cleared in zip(levels, train_level.map(levels)):
        print(level, "cleared" if cleared else "not yet")
```

Why this first:

* No change to the NEAT code: the `multiprocessing` pool just gets more
  cores (`Evaluator(workers=...)`).
* 32 levels run side by side, so the whole game takes as long as the hardest
  level, not the sum.
* Each container talks to Modal once, so there's no per-genome network
  overhead.

At 16 physical cores (32 emulators) a 1-1-sized run would take ~5 minutes
instead of 40.

### B. Fan out genomes across containers (if one level is still too slow)

For a stubborn level (say 8-4) keep the population in one small "driver"
function and send each generation's genomes to a worker function with
`.map`:

```python
@app.function(image=image, cpu=4)
def play_batch(level: str, genomes: list[dict]) -> list[dict]:
    env = make_marios_env(level)   # one env per container, reused across calls
    ...                             # run_genome for each genome dict

# in the driver, once per generation:
batches = [todo[i:i + 10] for i in range(0, len(todo), 10)]
results = play_batch.starmap((level, b) for b in batches)
```

Genomes are small JSON (`Genome.to_dict()`), so shipping them is cheap. Send
them in batches of ~10 so the per-call overhead (tens of milliseconds) stays
small compared with playing. A generation then takes about as long as its
slowest run plus a second or two, whatever the population size, which also
makes **bigger populations** (1,000+) affordable for hard levels.

Plan A is a few hours of work; plan B is a refactor of the evaluator behind
the same `Evaluator.run(genomes)` interface, so `train_neat.py` can pick
`--backend local|modal`.

### Checkpoints and results

* Volume layout: `/runs/neat/<level>/{population.json, best.json, log.csv}`.
* Commit every N generations; on restart, `train_level` resumes from
  `population.json` if it exists, so Modal's 24h timeout or a preemption
  only loses minutes.
* Pull results home with `modal volume get marlio-runs neat/ runs/` and
  copy each winner to `trained/neat-<w>-<s>.json`, then watch it with
  `play_neat.py` (and record it with `--record`).

### Alternatives considered

* **Ray** (local or on a cluster): same idea as plan B, but we'd have to run
  the cluster ourselves. Modal gives us the machines per second.
* **Bigger single VM** (e.g. a 64-core cloud box): simplest of all, plan A
  with `--workers 64`, but you pay while it idles and runs one level at a
  time unless we script it.
* **GitHub Actions matrix**: free minutes, 32 jobs, but 2-4 cores each and
  6-hour job limits. Fine for easy levels only.

## 4. Rough cost

Modal list prices (October 2026): **$0.0472 per physical core-hour**,
$0.008 per GiB-hour of memory, T4 GPU $0.59/h, L4 $0.80/h. The Starter
plan includes **$30/month** of free credit and up to 100 containers at once.

* 1-1 took ~40 min on 4 vCPUs ≈ 1.3 physical core-hours ≈ **$0.07**.
* Assume an average level needs ~10× that (longer levels, harder
  enemies, more generations): ~13 core-hours ≈ **$0.70** per level.
* Hard levels (mazes 4-4, 7-4, 8-4; water 2-2, 7-2; 8-x) might need 50-100×:
  **$3-7** each.
* **All 32 levels with NEAT: roughly $25-60**, i.e. about one month of free
  credit plus a little. Wall-clock: a few hours with plan A, because levels
  run in parallel.

These are estimates from one data point (1-1). The first real step should
measure 2-3 levels and re-estimate before launching all 32.

## 5. NEAT or PPO for all 32 levels?

**NEAT (MarI/O)** — what we have.

* Good: cheap (CPU only), easy to parallelise, small networks we can show in
  the overlay, faithful to MarI/O.
* Weak spots: fitness is "how far right", so **maze castles (4-4, 7-4, 8-4)**
  where you must take the right path and get looped back will stall; it
  memorises a level rather than learning to play, so each level gets its own
  network.
* MarI/O's 13x13 grid only sees tiles and enemy slots; fire bars, Bowser's
  fire and hammers may be invisible to it, so castles may need richer inputs.

**PPO with vectorised envs** — the plug-in the repo is designed for.

* Uses pixels (`make_env`, 84x84 grayscale, 4 stacked frames, frame skip 4).
  Here frame skip *does* help: the network runs every 4th frame and
  sees motion.
* Public results: the well-known PPO implementation on this same
  environment ([vietnh1009/uvipen Super-mario-bros-PPO-pytorch](https://www.sourcepulse.org/projects/1539793))
  reports clearing **31 of 32 levels**, all but 8-4, training one model per
  level with `--world --stage`.
* Needs a GPU for the network (T4/L4 is enough) plus 8-16 CPU emulators in a
  `gymnasium.vector.AsyncVectorEnv`. Expect a few to ~10 hours per hard level;
  roughly **$2-8 per level, $100-250 for all 32**.

**Recommendation**

1. Run **NEAT on all 32 levels with plan A** on Modal, warm-starting from
   the previous level. It's cheap, needs almost no new code, and keeps the
   MarI/O look.
2. Add **PPO** as the next algorithm (its own step/PR), and use it on the
   levels NEAT doesn't clear. That also gives the comparison we want:
   same levels, same Modal setup, two algorithms.
3. Treat 8-4 as a stretch goal for both.

## 6. Proposed steps (each its own branch/PR)

1. **4a. Level-agnostic NEAT**: drop the 1-1 x bonus, end-of-level via
   `flag_get`, compute the grid on think frames, `--resume` across levels,
   `trained/neat-<w>-<s>.json` naming.
2. **4b. Modal plan A**: `modal_train.py` with `train_level` + Volume
   checkpoints; run 1-2, 1-3, 1-4 and measure time and cost.
3. **4c. All 32 levels with NEAT**: launch once 4b's numbers look right;
   publish a table of cleared levels with generation, time and cost.
4. **4d. (if needed) Modal plan B**: genome fan-out for stubborn levels.
5. **5. PPO**: `marlio/agents/ppo/`, vectorised pixel envs, GPU on Modal,
   try the levels NEAT missed.

Before 4b you'll need a Modal account and `modal token new` in the Codespace
(or `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` as Codespace secrets).
