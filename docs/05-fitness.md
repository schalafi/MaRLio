# Step 5a: a fitness that doesn't punish waiting

Before training harder levels we changed how a run is scored. This is the
first half of the plan (option A of the research in the project files,
`marlio-fitness/fitness-options.md`); the second half adds a novelty score.

## The old fitness (MarI/O's)

```
fitness = rightmost x - frames / 2      (+1000 if flag_get or rightmost x > 3186)
```

Two problems showed up on World 2-1:

1. **Waiting scored worse than dying.** The piranha pipe at x~1950 needs
   Mario to wait for the plant to go down. Running in and dying at frame
   1000 scored `1950 - 500 = 1450`. Waiting 200 frames and getting 50 px
   further scored `2000 - 600 = 1400`, which is lower. Evolution was being
   pushed away from the behaviour the level needs.
2. **3186 is World 1-1's flag.** On other levels the +1000 bonus was given at
   the wrong place (or never).

## The new fitness

`fitness()` in `marlio/agents/neat/agent.py`, used by training and by the
live score in the overlay:

```
fitness = rightmost x
        - frames / 10          (TIME_PENALTY)
        - 100 if Mario died    (DEATH_PENALTY; dying = is_dying or is_dead, pits too)
        + 1000 if flag_get     (FLAG_BONUS)
```

| Run (from the example above) | Old | New |
|---|---|---|
| runs in, dies at x=1950 on frame 1000 | 1450 | 1750 |
| waits 200 frames, reaches x=2000 alive | 1400 | **1880** |

What each part does:

- **x is still the main driver.** One pixel of progress is still worth more
  than anything else per frame.
- **Time costs 1/10 per frame instead of 1/2.** A faster run still wins, but
  waiting a few seconds (60 frames per second) for an enemy is cheap.
- **Dying costs 100**, as much as 100 px of progress. A run that stops safely
  in front of a danger now beats one that walks into it a little further on.
- **The flag bonus comes only from the game's `flag_get`**, so it works on
  every level.

The run still ends the same way as in MarI/O: on death, on the flag, or after
`20 + frames/4` frames with no new rightmost point.

## What it changes for the saved genomes

Re-scored with the new function (same runs, nothing retrained):

| Genome | x reached | died | old | new |
|---|---|---|---|---|
| `trained/neat-1-1.json` (clears 1-1) | 3161 | no | 3467.5 | 4022.3 |
| 2-1 seed 2 best | 1950 | yes | 1489.0 | 1757.8 |
| 2-1 fresh (practice) best | 1734 | yes | 1346.5 | 1556.5 |

Both 2-1 champions get their score by running into the enemy and dying,
which is exactly what the death penalty now discourages.

Scores from the two functions aren't comparable, so `train_neat.py --resume`
now re-scores every genome of a saved population on its first generation.

## Trade-offs

- The weights (1/10, 100) are hand-picked. A larger death penalty makes the
  population more careful; too large and "stop before every enemy" can
  become a new dead end. That plateau is what the next step, a novelty
  score, is for.
- On easy levels like 1-1, a smaller time penalty may make the winning run a
  little slower than MarI/O's.
