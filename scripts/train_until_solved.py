"""Train NEAT on a level, starting over from another population when it gets stuck.

    python scripts/train_until_solved.py --level SuperMarioBros-1-3-v0 --out runs/neat-1-3 \\
        --start trained/neat-1-1-generations/seed0/population-gen067.json \\
        --start trained/neat-1-1-generations/seed1/population-gen086.json \\
        --start fresh:1 --start fresh:2

Each ``--start`` is one attempt, tried in order:

* a ``population.json`` path: warm start from that population (for example
  the one that cleared 1-1), re-scored on this level
* ``fresh:SEED``: a brand-new random population with that seed

Every attempt runs ``train_neat.py`` in its own folder (``OUT/attempt-1``,
``OUT/attempt-2``, ...), with everything it normally saves (``log.csv``,
``gen_NNN_best.json``, ``population.json``). An attempt is **stuck** when its
best x hasn't improved for ``--stall`` generations in a row (default 50).
Then it is stopped and the next attempt starts. Nothing is thrown away: a
stuck attempt's folder stays as it is and can be resumed later with
``train_neat.py --resume OUT/attempt-N/population.json``.

The script stops when an attempt clears the level, or when every ``--start``
has been tried. ``OUT/attempts.csv`` says how each attempt went, and
``OUT/best.json`` is the best genome over all attempts.
"""

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

TRAIN = Path(__file__).with_name("train_neat.py")


def read_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [row for row in csv.DictReader(f) if row.get("best_x")]


def stalled_for(rows: list[dict]) -> int:
    """Generations since the best x last went up."""
    best, since = -1, 0
    for row in rows:
        x = int(row["best_x"])
        if x > best:
            best, since = x, 0
        else:
            since += 1
    return since


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--level", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--start", action="append", required=True,
                        help="population.json or fresh:SEED; repeat for more attempts")
    parser.add_argument("--stall", type=int, default=50,
                        help="generations without a better x before trying the next start (default 50)")
    parser.add_argument("--generations", type=int, default=600, help="most generations per attempt")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--auto-release-a", type=int, default=0, metavar="FRAMES",
                        help="passed to train_neat.py (see there)")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "attempts.csv"
    new_summary = not summary_path.exists()
    summary = open(summary_path, "a", newline="")
    writer = csv.writer(summary)
    if new_summary:
        writer.writerow(["attempt", "start", "first_generation", "last_generation", "best_x", "result"])
    best_overall = -1

    for n, start in enumerate(args.start, 1):
        folder = out / f"attempt-{n}"
        if (folder / "log.csv").exists():
            print(f"attempt {n}: {folder} already exists, skipping it")
            continue
        cmd = [sys.executable, "-u", str(TRAIN), "--level", args.level, "--out", str(folder)]
        if start.startswith("fresh:"):
            cmd += ["--seed", start.split(":", 1)[1], "--generations", str(args.generations)]
        else:
            first = json.loads(Path(start).read_text()).get("generation", 0)
            cmd += ["--resume", start, "--generations", str(first + args.generations)]
        if args.workers:
            cmd += ["--workers", str(args.workers)]
        if args.auto_release_a:
            cmd += ["--auto-release-a", str(args.auto_release_a)]
        folder.mkdir(parents=True, exist_ok=True)
        print(f"attempt {n}: starting from {start}", flush=True)
        with open(folder / "train.log", "w") as console:
            proc = subprocess.Popen(cmd, stdout=console, stderr=subprocess.STDOUT)
            result = "finished"
            while proc.poll() is None:
                time.sleep(10)
                rows = read_log(folder / "log.csv")
                if any(r["cleared"] == "True" for r in rows):
                    continue  # train_neat.py stops by itself once the level is cleared
                if stalled_for(rows) >= args.stall:
                    # Let it finish saving population.json for this generation first.
                    log_time = (folder / "log.csv").stat().st_mtime
                    pop = folder / "population.json"
                    for _ in range(60):
                        if pop.exists() and pop.stat().st_mtime >= log_time:
                            break
                        time.sleep(1)
                    proc.terminate()
                    proc.wait()
                    result = f"stuck {args.stall} generations"
            rows = read_log(folder / "log.csv")
        if any(r["cleared"] == "True" for r in rows):
            result = "CLEARED"
        best_x = max((int(r["best_x"]) for r in rows), default=0)
        writer.writerow([n, start, rows[0]["generation"] if rows else "", rows[-1]["generation"] if rows else "",
                         best_x, result])
        summary.flush()
        print(f"attempt {n}: {result}, best x {best_x}", flush=True)
        if best_x > best_overall and (folder / "best.json").exists():
            best_overall = best_x
            shutil.copy(folder / "best.json", out / "best.json")
        if result == "CLEARED":
            print(f"Level cleared! Watch it: python scripts/play_neat.py {folder / 'best.json'}")
            break
    summary.close()


if __name__ == "__main__":
    main()
