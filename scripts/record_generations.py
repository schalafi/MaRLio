"""Record one clip per generation of a NEAT training run, to show how it learned.

    python scripts/record_generations.py runs/neat footage/1-1-seed0 --label neat-1-1-seed0

``train_neat.py`` saves the best genome of every generation
(``gen_NNN_best.json``) and a ``log.csv``. This script plays each of those
genomes once and saves, in the output folder:

* ``overlay/LABEL-genNNN.mp4``  the run with the network drawn on top
* ``clean/LABEL-genNNN.mp4``    the same run, game only (for your own graphics)
* ``genomes/gen_NNN_best.json`` the genome, re-scored with the current fitness
* ``index.csv``                 one row per generation: how far Mario got, how
  long the run was, whether it cleared the level, whether it was a new best
  ("milestone") and whether it is the same genome as an earlier generation
* ``LABEL-progress-montage.mp4`` the milestone clips back to back, each
  labelled "Generation N  best x X"
* ``log.csv``                   a copy of the training log

Many generations keep the previous generation's best genome unchanged
(``same_genome_as_gen`` in the index), so their clips are identical.

Training is deterministic for a given ``--seed``, so a run whose genomes were
lost can be recreated exactly with ``train_neat.py --seed N`` and recorded
with this script. Clips have sound unless ``--no-sound`` is given (needs
``uv sync --extra sound``); add ``--overlay-only`` to skip the clean clips.
"""

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

from marlio.agents.neat import Genome, run_genome
from marlio.env import DEFAULT_LEVEL, make_marios_env

PLAY = Path(__file__).with_name("play_neat.py")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def play(genome_path: Path, video: Path, level: str, sound: bool, overlay: bool) -> None:
    cmd = [sys.executable, str(PLAY), str(genome_path), "--level", level, "--no-render",
           "--outro", "1", "--record", str(video)]
    if sound:
        cmd.append("--sound")
    if not overlay:
        cmd.append("--no-overlay")
    subprocess.run(cmd, check=True, capture_output=True)


def label_clip(src: Path, dst: Path, text: str) -> None:
    font = f"fontfile={FONT}:" if Path(FONT).exists() else ""
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-i", str(src),
        "-vf", f"drawtext={font}text='{text}':x=(w-tw)/2:y=h-60:fontsize=34:fontcolor=white"
               ":box=1:boxcolor=black@0.6:boxborderw=12",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "44100", "-ac", "2", str(dst),
    ], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", help="a train_neat.py --out folder (or a genomes/ folder next to a log.csv)")
    parser.add_argument("out", help="folder for the clips")
    parser.add_argument("--label", default="neat", help="file name prefix for the clips")
    parser.add_argument("--level", default=DEFAULT_LEVEL)
    parser.add_argument("--no-sound", action="store_true")
    parser.add_argument("--overlay-only", action="store_true", help="skip the clips without the network")
    args = parser.parse_args()

    run, out = Path(args.run), Path(args.out)
    for sub in ("overlay", "genomes") + (() if args.overlay_only else ("clean",)):
        (out / sub).mkdir(parents=True, exist_ok=True)
    log_path = run / "log.csv" if (run / "log.csv").exists() else run.parent / "log.csv"
    log = {int(r["generation"]): r for r in csv.DictReader(open(log_path))}
    shutil.copy(log_path, out / "log.csv")

    env = make_marios_env(args.level)
    rows, seen, milestones, best_x = [], {}, [], -1
    for path in sorted(run.glob("gen_*_best.json")):
        gen = int(path.stem.split("_")[1])
        data = json.loads(path.read_text())
        key = json.dumps({k: v for k, v in data.items() if k != "fitness"}, sort_keys=True)
        result = run_genome(env, Genome.from_dict(data))
        # Re-score with the current fitness so the overlay's "Fitness a / b" compares like with like.
        data["fitness"] = round(result.fitness, 1)
        genome_path = out / "genomes" / path.name
        genome_path.write_text(json.dumps(data))

        x = int(log[gen]["best_x"]) if gen in log else result.rightmost
        milestone = x > best_x
        best_x = max(best_x, x)
        name = f"{args.label}-gen{gen:03d}.mp4"
        play(genome_path, out / "overlay" / name, args.level, not args.no_sound, overlay=True)
        if not args.overlay_only:
            play(genome_path, out / "clean" / name, args.level, not args.no_sound, overlay=False)
        if milestone:
            milestones.append((gen, x, out / "overlay" / name))
        rows.append({
            "generation": gen, "milestone": "yes" if milestone else "no", "best_x": x,
            "frames": result.frames, "cleared": "yes" if result.cleared else "no",
            "fitness_new": data["fitness"], "overlay_clip": f"overlay/{name}",
            "clean_clip": "" if args.overlay_only else f"clean/{name}",
            "same_genome_as_gen": seen.get(key, ""),
        })
        seen.setdefault(key, gen)
        print(f"gen {gen:3d}: x={result.rightmost} in {result.frames} frames"
              + (" CLEARED" if result.cleared else "") + (" (milestone)" if milestone else ""))
    env.close()

    with open(out / "index.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    parts = []
    for gen, x, clip in milestones:
        part = out / f".montage-{gen:03d}.mp4"
        label_clip(clip, part, f"Generation {gen}   best x {x}")
        parts.append(part)
    concat = out / ".montage.txt"
    concat.write_text("".join(f"file '{p.resolve()}'\n" for p in parts))
    montage = out / f"{args.label}-progress-montage.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0",
                    "-i", str(concat), "-c", "copy", str(montage)], check=True)
    for p in parts + [concat]:
        p.unlink()
    print(f"{len(rows)} generations, {len(seen)} distinct genomes, {len(milestones)} milestones -> {montage}")


if __name__ == "__main__":
    main()
