"""Train NEAT on Modal: one cloud container per run, results in a Modal Volume.

Run these from the repo, after ``uv sync --extra cloud`` and logging in to
Modal once with ``uv run modal token new``:

    # Start a run (it keeps going on Modal after this command returns)
    uv run modal run --detach marlio/cloud/modal_app.py::launch \\
        --run neat-1-4-test --level SuperMarioBros-1-4-v0 \\
        --start runs/neat-1-3-release/population.json --minutes 60

    # How every run is doing (reads the Volume, starts no container)
    uv run modal run marlio/cloud/modal_app.py::status

    # Copy a run's files to this computer
    uv run modal volume get marlio-runs neat-1-4-test runs/modal/

    # Record a run's best genome as an .mp4 with the network overlay
    uv run modal run marlio/cloud/modal_app.py::render --run neat-1-4-test

Each run plays its genomes in ``WORKERS`` game processes inside one
container, exactly like ``scripts/train_neat.py`` does on the Mac (the
emulator is the slow part, and a genome plays for well under a second, so
sending genomes to other containers one by one would cost more in network
round trips than it saves). What Modal adds is room: 16 cores per run, and
as many runs at the same time as we like.

Everything a run writes (``population.json``, ``best.json``, ``log.csv``,
``gen_NNN_best.json``, the console log) goes to the Volume ``marlio-runs``
under ``/<run>/``, committed every 30 s. If Modal stops the container
(preemption), the retry resumes from the last ``population.json``, and
``--max-minutes`` counts the time already used, so a run never trains for
longer than asked. ``modal-usage.json`` records the container's wall time
and an estimate of what it cost.

See ``docs/06-training-on-modal.md`` for the why and the prices.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path, PurePosixPath

# Workspaces made before 2025 default to an old image builder that insists on
# modal being a core dependency; the 2025.06 builder doesn't. Must be set
# before modal is imported.
os.environ.setdefault("MODAL_IMAGE_BUILDER_VERSION", "2025.06")

import modal  # noqa: E402

# The repo root, on your computer. In the container this file is
# /root/modal_app.py and the image is already built, so it isn't needed.
REPO = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/root")
RUNS = PurePosixPath("/runs")
CPU = 16  # physical cores; Modal gives 2 hardware threads per core
WORKERS = 2 * CPU  # one emulator per hardware thread
MEMORY_MB = 16 * 1024  # a game process needs about 200 MB, a population up to 1 GB

# Modal's prices (https://modal.com/pricing, checked 2026-11-17), for the
# cost estimate in modal-usage.json. The bill on modal.com is the real number.
USD_PER_CORE_SECOND = 0.0000131
USD_PER_GIB_SECOND = 0.00000222

app = modal.App("marlio")
volume = modal.Volume.from_name("marlio-runs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.13")
    # The same locked dependencies as everywhere else (uv.lock).
    .uv_sync(uv_project_dir=str(REPO), frozen=True)
    .add_local_python_source("marlio")
    .add_local_dir(REPO / "scripts", "/root/scripts")
)


def cost_per_hour(cpu: float = CPU, memory_mb: int = MEMORY_MB) -> float:
    return 3600 * (cpu * USD_PER_CORE_SECOND + memory_mb / 1024 * USD_PER_GIB_SECOND)


def _commit_every(seconds: float, stop: threading.Event) -> None:
    while not stop.wait(seconds):
        volume.commit()


@app.function(image=image, cpu=CPU, memory=MEMORY_MB, volumes={str(RUNS): volume},
              timeout=24 * 3600, retries=modal.Retries(max_retries=3, initial_delay=10.0))
def train(run: str, args: list[str], hard_minutes: float = 0) -> dict:
    """Run ``scripts/train_neat.py`` with ``args`` into ``/runs/<run>``, resuming if it was started before.

    ``hard_minutes`` (0 = none) is a wall-clock cap on the whole run,
    counted from its first start, retries included: at the deadline the
    trainer is stopped, keeping the last saved generation.
    """
    out = RUNS / run
    Path(out).mkdir(parents=True, exist_ok=True)
    deadline = None
    if hard_minutes:
        first = Path(out / "started_at")
        if not first.exists():
            first.write_text(str(time.time()))
        deadline = float(first.read_text()) + hard_minutes * 60
        if time.time() >= deadline:
            print(f"Run {run} is past its hard limit of {hard_minutes:g} minutes; not restarting.")
            return {"skipped": "past hard limit"}
    args = list(args)
    if Path(out / "population.json").exists():
        # A retry after preemption, or a second launch: carry on from the
        # last generation instead of the starting population.
        if "--resume" in args:
            i = args.index("--resume")
            del args[i:i + 2]
        args += ["--resume", str(out / "population.json")]
    cmd = [sys.executable, "-u", "/root/scripts/train_neat.py",
           "--out", str(out), "--workers", str(WORKERS), *args]
    print("$", shlex.join(cmd), flush=True)

    started = time.time()
    stop = threading.Event()
    committer = threading.Thread(target=_commit_every, args=(30, stop), daemon=True)
    committer.start()
    with open(out / "console.log", "a") as console:
        console.write(f"$ {shlex.join(cmd)}\n")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if deadline is not None:
            def stop_at_deadline():
                while proc.poll() is None and time.time() < deadline:
                    time.sleep(5)
                if proc.poll() is None:
                    print(f"Hard limit of {hard_minutes:g} minutes reached: stopping.", flush=True)
                    console.write(f"Hard limit of {hard_minutes:g} minutes reached: stopping.\n")
                    proc.terminate()
            threading.Thread(target=stop_at_deadline, daemon=True).start()
        for line in proc.stdout:
            print(line, end="", flush=True)
            console.write(line)
            console.flush()
        code = proc.wait()
    stop.set()

    usage_path = Path(out / "modal-usage.json")
    sessions = json.loads(usage_path.read_text()) if usage_path.exists() else []
    seconds = time.time() - started
    sessions.append({
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "seconds": round(seconds, 1),
        "cpu": CPU,
        "memory_gib": MEMORY_MB / 1024,
        "estimated_usd": round(seconds / 3600 * cost_per_hour(), 3),
        "exit_code": code,
        "hit_hard_limit": bool(deadline is not None and time.time() >= deadline),
    })
    usage_path.write_text(json.dumps(sessions, indent=1))
    volume.commit()
    if code != 0 and not sessions[-1]["hit_hard_limit"]:
        raise RuntimeError(f"train_neat.py exited with code {code}")
    return sessions[-1]


@app.function(image=image, cpu=2, memory=2048, volumes={str(RUNS): volume}, timeout=1800)
def record(run: str, level: str, auto_release_a: int, genome: str = "best.json") -> str:
    """Play a run's genome once and save it as ``/runs/<run>/<genome>.mp4`` with the overlay."""
    volume.reload()
    out = RUNS / run
    video = out / (Path(genome).stem + ".mp4")
    cmd = [sys.executable, "/root/scripts/play_neat.py", str(out / genome), "--level", level,
           "--record", str(video), "--no-render", "--auto-release-a", str(auto_release_a)]
    subprocess.run(cmd, check=True)
    volume.commit()
    return str(video)


# --- Commands run from your computer ------------------------------------------

@app.local_entrypoint()
def launch(run: str, level: str = "SuperMarioBros-1-4-v0", start: str = "", minutes: float = 60,
           hard_minutes: float = 0, selection: str = "fitness", auto_release_a: int = 30,
           seed: int = 0, extra: str = ""):
    """Start one training run on Modal.

    ``--start`` is a ``population.json`` on this computer to warm start from
    (it is uploaded to the Volume as ``/seeds/<run>.json``); leave it out for
    a fresh population with ``--seed``. ``--minutes`` caps the minutes of
    training; ``--hard-minutes`` caps the wall-clock time of the whole run,
    restarts included (0 = no cap). ``--selection nslc`` or ``dns`` turns on
    novelty search. ``--extra`` passes more flags to
    train_neat.py, e.g. ``--extra "--population 300"``.
    """
    # The time cap ends the run, not a generation count.
    args = ["--level", level, "--max-minutes", str(minutes), "--generations", "100000",
            "--selection", selection]
    if auto_release_a:
        args += ["--auto-release-a", str(auto_release_a)]
    if start:
        remote = f"/seeds/{run}.json"
        with volume.batch_upload(force=True) as batch:
            batch.put_file(start, remote)
        args += ["--resume", str(RUNS) + remote]
        print(f"Uploaded {start} to the Volume as {remote}")
    else:
        args += ["--seed", str(seed)]
    args += shlex.split(extra)
    fn = train.with_options(timeout=int(hard_minutes * 60) + 120) if hard_minutes else train
    call = fn.spawn(run, args, hard_minutes)
    print(f"Started run {run!r} on Modal ({CPU} cores, {WORKERS} games at a time): call {call.object_id}")
    longest = max(minutes, hard_minutes)
    print(f"About ${cost_per_hour():.2f} per hour, so about ${cost_per_hour() * minutes / 60:.2f} "
          f"for {minutes:g} minutes (at most ${cost_per_hour() * longest / 60:.2f}). Watch it with: uv run modal run marlio/cloud/modal_app.py::status")


@app.local_entrypoint()
def status():
    """Print the latest generation of every run on the Volume."""
    runs = [e.path for e in volume.listdir("/") if e.path != "seeds"]
    if not runs:
        print("No runs yet.")
    for run in sorted(runs):
        print(f"== {run}")
        try:
            log = b"".join(volume.read_file(f"{run}/log.csv")).decode().strip().splitlines()
        except Exception:  # not written yet
            print("   (no generations yet)")
            continue
        header = log[0].split(",")
        rows = [dict(zip(header, line.split(","))) for line in log[1:]]
        if rows:
            last = rows[-1]
            minutes = sum(float(r["seconds"]) for r in rows) / 60
            best_x = max(int(r["best_x"]) for r in rows)
            print(f"   generation {last['generation']}, {minutes:.1f} min of training, best x {best_x}, "
                  f"species {last['species']}, cleared {any(r['cleared'] == 'True' for r in rows)}")
        try:
            usage = json.loads(b"".join(volume.read_file(f"{run}/modal-usage.json")))
            print(f"   finished sessions: {sum(u['seconds'] for u in usage) / 60:.1f} min, "
                  f"about ${sum(u['estimated_usd'] for u in usage):.2f}")
        except Exception:  # written when the container finishes
            print("   still running (or stopped before it finished)")


@app.local_entrypoint()
def render(run: str, level: str = "SuperMarioBros-1-4-v0", auto_release_a: int = 30, genome: str = "best.json"):
    """Record a run's genome as an .mp4 on Modal; get it with ``modal volume get``."""
    print(record.remote(run, level, auto_release_a, genome))
