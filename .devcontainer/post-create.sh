#!/usr/bin/env bash
set -euo pipefail

# --- Python environment ------------------------------------------------------
# uv creates .venv from pyproject.toml + uv.lock and installs the project
# (editable) with the dev group.
if ! command -v uv >/dev/null; then
    python -m pip install --user --quiet uv
    export PATH="$HOME/.local/bin:$PATH"
fi
uv sync
uv run python -c "import marlio.env, gym_super_mario_bros; print('marlio environment ready')"

# --- OpenGL libraries for the game window (pyglet) ---------------------------
# The base image ships a Yarn apt source we don't need; when its signing key
# is out of date, `apt-get update` fails, so drop it first. A failure here
# must not undo the Python setup above, so it only warns.
sudo rm -f /etc/apt/sources.list.d/yarn.list
if ! { sudo apt-get update && sudo apt-get install -y --no-install-recommends libgl1 libglu1-mesa; }; then
    echo "WARNING: could not install libgl1/libglu1-mesa; game windows may not open." >&2
fi
