#!/usr/bin/env bash
set -euo pipefail

# uv creates .venv from pyproject.toml + uv.lock and installs the project
# (editable) with the dev group.
if ! command -v uv >/dev/null; then
    python -m pip install --user --quiet uv
    export PATH="$HOME/.local/bin:$PATH"
fi
uv sync
uv run python -c "import marlio.env, gym_super_mario_bros; print('marlio environment ready')"
