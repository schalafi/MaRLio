#!/usr/bin/env bash
set -euo pipefail

# OpenGL libraries the game window (pyglet) needs.
sudo apt-get update
sudo apt-get install -y --no-install-recommends libgl1 libglu1-mesa

pip install --upgrade pip
pip install -e ".[dev]"
