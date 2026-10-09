FROM mcr.microsoft.com/devcontainers/python:1-3.13-bookworm

# The base image ships a Yarn apt source whose signing key expired on
# 2026-01-23, so every `apt-get update` in the container fails, including the
# one inside the desktop-lite feature. We don't use Yarn: drop the source.
# Then add the OpenGL libraries the game window (pyglet) needs.
RUN rm -f /etc/apt/sources.list.d/yarn.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglu1-mesa \
    && rm -rf /var/lib/apt/lists/*
