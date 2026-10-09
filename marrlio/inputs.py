"""MarI/O's input grid: a 13x13 picture of the world around Mario.

MarI/O does not look at the screen pixels (256x240x3 = 184,320 numbers).
Instead it reads the game's memory (RAM) and builds a tiny grid centred on
Mario where every cell is one 16x16-pixel block of the level:

     1  -> a solid tile (ground, brick, pipe, ? block, ...)
    -1  -> an enemy (goomba, koopa, ...)
     0  -> empty space

That is 13 * 13 = 169 numbers instead of 184,320, which is what makes it
practical for NEAT to evolve networks from scratch: every input is a node in
the network, so the input must be small and already meaningful.

This module is a line-by-line port of the Super Mario Bros. branch of
SethBling's MarI/O Lua script (``getPositions``, ``getTile``, ``getSprites``
and ``getInputs``):
https://gist.github.com/SethBling/598639f8d5e8afb5453a0b9519be51ff

All functions take ``ram``: the NES's 2 KB of work RAM as a numpy array of
2048 bytes. With gym-super-mario-bros you get it from ``env.unwrapped.ram``.
"""

from dataclasses import dataclass

import numpy as np

# How many blocks the grid reaches out from Mario in each direction.
# 6 blocks left + Mario's block + 6 blocks right = 13 columns (same for rows).
BOX_RADIUS = 6
GRID_SIZE = 2 * BOX_RADIUS + 1
# Every cell of the grid is one 16x16-pixel block of the level.
TILE_SIZE = 16

# Cell values.
EMPTY = 0
SOLID = 1
ENEMY = -1

# --- Super Mario Bros. RAM addresses used by MarI/O -------------------------
# Mario's horizontal position is split in two bytes: which 256-pixel "page"
# of the level he is on, and the pixel inside that page.
PLAYER_PAGE = 0x006D
PLAYER_X_IN_PAGE = 0x0086
PLAYER_Y_ON_SCREEN = 0x03B8
# The game keeps up to 5 enemies alive at once, one per "slot".
ENEMY_SLOTS = 5
ENEMY_ACTIVE = 0x000F  # 0x0F..0x13: non-zero when the slot holds an enemy
ENEMY_PAGE = 0x006E  # 0x6E..0x72
ENEMY_X_IN_PAGE = 0x0087  # 0x87..0x8B
ENEMY_Y_ON_SCREEN = 0x00CF  # 0xCF..0xD3
# The visible part of the level is stored as two pages of 13 rows x 16
# columns of block ids, starting here. A non-zero id means "something solid".
TILE_BUFFER = 0x0500
TILE_ROWS = 13
TILE_COLUMNS = 16


@dataclass(frozen=True)
class Sprite:
    """An enemy's position in level pixel coordinates."""

    x: int
    y: int


def mario_position(ram: np.ndarray) -> tuple[int, int]:
    """Return Mario's (x, y) in level pixels, as MarI/O's ``getPositions``."""
    x = int(ram[PLAYER_PAGE]) * 0x100 + int(ram[PLAYER_X_IN_PAGE])
    y = int(ram[PLAYER_Y_ON_SCREEN]) + 16
    return x, y


def is_solid(ram: np.ndarray, x: int, y: int) -> bool:
    """Return whether the block at level pixel (x, y) is solid (``getTile``).

    MarI/O calls ``getTile(dx, dy)`` with an offset from Mario; here the
    caller passes the absolute point ``(marioX + dx, marioY + dy)`` instead.
    The +8 / -16 / -32 shifts are MarI/O's, lining the grid up with blocks.
    """
    x = x + 8
    y = y - 16
    page = (x // 256) % 2
    column = (x % 256) // TILE_SIZE
    row = (y - 32) // TILE_SIZE
    if row < 0 or row >= TILE_ROWS:
        return False
    address = TILE_BUFFER + page * TILE_ROWS * TILE_COLUMNS + row * TILE_COLUMNS + column
    return int(ram[address]) != 0


def enemy_positions(ram: np.ndarray) -> list[Sprite]:
    """Return the positions of the active enemies (``getSprites``)."""
    sprites = []
    for slot in range(ENEMY_SLOTS):
        if int(ram[ENEMY_ACTIVE + slot]) != 0:
            x = int(ram[ENEMY_PAGE + slot]) * 0x100 + int(ram[ENEMY_X_IN_PAGE + slot])
            y = int(ram[ENEMY_Y_ON_SCREEN + slot]) + 24
            sprites.append(Sprite(x, y))
    return sprites


def input_grid(ram: np.ndarray) -> np.ndarray:
    """Build the 13x13 MarI/O input grid (``getInputs``).

    Row 0 is the top of the grid and column 0 the left; Mario is in the
    centre cell ``[BOX_RADIUS, BOX_RADIUS]``. Mario himself is not drawn:
    the grid only shows the world around him.
    """
    mario_x, mario_y = mario_position(ram)
    sprites = enemy_positions(ram)
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int8)
    offsets = range(-BOX_RADIUS * TILE_SIZE, BOX_RADIUS * TILE_SIZE + 1, TILE_SIZE)
    for row, dy in enumerate(offsets):
        for column, dx in enumerate(offsets):
            x, y = mario_x + dx, mario_y + dy
            # 0x1B0 is MarI/O's cut-off for "below the bottom of the screen".
            if is_solid(ram, x, y) and y < 0x1B0:
                grid[row, column] = SOLID
            # An enemy within half a block of the cell centre overrides it.
            for sprite in sprites:
                if abs(sprite.x - x) <= 8 and abs(sprite.y - y) <= 8:
                    grid[row, column] = ENEMY
    return grid


def grid_to_text(grid: np.ndarray, mark_mario: bool = True) -> str:
    """Draw a grid as text: ``#`` solid, ``E`` enemy, ``.`` empty, ``M`` Mario."""
    symbols = {SOLID: "#", ENEMY: "E", EMPTY: "."}
    lines = []
    for row in range(grid.shape[0]):
        cells = [symbols[int(value)] for value in grid[row]]
        if mark_mario and row == grid.shape[0] // 2:
            cells[grid.shape[1] // 2] = "M"
        lines.append(" ".join(cells))
    return "\n".join(lines)
