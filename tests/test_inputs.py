import numpy as np

from marlio.inputs import (
    BOX_RADIUS,
    ENEMY,
    ENEMY_ACTIVE,
    ENEMY_PAGE,
    ENEMY_X_IN_PAGE,
    ENEMY_Y_ON_SCREEN,
    GRID_SIZE,
    PLAYER_PAGE,
    PLAYER_X_IN_PAGE,
    PLAYER_Y_ON_SCREEN,
    SOLID,
    TILE_BUFFER,
    TILE_COLUMNS,
    input_grid,
)


def blank_ram(mario_x=0x50, mario_screen_y=0x70):
    ram = np.zeros(2048, dtype=np.uint8)
    ram[PLAYER_PAGE] = mario_x // 256
    ram[PLAYER_X_IN_PAGE] = mario_x % 256
    ram[PLAYER_Y_ON_SCREEN] = mario_screen_y
    return ram


def test_empty_world_gives_empty_grid():
    grid = input_grid(blank_ram())
    assert grid.shape == (GRID_SIZE, GRID_SIZE)
    assert not grid.any()


def test_block_under_grid_centre_is_one_row_below():
    # Mario at x=0x50 (column 5 after MarI/O's +8 shift), y on screen 0x70:
    # grid centre row = (0x70 + 16 - 16 - 32) // 16 = 5 in the tile buffer.
    ram = blank_ram()
    ram[TILE_BUFFER + 6 * TILE_COLUMNS + 5] = 0x54  # any non-zero block id
    grid = input_grid(ram)
    assert grid[BOX_RADIUS + 1, BOX_RADIUS] == SOLID
    assert grid.sum() == SOLID


def test_enemy_next_to_mario():
    ram = blank_ram()
    enemy_x = 0x50 + 16  # one block to the right
    ram[ENEMY_ACTIVE] = 1
    ram[ENEMY_PAGE] = enemy_x // 256
    ram[ENEMY_X_IN_PAGE] = enemy_x % 256
    ram[ENEMY_Y_ON_SCREEN] = 0x70 + 16 - 24  # same height as the grid centre
    grid = input_grid(ram)
    assert grid[BOX_RADIUS, BOX_RADIUS + 1] == ENEMY
    assert (grid == ENEMY).sum() == 1


def test_real_level_start_has_ground_and_no_enemies():
    from marlio.env import make_marios_env

    env = make_marios_env()
    grid, _ = env.reset(seed=0)
    env.close()
    # World 1-1 starts on flat ground: two full rows of solid blocks
    # (except at the far left, which is before the start of the level).
    assert (grid[8, BOX_RADIUS:] == SOLID).all()
    assert (grid[9, BOX_RADIUS:] == SOLID).all()
    assert not (grid == ENEMY).any()
