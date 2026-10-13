import json
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import pytest

from marlio.agents.neat import N_INPUTS, N_OUTPUTS, Gene, Genome
from marlio.agents.neat.display import (
    BIAS_POSITION,
    BUTTON_X,
    GRID_CENTRE,
    HELD,
    HIDDEN_MAX_X,
    HIDDEN_MIN_X,
    HIDDEN_RADIUS,
    NetworkDisplay,
    button_centre,
    neuron_positions,
    value_text,
)
from marlio.agents.neat.genome import MAX_NODES
from marlio.agents.neat.network import Network
from marlio.video import VideoRecorder

BIAS = N_INPUTS - 1
RIGHT = MAX_NODES + 5
HIDDEN = N_INPUTS + 1
TRAINED = Path(__file__).parent.parent / "trained" / "neat-1-1.json"


def genome_with(*genes):
    return Genome(genes=list(genes), max_neuron=HIDDEN + 10, mutation_rates={})


def test_fixed_positions():
    positions = neuron_positions(genome_with(), N_INPUTS, N_OUTPUTS)
    assert positions[84] == GRID_CENTRE  # centre cell, where Mario is
    assert positions[0][0] < positions[1][0]  # row by row, left to right
    assert positions[BIAS] == BIAS_POSITION
    assert positions[MAX_NODES] == (BUTTON_X, button_centre(0)[1])


def test_hidden_neuron_sits_between_its_input_and_output():
    genome = genome_with(Gene(0, HIDDEN, 1.0, innovation=1), Gene(HIDDEN, RIGHT, 1.0, innovation=2))
    x, _ = neuron_positions(genome, N_INPUTS, N_OUTPUTS)[HIDDEN]
    assert HIDDEN_MIN_X <= x <= HIDDEN_MAX_X


def test_hidden_neurons_do_not_overlap():
    genome = Genome.from_dict(json.loads(TRAINED.read_text()))
    positions = neuron_positions(genome, N_INPUTS, N_OUTPUTS)
    hidden = [np.array(p) for n, p in positions.items() if N_INPUTS <= n < MAX_NODES]
    assert len(hidden) > 1
    closest = min(np.linalg.norm(a - b) for i, a in enumerate(hidden) for b in hidden[i + 1:])
    assert closest >= 2 * HIDDEN_RADIUS


@pytest.mark.parametrize("value, text", [(1.0, "1"), (-0.99, "-1"), (0.0, "0"), (0.72, ".7"), (-0.31, "-.3")])
def test_value_text(value, text):
    assert value_text(value) == text


def test_draw_scales_the_screen_and_lights_held_buttons():
    genome = genome_with(Gene(BIAS, RIGHT, 1.0, innovation=1))
    network = Network(genome, N_INPUTS, N_OUTPUTS)
    display = NetworkDisplay(genome, network, scale=2)
    screen = np.full((240, 256, 3), 200, dtype=np.uint8)
    before = display.draw(screen)
    assert before.shape == (480, 512, 3)
    network.evaluate(np.zeros(N_INPUTS - 1))  # bias -> right: "right" is held
    after = display.draw(screen)
    x, y = button_centre(5)
    corner = after[int(y * 2) - 6, int(x * 2) - 20]  # inside the button, away from its label
    assert tuple(corner) == HELD
    assert not np.array_equal(before, after)


def test_trained_genome_draws_with_info():
    genome = Genome.from_dict(json.loads(TRAINED.read_text()))
    network = Network(genome, N_INPUTS, N_OUTPUTS)
    frame = NetworkDisplay(genome, network).draw(np.zeros((240, 256, 3), np.uint8), ["NEAT", "Fitness 0"])
    assert frame.shape == (720, 768, 3)


@pytest.mark.parametrize("fps, expected_frames", [(60, 12), (30, 6)])
def test_recorder_writes_an_mp4_at_game_speed(tmp_path, fps, expected_frames):
    path = tmp_path / "run.mp4"
    recorder = VideoRecorder(path, fps=fps)
    for i in range(12):  # 12 game frames = 0.2 s
        recorder.add(np.full((64, 64, 3), i * 20, dtype=np.uint8))
    recorder.close()
    assert recorder.frames_written == expected_frames
    assert recorder.seconds == pytest.approx(0.2)
    probe = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-i", str(path)], capture_output=True, text=True)
    assert f"{fps} fps" in probe.stderr
    assert "h264" in probe.stderr


def test_recorder_rejects_a_rate_that_would_change_game_speed(tmp_path):
    with pytest.raises(ValueError):
        VideoRecorder(tmp_path / "run.mp4", fps=25)
