import subprocess

import imageio_ffmpeg
import numpy as np
import pytest

from marlio.video import VideoRecorder


def probe(path):
    return subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-i", str(path)], capture_output=True, text=True).stderr


def test_recorder_adds_an_aac_track_and_cleans_up(tmp_path):
    path = tmp_path / "run.mp4"
    recorder = VideoRecorder(path, fps=60)
    tone = (3000 * np.sin(np.arange(534) / 10)).astype(np.int16)
    for i in range(60):  # 1 s of game, with about 534 stereo samples per frame like the NES
        recorder.add(np.full((64, 64, 3), i * 4, dtype=np.uint8), np.stack([tone, tone], 1))
    recorder.close()
    info = probe(path)
    assert "h264" in info and "aac" in info
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run.mp4"]


def test_recorder_without_audio_has_no_audio_track(tmp_path):
    path = tmp_path / "run.mp4"
    recorder = VideoRecorder(path)
    for _ in range(6):
        recorder.add(np.zeros((64, 64, 3), np.uint8))
    recorder.close()
    assert "Audio:" not in probe(path)


def test_game_sound_follows_a_nes_py_run():
    pytest.importorskip("stable_retro")
    from marlio.controller import buttons_to_byte
    from marlio.env import make_marios_env
    from marlio.sound import GameSound

    env = make_marios_env()
    env.reset()
    sound = GameSound()
    sound.start(env.unwrapped.ram)
    right_and_jump = [1, 0, 0, 0, 0, 1]  # A, B, up, down, left, right
    chunks = []
    for _ in range(120):
        env.step(right_and_jump)
        chunks.append(sound.step(buttons_to_byte(right_and_jump), env.unwrapped.ram))
    audio = np.concatenate(chunks)
    assert audio.dtype == np.int16 and audio.shape[1] == 2
    assert len(audio) == pytest.approx(120 * sound.rate / 60.0988, rel=0.02)
    assert np.abs(audio).max() > 1000  # the music (and jumps) are playing
    env.close()
