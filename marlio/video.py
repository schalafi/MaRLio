"""Showing frames in a window and saving them as an .mp4 video.

Both work on plain RGB frames (height x width x 3 numpy arrays), so any agent
can use them, with or without an overlay drawn on the game.

The NES draws about 60 frames per second and every ``env.step`` advances the
game by exactly one frame, so a video at 60 fps plays at the real game
speed. At 30 fps every other frame is kept: same speed, half the file size.

``VideoRecorder`` can also take each frame's audio (see ``marlio.sound``)
and adds it to the .mp4 as an AAC soundtrack.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import numpy as np

# The NES's frame rate (NTSC: 60.0988 Hz, which every player rounds to 60).
NES_FPS = 60


class VideoRecorder:
    """Writes frames to an H.264 .mp4 that plays in browsers and on phones.

    Uses the ffmpeg binary bundled with the ``imageio-ffmpeg`` package, so
    nothing needs installing system-wide. Pass every game frame to
    ``add``: when ``fps`` is below 60 the recorder drops frames to keep the
    game at its real speed.

    To add sound, pass each game frame's audio too (int16, one row per
    sample, one column per channel; any number of samples per frame):
    ``add(frame, audio)``. It's added when the recorder closes.
    """

    def __init__(self, path: str | Path, fps: int = NES_FPS, quality_crf: int = 18):
        if NES_FPS % fps:
            raise ValueError(f"fps must divide {NES_FPS} (e.g. 60, 30, 20, 15), got {fps}")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.keep_every = NES_FPS // fps
        self.crf = quality_crf
        self.frames_seen = 0
        self.frames_written = 0
        self._writer = None
        self._audio: list[np.ndarray] = []
        self._video_path = self.path

    def add(self, frame: np.ndarray, audio: np.ndarray | None = None) -> None:
        if audio is not None:
            self._audio.append(np.asarray(audio, np.int16).reshape(len(audio), -1))
        if self.frames_seen % self.keep_every == 0:
            if self._writer is None:
                self._open(frame.shape[1], frame.shape[0])
            self._writer.send(np.ascontiguousarray(frame, dtype=np.uint8))
            self.frames_written += 1
        self.frames_seen += 1

    def _open(self, width: int, height: int) -> None:
        import imageio_ffmpeg

        # With sound, the picture goes to a temporary file first and the
        # sound is added when the recorder closes.
        self._video_path = self.path.with_suffix(".video.mp4") if self._audio else self.path
        self._writer = imageio_ffmpeg.write_frames(
            str(self._video_path),
            (width, height),
            fps=self.fps,
            codec="libx264",
            pix_fmt_out="yuv420p",  # the pixel format every player supports
            # crf: 0 is lossless, 23 is ffmpeg's default; 18 looks clean on
            # pixel art. A small keyframe interval makes seeking snappy.
            output_params=["-crf", str(self.crf), "-g", str(self.fps)],
            macro_block_size=2,
            ffmpeg_log_level="error",
        )
        self._writer.send(None)  # start the ffmpeg process

    @property
    def seconds(self) -> float:
        return self.frames_written / self.fps

    def close(self) -> None:
        if self._writer is None:
            return
        self._writer.close()
        self._writer = None
        if self._video_path != self.path:
            self._add_sound()

    def _add_sound(self) -> None:
        import imageio_ffmpeg

        audio = np.concatenate(self._audio)
        # The NES mix is quiet next to other videos; raise it so its loudest
        # moment is just below full scale (at most 4x).
        peak = int(np.abs(audio.astype(np.int32)).max()) or 1
        gain = min(4.0, 0.9 * 32767 / peak)
        audio = np.clip(audio.astype(np.float32) * gain, -32768, 32767).astype(np.int16)
        # The NES runs at 60.0988 frames per second, the video at exactly 60,
        # so the audio's sample rate is set to whatever makes it last exactly
        # as long as the video (a 0.16% change, which nobody can hear). That
        # keeps sound and picture in sync from start to end.
        rate = round(len(audio) / (self.frames_seen / NES_FPS))
        raw = self.path.with_suffix(".audio.raw")
        raw.write_bytes(audio.tobytes())
        video = self._video_path
        try:
            subprocess.run(
                [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
                 "-i", str(video),
                 "-f", "s16le", "-ar", str(rate), "-ac", str(audio.shape[1]), "-i", str(raw),
                 "-c:v", "copy", "-af", "aresample=48000", "-c:a", "aac", "-b:a", "160k",
                 "-shortest", str(self.path)],
                check=True,
            )
        finally:
            raw.unlink(missing_ok=True)
            video.unlink(missing_ok=True)


class LiveViewer:
    """Shows frames in a window at the game's real speed (60 fps by default).

    Uses the same pyglet window as gym-super-mario-bros's ``render_mode="human"``,
    so it works wherever that does (for example the Codespace's noVNC desktop).
    """

    def __init__(self, caption: str = "MaRLio", fps: float = NES_FPS):
        self.caption = caption
        self.frame_time = 1 / fps if fps > 0 else 0.0
        self._viewer = None
        self._next = 0.0

    def show(self, frame: np.ndarray) -> None:
        from nes_py._image_viewer import ImageViewer

        if self._viewer is None:
            self._viewer = ImageViewer(self.caption, frame.shape[0], frame.shape[1])
            self._next = time.perf_counter()
        self._viewer.show(np.ascontiguousarray(frame, dtype=np.uint8))
        # Wait until this frame's time is up, so the game isn't fast-forwarded.
        self._next += self.frame_time
        delay = self._next - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        elif delay < -5 * self.frame_time:
            # Far behind (the window was busy): carry on from now rather
            # than rushing through frames to catch up.
            self._next = time.perf_counter()

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
