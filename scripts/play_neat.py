"""Watch a trained NEAT genome play, with its network drawn over the game.

    python scripts/play_neat.py trained/neat-1-1.json
    python scripts/play_neat.py trained/neat-1-1.json --record videos/neat-1-1.mp4
    python scripts/play_neat.py trained/neat-1-1.json --record videos/neat-1-1.mp4 --no-render
    python scripts/play_neat.py trained/neat-1-1.json --record videos/neat-1-1.mp4 --sound

The window shows the game at its real speed (60 frames per second) with
MarI/O's view of the network on top: the input grid, the hidden neurons, the
connections lighting up and the buttons being pressed. ``--record`` also
saves exactly what the window shows to an .mp4. ``--no-render`` skips the
window, which makes recording much faster and works without a desktop.
``--sound`` adds the game's music and sound effects to the video (see
``marlio/sound.py``; needs ``uv sync --extra sound``).
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from marlio.agents.neat import Genome, run_genome
from marlio.agents.neat.agent import fitness as run_fitness
from marlio.agents.neat.display import NetworkDisplay
from marlio.controller import buttons_to_byte
from marlio.env import DEFAULT_LEVEL, make_marios_env
from marlio.video import NES_FPS, LiveViewer, VideoRecorder


def level_name(env_id: str) -> str:
    """"SuperMarioBros-1-1-v0" -> "World 1-1"."""
    parts = env_id.split("-")
    return f"World {parts[1]}-{parts[2]}" if len(parts) >= 4 else env_id


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("genome", help="a genome .json saved by train_neat.py")
    parser.add_argument("--level", default=DEFAULT_LEVEL)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--no-render", action="store_true", help="don't open a window")
    parser.add_argument("--no-overlay", action="store_true", help="show the game only, without the network")
    parser.add_argument("--record", metavar="MP4", help="also save the run as a video to this path")
    parser.add_argument("--fps", type=int, default=NES_FPS,
                        help="video frame rate: 60 keeps every game frame, 30 every other one (default 60)")
    parser.add_argument("--scale", type=int, default=3, help="enlarge the 256x240 screen this many times (default 3)")
    parser.add_argument("--outro", type=float, default=3.0,
                        help="seconds to keep showing the game after the level is cleared (default 3)")
    parser.add_argument("--sound", action="store_true",
                        help="add the game's music and sound effects to the --record video")
    args = parser.parse_args()
    if args.sound and not args.record:
        parser.error("--sound adds sound to the video, so it needs --record (the window has no sound)")

    genome = Genome.from_dict(json.loads(Path(args.genome).read_text()))
    env = make_marios_env(args.level)
    viewer = None if args.no_render else LiveViewer(f"MaRLio: NEAT on {level_name(args.level)}")
    recorder = VideoRecorder(args.record, fps=args.fps) if args.record else None
    sound = None
    if args.sound:
        from marlio.sound import GameSound

        sound = GameSound()

    def show(screen: np.ndarray, display: NetworkDisplay | None, info: list[str],
             audio: np.ndarray | None = None) -> None:
        if display is not None:
            frame = display.draw(screen, info)
        else:
            height, width = screen.shape[:2]
            frame = np.asarray(Image.fromarray(screen).resize(
                (width * args.scale, height * args.scale), Image.Resampling.NEAREST))
        if viewer is not None:
            viewer.show(frame)
        if recorder is not None:
            recorder.add(frame, audio)

    for run in range(args.runs):
        state = {"display": None, "rightmost": 0, "frames": 0}
        if sound is not None:
            # run_genome resets the env too; nes-py always resets to the same
            # snapshot, so this is the RAM the run starts from.
            env.reset()
            sound.start(env.unwrapped.ram)

        def on_frame(agent, info):
            if state["display"] is None and not args.no_overlay:
                state["display"] = NetworkDisplay(genome, agent.network, scale=args.scale)
            state["frames"] += 1
            state["rightmost"] = max(state["rightmost"], info["x_pos"])
            # The same score run_genome gives at the end, shown as it builds up.
            fitness = run_fitness(state["rightmost"], state["frames"],
                                  info["is_dying"] or info["is_dead"], info["flag_get"])
            state["info"] = [
                f"NEAT  {level_name(args.level)}",
                f"Fitness {fitness:.0f} / {genome.fitness:.0f}",
                "CLEARED!" if info["flag_get"] else f"x {info['x_pos']}",
            ]
            audio = sound.step(buttons_to_byte(agent.buttons), env.unwrapped.ram) if sound else None
            show(env.unwrapped.screen, state["display"], state["info"], audio)

        result = run_genome(env, genome, on_frame=on_frame)
        print(f"fitness {result.fitness:.1f}, reached x={result.rightmost} in {result.frames} frames"
              + (", LEVEL CLEARED" if result.cleared else ""))

        # After the flag, let the game play on (no buttons pressed) so the
        # run ends on Mario sliding down the pole, not mid-jump. The env
        # refuses to step once the episode is over, so this advances the
        # emulator directly.
        if result.cleared:
            for _ in range(int(args.outro * NES_FPS)):
                env.unwrapped._frame_advance(0)
                audio = sound.step(0, env.unwrapped.ram) if sound else None
                show(env.unwrapped.screen, state["display"], state["info"], audio)

    env.close()
    if viewer is not None:
        viewer.close()
    if recorder is not None:
        recorder.close()
        print(f"saved {recorder.path} ({recorder.seconds:.1f} s at {recorder.fps} fps"
              + (", with sound)" if sound else ")"))


if __name__ == "__main__":
    main()
