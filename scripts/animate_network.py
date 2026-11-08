"""Animate a NEAT network on its own, without the game: its parts, and how it evolves.

    python scripts/animate_network.py anatomy   videos/network-anatomy
    python scripts/animate_network.py evolution videos/network-evolution \\
        --genomes trained/neat-1-1-generations/seed0

Each command writes two files:

* ``NAME.mp4``  1920 x 1080, 30 fps, on a plain dark background
* ``NAME.mov``  the same with a transparent background (QuickTime Animation,
  lossless, ~10x smaller than ProRes 4444), to put over other footage in a
  video editor

``anatomy`` builds a network up piece by piece, with a caption for each part:
the 13 x 13 input grid, the bias, the 6 outputs (an NES controller), a
connection, a hidden neuron, and finally the trained 1-1 network
(``--genome``) thinking about one moment of the level, with signals running
along its connections.

``evolution`` shows the best network of each generation of a training run
(``gen_NNN_best.json`` files plus ``index.csv`` from
``record_generations.py``, or a ``train_neat.py --out`` folder), morphing
from one to the next. Connections and neurons that are new in a generation
flash gold, ones that are gone fade out, and weights that changed slide to
their new thickness and shade. Generations that kept the same best network
are skipped. A bar at the bottom shows how far into the level each one got.

The picture uses the same rules as the overlay drawn over the game
(``marlio/agents/neat/display.py``), and the same layout, just bigger: white
is +, black is -, a thicker line is a stronger weight.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from marlio.agents.neat import N_INPUTS, N_OUTPUTS, Genome, run_genome
from marlio.agents.neat.display import (
    AB_RADIUS, BIAS_POSITION, BIAS_RADIUS, CELL_SPACING, CONTROLLER, DPAD_CENTRE, DPAD_HALF_WIDTH,
    DPAD_REACH, GRID_CENTRE, HIDDEN_RADIUS, INPUT_HALF, OUTPUT_POSITIONS, neuron_positions, value_text,
)
from marlio.agents.neat.genome import MAX_NODES
from marlio.agents.neat.network import Network, sigmoid
from marlio.controller import BUTTONS
from marlio.env import make_marios_env
from marlio.inputs import BOX_RADIUS

WIDTH, HEIGHT, FPS = 1920, 1080, 30
U = 7  # screen pixels per NES pixel of the overlay layout
ORIGIN = (64, 40 - 34 * U)  # where NES (0, 0) lands, so the band (y 34..114) starts at y 40 + ...
NETWORK_TOP = 150  # ... pushed down to leave room for the title
SS = 2  # supersampling, for smooth lines and circles
BACKGROUND = (34, 40, 52)
GOLD = (255, 196, 40)
FADED = (150, 60, 60)
LEVEL_END_X = 3161  # where the 1-1 winner touches the flag
BIAS = N_INPUTS - 1


def lerp(a, b, t):
    return a + (b - a) * t


def ease(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def _shade(value: float) -> int:
    return int(min(max((value + 1) / 2 * 256, 0), 255))


@dataclass
class Link:
    into: int
    out: int
    weight: float
    alpha: float = 1.0
    grow: float = 1.0  # how much of the line is drawn, from its start
    glow: float = 0.0  # gold highlight, 0..1
    fade: float = 0.0  # red "going away" tint, 0..1


@dataclass
class Scene:
    """Everything on screen in one frame."""
    positions: dict[int, tuple[float, float]] = field(default_factory=dict)
    grid_alpha: float = 0.0
    dots: int = 169  # how many input dots are shown (they appear in reading order)
    inputs: dict[int, float] = field(default_factory=dict)  # input values (+1 block, -1 enemy)
    input_alpha: float = 1.0
    used_inputs: set[int] = field(default_factory=set)  # inputs with a connection get an outline
    bias_alpha: float = 0.0
    controller_alpha: float = 0.0
    pressed: dict[str, float] = field(default_factory=dict)  # button -> 0..1
    hidden: dict[int, tuple[float, float, float]] = field(default_factory=dict)  # id -> (value or nan, alpha, glow)
    links: list[Link] = field(default_factory=list)
    pulses: list[tuple[float, float, float, float]] = field(default_factory=list)  # x, y, alpha, sign
    title: str = ""
    caption: str = ""
    text_alpha: float = 1.0
    line_scale: float = 1.0  # thinner lines for big networks
    progress: float | None = None  # 0..1 for the "how far it got" bar
    progress_label: str = ""


class Renderer:
    def __init__(self):
        self.title_font = ImageFont.load_default(size=56)
        self.caption_font = ImageFont.load_default(size=36)
        self.small_font = ImageFont.load_default(size=28)
        self.neuron_font = ImageFont.load_default(size=3.4 * U * SS)
        self.button_font = ImageFont.load_default(size=4.5 * U * SS)

    @staticmethod
    def xy(x: float, y: float) -> tuple[float, float]:
        """NES layout coordinates -> supersampled canvas pixels."""
        return ((ORIGIN[0] + x * U) * SS, (NETWORK_TOP + (y - 34) * U) * SS)

    def draw(self, scene: Scene) -> Image.Image:
        big = Image.new("RGBA", (WIDTH * SS, HEIGHT * SS), (0, 0, 0, 0))
        d = ImageDraw.Draw(big, "RGBA")
        p = scene.positions

        def a8(alpha):
            return int(255 * min(max(alpha, 0), 1))

        def box(x0, y0, x1, y1, fill, outline=None, width=0.0, radius=0.0):
            (X0, Y0), (X1, Y1) = self.xy(x0, y0), self.xy(x1, y1)
            d.rounded_rectangle((X0, Y0, X1, Y1), radius=radius * U * SS, fill=fill, outline=outline,
                                width=round(width * U * SS))

        def circle(x, y, r, fill, outline=None, width=0.4):
            X, Y = self.xy(x, y)
            R = r * U * SS
            d.ellipse((X - R, Y - R, X + R, Y + R), fill=fill, outline=outline,
                      width=max(1, round(width * U * SS)) if outline else 0)

        # Input grid backdrop, Mario's cell.
        ga = scene.grid_alpha
        if ga > 0:
            cx, cy = GRID_CENTRE
            half = BOX_RADIUS * CELL_SPACING + CELL_SPACING / 2 + 1
            box(cx - half, cy - half, cx + half, cy + half, (70, 70, 70, a8(ga * 0.56)), (0, 0, 0, a8(ga)), 0.4, 1.5)
            m = CELL_SPACING / 2
            box(cx - m, cy - m, cx + m, cy + m, None, (230, 30, 30, a8(ga)), 0.7)

        # Controller case.
        ca = scene.controller_alpha
        if ca > 0:
            left, top, right, bottom = CONTROLLER
            box(left, top, right, bottom, (200, 200, 200, a8(ca * 0.6)), (60, 60, 60, a8(ca)), 0.4, 2.5)
            box(left + 2, top + 2, right - 2, bottom - 2, (25, 25, 25, a8(ca * 0.6)), None, 0, 1.5)

        # Connections.
        for link in scene.links:
            if link.alpha <= 0 or link.into not in p or link.out not in p:
                continue
            (x1, y1), (x2, y2) = p[link.into], p[link.out]
            if link.into < BIAS:
                x1 += INPUT_HALF
            else:
                x1 += BIAS_RADIUS if link.into == BIAS else HIDDEN_RADIUS
            if link.out < MAX_NODES:
                x2 -= HIDDEN_RADIUS
            x2, y2 = lerp(x1, x2, link.grow), lerp(y1, y2, link.grow)
            positive = link.weight > 0
            colour = np.array((255, 255, 255) if positive else (10, 10, 10), float)
            casing = np.array((0, 0, 0) if positive else (255, 255, 255), float)
            if link.glow > 0:
                colour = colour * (1 - link.glow) + np.array(GOLD) * link.glow
            if link.fade > 0:
                colour = colour * (1 - link.fade) + np.array(FADED) * link.fade
            width = (0.4 + 0.7 * abs(sigmoid(link.weight))) * U * SS * scene.line_scale
            pts = (*self.xy(x1, y1), *self.xy(x2, y2))
            d.line(pts, fill=(*casing.astype(int), a8(link.alpha * 0.75 * scene.line_scale)),
                   width=round(width + 0.7 * U * SS * scene.line_scale))
            d.line(pts, fill=(*colour.astype(int), a8(link.alpha)), width=max(1, round(width)))

        # Inputs.
        if ga > 0:
            for i in range(min(scene.dots, BIAS)):
                x, y = p[i]
                value = scene.inputs.get(i, 0.0) if scene.input_alpha > 0 else 0.0
                if value == 0:
                    circle(x, y, 0.6, (255, 255, 255, a8(ga * 0.3)))
                else:
                    shade = _shade(value)
                    ia = ga * scene.input_alpha
                    box(x - INPUT_HALF, y - INPUT_HALF, x + INPUT_HALF, y + INPUT_HALF,
                        (shade, shade, shade, a8(ia)), (0, 0, 0, a8(ia)) if shade > 128 else (255, 255, 255, a8(ia * 0.8)),
                        0.3, 0.5)
                if i in scene.used_inputs:
                    box(x - INPUT_HALF - 0.6, y - INPUT_HALF - 0.6, x + INPUT_HALF + 0.6, y + INPUT_HALF + 0.6,
                        None, (255, 255, 255, a8(ga * 0.9)), 0.35, 0.6)

        # Bias.
        if scene.bias_alpha > 0:
            x, y = p[BIAS]
            circle(x, y, BIAS_RADIUS, (255, 255, 255, a8(scene.bias_alpha)), (0, 0, 0, a8(scene.bias_alpha)))

        # Hidden neurons.
        for neuron, (value, alpha, glow) in scene.hidden.items():
            if alpha <= 0 or neuron not in p:
                continue
            x, y = p[neuron]
            if glow > 0:
                circle(x, y, HIDDEN_RADIUS + 1.6 * glow, (*GOLD, a8(alpha * 0.8 * glow)))
            shown = 0.0 if math.isnan(value) else value
            shade = _shade(shown)
            outline = tuple(int(lerp(c, g, glow)) for c, g in zip((0, 0, 0), GOLD))
            circle(x, y, HIDDEN_RADIUS, (shade, shade, shade, a8(alpha)), (*outline, a8(alpha)), 0.4 + 0.4 * glow)
            if not math.isnan(value):
                X, Y = self.xy(x, y)
                d.text((X, Y), value_text(value), font=self.neuron_font, anchor="mm",
                       fill=(0, 0, 0, a8(alpha)) if shade > 140 else (255, 255, 255, a8(alpha)))

        # Signals travelling along connections.
        for x, y, alpha, sign in scene.pulses:
            fill = (255, 255, 255) if sign > 0 else (10, 10, 10)
            ring = (0, 0, 0) if sign > 0 else (255, 255, 255)
            circle(x, y, 1.3, (*fill, a8(alpha)), (*ring, a8(alpha)), 0.35)

        # The controller's D-pad and buttons.
        if ca > 0:
            dx, dy = DPAD_CENTRE
            w, r = DPAD_HALF_WIDTH, DPAD_REACH
            box(dx - w, dy - r, dx + w, dy + r, (55, 55, 55, a8(ca)), (0, 0, 0, a8(ca)), 0.5, 1)
            box(dx - r, dy - w, dx + r, dy + w, (55, 55, 55, a8(ca)), (0, 0, 0, a8(ca)), 0.5, 1)
            arms = {"up": (dx - w, dy - r, dx + w, dy - w), "down": (dx - w, dy + w, dx + w, dy + r),
                    "left": (dx - r, dy - w, dx - w, dy + w), "right": (dx + w, dy - w, dx + r, dy + w)}
            for name, (x0, y0, x1, y1) in arms.items():
                held = scene.pressed.get(name, 0.0) * ca
                if held > 0:
                    box(x0 + 0.5, y0 + 0.5, x1 - 0.5, y1 - 0.5, (40, 140, 255, a8(held)), (255, 255, 255, a8(held)), 0.5, 1)
            circle(dx, dy, 1.4, (35, 35, 35, a8(ca)))
            for name in ("B", "A"):
                x, y = OUTPUT_POSITIONS[name]
                held = scene.pressed.get(name, 0.0)
                fill = tuple(int(lerp(u, v, held)) for u, v in zip((130, 25, 25), (255, 60, 50)))
                circle(x, y, AB_RADIUS, (*fill, a8(ca)), (255, 255, 255, a8(ca * held)) if held > 0.5 else (0, 0, 0, a8(ca)),
                       0.8 if held > 0.5 else 0.5)
                X, Y = self.xy(x, y + AB_RADIUS + 4)
                d.text((X, Y), name, font=self.button_font, anchor="mm", fill=(220, 70, 60, a8(ca)))

        image = big.reduce(SS)
        self._draw_text(image, scene)
        return image

    def _draw_text(self, image: Image.Image, scene: Scene) -> None:
        d = ImageDraw.Draw(image, "RGBA")
        ta = int(255 * min(max(scene.text_alpha, 0), 1))
        if scene.title:
            d.text((WIDTH // 2, 80), scene.title, font=self.title_font, anchor="mm", fill=(255, 255, 255, ta))
        if scene.caption:
            y = 800
            for line in scene.caption.split("\n"):
                d.text((WIDTH // 2, y), line, font=self.caption_font, anchor="mm", fill=(235, 235, 235, ta))
                y += 50
        if scene.progress is not None:
            x0, x1, y = 360, 1560, 1010
            d.rounded_rectangle((x0, y - 8, x1, y + 8), radius=8, fill=(255, 255, 255, 50))
            d.rounded_rectangle((x0, y - 8, x0 + (x1 - x0) * min(scene.progress, 1), y + 8), radius=8,
                                fill=(255, 255, 255, 230))
            d.text((x0 - 20, y), "start", font=self.small_font, anchor="rm", fill=(200, 200, 200, 255))
            d.text((x1 + 20, y), "flag", font=self.small_font, anchor="lm", fill=(200, 200, 200, 255))
            if scene.progress_label:
                d.text((WIDTH // 2, y - 40), scene.progress_label, font=self.small_font, anchor="mm",
                       fill=(220, 220, 220, 255))


class Writer:
    """Pipes frames to ffmpeg: an .mp4 on the dark background and a transparent .mov."""

    def __init__(self, stem: Path):
        stem.parent.mkdir(parents=True, exist_ok=True)
        common = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS)]
        self.mp4 = subprocess.Popen(common + ["-pix_fmt", "rgb24", "-i", "-", "-c:v", "libx264", "-crf", "16",
                                              "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                                              str(stem.with_suffix(".mp4"))], stdin=subprocess.PIPE)
        self.mov = subprocess.Popen(common + ["-pix_fmt", "rgba", "-i", "-", "-c:v", "qtrle", "-pix_fmt", "argb",
                                              str(stem.with_suffix(".mov"))],
                                    stdin=subprocess.PIPE)
        self.background = Image.new("RGBA", (WIDTH, HEIGHT), (*BACKGROUND, 255))
        self.frames = 0

    def add(self, frame: Image.Image) -> None:
        self.mov.stdin.write(frame.tobytes())
        self.mp4.stdin.write(Image.alpha_composite(self.background, frame).convert("RGB").tobytes())
        self.frames += 1

    def close(self) -> None:
        for proc in (self.mp4, self.mov):
            proc.stdin.close()
            proc.wait()


def fixed_positions() -> dict[int, tuple[float, float]]:
    """Inputs, bias and outputs: they never move."""
    return neuron_positions(Genome(), N_INPUTS, N_OUTPUTS)


def sample_moment(genome: Genome, frame: int) -> tuple[dict[int, float], dict[int, float]]:
    """Play ``genome`` on 1-1 and return (input values, all neuron values) at ``frame``."""
    env = make_marios_env("SuperMarioBros-1-1-v0")
    grab: dict = {}

    def on_frame(agent, info, stats):
        if stats.frames == frame:
            grab["values"] = dict(agent.network.values)

    run_genome(env, genome, on_frame=on_frame)
    env.close()
    values = grab["values"]
    return {i: values[i] for i in range(BIAS) if values.get(i, 0) != 0}, values


def layers(genome: Genome) -> dict[int, int]:
    """Signal depth of each neuron: inputs 0, then one more per connection (loops ignored)."""
    depth = {i: 0 for i in range(N_INPUTS)}
    enabled = [g for g in genome.genes if g.enabled]
    for _ in range(20):
        for g in enabled:
            if g.into in depth and g.out != g.into:
                if g.out not in depth or depth[g.out] < depth[g.into] + 1 <= 10:
                    depth[g.out] = depth[g.into] + 1
    return depth


# ---------------------------------------------------------------- anatomy


def anatomy(stem: Path, genome_path: Path, moment: int) -> None:
    genome = Genome.from_dict(json.loads(genome_path.read_text()))
    inputs, values = sample_moment(genome, moment)
    fixed = fixed_positions()
    renderer, writer = Renderer(), Writer(stem)

    def frames(seconds: float):
        n = int(seconds * FPS)
        for k in range(n):
            yield k / max(n - 1, 1)

    def emit(scene: Scene):
        writer.add(renderer.draw(scene))

    def text_fade(t: float, seconds: float) -> float:
        edge = 0.4 / seconds
        return min(ease(t / edge), ease((1 - t) / edge))

    base = dict(positions=fixed)
    # 1. Inputs.
    title, caption = "The inputs", ("What the network sees: the 13 x 13 tiles around Mario (the red square).\n"
                                    "White = solid block (+1), black = enemy (-1), dot = empty (0).")
    for t in frames(7):
        emit(Scene(**base, grid_alpha=ease(t * 4), dots=int(169 * ease(t * 2.5)), inputs=inputs,
                   input_alpha=ease((t - 0.45) * 4), title=title, caption=caption, text_alpha=text_fade(t, 7)))
    common = dict(**base, grid_alpha=1, inputs=inputs)
    # 2. Bias.
    for t in frames(4):
        emit(Scene(**common, bias_alpha=ease(t * 4), title="The bias",
                   caption="One extra input that is always 1,\nso a neuron can be on even when the screen is empty.",
                   text_alpha=text_fade(t, 4)))
    common["bias_alpha"] = 1
    # 3. Outputs.
    for t in frames(6):
        order = ["up", "right", "down", "left", "B", "A"]
        k = int(t * 8) - 1
        pressed = {order[k]: 1.0} if 0 <= k < len(order) else {}
        emit(Scene(**common, controller_alpha=ease(t * 4), pressed=pressed, title="The outputs",
                   caption="Six buttons. Each one is pressed while its neuron is above 0.", text_alpha=text_fade(t, 6)))
    common["controller_alpha"] = 1
    # 4. Connections: pick an input that is on right now, to a button.
    on = sorted(inputs, key=lambda i: (-inputs[i], abs(fixed[i][1] - GRID_CENTRE[1])))
    src_pos = on[0] if on else 97
    src_neg = next((i for i in inputs if inputs[i] < 0), BIAS)
    out_a, out_right = MAX_NODES + BUTTONS.index("A"), MAX_NODES + BUTTONS.index("right")
    l1 = Link(src_pos, out_right, 1.8)
    l2 = Link(src_neg, out_a, -1.2)
    for t in frames(8):
        l1.grow = ease(t * 3)
        l2.grow = ease((t - 0.45) * 3)
        l2.alpha = 1.0 if t > 0.45 else 0.0
        # An enemy (-1) times a negative weight is positive: A (jump) gets pressed.
        pressed = {"right": ease((t - 0.3) * 6), "A": ease((t - 0.75) * 6)}
        emit(Scene(**common, links=[l1, l2], pressed=pressed, title="A connection (a gene)",
                   caption="Carries a value from one neuron to another, multiplied by its weight.\n"
                           "White = positive weight, black = negative, thicker = stronger.",
                   text_alpha=text_fade(t, 8)))
    # 5. Hidden neuron: split l1.
    hidden_id = N_INPUTS + 1
    hx, hy = 140, 62
    positions = dict(fixed)
    positions[hidden_id] = (hx, hy)
    for t in frames(9):
        split = ease((t - 0.15) * 4)
        a = Link(src_pos, hidden_id, 1.0, grow=1.0, alpha=split, glow=1 - ease((t - 0.5) * 3))
        b = Link(hidden_id, out_right, 1.8, grow=1.0, alpha=split, glow=1 - ease((t - 0.5) * 3))
        old = Link(l1.into, l1.out, l1.weight, alpha=1 - split, fade=split)
        hv = sigmoid(1.0 * inputs.get(src_pos, 1.0)) if t > 0.6 else float("nan")
        emit(Scene(positions=positions, grid_alpha=1, inputs=inputs, bias_alpha=1, controller_alpha=1,
                   links=[old, a, b, l2], hidden={hidden_id: (hv, split, 1 - ease((t - 0.5) * 3))},
                   pressed={"right": 1.0, "A": 1.0}, title="A hidden neuron",
                   caption="Adds up everything that reaches it and squashes the sum to between -1 and +1.\n"
                           "Evolution adds one by splitting a connection in two.",
                   text_alpha=text_fade(t, 9)))
    # 6. The trained network thinking.
    full = neuron_positions(genome, N_INPUTS, N_OUTPUTS)
    enabled = [g for g in genome.genes if g.enabled]
    depth = layers(genome)
    max_depth = max((depth.get(g.out, 1) for g in enabled), default=1)
    pressed = {name: float(values.get(MAX_NODES + o, 0) > 0) for o, name in enumerate(BUTTONS)}
    n_hidden = len({n for g in enabled for n in (g.into, g.out) if N_INPUTS <= n < MAX_NODES})
    seconds = 12
    for t in frames(seconds):
        appear = ease(t * 3)
        links = [Link(g.into, g.out, g.weight, alpha=appear * (1.0 if values.get(g.into, 0) != 0 else 0.25))
                 for g in enabled]
        cycle = (t * seconds / 2.5) % 1.0  # a wave of signals every 2.5 s, layer by layer
        pulses = []
        if t > 0.25:
            for g in enabled:
                v = values.get(g.into, 0)
                if v == 0:
                    continue
                start = (depth.get(g.into, 0)) / (max_depth + 1)
                local = (cycle - start) * (max_depth + 1)
                if 0 <= local <= 1:
                    (x1, y1), (x2, y2) = full[g.into], full[g.out]
                    pulses.append((lerp(x1, x2, local), lerp(y1, y2, local), 1.0, g.weight * v))
        hidden = {n: (values.get(n, 0.0) if t > 0.25 else float("nan"), appear, 0.0)
                  for n in full if N_INPUTS <= n < MAX_NODES}
        emit(Scene(positions=full, grid_alpha=1, inputs=inputs, bias_alpha=1, controller_alpha=1, links=links,
                   hidden=hidden, pulses=pulses, pressed=pressed if t > 0.25 else {}, line_scale=0.7,
                   used_inputs=set(), title="The whole network",
                   caption=f"The network that clears World 1-1: {n_hidden} hidden neurons, {len(enabled)} connections.\n"
                           "Every frame the inputs flow through it and the controller presses the buttons.",
                   text_alpha=text_fade(t, seconds)))
    writer.close()
    print(f"saved {stem}.mp4 and {stem}.mov ({writer.frames / FPS:.1f} s)")


# ---------------------------------------------------------------- evolution


def load_generations(folder: Path) -> list[tuple[int, Genome, int]]:
    """(generation, genome, best x) for every generation with a new best genome."""
    genome_dir = folder / "genomes" if (folder / "genomes").is_dir() else folder
    index = folder / "index.csv" if (folder / "index.csv").exists() else folder.parent / "index.csv"
    log = folder / "log.csv" if (folder / "log.csv").exists() else folder.parent / "log.csv"
    best_x = {int(r["generation"]): int(r["best_x"]) for r in csv.DictReader(open(log))}
    rows = []
    seen = set()
    for path in sorted(genome_dir.glob("gen_*_best.json")):
        gen = int(path.stem.split("_")[1])
        data = json.loads(path.read_text())
        key = json.dumps(data["genes"], sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        rows.append((gen, Genome.from_dict(data), best_x.get(gen, 0)))
    del index
    return rows


def evolution(stem: Path, folder: Path, hold: float, morph: float) -> None:
    generations = load_generations(folder)
    fixed = fixed_positions()
    renderer, writer = Renderer(), Writer(stem)
    layouts = [neuron_positions(g, N_INPUTS, N_OUTPUTS) for _, g, _ in generations]

    def genes(genome: Genome) -> dict[int, object]:
        return {g.innovation: g for g in genome.genes if g.enabled}

    def hidden_of(genome: Genome) -> set[int]:
        return {n for g in genome.genes if g.enabled for n in (g.into, g.out) if N_INPUTS <= n < MAX_NODES}

    def inputs_of(genome: Genome) -> set[int]:
        return {g.into for g in genome.genes if g.enabled and g.into < BIAS}

    def scene_for(i: int, t: float, phase: str) -> Scene:
        """``phase`` "hold": generation i, new parts glowing (t from 0 to 1 fades the glow).
        ``phase`` "morph": from generation i to i + 1, t from 0 to 1."""
        gen, genome, x = generations[i]
        if phase == "hold":
            prev = generations[i - 1][1] if i > 0 else Genome()
            now_genes, before = genes(genome), genes(prev)
            glow = 1 - ease(t * 1.6)
            links = [Link(g.into, g.out, g.weight, glow=glow if inn not in before else 0.0)
                     for inn, g in now_genes.items()]
            new_hidden = hidden_of(genome) - hidden_of(prev)
            hidden = {n: (float("nan"), 1.0, glow if n in new_hidden else 0.0) for n in hidden_of(genome)}
            positions = {**fixed, **layouts[i]}
            label_gen, label_x = gen, x
        else:
            nxt_gen, nxt, nxt_x = generations[i + 1]
            s = ease(t)
            a, b = genes(genome), genes(nxt)
            links = []
            for inn in a.keys() | b.keys():
                ga, gb = a.get(inn), b.get(inn)
                if ga and gb:
                    links.append(Link(gb.into, gb.out, lerp(ga.weight, gb.weight, s)))
                elif ga:
                    links.append(Link(ga.into, ga.out, ga.weight, alpha=1 - s, fade=s))
                else:
                    links.append(Link(gb.into, gb.out, gb.weight, grow=s, alpha=1.0, glow=1.0))
            ha, hb = hidden_of(genome), hidden_of(nxt)
            hidden = {}
            for n in ha | hb:
                if n in ha and n in hb:
                    hidden[n] = (float("nan"), 1.0, 0.0)
                elif n in ha:
                    hidden[n] = (float("nan"), 1 - s, 0.0)
                else:
                    hidden[n] = (float("nan"), s, 1.0)
            positions = dict(fixed)
            la, lb = layouts[i], layouts[i + 1]
            for n in ha | hb:
                pa, pb = la.get(n), lb.get(n)
                pa, pb = pa or pb, pb or pa
                positions[n] = (lerp(pa[0], pb[0], s), lerp(pa[1], pb[1], s))
            label_gen = gen if s < 0.5 else nxt_gen
            label_x = int(lerp(x, nxt_x, s))
            genome = genome if s < 0.5 else nxt
        n_hidden = len(hidden_of(genome))
        n_links = len(genes(genome))
        return Scene(positions=positions, grid_alpha=1, bias_alpha=1, controller_alpha=1, links=links,
                     hidden=hidden, used_inputs=inputs_of(genome), line_scale=0.55,
                     title=f"Generation {label_gen}",
                     caption=f"{n_hidden} hidden neuron{'s' if n_hidden != 1 else ''}   ·   {n_links} connections",
                     progress=label_x / LEVEL_END_X, progress_label=f"reached x = {label_x}")

    for i in range(len(generations)):
        n = int((hold * (1.6 if i in (0, len(generations) - 1) else 1)) * FPS)
        for k in range(n):
            writer.add(renderer.draw(scene_for(i, k / max(n - 1, 1), "hold")))
        if i + 1 < len(generations):
            n = int(morph * FPS)
            for k in range(n):
                writer.add(renderer.draw(scene_for(i, k / max(n - 1, 1), "morph")))
    writer.close()
    print(f"saved {stem}.mp4 and {stem}.mov ({writer.frames / FPS:.1f} s, {len(generations)} networks)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", choices=("anatomy", "evolution"))
    parser.add_argument("out", help="output path without extension; .mp4 and .mov are written")
    parser.add_argument("--genome", default="trained/neat-1-1.json", help="anatomy: the network to show at the end")
    parser.add_argument("--moment", type=int, default=520, help="anatomy: which frame of its 1-1 run to show")
    parser.add_argument("--genomes", default="trained/neat-1-1-generations/seed0",
                        help="evolution: a folder with gen_NNN_best.json (or genomes/) and log.csv")
    parser.add_argument("--hold", type=float, default=2.5, help="evolution: seconds on each network")
    parser.add_argument("--morph", type=float, default=1.2, help="evolution: seconds from one network to the next")
    args = parser.parse_args()
    out = Path(args.out)
    if args.what == "anatomy":
        anatomy(out, Path(args.genome), args.moment)
    else:
        evolution(out, Path(args.genomes), args.hold, args.morph)


if __name__ == "__main__":
    main()
