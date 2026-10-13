"""Drawing a NEAT network on top of the game, in the spirit of MarI/O.

This started as a port of MarI/O's ``displayGenome``: the same parts in the
same places (input grid on the left, hidden neurons in the middle, buttons
on the right), and the same way of placing hidden neurons. It is drawn
larger than MarI/O's so every neuron can show its value as a number.

Positions are in NES screen pixels (256 x 240). The screen is scaled up
``scale`` times and the network is drawn at that size (with 2x
supersampling, so circles and lines come out smooth), which keeps it sharp
in a window and in a video.

What you see:

* **Input grid** (left): the 13 x 13 cells the network sees. A white ``1``
  is a solid block, a black ``-1`` an enemy; empty cells are small dots.
  The red square in the middle is Mario.
* **Bias** (the circle at the grid's bottom-right corner): the input that is
  always 1.
* **Hidden neurons** (middle): one circle each, showing its current value,
  from black (-1) through grey (0) to white (+1).
* **Buttons** (right): one square button per output, blue while the
  network holds that button.
* **Connections**: green for a positive weight, red for a negative one,
  thicker for a stronger weight. A connection is drawn faint while the
  neuron feeding it is 0 and bright while that neuron is sending a signal.
* **Info box** (left, under the grid): whatever lines the caller passes,
  such as the level and the fitness so far.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from marlio.agents.neat.genome import MAX_NODES, Genome
from marlio.agents.neat.network import Network, sigmoid
from marlio.controller import BUTTONS
from marlio.inputs import BOX_RADIUS

# Layout, in NES pixels. The top 34 rows are left free for the game's own
# score / coins / world / time.
GRID_CENTRE = (58, 88)
CELL_SPACING = 8
INPUT_RADIUS = 3.5
BIAS_POSITION = (116, 140)
HIDDEN_START = (165, 60)
HIDDEN_MIN_X, HIDDEN_MAX_X = 131, 202
HIDDEN_MIN_Y, HIDDEN_MAX_Y = 42, 138
HIDDEN_RADIUS = 5.5
# How far MarI/O pushes a hidden neuron right of the neurons feeding it
# (MarI/O uses 40 for a 130-pixel-wide band; this band is narrower).
HIDDEN_PUSH = 22
BUTTON_X, BUTTON_WIDTH, BUTTON_HEIGHT = 216, 36, 11
BUTTON_TOP, BUTTON_SPACING = 42, 15
INFO_TOP = 146
NETWORK_BOTTOM = 150

SUPERSAMPLE = 2

POSITIVE = (40, 210, 70)
NEGATIVE = (235, 50, 40)
HELD = (30, 110, 255)
MARIO = (230, 30, 30)


def button_centre(o: int) -> tuple[float, float]:
    return BUTTON_X + BUTTON_WIDTH / 2, BUTTON_TOP + BUTTON_SPACING * o + BUTTON_HEIGHT / 2


def neuron_positions(genome: Genome, n_inputs: int, n_outputs: int) -> dict[int, tuple[float, float]]:
    """Place every neuron on the screen.

    Inputs and outputs have fixed places. Hidden neurons are placed like
    MarI/O does: they all start in one spot and are then pulled, 4 times
    over, towards the neurons they connect to, while being pushed right of
    the neurons that feed them and left of the ones they feed. Finally,
    neurons that ended up on top of each other are pushed apart so each one
    (and its number) can be read. The layout only depends on the genome, so
    it is computed once.
    """
    positions: dict[int, list[float]] = {}
    i = 0
    for dy in range(-BOX_RADIUS, BOX_RADIUS + 1):
        for dx in range(-BOX_RADIUS, BOX_RADIUS + 1):
            positions[i] = [GRID_CENTRE[0] + CELL_SPACING * dx, GRID_CENTRE[1] + CELL_SPACING * dy]
            i += 1
    positions[n_inputs - 1] = list(BIAS_POSITION)
    for o in range(n_outputs):
        positions[MAX_NODES + o] = [BUTTON_X, button_centre(o)[1]]

    def is_hidden(neuron: int) -> bool:
        return n_inputs <= neuron < MAX_NODES

    enabled = [gene for gene in genome.genes if gene.enabled]
    hidden = []
    for gene in enabled:
        for neuron in (gene.into, gene.out):
            if is_hidden(neuron) and neuron not in positions:
                positions[neuron] = list(HIDDEN_START)
                hidden.append(neuron)

    def clamp(c: list[float]) -> None:
        c[0] = min(max(c[0], HIDDEN_MIN_X), HIDDEN_MAX_X)
        c[1] = min(max(c[1], HIDDEN_MIN_Y), HIDDEN_MAX_Y)

    for _ in range(4):
        for gene in enabled:
            c1, c2 = positions[gene.into], positions[gene.out]
            if is_hidden(gene.into):
                c1[0] = 0.75 * c1[0] + 0.25 * c2[0]
                c1[1] = 0.75 * c1[1] + 0.25 * c2[1]
                if c1[0] >= c2[0]:
                    c1[0] -= HIDDEN_PUSH
                clamp(c1)
            if is_hidden(gene.out):
                c2[0] = 0.25 * c1[0] + 0.75 * c2[0]
                c2[1] = 0.25 * c1[1] + 0.75 * c2[1]
                if c1[0] >= c2[0]:
                    c2[0] += HIDDEN_PUSH
                clamp(c2)

    _spread_apart([positions[n] for n in hidden], 2 * HIDDEN_RADIUS + 1.5, clamp)
    return {neuron: (x, y) for neuron, (x, y) in positions.items()}


def _spread_apart(points: list[list[float]], min_distance: float, clamp, rounds: int = 200) -> None:
    """Nudge points apart until no two are closer than ``min_distance``."""
    rng = np.random.default_rng(0)  # only breaks ties between identical points
    for _ in range(rounds):
        moved = False
        for a in range(len(points)):
            for b in range(a + 1, len(points)):
                p, q = points[a], points[b]
                dx, dy = q[0] - p[0], q[1] - p[1]
                distance = (dx * dx + dy * dy) ** 0.5
                if distance >= min_distance:
                    continue
                if distance < 1e-6:
                    dx, dy = rng.normal(size=2)
                    distance = (dx * dx + dy * dy) ** 0.5
                push = (min_distance - distance) / 2 + 0.01
                ux, uy = dx / distance, dy / distance
                p[0] -= ux * push
                p[1] -= uy * push
                q[0] += ux * push
                q[1] += uy * push
                clamp(p)
                clamp(q)
                moved = True
        if not moved:
            return


def value_text(value: float) -> str:
    """A neuron's value in at most 3 characters: "1", "-1", ".7", "-.3", "0"."""
    if abs(value) >= 0.95:
        return "1" if value > 0 else "-1"
    if abs(value) < 0.05:
        return "0"
    text = f"{value:.1f}"
    return text.replace("0.", ".")


def _shade(value: float) -> int:
    return int(min(max((value + 1) / 2 * 256, 0), 255))


class NetworkDisplay:
    """Draws one genome's network, with its current neuron values, on a frame."""

    def __init__(self, genome: Genome, network: Network, scale: int = 3):
        self.genome = genome
        self.network = network
        self.scale = scale
        self.positions = neuron_positions(genome, network.n_inputs, network.n_outputs)
        u = scale * SUPERSAMPLE  # pixels per NES pixel while drawing the network
        self.input_font = ImageFont.load_default(size=4 * u)
        self.hidden_font = ImageFont.load_default(size=4.5 * u)
        self.button_font = ImageFont.load_default(size=6 * u)
        self.info_font = ImageFont.load_default(size=7 * scale)
        self._layer: Image.Image | None = None
        self._layer_key: tuple | None = None
        self._layer_box: tuple[int, int, int, int] = (0, 0, 0, 0)

    def draw(self, screen: np.ndarray, info: list[str] | None = None) -> np.ndarray:
        """Return the screen scaled up, with the network (and an info box) on top.

        ``screen`` is the NES screen (240 x 256 x 3). ``info`` is a few short
        lines of text for a box under the input grid.
        """
        s = self.scale
        base = Image.fromarray(screen).resize(
            (screen.shape[1] * s, screen.shape[0] * s), Image.Resampling.NEAREST
        )
        # The network only changes when it thinks (every few frames), so its
        # picture is redrawn only then, and only the part of the screen it
        # covers is blended. This keeps a live window at 60 fps.
        key = tuple(self.network.values.values())
        if key != self._layer_key:
            # Everything but the info box is above NETWORK_BOTTOM.
            big = self._draw_network((base.width * SUPERSAMPLE, NETWORK_BOTTOM * s * SUPERSAMPLE))
            layer = big.reduce(SUPERSAMPLE)
            self._layer_box = layer.getbbox()
            self._layer = layer.crop(self._layer_box)
            self._layer_key = key
        base.paste(self._layer, self._layer_box[:2], mask=self._layer)
        if info:
            self._draw_info(base, info)
        return np.asarray(base)

    def _draw_info(self, image: Image.Image, lines: list[str]) -> None:
        s = self.scale
        draw = ImageDraw.Draw(image, "RGBA")
        pad, line_height = 3 * s, 9 * s
        width = max(draw.textlength(line, font=self.info_font) for line in lines)
        x0, y0 = 4 * s, INFO_TOP * s
        draw.rounded_rectangle(
            (x0, y0, x0 + width + 2 * pad, y0 + line_height * len(lines) + pad),
            radius=2 * s, fill=(0, 0, 0, 0xB0),
        )
        for row, text in enumerate(lines):
            draw.text((x0 + pad, y0 + pad // 2 + line_height * row), text,
                      fill=(255, 255, 255, 255), font=self.info_font)

    def _draw_network(self, size: tuple[int, int]) -> Image.Image:
        u = self.scale * SUPERSAMPLE
        layer = Image.new("RGBA", size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer, "RGBA")  # blend shapes drawn over each other
        values = self.network.values
        n_inputs = self.network.n_inputs
        positions = self.positions

        def circle(x, y, r, fill, outline=(0, 0, 0, 255)):
            draw.ellipse(((x - r) * u, (y - r) * u, (x + r) * u, (y + r) * u),
                         fill=fill, outline=outline, width=max(1, u // 3))

        def label(x, y, text, font, fill):
            draw.text((x * u, y * u), text, fill=fill, font=font, anchor="mm")

        # Backdrop behind the input grid, and Mario's cell.
        cx, cy = GRID_CENTRE
        half = BOX_RADIUS * CELL_SPACING + CELL_SPACING / 2 + 1
        draw.rounded_rectangle(((cx - half) * u, (cy - half) * u, (cx + half) * u, (cy + half) * u),
                               radius=2 * u, fill=(40, 40, 40, 0x90), outline=(0, 0, 0, 255), width=u // 2)
        m = CELL_SPACING / 2
        draw.rectangle(((cx - m) * u, (cy - m) * u, (cx + m) * u, (cy + m) * u), outline=MARIO, width=u)

        # Connections first, so neurons sit on top of their ends.
        for gene in self.genome.genes:
            if not gene.enabled:
                continue
            (x1, y1), (x2, y2) = positions[gene.into], positions[gene.out]
            x1 += INPUT_RADIUS if gene.into < n_inputs else HIDDEN_RADIUS
            if gene.out < MAX_NODES:
                x2 -= HIDDEN_RADIUS
            active = values.get(gene.into, 0.0) != 0
            colour = POSITIVE if gene.weight > 0 else NEGATIVE
            width = (0.5 + 1.0 * abs(sigmoid(gene.weight))) * u * (1 if active else 0.6)
            draw.line((x1 * u, y1 * u, x2 * u, y2 * u),
                      fill=(*colour, 235 if active else 35), width=max(1, round(width)))

        # Inputs: a numbered circle for every cell that isn't empty, a dot otherwise.
        for neuron in range(n_inputs - 1):
            x, y = positions[neuron]
            value = values.get(neuron, 0.0)
            if value == 0:
                circle(x, y, 0.8, (255, 255, 255, 0x50), outline=None)
                continue
            shade = _shade(value)
            circle(x, y, INPUT_RADIUS, (shade, shade, shade, 255))
            label(x, y, value_text(value), self.input_font, (0, 0, 0) if shade > 140 else (255, 255, 255))

        # Bias and hidden neurons.
        for neuron, (x, y) in positions.items():
            if neuron < n_inputs - 1 or neuron >= MAX_NODES:
                continue
            value = values.get(neuron, 0.0)
            shade = _shade(value)
            circle(x, y, HIDDEN_RADIUS, (shade, shade, shade, 255))
            label(x, y, value_text(value), self.hidden_font, (0, 0, 0) if shade > 140 else (255, 255, 255))

        # Buttons, blue while held.
        for o, name in enumerate(BUTTONS):
            held = values.get(MAX_NODES + o, 0.0) > 0
            top = BUTTON_TOP + BUTTON_SPACING * o
            draw.rounded_rectangle(
                (BUTTON_X * u, top * u, (BUTTON_X + BUTTON_WIDTH) * u, (top + BUTTON_HEIGHT) * u),
                radius=1.5 * u, fill=HELD + (255,) if held else (60, 60, 60, 230),
                outline=(255, 255, 255, 255) if held else (0, 0, 0, 255), width=max(1, u // 2),
            )
            label(*button_centre(o), name, self.button_font, (255, 255, 255) if held else (170, 170, 170))
        return layer
