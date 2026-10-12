"""Drawing a NEAT network on top of the game, as MarI/O does.

This is a port of MarI/O's ``displayGenome`` and ``displayBanner``. All
positions below are MarI/O's, in NES screen pixels (256 x 240); the drawing
is done on a copy of the screen scaled up ``scale`` times so lines and text
stay sharp.

What you see:

* **Input grid** (top left): the 13 x 13 cells the network sees. White is a
  solid block, black an enemy; empty cells are not drawn. The red mark in
  the middle is Mario.
* **Bias** (the lone square under the grid): the input that is always 1.
* **Hidden neurons** (middle): one square each, grey shade = its value,
  from black (-1) to white (+1).
* **Outputs** (right): one square per button, with the button's name, which
  turns blue while the network holds that button.
* **Connections**: green for a positive weight, red for a negative one,
  darker for a stronger weight. A connection is drawn faint while the neuron
  feeding it is 0 and bright while that neuron is sending a signal.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from marlio.agents.neat.genome import MAX_NODES, Genome
from marlio.agents.neat.network import Network, sigmoid
from marlio.controller import BUTTONS
from marlio.inputs import BOX_RADIUS

# Where MarI/O puts things, in NES pixels.
GRID_CENTRE = (50, 70)
CELL_SPACING = 5
BIAS_POSITION = (80, 110)
OUTPUT_X = 220
HIDDEN_START = (140, 40)
HIDDEN_MIN_X, HIDDEN_MAX_X = 90, 220


def neuron_positions(genome: Genome, n_inputs: int, n_outputs: int) -> dict[int, tuple[float, float]]:
    """Place every neuron on the screen (MarI/O's layout).

    Inputs and outputs have fixed places. Every hidden neuron starts in the
    same spot and is then pulled, 4 times over, towards the neurons it is
    connected to, while being kept to the right of the neurons that feed it
    and to the left of the ones it feeds. The layout only depends on the
    genome, so it is computed once.
    """
    positions: dict[int, list[float]] = {}
    i = 0
    for dy in range(-BOX_RADIUS, BOX_RADIUS + 1):
        for dx in range(-BOX_RADIUS, BOX_RADIUS + 1):
            positions[i] = [GRID_CENTRE[0] + CELL_SPACING * dx, GRID_CENTRE[1] + CELL_SPACING * dy]
            i += 1
    positions[n_inputs - 1] = list(BIAS_POSITION)
    for o in range(n_outputs):
        positions[MAX_NODES + o] = [OUTPUT_X, 30 + 8 * (o + 1)]

    def is_hidden(neuron: int) -> bool:
        return n_inputs <= neuron < MAX_NODES

    enabled = [gene for gene in genome.genes if gene.enabled]
    for gene in enabled:
        for neuron in (gene.into, gene.out):
            if is_hidden(neuron):
                positions.setdefault(neuron, list(HIDDEN_START))

    for _ in range(4):
        for gene in enabled:
            c1, c2 = positions[gene.into], positions[gene.out]
            if is_hidden(gene.into):
                c1[0] = 0.75 * c1[0] + 0.25 * c2[0]
                if c1[0] >= c2[0]:
                    c1[0] -= 40
                c1[0] = min(max(c1[0], HIDDEN_MIN_X), HIDDEN_MAX_X)
                c1[1] = 0.75 * c1[1] + 0.25 * c2[1]
            if is_hidden(gene.out):
                c2[0] = 0.25 * c1[0] + 0.75 * c2[0]
                if c1[0] >= c2[0]:
                    c2[0] += 40
                c2[0] = min(max(c2[0], HIDDEN_MIN_X), HIDDEN_MAX_X)
                c2[1] = 0.25 * c1[1] + 0.75 * c2[1]
    return {neuron: (x, y) for neuron, (x, y) in positions.items()}


def _cell_colour(value: float) -> tuple[int, int, int, int]:
    """Grey from black (-1) to white (+1); faint when the value is 0."""
    shade = int(min(max((value + 1) / 2 * 256, 0), 255))
    return (shade, shade, shade, 0x50 if value == 0 else 0xFF)


def _connection_colour(weight: float, active: bool) -> tuple[int, int, int, int]:
    """Green for positive, red for negative weights, darker when stronger."""
    other = 0x80 - int(abs(sigmoid(weight)) * 0x80)
    alpha = 0xA0 if active else 0x20
    if weight > 0:
        return (other, 0x80, 0, alpha)
    return (0x80, other, 0, alpha)


class NetworkDisplay:
    """Draws one genome's network, with its current neuron values, on a frame."""

    def __init__(self, genome: Genome, network: Network, scale: int = 3):
        self.genome = genome
        self.network = network
        self.scale = scale
        self.positions = neuron_positions(genome, network.n_inputs, network.n_outputs)
        self.font = ImageFont.load_default(size=8 * scale)
        self.banner_font = ImageFont.load_default(size=10 * scale)
        self.line_width = max(1, round(scale / 2))
        self._layer: Image.Image | None = None
        self._layer_key: tuple | None = None
        self._layer_box: tuple[int, int, int, int] = (0, 0, 0, 0)

    def _box(self, draw: ImageDraw.ImageDraw, x0, y0, x1, y1, outline, fill) -> None:
        s = self.scale
        draw.rectangle((x0 * s, y0 * s, (x1 + 1) * s - 1, (y1 + 1) * s - 1), fill=fill, outline=outline)

    def draw(self, screen: np.ndarray, banner: list[str] | None = None) -> np.ndarray:
        """Return the screen scaled up, with the network (and a banner) on top.

        ``screen`` is the NES screen (240 x 256 x 3). ``banner`` is up to two
        lines of text for a white strip across the top of the screen, like
        MarI/O's "Gen ... species ... genome" / "Fitness: ..." banner.
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
            layer = self._draw_network(base.size)
            self._layer_box = layer.getbbox()
            self._layer = layer.crop(self._layer_box)
            self._layer_key = key
        base.paste(self._layer, self._layer_box[:2], mask=self._layer)
        if banner:
            draw = ImageDraw.Draw(base, "RGBA")
            self._box(draw, 0, 0, 255, 12 * len(banner) + 1, None, (255, 255, 255, 0xD0))
            for row, text in enumerate(banner):
                draw.text((2 * s, (1 + 12 * row) * s), text, fill=(0, 0, 0, 255), font=self.banner_font)
        return np.asarray(base)

    def _draw_network(self, size: tuple[int, int]) -> Image.Image:
        s = self.scale
        layer = Image.new("RGBA", size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer, "RGBA")  # blend shapes drawn over each other
        values = self.network.values
        n_inputs = self.network.n_inputs

        # Grey backdrop behind the input grid.
        cx, cy = GRID_CENTRE
        r = BOX_RADIUS * CELL_SPACING
        self._box(draw, cx - r - 3, cy - r - 3, cx + r + 2, cy + r + 2, (0, 0, 0, 255), (0x80, 0x80, 0x80, 0x80))

        # Neurons: every input that isn't 0, and every other neuron.
        for neuron, (x, y) in self.positions.items():
            value = values.get(neuron, 0.0)
            if neuron >= n_inputs - 1 or value != 0:
                colour = _cell_colour(value)
                self._box(draw, x - 2, y - 2, x + 2, y + 2, (0, 0, 0, colour[3]), colour)

        # Connections, from the right edge of one square to the left of the next.
        for gene in self.genome.genes:
            if not gene.enabled:
                continue
            (x1, y1), (x2, y2) = self.positions[gene.into], self.positions[gene.out]
            colour = _connection_colour(gene.weight, values.get(gene.into, 0.0) != 0)
            draw.line(((x1 + 1) * s, (y1 + 0.5) * s, (x2 - 3) * s, (y2 + 0.5) * s),
                      fill=colour, width=self.line_width)

        # Button names, blue while held.
        for o, name in enumerate(BUTTONS):
            held = values.get(MAX_NODES + o, 0.0) > 0
            colour = (0, 0, 255, 255) if held else (0, 0, 0, 255)
            draw.text(((OUTPUT_X + 3) * s, (24 + 8 * (o + 1)) * s), name, fill=colour, font=self.font)

        # Mario's place in the grid.
        self._box(draw, cx - 1, cy + 1, cx + 1, cy + 8, None, (255, 0, 0, 0x80))
        return layer
