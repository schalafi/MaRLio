"""Drawing a NEAT network on top of the game, in the spirit of MarI/O.

This started as a port of MarI/O's ``displayGenome``: the same parts in the
same places (input grid on the left, hidden neurons in the middle, buttons
on the right), and the same way of placing hidden neurons. It is kept to
the top third of the screen so the level stays visible, and hidden neurons
show their value as a number.

Positions are in NES screen pixels (256 x 240). The screen is scaled up
``scale`` times and the network is drawn at that size (with 2x
supersampling, so circles and lines come out smooth), which keeps it sharp
in a window and in a video.

What you see:

* **Input grid** (left): the 13 x 13 cells the network sees. A white square
  is a solid block, a black square an enemy; empty cells are small dots.
  The red square in the middle is Mario.
* **Bias** (the small white circle at the grid's bottom-right corner): the
  input that is always 1.
* **Hidden neurons** (middle): one circle each, showing its current value,
  from black (-1) through grey (0) to white (+1).
* **Controller** (right): the 6 outputs drawn as an NES controller. A D-pad
  arm turns blue, and the red A or B button lights up, while the network
  holds it.
* **Connections**: only those carrying a signal right now are drawn. White
  for a positive weight, black for a negative one (the same rule as the
  neurons: white is +, black is -), thicker for a stronger weight, each with
  an outline of the opposite shade.
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

# Layout, in NES pixels. The network fits in a band across the top third of
# the screen (rows 34-112), under the game's own score / coins / world /
# time, so the lower two thirds, where Mario plays, stay clear.
GRID_CENTRE = (44, 74)
CELL_SPACING = 5.5
INPUT_HALF = 2  # input squares are 4 x 4
BIAS_POSITION = (86, 108)
BIAS_RADIUS = 2.5
HIDDEN_START = (140, 60)
HIDDEN_MIN_X, HIDDEN_MAX_X = 98, 170
HIDDEN_MIN_Y, HIDDEN_MAX_Y = 40, 108
HIDDEN_RADIUS = 4
# How far MarI/O pushes a hidden neuron right of the neurons feeding it
# (MarI/O uses 40 for a 130-pixel-wide band; this band is narrower).
HIDDEN_PUSH = 22
INFO_TOP = 116
NETWORK_BOTTOM = 114

# The outputs are drawn as an NES controller: D-pad on the left, B and A on
# the right.
CONTROLLER = (182, 54, 252, 96)  # left, top, right, bottom
DPAD_CENTRE = (199, 75)
DPAD_HALF_WIDTH = 3.5  # half the width of each arm
DPAD_REACH = 10.5  # from the centre to the end of each arm
AB_RADIUS = 6
OUTPUT_POSITIONS = {
    "A": (240, 72),
    "B": (223, 72),
    "up": (DPAD_CENTRE[0], DPAD_CENTRE[1] - 7),
    "down": (DPAD_CENTRE[0], DPAD_CENTRE[1] + 7),
    "left": (DPAD_CENTRE[0] - 7, DPAD_CENTRE[1]),
    "right": (DPAD_CENTRE[0] + 7, DPAD_CENTRE[1]),
}

SUPERSAMPLE = 2

# One colour rule for the whole picture: white means +, black means -.
# Neurons go from black (-1) to white (+1), solid blocks are white (+1) and
# enemies black (-1), and so a connection with a positive weight is white
# and one with a negative weight is black. Each line gets an outline of the
# opposite shade (dark for white lines, light for black ones) so it shows on
# sky, clouds, bricks and the dark grid alike.
POSITIVE = (255, 255, 255)
NEGATIVE = (10, 10, 10)
POSITIVE_CASING = (0, 0, 0, 190)
NEGATIVE_CASING = (255, 255, 255, 190)
HELD = (40, 140, 255)  # a pressed D-pad arm
A_B_UP = (130, 25, 25)  # NES A/B red, not pressed...
A_B_DOWN = (255, 60, 50)  # ... and pressed
MARIO = (230, 30, 30)


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
        positions[MAX_NODES + o] = list(OUTPUT_POSITIONS[BUTTONS[o]])

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
        self.hidden_font = ImageFont.load_default(size=3.4 * u)
        self.button_font = ImageFont.load_default(size=4.5 * u)
        self.info_font = ImageFont.load_default(size=5.5 * scale)
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
        pad, line_height = 2 * s, 7 * s
        width = max(draw.textlength(line, font=self.info_font) for line in lines)
        x0, y0 = 3 * s, INFO_TOP * s
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

        def box(x0, y0, x1, y1, fill, outline=None, width=0, radius=0):
            draw.rounded_rectangle((x0 * u, y0 * u, x1 * u, y1 * u), radius=radius * u,
                                   fill=fill, outline=outline, width=round(width * u))

        def circle(x, y, r, fill, outline=(0, 0, 0, 255), width=0.4):
            draw.ellipse(((x - r) * u, (y - r) * u, (x + r) * u, (y + r) * u),
                         fill=fill, outline=outline, width=max(1, round(width * u)))

        def label(x, y, text, font, fill):
            draw.text((x * u, y * u), text, fill=fill, font=font, anchor="mm")

        # Backdrop behind the input grid, Mario's cell, and the controller.
        cx, cy = GRID_CENTRE
        half = BOX_RADIUS * CELL_SPACING + CELL_SPACING / 2 + 1
        box(cx - half, cy - half, cx + half, cy + half, (70, 70, 70, 0x90), (0, 0, 0, 255), 0.4, 1.5)
        m = CELL_SPACING / 2
        box(cx - m, cy - m, cx + m, cy + m, None, MARIO, 0.7)
        left, top, right, bottom = CONTROLLER
        box(left, top, right, bottom, (200, 200, 200, 150), (60, 60, 60, 255), 0.4, 2.5)  # grey case (see-through, so Mario can be seen behind it)
        box(left + 2, top + 2, right - 2, bottom - 2, (25, 25, 25, 150), None, 0, 1.5)  # black face

        # Connections. Only the ones carrying a signal right now (the neuron
        # feeding them isn't 0) are drawn, thicker for stronger weights.
        for gene in self.genome.genes:
            if not gene.enabled or values.get(gene.into, 0.0) == 0:
                continue
            (x1, y1), (x2, y2) = positions[gene.into], positions[gene.out]
            if gene.into < n_inputs - 1:
                x1 += INPUT_HALF
            else:
                x1 += BIAS_RADIUS if gene.into == n_inputs - 1 else HIDDEN_RADIUS
            if gene.out < MAX_NODES:
                x2 -= HIDDEN_RADIUS
            positive = gene.weight > 0
            points = (x1 * u, y1 * u, x2 * u, y2 * u)
            width = (0.4 + 0.7 * abs(sigmoid(gene.weight))) * u
            draw.line(points, fill=POSITIVE_CASING if positive else NEGATIVE_CASING, width=round(width + 0.7 * u))
            draw.line(points, fill=(*(POSITIVE if positive else NEGATIVE), 255), width=max(1, round(width)))

        # Inputs: a white square for a solid block, a black one for an enemy,
        # a dot for an empty cell.
        for neuron in range(n_inputs - 1):
            x, y = positions[neuron]
            value = values.get(neuron, 0.0)
            if value == 0:
                circle(x, y, 0.6, (255, 255, 255, 0x40), outline=None)
                continue
            shade = _shade(value)
            box(x - INPUT_HALF, y - INPUT_HALF, x + INPUT_HALF, y + INPUT_HALF,
                (shade, shade, shade, 255), (0, 0, 0, 255) if shade > 128 else (255, 255, 255, 200), 0.3, 0.5)

        # Bias and hidden neurons: circles showing their value.
        for neuron, (x, y) in positions.items():
            if neuron < n_inputs - 1 or neuron >= MAX_NODES:
                continue
            value = values.get(neuron, 0.0)
            shade = _shade(value)
            if neuron == n_inputs - 1:
                circle(x, y, BIAS_RADIUS, (255, 255, 255, 255))
                continue
            circle(x, y, HIDDEN_RADIUS, (shade, shade, shade, 255))
            label(x, y, value_text(value), self.hidden_font, (0, 0, 0) if shade > 140 else (255, 255, 255))

        # The controller: D-pad arms light up blue, A and B light up red.
        held = {name: values.get(MAX_NODES + o, 0.0) > 0 for o, name in enumerate(BUTTONS)}
        dx, dy = DPAD_CENTRE
        w, r = DPAD_HALF_WIDTH, DPAD_REACH
        box(dx - w, dy - r, dx + w, dy + r, (55, 55, 55, 255), (0, 0, 0, 255), 0.5, 1)
        box(dx - r, dy - w, dx + r, dy + w, (55, 55, 55, 255), (0, 0, 0, 255), 0.5, 1)
        arms = {
            "up": (dx - w, dy - r, dx + w, dy - w),
            "down": (dx - w, dy + w, dx + w, dy + r),
            "left": (dx - r, dy - w, dx - w, dy + w),
            "right": (dx + w, dy - w, dx + r, dy + w),
        }
        for name, (x0, y0, x1, y1) in arms.items():
            if held[name]:
                box(x0 + 0.5, y0 + 0.5, x1 - 0.5, y1 - 0.5, HELD + (255,), (255, 255, 255, 255), 0.5, 1)
        circle(dx, dy, 1.4, (35, 35, 35, 255), outline=None)
        for name in ("B", "A"):
            x, y = OUTPUT_POSITIONS[name]
            pressed = held[name]
            circle(x, y, AB_RADIUS, A_B_DOWN + (255,) if pressed else A_B_UP + (255,),
                   (255, 255, 255, 255) if pressed else (0, 0, 0, 255), 0.8 if pressed else 0.5)
            label(x, y + AB_RADIUS + 4, name, self.button_font, A_B_DOWN if pressed else (170, 40, 40))
        return layer
