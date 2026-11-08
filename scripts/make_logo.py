"""Draw the MaRLio logo: the name written as a neural network (nodes and edges).

    python scripts/make_logo.py                  # writes runs/logo/marlio-logo.png
    python scripts/make_logo.py --out docs/images --force   # replace the saved logo

This is the recipe behind ``docs/images/marlio-logo.png``. It is drawn in code
(Pillow) so it can be rebuilt exactly or tweaked:

* **Letters**: original chunky, tall, condensed retro shapes (in the spirit of
  an NES title screen, not a copy of any real logo). Each letter is a little
  network: its strokes are edges, its corners and stroke ends are neurons
  (white rings with a coloured "signal" dot). ``GLYPHS`` holds the skeletons in
  grid units; ``TALL`` stretches them vertically and ``GAP`` spaces them.
* **Colours**: cream letters, "RL" in light blue (the reinforcement-learning
  part of the name), a dark drop shadow down and to the right.
* **Hidden links**: faint lines join random neurons of neighbouring letters
  (fixed seed, so the picture never changes).
* **Sign**: a rounded orange plate with a darker rim and four rivets, on a
  sky-blue background with white clouds.
* Everything is drawn at 3x and scaled down (``SS``) for smooth edges.

It won't overwrite an existing file unless you pass ``--force``.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw

SS = 3  # supersampling

SKY = (156, 166, 255)
PLATE = (214, 84, 30)
PLATE_EDGE = (150, 52, 14)
CREAM = (253, 222, 200)
BLUE = (110, 168, 255)
SHADOW = (24, 14, 10)
NODE_RING = (24, 14, 10)
FAINT = (176, 64, 20)

# Letter skeletons in grid units (x right, y down). Cap height 0..8, x-height starts at 3.
# Each letter: (width, strokes, lone nodes). A stroke is a chain of points.
GLYPHS = {
    "M": (5.0, [[(0, 8), (0, 0), (2.5, 4.2), (5, 0), (5, 8)]], []),
    "a": (4.0, [[(0.3, 3), (4, 3), (4, 8)], [(4, 5.4), (0, 5.4), (0, 8), (4, 8)]], []),
    "R": (4.6, [[(0, 8), (0, 0), (4.6, 0), (4.6, 4), (0, 4)], [(2.6, 4), (4.6, 8)]], []),
    "L": (4.0, [[(0, 0), (0, 8), (4, 8)]], []),
    "i": (0.0, [[(0, 3), (0, 8)]], [(0, 0.8)]),
    "o": (4.0, [[(0, 3), (4, 3), (4, 8), (0, 8), (0, 3)]], []),
}
WORD = [("M", CREAM), ("a", CREAM), ("R", BLUE), ("L", BLUE), ("i", CREAM), ("o", CREAM)]
GAP = 2.6
TALL = 1.3  # stretch letters vertically: tall, condensed retro title letters


def layout(word):
    """Place each letter; returns [(colour, strokes, lone, nodes)] in units, and the total width."""
    x, placed = 0.0, []
    for ch, colour in word:
        width, strokes, lone = GLYPHS[ch]
        s = [[(x + px, py) for px, py in stroke] for stroke in strokes]
        l = [(x + px, py) for px, py in lone]
        nodes = sorted({p for stroke in s for p in stroke} | set(l))
        placed.append((colour, s, l, nodes))
        x += width + GAP
    return placed, x - GAP


def draw_word(img, origin, unit, word=WORD, faint_net=True, seed=4):
    d = ImageDraw.Draw(img)
    ox, oy = origin
    P = lambda p: (ox + p[0] * unit, oy + p[1] * unit * TALL)  # noqa: E731
    placed, _ = layout(word)
    stroke_w = round(1.45 * unit)
    node_r = 0.58 * unit
    shadow = (0.5 * unit, 0.6 * unit)

    # Faint "hidden" connections between neighbouring letters, behind everything.
    if faint_net:
        rng = random.Random(seed)
        for (_, _, _, a), (_, _, _, b) in zip(placed, placed[1:]):
            for _ in range(3):
                p, q = rng.choice(a), rng.choice(b)
                d.line([P(p), P(q)], fill=FAINT, width=max(2, round(0.16 * unit)))

    def strokes(offset, colour_fn, width, radius_scale):
        for colour, s, lone, nodes in placed:
            c = colour_fn(colour)
            for stroke in s:
                pts = [(P(p)[0] + offset[0], P(p)[1] + offset[1]) for p in stroke]
                d.line(pts, fill=c, width=width, joint="curve")
                for x, y in pts:  # square-ish caps so stroke ends look blocky
                    h = width / 2
                    d.rectangle([x - h, y - h, x + h, y + h], fill=c)

    def nodes(offset, fill_fn, ring):
        for colour, s, lone, ns in placed:
            for p in ns:
                x, y = P(p)
                x += offset[0]
                y += offset[1]
                r = node_r
                if ring:
                    d.ellipse([x - r, y - r, x + r, y + r], fill=fill_fn(colour), outline=NODE_RING,
                              width=max(2, round(0.2 * unit)))
                else:
                    d.ellipse([x - r, y - r, x + r, y + r], fill=fill_fn(colour))

    # Shadow, then letters, then neurons on top.
    strokes(shadow, lambda c: SHADOW, stroke_w, 1)
    nodes(shadow, lambda c: SHADOW, False)
    strokes((0, 0), lambda c: c, stroke_w, 1)
    nodes((0, 0), lambda c: (255, 255, 255) if c == CREAM else (225, 238, 255), True)
    # A dark "signal" dot inside each neuron, like an active unit.
    for colour, s, lone, ns in placed:
        for p in ns:
            x, y = P(p)
            r = 0.2 * unit
            d.ellipse([x - r, y - r, x + r, y + r], fill=PLATE if colour == CREAM else (39, 99, 216))


def plate(width, height, unit_px):
    """The sign: a rounded orange plate with a darker rim and four rivets."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    rim = round(unit_px * 0.35)
    d.rounded_rectangle([0, 0, width - 1, height - 1], radius=rim * 2, fill=PLATE_EDGE)
    d.rounded_rectangle([rim // 2, rim // 2, width - 1 - rim, height - 1 - rim], radius=rim * 2, fill=PLATE)
    r = unit_px * 0.28
    for x, y in [(rim * 2.2, rim * 2.2), (width - rim * 3, rim * 2.2), (rim * 2.2, height - rim * 3), (width - rim * 3, height - rim * 3)]:
        d.ellipse([x - r, y - r, x + r, y + r], fill=SHADOW)
        d.ellipse([x - r * .45, y - r * .45 - r * .2, x + r * .1, y - r * .2 + r * .1], fill=CREAM)
    return img


def make_plate(unit: int):
    _, word_w = layout(WORD)
    pad_x, pad_y = 2.6, 2.4
    w = round((word_w + 2 * pad_x + 0.5) * unit)
    h = round((8 * TALL + 2 * pad_y + 0.6) * unit)
    img = plate(w, h, unit)
    draw_word(img, (pad_x * unit, pad_y * unit), unit)
    return img


def sky_background(w, h):
    img = Image.new("RGBA", (w, h), SKY + (255,))
    d = ImageDraw.Draw(img)
    # a few blocky clouds made of circles, in keeping with the node theme
    for cx, cy, s in [(0.12, 0.2, 1.0), (0.86, 0.14, 0.8), (0.9, 0.82, 0.7), (0.1, 0.84, 0.6)]:
        x, y, r = cx * w, cy * h, 0.035 * w * s
        for dx, dy, k in [(-1.2, .2, .8), (0, 0, 1), (1.2, .2, .8), (-.6, -.5, .7), (.6, -.5, .7)]:
            rr = r * k
            d.ellipse([x + dx * r - rr, y + dy * r - rr, x + dx * r + rr, y + dy * r + rr], fill=(255, 255, 255, 255))
        d.rectangle([x - 2 * r, y + .2 * r, x + 2 * r, y + .8 * r], fill=(255, 255, 255, 255))
    return img


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("runs/logo"))
    parser.add_argument("--unit", type=int, default=40, help="size of one grid unit in pixels")
    parser.add_argument("--force", action="store_true", help="overwrite an existing marlio-logo.png")
    args = parser.parse_args()
    target = args.out / "marlio-logo.png"
    if target.exists() and not args.force:
        raise SystemExit(f"{target} exists; pass --force to replace it")
    args.out.mkdir(parents=True, exist_ok=True)

    sign = make_plate(args.unit * SS)
    sign_small = sign.resize((sign.width // SS, sign.height // SS), Image.LANCZOS)
    w, h = round(sign_small.width * 1.5), round(sign_small.height * 2.0)
    bg = sky_background(w * SS, h * SS)
    bg.alpha_composite(sign, ((bg.width - sign.width) // 2, (bg.height - sign.height) // 2))
    bg.convert("RGB").resize((w, h), Image.LANCZOS).save(target)
    print("wrote", target)


if __name__ == "__main__":
    main()
