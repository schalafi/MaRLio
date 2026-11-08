# The MaRLio logos

**The official logo is `docs/images/marlio-logo.png`.** It is drawn in code by
`scripts/make_logo.py`, which rebuilds it exactly, so it can be tweaked and
regenerated.

| File | Size | Status | What it is |
|------|------|--------|------------|
| `marlio-logo.png` | 2418 × 1264 | **Official** | The name on an orange sign, each letter drawn as a small neural network. |
| `MaRLio_Final.png` | 1920 × 1080 | Secondary idea (too complicated for the main logo) | A green circuit-board sign with small networks growing out of its top edge, bigger from left to right, to show networks evolving. It was made with an image generator, so there is no recipe and it can't be rebuilt. Keep the file as it is. |

## Rebuilding `marlio-logo.png`

```bash
uv run python scripts/make_logo.py                    # writes runs/logo/marlio-logo.png
uv run python scripts/make_logo.py --out docs/images --force   # replace the saved logo
```

The default output is `runs/logo/`, and the script won't overwrite an existing
file without `--force`, so the copy in `docs/images/` is safe. With the default
options the output is byte-for-byte identical to `docs/images/marlio-logo.png`
(checked with `cmp`). The only dependency is Pillow, which the project already
uses.

## The recipe

**Idea.** The style is a retro game title screen with an original lettering, not a copy of
any real logo. Every letter is also a neural network, like the ones NEAT
evolves in this project: the strokes are connections and the corners and
stroke ends are neurons.

**Letters.** Each letter is a skeleton of points on a grid (`GLYPHS` in the
script). The cap height is 8 units and lowercase letters start at 3:

| Letter | Strokes (chains of points) |
|--------|----------------------------|
| M | (0,8) (0,0) (2.5,4.2) (5,0) (5,8) |
| a | (0.3,3) (4,3) (4,8); and (4,5.4) (0,5.4) (0,8) (4,8) |
| R | (0,8) (0,0) (4.6,0) (4.6,4) (0,4); and the leg (2.6,4) (4.6,8) |
| L | (0,0) (0,8) (4,8) |
| i | (0,3) (0,8), plus a lone neuron at (0,0.8) for the dot |
| o | the rectangle (0,3) (4,3) (4,8) (0,8) |

- The letters are stretched 1.3× vertically (`TALL`) to make them tall and condensed, with 2.6 units between letters (`GAP`).
- One unit is 40 px (`--unit`). Strokes are 1.45 units thick, with square caps so their ends look blocky.
- Each neuron is a circle of radius 0.58 units with a dark ring (0.2 units). Inside it is a white fill, or pale blue in "RL", and a coloured "signal" dot of radius 0.2 units.

**Shadow.** Letters and neurons are first drawn in near-black, offset by
(0.5, 0.6) units down and to the right, then drawn again in colour on top.

**Hidden connections.** Faint lines (0.16 units) join three random pairs of
neurons between each pair of neighbouring letters, behind everything. The
random seed is fixed (4), so they're always the same.

**Colours.**

| Part | Hex |
|------|-----|
| Sky background | `#9CA6FF` |
| Sign | `#D6541E` |
| Sign rim | `#96340E` |
| Letters | `#FDDEC8` (cream) |
| "RL" letters | `#6EA8FF` (light blue: the reinforcement-learning part of the name) |
| Shadow, neuron rings, rivets | `#180E0A` |
| Faint hidden connections | `#B04014` |
| Neuron fill | `#FFFFFF`; `#E1EEFF` in "RL" |
| Signal dots | `#D6541E`; `#2763D8` in "RL" |

**Sign.**
- A rounded rectangle with the rim colour, and an inner plate inset by 0.35 units. The corner radius is 0.7 units.
- Four round rivets, near-black with a small cream highlight.
- Padding around the word is 2.6 units at the sides and 2.4 units at the top and bottom.

**Background.**
- Sky blue, 1.5× the sign's width and 2× its height, with the sign centred.
- Four white clouds, each made of overlapping circles on a flat base.

**Smoothing.** Everything is drawn at 3× size and scaled down with Lanczos
(`SS = 3`), so edges are smooth.
