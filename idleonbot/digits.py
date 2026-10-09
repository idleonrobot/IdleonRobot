"""The game's small pixel-font digits (the Hoops score, the Darts wind speed)."""
import numpy as np

_GLYPHS = {
    0: ".#####. ##...## ##...## ##...## ##...## ##...## ##...## ##...## .#####.",
    1: "..##.. .###.. #.##.. ..##.. ..##.. ..##.. ..##.. ..##.. ######",
    2: ".#####. ##...## ##...## ....### ...###. ..###.. .###... ###.... #######",
    3: ".#####. ##...## ##...## .....## ...###. .....## ##...## ##...## .#####.",
    4: "##..##. ##..##. ##..##. ##..##. ####### ....##. ....##. ....##. ....##.",
    5: "####### ##..... ##..... ######. ##...## .....## ##...## ##...## .#####.",
    6: ".#####. ##...## ##....# ######. ##...## ##...## ##...## ##...## .#####.",
    7: "####### .....## .....## ....##. ....##. ...##.. ...##.. ..##... ..##...",
    8: ".#####. ##...## ##...## ##...## .#####. ##...## ##...## ##...## .#####.",
    9: ".#####. ##...## ##...## ##...## ##...## .###### #....## ##...## .#####.",
}
DIGITS = {d: np.array([[c == "#" for c in row] for row in g.split()]) for d, g in _GLYPHS.items()}


def native(img, w, h, s):
    """A grab made at scale s, sampled back to w x h native pixels."""
    ys = np.minimum((np.arange(h) + 0.5) * s, img.shape[0] - 1).astype(int)
    xs = np.minimum((np.arange(w) + 0.5) * s, img.shape[1] - 1).astype(int)
    return img[ys][:, xs]


def read_number(mask, strict=False):
    """The number written at the left of a mask of text pixels (native scale). Reading stops
    at a gap wider than the space between digits, or at a character that isn't a digit
    (strict: then the whole reading fails). None if there's no number."""
    cols = np.flatnonzero(mask.any(0))
    value, last = None, None
    for run in np.split(cols, np.flatnonzero(np.diff(cols) > 1) + 1) if len(cols) else []:
        if last is not None and run[0] - last > 2:
            break
        g = mask[:, run[0]:run[-1] + 1]
        rows = np.flatnonzero(g.any(1))
        g = g[rows[0]:rows[-1] + 1]
        best, err = None, 0.1
        for d, t in DIGITS.items():
            if t.shape == g.shape and (e := (t != g).mean()) <= err:
                best, err = d, e
        if best is None:
            if strict:
                return None
            break
        value, last = (value or 0) * 10 + best, run[-1]
    return value
