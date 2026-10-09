"""Fishing minigame bot.

Holding the mouse fills the power meter; releasing casts the bobber, which lands on
the water bar further out the longer it was held. Landing on a fish scores (green 1,
eel 2, squid 3, whale 5) - even if the fish is over a mine; landing on a mine ends the
game; landing in empty water just misses (the game goes on, but the run of landings
that brings the better fish starts again). From the 7th cast or so the fish drift
back and forth, and from about the 17th the mines do too. Several fish can be out at
once (green, plus eels, squid and whales as the run of landings grows) and overlap.

The Megalodon: catch two whales in a row (no misses), then leave whales alone and keep
catching other fish until it turns up. A miss starts the run of whales again. Its
colours aren't those of its wiki icon, so while waiting for it anything big and
unrecognised is taken for it. The bot stops once it's caught (--keep-going to carry on).

The bot waits for a fresh round (bobber gone), watches the fish, then picks the cast
(when, and where on the fish) with the best odds: the chance of catching it times its
value, against the chance of missing and, heavily, of landing on a mine.

    python -m idleonbot.fishing          # F10 to stop
"""
import argparse
import math
import time

import numpy as np

NATIVE_W = 960
# Everything's found relative to the left end of the dark bottom edge of the water
# bar (native (328, 299)).
ANCHOR_COLOR = (19, 38, 74)
SEARCH = (100, 150, 860, 470)            # native box to look for the bar in
STRIP = (-48, -74, 312, 13)              # grabbed box, relative to the anchor
METER_X, METER_Y = -35, (-63, 5)         # meter column and rows, relative to the anchor
METER_FILL = (255, 112, 119)
CAST_POINT = (198, -28)                  # where to click (the sky above the bar)
BOB_COLOR = (252, 80, 63)
MINE_COLORS = [(240, 66, 66), (170, 150, 152)]
MINE_ALL = MINE_COLORS + [(125, 93, 93), (51, 23, 23), (204, 187, 189)]
BOB_ALL = [BOB_COLOR, (229, 40, 23), (102, 15, 7), (243, 245, 190), (255, 255, 255), (255, 148, 135),
           (80, 81, 55)]
# Each fish's two main colours.
# Anything else on the bar that isn't background counts as a fish of unknown kind.
FISH_KINDS = {
    "green": [(176, 255, 219), (71, 164, 128)],
    "eel": [(234, 213, 122), (145, 111, 61)],
    "squid": [(237, 204, 240), (147, 89, 161)],
    "whale": [(188, 224, 251), (100, 107, 145)],
    "megalodon": [(255, 202, 174), (198, 113, 91)],
}
FISH_COLORS = [c for cs in FISH_KINDS.values() for c in cs]
FISH_OUTLINES = [(12, 59, 40), (40, 26, 12), (49, 24, 60), (24, 34, 55), (42, 10, 7)]   # (a sprite's
                                         # third colour; known fish shouldn't also show as unknown)
OBJ_ROWS = (-41, 2)                      # rows (relative to the anchor) fish and mines sit in
BELOW_ROWS = (4, 13)                     # plain water under the bar: pets hanging down show here

# Models (native px, s).
# Landing x from the bot's hold (s): x = a + b m + c m^2 with m = MK (hold - HOLD_BIAS + MH0)^2.
# Fitted on 210 casts: 1-3 px rms everywhere.
MK, MH0 = 75.1271, -0.04052
XM = (352.847, 4.61328, 0.00906)         # x = a + b m + c m^2
METER_TO_X = (346.59, 2.4403, 0.02609)   # landing x from the meter's height in px (exact, ~1 px)
FLIGHT = (0.001981, -0.022721)           # release -> landing = FLIGHT[0] x + FLIGHT[1]
X_MIN, X_MAX = 352.0, 590.0              # castable range (a tap lands ~347, full ~598; near full
                                         # one game frame of holding moves it ~7 px)
HOLD_BIAS = 0.03646
FISH_W = 1.135                           # rad/s of every moving fish (and mine) - the same all game
FISH_AMP = 13.0                          # their swing (px either side) at first; it grows during a
                                         # game (13 until ~2 min in, then 15, 16, 19, 20 by 2.5 min)
                                         # and every fish in a round has the same - so it's learned
                                         # as the game goes (Track.amp), not fitted per fish
FISH_PX = {"green": 17, "eel": 19, "squid": 26, "whale": 35, "megalodon": 38, "fish": 17}
                                         # widths (in their two main colours). A landing within
                                         # half that + 1.5 px of a fish's centre catches it, even
                                         # over a mine
MINE_PAD = 4.0                           # a landing within a mine's edges + this sets it off
MINE_PX = 29                             # a mine's width; from ~the 17th cast they drift like the
                                         # fish
LAND_SD = 2.5                            # landing scatter (px), plus a frame of hold's worth
MAX_BOOM = 0.002                         # never take more than this chance of a mine...
MAX_RISKY = 0.02                         # ...unless nothing's been safe for a long while
BOOM_COST, MISS_COST = 300.0, 2.0        # in fish points
VALUES = {"green": 1, "eel": 2, "squid": 3, "whale": 5, "megalodon": 50}
# ("fish" = something unrecognised on the bar: all five real kinds are known by colour, so
# it's ignored for planning - it's only logged)
STATIC_PX = 0.6                          # fish that moved less than this while watched is still
WATCH_S = 0.9                            # watch a fish this long before casting at it
WATCH_STILL_S = 1.6                      # ... to call it still once fish have been moving (near
                                         # the end of its swing a moving fish hardly moves)
HORIZON = 6.0                            # look this far ahead for a good moment
REPLAN_S = 0.1                           # plan this often (it takes ~30 ms); the cast itself is
                                         # still timed to the ms
GIVE_UP_S = 30.0                         # after this with nothing good, take more risk / cast into
                                         # empty water rather than wait for ever


def hit_r(kind):
    """How close to a fish's centre a landing must be to catch it."""
    return FISH_PX.get(kind, 17) / 2 + 1.5


def catch_needed(waited):
    """How sure of a catch it wants before casting: picky while there's time (fish turn
    and drift away from mines, so better moments come), less so later."""
    return 0.9 if waited < 10 else 0.75 if waited < 20 else 0.6 if waited < GIVE_UP_S else 0.0


def match(img, colors, tol=2):
    m = np.zeros(img.shape[:2], bool)
    for c in colors:
        m |= (np.abs(img.astype(np.int16) - c) <= tol).all(-1)
    return m


def keys_of(img):
    """Each pixel's colour as one int (r << 16 | g << 8 | b): exact colour matches are then
    one comparison instead of three, and an image is converted once for all colours."""
    i = img.astype(np.int32)
    return (i[..., 0] << 16) | (i[..., 1] << 8) | i[..., 2]


def pack(colors):
    return np.array([(r << 16) | (g << 8) | b for r, g, b in colors], np.int32)


def has(keys, packed):
    return np.isin(keys, packed)


def runs(cols, gap=5):
    """Groups of nearby column indices -> [(first, last)]."""
    cols = np.unique(cols)
    out = []
    for c in cols:
        if out and c - out[-1][1] <= gap:
            out[-1][1] = c
        else:
            out.append([c, c])
    return [tuple(map(int, r)) for r in out]


def hold_for(x):
    """How long to hold (s) to land the bobber at native x (works on arrays)."""
    a, b, c = XM
    x = np.clip(x, X_MIN, X_MAX)
    m = (-b + np.sqrt(b * b - 4 * c * (a - x))) / (2 * c)
    return np.sqrt(np.maximum(m, 0) / MK) - MH0 + HOLD_BIAS


def landing_sd(x):
    """Scatter of a landing aimed at x: a base, plus how far a 1/60 s frame of holding moves
    it (the game counts the hold in whole frames)."""
    a, b, c = XM
    m = (-b + np.sqrt(b * b - 4 * c * (a - np.clip(x, X_MIN, X_MAX)))) / (2 * c)
    slope = (b + 2 * c * m) * 2 * np.sqrt(MK * np.maximum(m, 0.1))      # px per s of hold
    return np.sqrt(LAND_SD ** 2 + (slope * 0.0048) ** 2)


def kind_of(color):
    """Name an unrecognised fish by the nearest known colour (if it's close)."""
    best = min(((sum((a - b) ** 2 for a, b in zip(color, c)), k) for k, cs in FISH_KINDS.items() for c in cs))
    return best[1] if best[0] < 40 ** 2 else "fish"


def _phi(z):
    """Standard normal CDF (Abramowitz-Stegun 7.1.26, |error| < 1.5e-7)."""
    x = np.abs(z) / math.sqrt(2)
    t = 1 / (1 + 0.3275911 * x)
    y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
             + 0.254829592) * t * np.exp(-x * x)
    return 0.5 * (1 + np.sign(z) * y)


def p_between(lo, hi, mu, sd):
    return np.clip(_phi((hi - mu) / sd) - _phi((lo - mu) / sd), 0, 1)


def flight(x):
    return FLIGHT[0] * x + FLIGHT[1]


class Track:
    """One fish's (or mine's) centre over time; predicts it with a fixed-frequency sine of
    the game's current swing (Track.amp)."""

    amp = FISH_AMP                       # the swing now (learned during a game: learn_swing)
    overlap_t = -1e9                     # (mines) when last seen overlapping another one

    def __init__(self, x, t, kind, size):
        self.pts = [(t, x)]
        self.kinds = {kind: 1}
        self.size = size
        self.fit = None
        self.lo = self.hi = x
        self.rms = 0.0

    def add(self, t, x, kind):
        self.pts.append((t, x))
        self.kinds[kind] = self.kinds.get(kind, 0) + 1
        self.fit = None
        self.lo, self.hi = min(self.lo, x), max(self.hi, x)

    @property
    def kind(self):
        return max(self.kinds, key=self.kinds.get)

    def span(self):
        return self.pts[-1][0] - self.pts[0][0]

    def moving(self):
        return self.hi - self.lo > STATIC_PX

    def _fit(self):
        if self.fit is None:
            # the last few seconds are plenty (and a long wait would make the fit slow)
            p = np.array(self.pts[-600:])
            p = p[p[:, 0] >= p[-1, 0] - 4.0]
            if len(p) > 120:
                p = p[np.linspace(0, len(p) - 1, 120).astype(int)]
            t0 = p[-1, 0]
            if not self.moving():
                self.fit = (t0, float(p[:, 1].mean()), 0.0, 0.0, FISH_W)
            else:
                # x = c + amp sin(w t + phi): only where it is in its swing is fitted
                t = p[:, 0] - t0
                amp = Track.amp
                phis = np.linspace(0, 2 * np.pi, 120, endpoint=False)
                curve = amp * np.sin(FISH_W * t[None, :] + phis[:, None])              # (P, N)
                c = (p[None, :, 1] - curve).mean(-1)
                err = ((c[:, None] + curve - p[None, :, 1]) ** 2).mean(-1)
                j = int(np.argmin(err))
                # c + A sin(w t + phi) = c + (A cos phi) sin(w t) + (A sin phi) cos(w t)
                self.fit = (t0, float(c[j]), amp * math.cos(phis[j]), amp * math.sin(phis[j]), FISH_W)
                self.rms = float(math.sqrt(err[j]))
        return self.fit

    def sd(self, t):
        """How unsure its position at time t is (px)."""
        if not self.moving():
            return 1.0 + 0 * np.asarray(t, float)
        self._fit()
        return 2.0 + self.rms + 0.15 * np.abs(self.speed(t))

    def at(self, t):
        t0, c, a, b, w = self._fit()
        u = w * (np.asarray(t) - t0)
        return c + a * np.sin(u) + b * np.cos(u)

    def speed(self, t):
        t0, c, a, b, w = self._fit()
        u = w * (np.asarray(t) - t0)
        return w * (a * np.cos(u) - b * np.sin(u))

    def swing_seen(self):
        """Its own swing, fitted freely (fixed speed) - only once it's been watched long
        enough to tell (half a cycle); else None."""
        p = np.array(self.pts[-900:])
        p = p[p[:, 0] >= p[-1, 0] - 6.0]
        if len(p) < 20 or p[-1, 0] - p[0, 0] < 2.6 or np.ptp(p[:, 1]) < 8:
            return None
        if len(p) > 150:
            p = p[np.linspace(0, len(p) - 1, 150).astype(int)]
        t = p[:, 0] - p[-1, 0]
        A = np.c_[np.ones_like(t), np.sin(FISH_W * t), np.cos(FISH_W * t)]
        c = np.linalg.lstsq(A, p[:, 1], rcond=None)[0]
        if np.sqrt(np.mean((A @ c - p[:, 1]) ** 2)) > 1.2:
            return None                          # (hidden part of the time, or not a clean swing)
        return float(math.hypot(c[1], c[2]))


class Background:
    """Learns the plain strip (sky, water, bar) from snapshots at the start of rounds, so
    fish of any colour show up as differences. Until there are enough snapshots it goes
    by the colours of the first one away from the (green) fish and mines: eels already
    come in the 4th round."""

    def __init__(self):
        self.shots = []
        self.bg = None
        self.palette = None

    def add(self, img, objects=()):
        """objects: native x ranges (in img columns) with things on them."""
        keep = np.ones(img.shape[1], bool)
        for x0, x1 in objects:
            keep[max(0, int(x0)):max(0, int(x1))] = False
        cols = keys_of(img)[:, keep]
        pal = np.unique(cols)
        self.palette = pal if self.palette is None else np.union1d(self.palette, pal)
        self.shots = (self.shots + [img.copy()])[-9:]
        if len(self.shots) >= 4:
            key = keys_of(np.stack(self.shots))
            ks = np.sort(key, axis=0)
            # value held by the middle snapshot, and how many agree with it
            mid = ks[len(ks) // 2]
            agree = (key == mid[None]).sum(0)
            self.bg = np.where(agree >= max(3, len(ks) // 2 + 1), mid, -1)

    def diff(self, key):
        """key: keys_of(the strip image)."""
        if self.bg is None:
            return None if self.palette is None else ~np.isin(key, self.palette)
        return (self.bg >= 0) & (key != self.bg)


P_METER, P_BOB, P_MINE, P_MINE_ALL = pack([METER_FILL]), pack([BOB_COLOR]), pack(MINE_COLORS), pack(MINE_ALL)
P_KINDS = {kind: pack(colors) for kind, colors in FISH_KINDS.items()}
P_NOT_UNKNOWN = pack(MINE_ALL + BOB_ALL + FISH_OUTLINES + FISH_COLORS)   # (for the unknown-fish check)


def read_strip(img, bg=None):
    """Parse a native-scale strip image (grabbed at STRIP relative to the anchor)."""
    ox, oy = -STRIP[0], -STRIP[1]            # anchor position inside img
    key = keys_of(img)
    meter = int(has(key[oy + METER_Y[0]:oy + METER_Y[1], ox + METER_X], P_METER).sum())
    r0, r1 = oy + OBJ_ROWS[0], oy + OBJ_ROWS[1]
    band = img[r0:r1, ox:]
    kb = key[r0:r1, ox:]
    ys, xs = np.nonzero(has(key, P_BOB))
    bob = (float(xs.mean()) - ox, float(ys.mean()) - oy) if len(xs) >= 8 else None
    mm = has(kb, P_MINE)
    masks = {kind: has(kb, pk) for kind, pk in P_KINDS.items()}
    fm = np.zeros(band.shape[:2], bool)
    for km in masks.values():
        fm |= km
    mine_cols = has(kb, P_MINE_ALL).any(0)
    mines, overlaps = [], []
    fish_cols = fm.any(0)
    for a, b in runs(np.nonzero(mm)[1]):
        if mm[:, a:b + 1].sum() < 20:
            continue
        w = b - a + 1
        if w >= MINE_PX + 3:             # two (or more) overlapping: one at each end (a mine is
            n = max(2, int(round(w / MINE_PX)))   # 29-31 px; nothing makes one look wider)
            overlaps.append((a + b) / 2)
            mines += [(a + (MINE_PX - 1) / 2 + k * (w - MINE_PX) / max(n - 1, 1), MINE_PX) for k in range(n)]
        elif w < MINE_PX - 3:            # partly behind a fish: place it by the end that shows
            left, right = fish_cols[max(0, a - 4):a].any(), fish_cols[b + 1:b + 5].any()
            if left and not right:
                mines.append((b - (MINE_PX - 1) / 2, MINE_PX))
            elif right and not left:
                mines.append((a + (MINE_PX - 1) / 2, MINE_PX))
            else:
                mines.append(((a + b) / 2, MINE_PX + 8))     # unsure where its middle is: wider
        else:
            mines.append(((a + b) / 2, w))
    fish = []
    for kind, km in masks.items():
        w = FISH_PX[kind]
        groups = []                      # pieces of one fish (another may cover its middle)
        for a, b in runs(np.nonzero(km.any(0))[0], gap=1):
            if groups and b - groups[-1][0] + 1 <= w + 1:
                groups[-1][1] = b
            else:
                groups.append([a, b])
        others = (fm & ~km).any(0) | mine_cols
        for a, b in groups:
            n = int(km[:, a:b + 1].sum())
            if n < 10:
                continue
            x, info = (a + b) / 2, None
            if b - a + 1 < w - 2:        # partly hidden: place it by the end that shows
                left, right = others[max(0, a - 4):a].any(), others[b + 1:b + 5].any()
                if left and not right:
                    x = b - (w - 1) / 2
                elif right and not left:
                    x = a + (w - 1) / 2
                else:
                    info = "unsure"
            fish.append((x, kind, n, info))
    other = []
    d = bg.diff(key) if bg is not None else None
    if d is not None:
        below = d[oy + BELOW_ROWS[0]:oy + BELOW_ROWS[1], ox:].any(0)
        # (mines and the bobber are matched by their own colours, so a fish over a mine or
        # under the bobber still shows)
        db = d[r0:r1, ox:] & ~has(kb, P_NOT_UNKNOWN)
        for a, b in runs(np.nonzero(db.sum(0) >= 2)[0], gap=3):
            n = int(db[:, a:b + 1].sum())
            if not (8 <= b - a + 1 <= 70 and n >= 25) or below[a:b + 1].mean() > 0.2:
                continue                 # too small, or a pet hanging over the bar
            x = (a + b) / 2
            if x < 17:
                continue                 # by the rod (its line is steep there); can't cast that short
            if any(abs(x - f) < 12 for f, _, _, _ in fish) or db[:, a:b + 1].sum(0).max() <= 3:
                continue                 # a known fish's edge, or the fishing line (thin)
            px = band[:, a:b + 1][db[:, a:b + 1]]
            vals, counts = np.unique(px.reshape(-1, 3), axis=0, return_counts=True)
            main = tuple(int(v) for v in vals[counts.argmax()])
            if bg.palette is not None and np.isin(pack([main]), bg.palette)[0]:
                continue                 # mostly background colours: not a thing
            other.append((x, kind_of(main), n, ("colour", main, b - a + 1)))
    return {"meter": meter, "bob": bob, "mines": mines, "mine_overlaps": overlaps, "fish": fish + other}


def find_anchor(grab, cl, ct, s):
    """Screen position of the bar's anchor pixel, or None."""
    x0, y0, x1, y1 = SEARCH
    img = grab(cl + int(x0 * s), ct + int(y0 * s), int((x1 - x0) * s), int((y1 - y0) * s))
    m = match(img, [ANCHOR_COLOR], tol=3)
    counts = m.sum(1)
    rows = np.flatnonzero(counts > 250 * s)
    if not len(rows):
        return None
    r = rows[0]
    xs = np.flatnonzero(m[r])
    return cl + int(x0 * s) + int(xs.min()), ct + int(y0 * s) + int(r)


def mine_zones(mines, tl, mines_move):
    """Each mine's predicted centre at the landing times tl, and how far either side of it
    a landing sets it off (wider when it's moving or hasn't been watched long)."""
    out = []
    zero = 0 * np.asarray(tl, float)
    for tr in mines:
        w = tr.size
        if tr.span() >= WATCH_S and (tr.moving() or not mines_move or tr.span() >= WATCH_STILL_S):
            c = tr.at(tl) + zero
            half = w / 2 + MINE_PAD + (tr.sd(tl) if tr.moving() else 0) + zero
        else:                            # not watched long: anywhere it could swing to
            c = tr.pts[-1][1] + zero
            half = w / 2 + MINE_PAD + (2 * Track.amp + 2 if mines_move else 2) + zero
        out.append((c, half))
    return out


def learn_swing(tracks):
    """Update the game's swing from fish watched long enough (it only grows)."""
    seen = [a for a in (tr.swing_seen() for tr in tracks) if a is not None and 10 < a < 60]
    if seen:
        new = float(np.median(seen))
        if new > Track.amp + 0.3:
            Track.amp = new
            for tr in tracks:
                tr.fit = None
            return True
    return False


def match_tracks(tracks, dets, now, gate):
    """Pair sightings (x, size, kind) with tracks by where each track expects its thing to be
    now (nearest first, one each, same kind); unmatched sightings get None (a new track)."""
    pred = [float(tr.at(now)) if len(tr.pts) >= 5 else tr.pts[-1][1] for tr in tracks]
    pairs = sorted((abs(d[0] - pred[j]), i, j) for i, d in enumerate(dets) for j, tr in enumerate(tracks)
                   if tr.kind == d[2] and abs(d[0] - pred[j]) < gate)
    used_d, used_t, out = set(), set(), []
    for _, i, j in pairs:
        if i not in used_d and j not in used_t:
            used_d.add(i), used_t.add(j)
            out.append((tracks[j], dets[i]))
    out += [(None, d) for i, d in enumerate(dets) if i not in used_d]
    return out


def plan_cast(tracks, mines, now, fish_move=False, values=VALUES, horizon=HORIZON, max_boom=MAX_BOOM,
              mines_move=False):
    """The best cast over the next `horizon` s, scoring each landing spot by what it'd
    catch: where fish overlap, the lower value (we can't tell which the game gives);
    a mine unless the landing is well on a fish; else a miss. mines: Tracks (they move
    late on). Fish not watched long enough aren't targets. Returns a dict
    (t_press, hold, aim, p_catch, p_boom, target, fish_x, all_at_land) or None."""
    tracks = [tr for tr in tracks if tr.kind in values]
    def ready(tr):
        return (tr.span() >= (WATCH_STILL_S if fish_move and not tr.moving() else WATCH_S)
                and not (tr.moving() and tr.rms > 4))       # (a fit that makes no sense yet)
    tracks = [tr for tr in tracks if ready(tr)]
    if not tracks:
        return None
    moving = any(tr.moving() for tr in tracks) or (mines_move and mines)

    def score(dts, aims, step=2.0):
        """EV, mine risk and catch chance for pressing at now + dts (D) aiming at aims (A)."""
        e = np.arange(-16.0, 17.0, step)
        hold, fl, sdl = hold_for(aims), flight(aims), landing_sd(aims)
        w = np.exp(-0.5 * (e[None, :] / sdl[:, None]) ** 2)
        w /= w.sum(1, keepdims=True)                                  # (A, E)
        land = (aims[:, None] + e[None, :])[None]                     # (1, A, E)
        tl = now + dts[:, None] + (hold + fl)[None, :]                # (D, A) landing times
        val = np.full(tl.shape + e.shape, np.inf)
        miss_all = np.ones(tl.shape + e.shape)
        for tr in tracks:
            d = np.abs(land - tr.at(tl)[..., None])
            pc = _phi((hit_r(tr.kind) - d) / tr.sd(tl)[..., None])
            val = np.where(pc > 0.05, np.minimum(val, values.get(tr.kind, 1)), val)
            miss_all *= 1 - pc
        p_any = 1 - miss_all
        val = np.where(np.isfinite(val), val, 0)
        zone = np.zeros(val.shape, bool)
        for c, half in mine_zones(mines, tl, mines_move):
            zone |= np.abs(land - c[..., None]) <= half[..., None]
        boom = miss_all * zone           # no fish there (or we miss it) and a mine is
        out = p_any * val - boom * BOOM_COST - miss_all * ~zone * MISS_COST
        ev = (w[None] * out).sum(-1)
        pb = (w[None] * boom).sum(-1)
        pcatch = (w[None] * p_any).sum(-1)
        ev = np.where(pb <= max_boom, ev - 0.05 * dts[:, None], -np.inf)
        return ev, pb, pcatch, tl, hold

    # coarse over the next few seconds, then fine around the best
    dts = np.arange(0.03, horizon, 0.15) if moving else np.array([0.03])
    ev, pb, pc, tl, hold = score(dts, np.arange(X_MIN, X_MAX + 0.01, 5.0), step=4.0)
    i, j = np.unravel_index(np.argmax(ev), ev.shape)
    if not np.isfinite(ev[i, j]):
        return None
    # The earliest moment that's nearly as good: always holding out for the very best one
    # in view meant a slightly better one kept turning up further on, and it never cast.
    best_per_dt = ev.max(1)
    i = int(np.flatnonzero(best_per_dt >= ev[i, j] - (0.05 + 0.05 * abs(ev[i, j])))[0])
    j = int(np.argmax(ev[i]))
    dt0, aim0 = dts[i], X_MIN + 5.0 * j
    dts = np.unique(np.clip(dt0 + np.arange(-0.12, 0.121, 0.03), 0.03, None)) if moving else dts
    aims = np.clip(aim0 + np.arange(-5.0, 5.01, 1.0), X_MIN, X_MAX)
    ev, pb, pc, tl, hold = score(dts, aims)
    i, j = np.unravel_index(np.argmax(ev), ev.shape)
    if not np.isfinite(ev[i, j]):
        return None
    t_land = float(tl[i, j])
    fx = [float(tr.at(t_land)) for tr in tracks]
    k = int(np.argmin([abs(x - aims[j]) for x in fx]))
    return {"t_press": now + float(dts[i]), "hold": float(hold[j]), "aim": float(aims[j]),
            "p_catch": float(pc[i, j]), "p_boom": float(pb[i, j]), "target": tracks[k], "fish_x": fx[k],
            "all_at_land": [(tr.kind, x) for tr, x in zip(tracks, fx)]}


def plan_miss(tracks, mines, now, horizon=3.0, max_boom=MAX_BOOM / 2, mines_move=False):
    """A cast into empty water (when nothing's catchable): (t_press, hold, aim_x) or None."""
    aims = np.arange(X_MIN, X_MAX, 2.0)
    for dt in np.arange(0.03, horizon, 0.1):
        tp = now + dt
        hold = hold_for(aims)
        tl = tp + hold + flight(aims)
        sd = landing_sd(aims)
        p_fish = 0 * aims
        for tr in tracks:
            if tr.kind not in VALUES:
                continue
            fx = tr.at(tl)
            r = hit_r(tr.kind)
            p_fish += p_between(fx - r - 4, fx + r + 4, aims, sd + 2)
        p_boom = 0 * aims
        for c, half in mine_zones(mines, tl, mines_move):
            p_boom += p_between(c - half, c + half, aims, sd)
        risk = p_fish * 10 + p_boom * BOOM_COST
        k = int(np.argmin(risk))
        if p_fish[k] < 0.01 and p_boom[k] < max_boom:
            return float(tp), float(hold[k]), float(aims[k])
    return None


def run_bot(args):
    from . import inputs
    from .capture import ScreenGrabber
    from .hotkeys import VK_F10, HotkeyPoller
    from .window import GameWindow

    win = GameWindow()
    hk = HotkeyPoller(VK_F10)

    print("Fishing bot running: start a fishing game (F10 stops it).")
    print("(going for the Megalodon: after two whales in a row it leaves whales alone and catches "
          "other fish until it turns up)")
    anchor, anchor_checked = None, 0.0
    bg = Background()
    state = "idle"                  # idle -> watching (a round) -> cast -> idle
    tracks, mines, t_round = [], [], None    # (fish and mine Tracks this round)
    mines_move = False              # mines have started drifting this game
    planned = None                  # (when, how many fish, the plan)
    swing_checked = 0.0
    Track.amp = FISH_AMP
    unknown_told, unknown_round = 0, None   # (say so when something unrecognised turns up)
    told_wait = 0                   # last "still waiting" message (s into the round)
    cast = None
    no_bob_since = None
    score = 0
    fish_move = False               # fish have started moving this game
    whales = 0                      # whales caught with no miss in between
    trophy = False                  # caught the Megalodon
    seen_kinds = set()

    def new_game():
        nonlocal score, fish_move, whales, mines_move
        score, fish_move, whales, mines_move = 0, False, 0, False
        Track.amp = FISH_AMP

    def finish_cast():
        """Book the last cast's outcome."""
        nonlocal cast, score, whales, trophy
        if cast is None:
            return
        if cast.get("caught"):
            score += VALUES.get(cast["caught"], 1)
            if cast["caught"] == "whale":
                whales += 1
                if whales == 2:
                    print("  (two whales in a row: now it leaves whales alone and keeps catching other "
                          "fish until the Megalodon turns up)")
            if cast["caught"] == "megalodon":
                print("  *** caught the Megalodon! ***")
                whales, trophy = 0, True
        elif "caught" in cast:
            if whales >= 2:
                print("  (a miss: the run of whales starts again)")
            whales = 0                   # a miss breaks the run of whales
        cast = None

    with ScreenGrabber() as grabber:
        try:
            while True:
                now = time.perf_counter()
                if hk.pressed(VK_F10):
                    break
                if trophy and not args.keep_going:
                    print("That's the Megalodon trophy - stopping.")
                    print("If this trophy is worth as much as a supporter pack to you, please consider tipping here! Ko-fi.com/idlerobot")
                    break
                if not win.is_foreground() or win.is_minimized():
                    time.sleep(0.03)
                    continue
                cl, ct, cw, ch = win.client_rect()
                s = cw / NATIVE_W
                if anchor is None:
                    if now - anchor_checked < 0.5:
                        time.sleep(0.02)
                        continue
                    anchor_checked = now
                    anchor = find_anchor(grabber.grab, cl, ct, s)
                    if anchor is None:
                        if state == "cast" or score:
                            finish_cast()
                            print(f"Game over? About {score} points this game." if score else "Game over?")
                            new_game()
                        state = "idle"
                        continue
                ax, ay = anchor
                raw = grabber.grab(ax + int(STRIP[0] * s), ay + int(STRIP[1] * s),
                                   int((STRIP[2] - STRIP[0]) * s), int((STRIP[3] - STRIP[1]) * s))
                if s != 1:
                    h, w = STRIP[3] - STRIP[1], STRIP[2] - STRIP[0]
                    raw = raw[(np.arange(h) * s).astype(int)][:, (np.arange(w) * s).astype(int)]
                img = np.ascontiguousarray(raw)
                # still the bar?
                if match(img[-STRIP[1]:-STRIP[1] + 1, -STRIP[0]:-STRIP[0] + 250], [ANCHOR_COLOR], 3).mean() < 0.6:
                    anchor = None
                    continue
                seen = read_strip(img, bg)

                if cast is not None:           # follow the last cast until it's back in
                    if ("meter_x" not in cast and seen["meter"] > 0 and now > cast["t_release"] + 0.1
                            and now < cast["t_release"] + 0.6):
                        a_, b_, c_ = METER_TO_X  # where the meter says it'll land
                        mx = seen["meter"]
                        cast["meter_x"] = a_ + b_ * mx + c_ * mx * mx
                        if abs(cast["meter_x"] - cast["aim"]) > 20:
                            print(f"  !! the game saw a different hold: the meter says it'll land at "
                                  f"{cast['meter_x']:.0f}, not {cast['aim']:.0f} (a mouse click while the bot "
                                  f"was holding the button cuts it short)")
                    b, last_bob = seen["bob"], cast.get("last_bob")
                    cast["last_bob"] = b
                    if (b is not None and last_bob is not None and b[1] > -17 and abs(b[0] - last_bob[0]) < 1
                            and "land_x" not in cast and now > cast["t_release"] + 0.3):
                        cast["land_x"], cast["t_land"] = b[0] + 328, now
                        cast.pop("last_bob")
                    if "land_x" in cast and "caught" not in cast and now > cast["t_land"] + 0.6:
                        # A caught fish has faded out by now: which one's gone? (Fish stop
                        # moving when the bobber lands.)
                        there = [(k, f + 328) for f, k, _, _ in seen["fish"]]
                        gone = [k for k, x in cast["all_at_land"]
                                if not any(k2 == k and abs(x2 - x) < 25 for k2, x2 in there)]
                        cast["caught"] = gone[0] if gone else None
                        what = f" - caught the {gone[0]}" if gone else " - missed"
                        if gone and cast["target"] and gone[0] != cast["target"]:
                            what += f" (aimed at the {cast['target']})"
                        print(f"  landed at {cast['land_x']:.0f} (aimed {cast['aim']:.0f}){what}")
                if seen["bob"] is not None:
                    no_bob_since = None
                    if state == "watching":
                        state = "idle"
                    time.sleep(0.005)
                    continue
                no_bob_since = no_bob_since or now
                if now - no_bob_since < 0.15:
                    time.sleep(0.005)
                    continue
                if state == "cast" and (cast is None or "land_x" in cast or now > cast["t_release"] + 2.5):
                    state = "idle"             # it's back in (a long cast flies off the top a while)

                if state == "idle":            # a fresh round
                    if seen["meter"] > 0 or not seen["fish"]:
                        time.sleep(0.005)
                        continue
                    finish_cast()
                    ox = -STRIP[0]
                    bg.add(img, [(ox + x - 16, ox + x + 17) for x, _, _, _ in seen["fish"]] +
                           [(ox + m - w / 2 - 3, ox + m + w / 2 + 4) for m, w in seen["mines"]])
                    state, t_round, tracks, mines, planned, told_wait = "watching", now, [], [], None, 0
                if state != "watching":
                    continue
                mine_dets = [(m + 328, w, "mine") for m, w in seen["mines"]]
                for tr, (m, w, _) in match_tracks(mines, mine_dets, now, gate=16):
                    if tr is None:
                        mines.append(Track(m, now, "mine", w))
                    else:
                        tr.add(now, m, "mine")
                        tr.size = max(tr.size, w) if tr.span() > 0.3 else w
                # Mines don't go away during a round: one out of sight is behind a fish (they're
                # drawn behind them), so keep predicting it - unless it should be visible (no
                # fish where it'd be) or another track is following the same mine.
                # Two mines on top of each other: don't merge a new track there into the old one
                # as a duplicate - being new, it's treated as possibly anywhere in its swing until
                # watched a while.
                for c in seen["mine_overlaps"]:
                    near = [tr for tr in mines if abs(tr.pts[-1][1] - (c + 328)) < 20]
                    if len([tr for tr in near if len(tr.pts) >= 5]) < 2:
                        for tr in near:
                            tr.overlap_t = now
                fish_now = [(x + 328, FISH_PX.get(k, 17)) for x, k, _, _ in seen["fish"]]
                keep = []
                for tr in sorted(mines, key=lambda tr: -len(tr.pts)):
                    x = float(tr.at(now)) if len(tr.pts) >= 5 else tr.pts[-1][1]
                    hidden = any(abs(x - f) < fw / 2 + 10 for f, fw in fish_now)
                    stale = now - tr.pts[-1][0] > 0.4
                    dup = any(abs(x - (float(k.at(now)) if len(k.pts) >= 5 else k.pts[-1][1])) < 10 for k in keep)
                    if (dup and now - tr.overlap_t > 1.5) or (stale and (len(tr.pts) < 5 or not hidden)):
                        continue
                    keep.append(tr)
                mines = keep
                mines_move = mines_move or any(tr.moving() and tr.span() > 0.5 for tr in mines)
                mega_mode = whales >= 2
                dets = []
                for x, kind, size, info in seen["fish"]:
                    x += 328
                    if isinstance(info, tuple):  # something on the bar not matching any fish's colours
                        _, color, width = info
                        if mega_mode and width >= 24:   # waiting for the Megalodon: big and new = it
                            kind = "megalodon"   # (its colours don't match its wiki icon)
                        if unknown_told < 5 and unknown_round != t_round:
                            unknown_told, unknown_round = unknown_told + 1, t_round
                            print(f"  (something unrecognised on the bar at {x:.0f}: colour {color}, {width} px wide"
                                  f"{' - taking it for the Megalodon' if kind == 'megalodon' else ''})")
                    if kind not in seen_kinds:
                        seen_kinds.add(kind)
                        if kind not in ("green", "fish"):
                            print(f"  (first {kind} seen)")
                    if info != "unsure":
                        dets.append((x, size, kind))
                for tr, (x, size, kind) in match_tracks(tracks, dets, now, gate=15):
                    if tr is None:
                        tracks.append(Track(x, now, kind, size))
                    else:
                        tr.add(now, x, kind)
                        tr.size = max(tr.size, size)
                tracks = [tr for tr in tracks if now - tr.pts[-1][0] < 2.0]   # (may be half hidden a while)
                fish_move = fish_move or any(tr.moving() and tr.span() > 0.5 for tr in tracks)
                if fish_move and now - swing_checked > 0.5:
                    swing_checked = now
                    if learn_swing(tracks):
                        print(f"  (the fish now swing {2 * Track.amp:.0f} px end to end)")

                # The Megalodon: after two whales in a row, leave whales alone (catching other
                # fish) until it comes.
                kinds = {tr.kind for tr in tracks if tr.kind in VALUES}
                setup = mega_mode and "whale" in kinds and "megalodon" not in kinds
                values = dict(VALUES, whale=-30) if mega_mode else VALUES
                miss = None
                waited = now - t_round
                late = waited > GIVE_UP_S
                # A crowded bar can leave nothing under the mine limit for a long time (no fish,
                # no empty water): rather than wait for ever, slowly accept a little more.
                max_boom = MAX_BOOM if not late else min(MAX_RISKY, MAX_BOOM * 4 ** (waited / GIVE_UP_S))
                if waited >= told_wait + 10:     # say it's still going, every 10 s
                    told_wait += 10
                    best = planned[2] if planned else None
                    if best:
                        mine = f", {best['p_boom']:.1%} mine" if best["p_boom"] >= 0.0005 else ""
                        what = f"best so far {best['p_catch']:.0%} to catch the {best['target'].kind}{mine}"
                    else:
                        what = "nothing clear of the mines"
                    print(f"  (waiting {waited:.0f} s for a good cast: {len(mines)} mines, {what}; wants "
                          f"{catch_needed(waited):.0%} now{', taking a little more mine risk' if late else ''})")
                if planned is None or now - planned[0] >= REPLAN_S or planned[1] != len(tracks):
                    planned = (now, len(tracks), plan_cast(tracks, mines, now, fish_move, values=values,
                                                           max_boom=max_boom, mines_move=mines_move))
                p = planned[2]
                if p is not None and p["p_catch"] < catch_needed(waited):
                    p = None                     # wait for something better (it usually comes)
                if late and p is not None and p["p_catch"] < 0.3:
                    # the "best cast" is really a miss: miss properly, well clear of mines
                    miss = plan_miss(tracks, mines, now, max_boom=max_boom / 2, mines_move=mines_move)
                    if miss is None and p["p_boom"] > MAX_BOOM:
                        time.sleep(0.005)        # no clear water either: keep waiting
                        continue
                elif p is None and late:
                    miss = plan_miss(tracks, mines, now, max_boom=max_boom / 2, mines_move=mines_move)
                    if miss is None:
                        time.sleep(0.005)
                        continue
                elif p is None:
                    time.sleep(0.005)
                    continue
                if miss:
                    t_press, hold, aim = miss
                    tr = None
                    t_land = t_press + hold + flight(aim)
                    all_at_land = [(t.kind, float(t.at(t_land))) for t in tracks if t.kind in VALUES]
                else:
                    t_press, hold, aim, tr, fish_x = p["t_press"], p["hold"], p["aim"], p["target"], p["fish_x"]
                    p_catch, p_boom, all_at_land = p["p_catch"], p["p_boom"], p["all_at_land"]
                # Wait for the moment; replan until close (the fit keeps improving).
                if t_press - time.perf_counter() > 0.06:
                    time.sleep(0.005)
                    continue
                while time.perf_counter() < t_press:
                    pass
                cursor = inputs.get_cursor()
                inputs.mouse_down(ax + int(CAST_POINT[0] * s), ay + int(CAST_POINT[1] * s))
                t0 = time.perf_counter()
                while time.perf_counter() < t0 + hold:
                    pass
                inputs.mouse_up()
                t1 = time.perf_counter()
                inputs.set_cursor(*cursor)
                state = "cast"
                cast = {"t_release": t1, "aim": aim, "target": tr.kind if tr else None,
                        "all_at_land": all_at_land}
                if setup and tr:
                    print("  (leaving the whale alone for the Megalodon)")
                if tr:
                    print(f"cast: hold {hold * 1000:.0f} ms for the {tr.kind} at {fish_x:.0f}"
                          f"{' (moving)' if tr.moving() else ''}, aiming {aim:.0f} "
                          f"({p_catch:.0%} to catch{f', {p_boom:.1%} mine' if p_boom >= 0.0005 else ''})")
                else:
                    print(f"  nothing safe to aim at for {GIVE_UP_S:.0f} s - casting into empty water at {aim:.0f}")
        except KeyboardInterrupt:
            pass
        finally:
            finish_cast()
    print("Bye!")


def main():
    ap = argparse.ArgumentParser(description="Idleon fishing bot")
    ap.add_argument("--keep-going", action="store_true", help="keep fishing after catching the Megalodon")
    run_bot(ap.parse_args())


if __name__ == "__main__":
    main()
