"""Mining minigame bot (the mine cart).

The minigame: a mine cart sits at a fixed spot while the track scrolls left,
faster and faster. Click to jump; click again in the air to slam straight down.
Slamming onto an ore node collects it (copper +1, platinum +2; starfire's
worth more) and bounces the cart back up; being on the track with the cart's middle
over a pit ends the run.

Measured from a recording (native 960x572 client; y down; times as seen on screen):
  * Jump: leaves the track 11 ms after the click at 249 px/s, gravity 480 px/s^2
    -> 65 px high, 1.05 s in the air.                                 (rms 1.3 px)
  * Slam: 18 ms after the click the cart drops at a steady 360 px/s.   (rms 3.6 px)
  * Ore bounce: from the ore's top (22 px above the track) up at 241 px/s, same
    gravity -> 61 px above the ore, ~1.09 s until it's back on the track. (rms 2 px)
  * Scroll speed: ~77 px/s at the start, ~143 px/s after 26 s.
  * Pits are ~50 px wide (and come in runs later); ores ~28 px wide, often just
    before a pit - slam the ore and the bounce carries you over it.
  * A slam collects an ore when the ore is under the front half of the cart; so does
    coming down onto it without slamming. Coming down on an ore's edge doesn't collect
    it: the cart lands on the track beside it (harmless when the ore's behind).
  * Running into an ore along the track ends the run, as does the cart's middle being
    over a pit while it's on the track.
The picture sometimes freezes for a moment; the game's clock doesn't (afterwards
everything has jumped on), so frozen frames are just ignored.

Each frame it reads the cart's height, the pits and the ores, estimates the
scroll speed, and plans the best sequence of jumps and slams over the next
~2.5 s (dynamic programming over click times: survive first, then collect as
many ores as possible; every click is chosen so that being +-30 ms off still
works).

Usage:
    python -m idleonbot.mining          # F10 to stop
"""
import argparse
import collections
import math
import time

import numpy as np

NATIVE_W = 960

TRACK_FRONT = np.array([166, 131, 110], np.int16)   # the track's front face
TRACK_TOP = np.array([208, 162, 123], np.int16)     # its top surface
PIT_FRONT = np.array([83, 61, 51], np.int16)        # a pit's front face
TRACK_BOTTOM = np.array([119, 91, 75], np.int16)    # the strip under the front face
RIM = np.array([93, 116, 145], np.int16)            # the cart's rim and wheels
COPPER = np.array([255, 131, 39], np.int16)         # copper ore's orange glint (+1)
PLATINUM = np.array([21, 154, 223], np.int16)       # platinum ore's blue glint (+2)
ORE_VALUES = ((COPPER, 1.0), (PLATINUM, 2.0))
UNKNOWN_ORE_VALUE = 2.0                             # (starfire, +3, not seen yet)
COLOR_TOL = 12

# Geometry in native px: x in client coordinates, y relative to the middle row of the
# track's front face (FY; y=195 when the minigame was recorded, track x 39..400).
GROUND = -5            # the cart's lowest pixel while it's on the track
ORE_TOP = -27          # ... while it's standing on an ore
ORE_BAND = (-19, -7)   # rows just above the track where the ores sit
ROI_UP = 150           # how far above FY the cart can go
CART_X = (45, 150)     # where to look for the cart, relative to the track's left end
VIEW_LEFT = 70         # ignore the faded far-left end of the track (relative to its left end)
# The right end of the track is drawn squashed: things there are really further right than
# they look. Measured (native x; same at every speed): ore centres are true up to x~362 and
# pits' left edges up to ~335; pits' right edges up to ~390. Seen -> true:
ORE_SEEN = (362, 366, 371, 375, 378, 381, 384)
ORE_TRUE = (362, 366, 376, 387, 398, 411, 424)
PIT_SEEN = (335, 342, 347, 350, 353, 355, 358)
PIT_TRUE = (335, 345, 355, 365, 375, 385, 395)
SPEED_RIGHT = 332      # the speed estimate only looks left of this
EDGE_COST = 0.5        # extra cost of coming down next to an ore (edges are harmless, but it
                       # leaves the cart right beside it)
# The scroll speed follows the same curve every game (measured over three live runs, the
# same whatever the score): (seconds since the start, px/s). Past the end: +0.6 px/s per s.
SPEED_CURVE = [(0, 77), (5, 93), (10, 114), (15, 126), (20, 133), (25, 142), (30, 147), (35, 153), (40, 156), (45, 157), (50, 175), (55, 191), (60, 201), (65, 211), (70, 221), (75, 224), (80, 232), (85, 236), (90, 247), (95, 247), (100, 247), (105, 252), (110, 253), (115, 260), (120, 265), (125, 268)]
SQUASH_ERR = 0         # +- px we may be out for something still in the squashed part

# Physics (px, s, y down)
G = 480.2
JUMP_V, JUMP_DELAY = 249.3, 0.011
BOUNCE_V = 243.0
SLAM_V, SLAM_DELAY = 359.5, 0.018

ORE_W, PIT_W = 28, 50   # (pits come single or double width)
CART_HALF = 19          # the cart is 38 px wide
DEATH_DX = 3            # the point that mustn't be over a pit, relative to the cart's centre
PIT_MARGIN = 3          # ... with this much to spare
HIT = (-3, 17)          # ore centre - cart centre when a slam reaches the ore: collected
FRONT = (18, 33)        # an ore this far ahead of the cart centre is touching its front

# Planner
DT = 0.01
HORIZON = 2.0           # (things only come into view ~1.5 s ahead at speed)
ROBUST = 4              # steps (40 ms) either side a planned future click may be off and
                        # still work...
ROBUST_NOW = 2          # ...and the click we're about to make (we know more by then)
EDGE_PAD = 5            # aim this many steps into a window of good click times
REACT = 3               # steps (30 ms) after landing before a jump: the jump's clicked a
                        # little after the predicted landing (never before: then it's a slam)
LAND_JUMP_SLAM = 0.02   # s after a slam's predicted landing to click the next jump
LAND_JUMP_NAT = 0.03    # ... after coming down by itself (its landing is less certain)
LAND_SLACK = 1          # steps a landing may come early/late (the bounce height varies)
SLAM_LAND_SLACK = 0     # ... after a slam
RISKY_SLACK = 4         # a landing that wouldn't survive being this many steps off...
RISK = 0.6              # ...costs this much: worth it for an ore, not for nothing
SEEN_UP_TO = 395        # track right of this (true x) isn't in view yet...
UNSEEN_RISK = 0.5       # ...so landing where it was costs this (it may be a pit)
DEATH = -1000.0
UNKNOWN = -30.0         # (below this, no option is safe with slack to spare)
CLICK_POS = (317, 271)  # where to click (native client px)


def match(img, color, tol=COLOR_TOL):
    return np.abs(img.astype(np.int16) - color).sum(-1) < tol


def runs(mask):
    e = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return list(zip(e[::2].tolist(), e[1::2].tolist()))


def bridge(mask, gap):
    """Fill gaps of up to `gap` False values between True runs."""
    out = mask.copy()
    for a, b in runs(~mask):
        if 0 < a and b < len(mask) and b - a <= gap:
            out[a:b] = True
    return out


def to_native(img, s):
    """Nearest-neighbour resample a screen grab to native (960-wide) pixels."""
    if abs(s - 1) < 1e-3:
        return img
    h, w = int(img.shape[0] / s), int(img.shape[1] / s)
    ri = np.minimum((np.arange(h) * s + s / 2).astype(int), img.shape[0] - 1)
    ci = np.minimum((np.arange(w) * s + s / 2).astype(int), img.shape[1] - 1)
    return img[ri][:, ci]


def find_track(img):
    """(FY, TX0, TX1) for the minigame's track in a native client image, or None.
    The track's front face is a ~360 px long, 5-row band of one colour (pits in a
    darker one), with its top surface above and a darker strip below."""
    m = match(img, TRACK_FRONT) | match(img, PIT_FRONT)
    rows = np.flatnonzero(m.sum(1) > 250)
    for a, b in runs(np.isin(np.arange(img.shape[0]), rows)):
        fy = a + 3
        if not 4 <= b - a <= 6 or fy < 12 or fy + 3 >= img.shape[0]:
            continue
        xs = np.flatnonzero(m[fy])
        x0, x1 = int(xs.min()), int(xs.max())
        if not 356 <= x1 - x0 <= 366:   # (it's 361; less if a pit is fading out at an end)
            continue
        top = img[fy - 7, x0 + 10:x1 - 10]
        bottom = img[fy + 3, x0 + 10:x1 - 10]
        if (match(top, TRACK_TOP).mean() > 0.3 and
                (match(bottom, TRACK_BOTTOM) | match(bottom, PIT_FRONT)).mean() > 0.7):
            return fy, x0, x1
    return None


class Reader:
    """Reads the cart, pits and ores from a native-scale grab of the minigame."""

    def __init__(self, fy, tx0, tx1):
        self.fy, self.tx0, self.tx1 = fy, tx0, tx1
        # the region to grab: native client x0, x1, y0, y1
        self.box = (tx0, tx1 + 3, fy - ROI_UP, fy + 4)

    def read(self, img):
        """-> dict(cy, cx, pits, ores, pit_mask, track_ok). x in native client px, y
        relative to FY. cy/cx are None when the cart isn't visible."""
        ox, oy = self.box[0], self.box[2]
        fy = self.fy - oy
        out = {"cy": None, "cx": None}
        # Cart: the rim/wheel colour in the cart's column.
        c0, c1 = self.tx0 + CART_X[0] - ox, self.tx0 + CART_X[1] - ox
        ys, xs = np.nonzero(match(img[:fy - 1, c0:c1], RIM))
        if len(ys) >= 10:
            out["cy"] = int(ys.max()) - fy
            out["cx"] = (xs.min() + xs.max()) / 2 + c0 + ox
        # Pits: gaps in the track's front face.
        row = img[fy, :self.tx1 - ox + 1]
        track = match(row, TRACK_FRONT)
        # (late in a game the track can be 80% pits; when the game ends it fades away)
        out["track_ok"] = (track | match(row, PIT_FRONT)).mean() > 0.5
        left = self.tx0 + VIEW_LEFT - ox
        pit = bridge(~track, 3)
        pit[:left] = False
        out["pit_mask"] = pit[left:SPEED_RIGHT - ox]
        pits = []
        for a, b in runs(pit):
            if b - a < 8:
                continue
            a, b = a + ox, b + ox
            a = float(np.interp(a, PIT_SEEN, PIT_TRUE)) if a > PIT_SEEN[0] else a
            if b >= self.tx1 - 10:   # still coming into view: assume a single one for now
                b = max(b, a + PIT_W)   # (landing beyond what we've seen is penalised anyway)
            pits.append((round(a), b))
        out["pits"] = pits
        # Ores: anything bright just above the track that isn't the track.
        band = img[fy + ORE_BAND[0]:fy + ORE_BAND[1] + 1]
        bright = ((band.astype(np.int16).sum(-1) > 120) & ~match(band, TRACK_TOP)
                  & ~match(band, TRACK_FRONT))
        cols = bright.sum(0) >= 2
        right = self.tx1 - 3 - ox
        cols[:left] = False
        cols[right:] = False
        if out["cy"] is not None and out["cy"] >= ORE_BAND[0] - 2:   # the cart is down there too
            cx = int(out["cx"] - ox)
            cols[max(cx - CART_HALF - 3, 0):cx + CART_HALF + 3] = False
        ores = []
        for a, b in runs(bridge(cols, 4)):
            entering = b >= right
            if b - a >= 18 or (entering and b - a >= 4):
                value = next((val for col, val in ORE_VALUES if match(band[:, a:b], col).any()),
                             UNKNOWN_ORE_VALUE)
                if entering:
                    c = a + ORE_W / 2
                elif ORE_W + 4 < b - a < 2 * ORE_W:   # a collected nugget flying off its left
                    c = b - ORE_W / 2
                else:
                    c = (a + b) / 2
                c += ox
                if c > ORE_SEEN[0]:
                    c = (float(np.interp(c, ORE_SEEN, ORE_TRUE)) if c <= ORE_SEEN[-1]
                         else ORE_TRUE[-1] + (c - ORE_SEEN[-1]) * 3.3)
                ores.append((c, value))
        out["ores"] = ores
        return out


class Speed:
    """Scroll speed: the known curve, corrected by how far the pit pattern moved since
    ~0.4 s ago."""

    def __init__(self):
        self.hist = collections.deque()
        self.est = collections.deque()
        self.v = 80.0
        self.t_start = None      # when the game started (set before the speed is used)

    @staticmethod
    def curve(dt):
        ts, vs = zip(*SPEED_CURVE)
        if dt > ts[-1]:
            return vs[-1] + 0.6 * (dt - ts[-1])
        return float(np.interp(dt, ts, vs))

    def add(self, t, mask):
        self.hist.append((t, mask))
        while len(self.hist) > 2 and t - self.hist[1][0] >= 0.4:
            self.hist.popleft()
        t0, m0 = self.hist[0]
        dt = t - t0
        n = len(mask)
        if dt < 0.3 or m0.sum() < 15 or mask.sum() < 15 or len(m0) != n:
            return
        lo, hi = int(0.6 * self.v * dt), min(int(1.6 * self.v * dt) + 2, n - 60)
        if hi - lo < 3:
            return
        # pixels that disagree when the old pattern is moved left by s
        errs = np.array([np.count_nonzero(m0[s:] != mask[:n - s]) for s in range(lo, hi)])
        i = int(errs.argmin())
        far = np.abs(np.arange(len(errs)) - i) > 8
        if errs[i] <= 4 and (not far.any() or errs[far].min() >= errs[i] + 10) and 0 < i < len(errs) - 1:
            # sub-pixel: parabola through the minimum and its neighbours
            a, b, c = errs[i - 1], errs[i], errs[i + 1]
            d = 0.5 * (a - c) / (a - 2 * b + c) if a - 2 * b + c > 0 else 0.0
            self.est.append((t, (lo + i + d) / dt))
        while self.est and t - self.est[0][0] > 6:
            self.est.popleft()
        self.v = self.at(t)

    def at(self, t):
        """The speed now: the known curve, shifted by how far our recent measurements are
        from it."""
        recent = [(et, ev) for et, ev in self.est if et > t - 5]
        off = float(np.median([ev - self.curve(et - self.t_start) for et, ev in recent])) if len(recent) >= 3 else 0.0
        return self.curve(t - self.t_start) + max(min(off, 25.0), -25.0)


class Cart:
    """Follows the cart's height and fits its arc. Modes: ground, air (on a jump or
    bounce arc), slam (dropping after a click in the air).

    An arc is fitted with gravity and launch speed known: a jump leaves the track
    (y = GROUND) at JUMP_V; a bounce leaves wherever the cart hit the ore (usually
    ORE_TOP, lower after a slam from low down) at BOUNCE_V. The launch happened
    between the last frame before the change and the first one after it."""

    def __init__(self):
        self.mode = None
        self.kind = None          # "jump" / "bounce"
        self.samples = []         # (t, y) on the current arc
        self.window = None        # (earliest, latest) launch time
        self.last_t = self.last_y = None
        self.t_launch = self.y_launch = None
        self.t_slam = self.slam_y = self.slam_max = None
        self.expect_jump = None   # when we clicked to jump as it lands (maybe no frame on the
                                  # track in between)

    def update(self, t, y):
        """Returns "bounce"/"land"/None for what just happened."""
        if y is None:
            return None
        event = None
        prev_t = self.last_t if self.last_t is not None else t - 0.02
        if (self.expect_jump is not None and self.mode in ("air", "slam") and y < GROUND - 1
                and self.last_y is not None and y < self.last_y - 1 and t > self.expect_jump):
            # going up again after our jump click: it landed and jumped between frames
            self._launch("jump", max(prev_t, self.expect_jump), t, y)
            self.last_t, self.last_y = t, y
            self._fit()
            return "land"
        if y >= GROUND - 1:
            if self.mode not in ("ground", None):
                event = "land"
            self.mode, self.samples = "ground", []
        elif self.mode in ("ground", None):
            self._launch("jump", prev_t, t, y)
        elif self.mode == "slam":
            self.slam_max = max(self.slam_max, y)
            # on its way up again (after it had started down: a slam from just above an ore
            # hardly drops at all before bouncing)
            if y < self.slam_max - 2 and (self.slam_max > self.slam_y + 4 or t - self.t_slam > 0.06):
                self._launch("bounce", prev_t, t, y)
                event = "bounce"
            elif (t - self.t_slam > 0.08 and self.t_launch is not None and
                    y < self.y_at(t, anyway=True) + 3):
                self.mode = "air"   # the slam didn't happen: still on the same arc
                self.samples.append((t, y))
        else:
            pred = self.y_at(t)
            if pred is not None and abs(y - pred) > 6 and len(self.samples) >= 2:
                if y > pred:        # dropping faster than the arc: someone else's slam
                    self.slammed(prev_t, self.last_y)
                    self.slam_max = y
                else:               # thrown up: bounced off something
                    self._launch("bounce", prev_t, t, y)
                    event = "bounce"
            else:
                self.samples.append((t, y))
        self.last_t, self.last_y = t, y
        if self.mode == "air":
            self._fit()
        return event

    def _launch(self, kind, t0, t1, y):
        self.mode, self.kind, self.samples, self.window = "air", kind, [(t1, y)], (t0, t1)
        if kind == "jump":
            self.expect_jump = None

    def slammed(self, t, y=None):
        self.mode, self.t_slam = "slam", t
        self.slam_y = self.slam_max = y if y is not None else (self.last_y or 0)

    def _fit(self):
        t = np.array([s[0] for s in self.samples])
        y = np.array([s[1] for s in self.samples], float)
        v = JUMP_V if self.kind == "jump" else BOUNCE_V
        t0, t1 = self.window
        cand = np.arange(min(t0, t1 - 0.005), t1 + 1e-9, 0.001)
        q = t[None, :] - cand[:, None]
        shape = -v * q + G / 2 * q * q
        if self.kind == "jump":
            y0 = np.full(len(cand), float(GROUND))
        else:   # the height it bounced from: whatever fits best (but not below the track)
            y0 = np.minimum((y[None, :] - shape).mean(1), GROUND)
        err = ((y0[:, None] + shape - y[None, :]) ** 2).sum(1)
        i = int(err.argmin())
        self.t_launch, self.y_launch, self.v0 = float(cand[i]), float(y0[i]), v
        if self.kind == "bounce" and len(t) >= 5 and t[-1] - t[0] >= 0.15:
            # bounces vary in height (60-75 px): once there's enough of it, fit its speed too
            q = t - self.t_launch
            A = np.c_[np.ones_like(q), -q]
            (y0f, vf), *_ = np.linalg.lstsq(A, y - G / 2 * q * q, rcond=None)
            if 200 < vf < 300:
                self.y_launch, self.v0 = float(y0f), float(vf)

    def landing(self):
        """When the current arc comes down onto the track by itself."""
        v, y0 = self.v0, self.y_launch
        return self.t_launch + (v + math.sqrt(v * v + 2 * G * (GROUND - y0))) / G

    def y_at(self, t, anyway=False):
        if (self.mode != "air" and not anyway) or self.t_launch is None:
            return None
        q = t - self.t_launch
        return self.y_launch - self.v0 * q + G / 2 * q * q

    def vy_at(self, t):
        return -self.v0 + G * (t - self.t_launch)


class OreTracker:
    """Remembers ores for a moment when we can't see them (behind the cart, or a missed
    frame) and forgets the ones the cart collects."""
    GRACE = 0.1

    def __init__(self):
        self.ores = []    # [x centre at time self.t, value, last seen]
        self.t = None

    def update(self, t, seen, v, hidden):
        """seen: [(x, value)]; hidden: (x0, x1) span the reader couldn't see, or None."""
        new = [[x, val, t] for x, val in seen]
        if self.t is not None:
            dx = v * (t - self.t)
            for x, val, last in self.ores:
                x -= dx
                if any(abs(x - n[0]) < 16 for n in new):   # (ores are >= 40 px apart)
                    continue
                if (hidden and hidden[0] - 6 <= x <= hidden[1] + 6) or t - last < self.GRACE:
                    new.append([x, val, last])
        self.ores = sorted(new)
        self.t = t

    def collected(self, cx):
        """Forget the ore the cart just bounced off (if we still think it's there)."""
        near = [o for o in self.ores if HIT[0] - 6 <= o[0] - cx <= HIT[1] + 3]
        if near:
            self.ores.remove(min(near, key=lambda o: abs(o[0] - cx - sum(HIT) / 2)))


# ----------------------------------------------------------------------------- planning

class ArcTable:
    """Where things happen on an arc y(tau) = y0 + vy0 (tau-d) + G/2 (tau-d)^2 (y0 until d),
    in DT steps from the arc's start: for a slam clicked at step i, when it reaches an ore's
    top (ko, -1 if it's already below that) and the track (kg); when the arc itself comes
    down through an ore's top (n_o) and onto the track (n_g); when it rises past an ore's
    top (r_o, or None)."""

    def __init__(self, y0, vy0, delay=0.0, first=0):
        def y(tau):
            q = np.maximum(tau - delay, 0)
            return y0 + vy0 * q + G / 2 * q * q

        def cross(level, rising):
            disc = vy0 * vy0 + 2 * G * (level - y0)
            if disc < 0:
                return None
            r = (-vy0 - math.sqrt(disc)) / G if rising else (-vy0 + math.sqrt(disc)) / G
            return delay + r if r > 0 else None

        t_land = cross(GROUND, False) or delay
        self.n_g = int(round(t_land / DT))
        t_o = cross(ORE_TOP, False)
        self.n_o = int(round(t_o / DT)) if t_o is not None else 0
        t_r = cross(ORE_TOP, True) if y0 > ORE_TOP else None
        self.r_o = int(round(t_r / DT)) if t_r is not None else None
        i = np.arange(first, max(int((t_land - SLAM_DELAY) / DT), first))
        ts = i * DT + SLAM_DELAY
        ys = y(ts)
        self.ko = np.where(ys <= ORE_TOP, np.round((ts + (ORE_TOP - ys) / SLAM_V) / DT), -1).astype(int)
        self.kg = np.round((ts + (GROUND - ys) / SLAM_V) / DT).astype(int)


JUMP_ARC = ArcTable(GROUND, -JUMP_V, JUMP_DELAY)
BOUNCE_ARC = ArcTable(ORE_TOP, -BOUNCE_V, 0.0, first=1)
PAD = 150


def sliding_min(a, r, axis=-1):
    """min over a window of +-r along axis, edges padded with the edge values."""
    if r <= 0:
        return a
    a = np.moveaxis(np.asarray(a), axis, -1)
    p = np.pad(a, [(0, 0)] * (a.ndim - 1) + [(r, r)], mode="edge")
    n = a.shape[-1]
    out = p[..., :n].copy()
    for k in range(1, 2 * r + 1):
        np.minimum(out, p[..., k:k + n], out=out)
    return np.moveaxis(out, -1, axis)


class Plan:
    """Dynamic programming over the next HORIZON seconds. World at time t0: pits
    [(a, b)] and ores [(x, value)] in native x, scrolling left at v px/s; cart centre cx."""

    def __init__(self, pits, ores, v, cx):
        self.n = n = int(HORIZON / DT)
        self.np_ = np_ = n + PAD
        t = np.arange(np_) * DT
        p = cx + DEATH_DX
        self.pit_bad = np.zeros(np_, bool)
        for a, b in pits:
            u = SQUASH_ERR if a > PIT_SEEN[0] else 0     # (seen in the squashed right end)
            self.pit_bad |= (a - u - v * t - PIT_MARGIN <= p) & (p <= b - v * t + PIT_MARGIN)
        self.pit_bad[n:] = False
        # landing (then being stuck on the track for REACT) where we can't see yet
        self.unseen = p + v * (t + REACT * DT) > SEEN_UP_TO
        ores = sorted(ores)
        self.hit = np.full(np_, -1)
        self.foot = np.full(np_, -1)
        self.front = np.zeros(np_, bool)
        for j, (x, _) in enumerate(ores):
            rel = x - v * t - cx
            u = SQUASH_ERR if x > ORE_SEEN[0] else 0     # (seen in the squashed right end)
            self.hit[(rel - u >= HIT[0]) & (rel + u <= HIT[1])] = j
            self.foot[np.abs(rel) <= CART_HALF + ORE_W / 2 + u] = j
            self.front |= (rel + u >= FRONT[0]) & (rel - u <= FRONT[1])
        for a in (self.hit, self.foot):
            a[n:] = -1
        self.front[n:] = False
        self.ore_val = np.array([val for _, val in ores] + [0.0])
        self.Vg = np.zeros(np_)
        self.Vl = np.zeros(np_)     # value of landing on the track at a step
        self.Vl_nat = self.Vl_slam = self.Vl
        self.Vb = np.zeros(np_)
        # landing at k: stuck on the track until k + REACT
        bad_soon = np.zeros(np_, bool)
        clash_soon = np.zeros(np_, bool)
        for d in range(REACT + 1):
            bad_soon[:np_ - d] |= self.pit_bad[d:]
            clash_soon[:np_ - d] |= self.front[d:]
        ks = np.arange(n)
        kb = np.flatnonzero(self.hit[:n] >= 0)
        for _ in range(15):
            vj = self._arc(JUMP_ARC, ks, np.full(n, -1))
            vb = self._arc(BOUNCE_ARC, kb, self.hit[kb]) if len(kb) else None
            vjr = sliding_min(vj, ROBUST)
            self.Vj_raw = np.zeros(np_)
            self.Vj_raw[:n] = vj
            vg = self._ground(vjr)
            done = np.array_equal(vg, self.Vg[:n]) and (vb is None or np.array_equal(vb, self.Vb[kb]))
            self.Vg[:n] = vg
            self.Vl[:n] = np.where(bad_soon[:n] | clash_soon[:n], DEATH, self.Vg[REACT:n + REACT])
            wide = sliding_min(self.Vl, RISKY_SLACK)
            self.Vl_nat = sliding_min(self.Vl, LAND_SLACK)
            self.Vl_slam = sliding_min(self.Vl, SLAM_LAND_SLACK)
            self.Vl_nat = self.Vl_nat - RISK * (wide < self.Vl_nat - 0.5) - UNSEEN_RISK * self.unseen
            self.Vl_slam = self.Vl_slam - RISK * (wide < self.Vl_slam - 0.5) - UNSEEN_RISK * self.unseen
            if vb is not None:
                self.Vb[kb] = vb
            if done:
                break

    def _ground(self, vjr):
        vg = np.empty(self.n)
        nxt = 0.0
        for k in range(self.n - 1, -1, -1):
            if self.pit_bad[k]:
                nxt = DEATH
            elif self.front[k]:     # running into an ore
                nxt = DEATH
            else:
                nxt = max(vjr[k], nxt)
            vg[k] = nxt
        return vg

    def options(self, arc, ks, excl):
        """Values of slamming at each step of `arc` started at each of steps ks (rows), and
        of letting it come down by itself. Ores with index <= excl are already collected."""
        ks = np.asarray(ks)[:, None]
        ex = np.asarray(excl)[:, None]
        KO = ks + np.maximum(arc.ko, 0)[None, :]
        KG = ks + arc.kg[None, :]
        has = (arc.ko >= 0)[None, :]
        h = self.hit[KO]
        ore = has & (h > ex)
        # anything else: it lands on the track (on an ore's edge too), and then must not be
        # over a pit or running into an ore (both in Vl)
        near = (has & (self.foot[KO] > ex)) | (self.foot[KG] > ex)
        slam = np.where(ore, self.ore_val[h] + self.Vb[KO], self.Vl_slam[KG] - EDGE_COST * near)
        NO, NG = ks[:, 0] + arc.n_o, ks[:, 0] + arc.n_g
        # coming down onto an ore by itself collects it too
        hn = self.hit[NO]
        near = (self.foot[NO] > excl) | (self.foot[NG] > excl)
        nat = np.where(hn > excl, self.ore_val[hn] + self.Vb[NO], self.Vl_nat[NG] - EDGE_COST * near)
        if arc.r_o is not None:  # rising off the track: an ore at the front before we're
            # above ore height would hit us (the whole climb, not just its end)
            fc = np.concatenate(([0], np.cumsum(self.front)))
            k = ks[:, 0]
            bad = fc[k + arc.r_o + 1] - fc[k] > 0
            slam = np.where(bad[:, None], DEATH, slam)
            nat = np.where(bad, DEATH, nat)
        return slam, nat

    def _arc(self, arc, ks, excl):
        slam, nat = self.options(arc, ks, excl)
        if slam.shape[1]:
            return np.maximum(sliding_min(slam, ROBUST, axis=1).max(1), nat)
        return nat

    def jump_values_after(self, k0, first):
        """Like ground_values, for a cart that lands on the track at step k0."""
        vals = sliding_min(self.Vj_raw[:self.n].copy(), ROBUST_NOW)
        vals[:k0] = -np.inf
        bad = np.flatnonzero(self.pit_bad[k0:self.n] | self.front[k0:self.n])
        if len(bad):
            vals[max(k0 + bad[0], first + 1):] = -np.inf
        return vals

    def ground_values(self, first):
        """Per-step values of jumping from the track now (step 0 = now), -inf once a pit
        (or an ore) makes staying on the track impossible."""
        vals = self.Vj_raw[:self.n].copy()
        vals = sliding_min(vals, ROBUST_NOW)
        bad = np.flatnonzero(self.pit_bad[:self.n] | self.front[:self.n])
        if len(bad):
            # already too close to a pit (or an ore)? then jumping right away is the best bet
            vals[max(bad[0], first + 1):] = -np.inf
        return vals


def choose(values, first, committed=None, late=False):
    """Pick a click step from per-step option values (index = DT steps from now): the
    soonest window of best values, EDGE_PAD steps in (or its middle if it's short).
    Keeps a previously chosen step if it's still one of the best. Returns (step, value,
    due) - due: the chosen moment has come (click now)."""
    v = np.asarray(values, float)
    v[:first] = -np.inf
    best = v.max() if len(v) else -np.inf
    if not np.isfinite(best):
        return None, best, False
    good = v >= best - 1e-6
    if committed is not None and committed < len(v):
        due = committed <= first
        committed = max(committed, first)   # due already: as soon as we can
        if good[committed]:
            return committed, best, due
    # late: the end of the window (for jumps: stay on the track while it's safe - more
    # of what's coming is in view by then, and its end is fixed by the next pit)
    a, b = runs(good)[0]
    pad = min(EDGE_PAD, (b - a - 1) // 2)
    step = b - 1 - pad if late else a + pad
    return step, best, step <= first


def decide(plan, mode, y_now, vy_now, first, committed=None):
    """The next click: (step, value, due, kind, aim) - step in DT steps from the frame (None:
    no click, let the cart come down by itself), kind "jump"/"slam", aim what a slam is
    expected to come down on ("ore"/"edge"/"track")."""
    if mode == "ground":
        vals = plan.ground_values(first)
        step, best, due = choose(vals, first, committed, late=True)
        if best <= UNKNOWN:   # nothing's safe with slack to spare: best raw option
            raw = plan.Vj_raw[:plan.n].copy()
            raw[np.isneginf(vals)] = -np.inf
            step, best, due = choose(raw, first, committed, late=True)
        return step, best, due, "jump", "jump"
    arc = ArcTable(y_now, vy_now)
    slam, nat = plan.options(arc, [0], [-1])
    raw = slam[0]
    slam = sliding_min(slam, ROBUST_NOW, axis=1)[0] if slam.shape[1] else raw
    step, best, due = choose(slam, first, committed) if len(slam) else (None, -np.inf, False)
    if best <= UNKNOWN and nat[0] <= UNKNOWN and len(raw):
        step, best, due = choose(raw, first, committed)   # best raw option
    if best <= nat[0]:
        return None, nat[0], False, "slam", None          # just let it come down
    ko = arc.ko[step] if step < len(arc.ko) else -1       # what we expect to come down on
    label = ("ore" if ko >= 0 and plan.hit[ko] >= 0 else
             "edge" if ko >= 0 and plan.foot[ko] >= 0 else "track")
    return step, best, due, "slam", label


# ----------------------------------------------------------------------------- the bot

def run_bot():
    from . import inputs
    from .capture import ScreenGrabber
    from .hotkeys import VK_F10, HotkeyPoller
    from .window import GameWindow

    win = GameWindow()
    hk = HotkeyPoller(VK_F10)

    print("Mining bot running: start a mining game (F10 stops it).")
    track = reader = None

    def new_game():
        return dict(speed=Speed(), cart=Cart(), ores=OreTracker(), last_sig=None,
                    last_change=None, frozen_since=None, commit=None, pending=None,
                    started=None, lost_since=None, score=0, cx=None)

    g = new_game()
    frame_dt, last_now = 0.02, None

    with ScreenGrabber() as grabber:
        try:
            while True:
                now = time.perf_counter()
                if hk.pressed(VK_F10):
                    break
                if not win.is_foreground() or win.is_minimized():
                    time.sleep(0.03)
                    last_now = None
                    continue
                cl, ct, cw, ch = win.client_rect()
                s = cw / NATIVE_W
                if last_now is not None:
                    frame_dt = 0.9 * frame_dt + 0.1 * min(now - last_now, 0.1)
                last_now = now

                if track is None:
                    full = to_native(grabber.grab(cl, ct, cw, ch), s)
                    track = find_track(full)
                    if track is None:
                        time.sleep(0.05)
                        continue
                    reader = Reader(*track)
                x0, x1, y0, y1 = reader.box
                img = to_native(grabber.grab(cl + int(x0 * s), ct + int(y0 * s),
                                             int((x1 - x0) * s), int((y1 - y0) * s)), s)
                r = reader.read(img)

                # Game over: the track's gone, or everything has stood still for a while.
                if not r["track_ok"]:
                    g["lost_since"] = g["lost_since"] or now
                    if now - g["lost_since"] > 0.5:
                        if g["started"]:
                            print(f"Game over? ({g['score']} ores collected)")
                        track, g = None, new_game()
                    continue
                g["lost_since"] = None

                # Freezes: the picture stops completely (not just between game frames).
                sig = (r["cy"], tuple(r["pits"]), tuple(round(o[0]) for o in r["ores"]))
                moving_stuff = bool(r["pits"] or r["ores"]) or (r["cy"] is not None and r["cy"] < GROUND - 1)
                # An unchanged picture tells us nothing new (we often grab faster than the game
                # draws; or it's the start of a freeze): don't let it move the trackers.
                static = sig == g["last_sig"] and moving_stuff
                if not static:
                    # (the game keeps its own time through a freeze: afterwards everything has
                    # jumped on to where it would have been, so our clock just keeps going)
                    if g["frozen_since"] is not None and g["pending"]:
                        # a click made during the freeze only takes effect now
                        g["pending"] = (g["pending"][0], now)
                    g["last_sig"], g["last_change"], g["frozen_since"] = sig, now, None
                elif now - g["last_change"] > 0.05:
                    g["frozen_since"] = g["frozen_since"] or now
                    if now - g["last_change"] > 1.5 and g["started"]:
                        print(f"Game over ({g['score']} ores collected)")
                        track, g = None, new_game()
                    continue
                gt = now

                if r["cy"] is None:
                    continue
                if g["started"] is None:
                    g["started"] = gt
                    g["speed"].t_start = gt
                    print("Game on!")
                g["cx"] = r["cx"] if r["cy"] >= GROUND - 1 else (g["cx"] or r["cx"])
                cx = g["cx"]
                cart = g["cart"]
                if not static:
                    g["speed"].add(gt, r["pit_mask"])
                v = g["speed"].at(gt)
                hidden = (cx - CART_HALF - 3, cx + CART_HALF + 3) if r["cy"] >= ORE_BAND[0] - 2 else None
                event = None
                if not static:
                    g["ores"].update(gt, r["ores"], v, hidden)
                    event = cart.update(gt, r["cy"])
                if event == "bounce":
                    g["score"] += 1
                    g["ores"].collected(cx)
                    print(f"  ore! ({g['score']})")
                if event:
                    g["commit"] = None

                # A click we've made that hasn't shown up yet: wait for it (for 80 ms of moving
                # picture - if it froze, the click only takes effect once it's going again, and
                # clicking again would turn a jump into a slam)
                if g["pending"] and gt - g["pending"][1] < 0.08:
                    if not (g["pending"][0] == "jump" and cart.mode == "air"):
                        continue
                g["pending"] = None
                landing = None     # when a slam onto the track will land
                if cart.mode == "slam":
                    if g.get("slam_aim") != "track" or cart.t_slam is None:
                        continue
                    t0 = cart.t_slam + SLAM_DELAY
                    y0 = cart.y_at(t0, anyway=True) if cart.t_launch is not None else cart.slam_y
                    if y0 is None:
                        continue
                    landing = t0 + max(GROUND - y0, 0) / SLAM_V + LAND_JUMP_SLAM
                elif cart.mode == "air" and cart.t_launch is None:
                    continue

                def click(kind, label):
                    cursor = inputs.get_cursor()
                    inputs.mouse_down(cl + int(CLICK_POS[0] * s), ct + int(CLICK_POS[1] * s))
                    inputs.mouse_up()
                    inputs.set_cursor(*cursor)
                    t = time.perf_counter()
                    g["pending"] = (kind, t)
                    g["commit"] = None
                    if kind == "slam":
                        cart.slammed(t)
                        g["slam_aim"] = label
                    elif cart.mode != "ground":
                        cart.expect_jump = t

                def click_at(t_click, kind, label):
                    """Wait for the moment (spinning: sleep is too coarse on Windows), then click."""
                    while t_click - time.perf_counter() > 0.001:
                        pass
                    click(kind, label)

                def scene():
                    """What the plan was based on, in track coordinates (so it can be compared
                    with a later frame)."""
                    return (gt, [(a, b) for a, b in r["pits"] if b > cx - 30],
                            [o[0] for o in g["ores"].ores if o[0] > cx - 30])

                def same_scene(old):
                    t0, pits0, ores0 = old
                    dx = g["speed"].v * (gt - t0)          # how far things have moved since
                    pits0 = [(a - dx, b - dx) for a, b in pits0 if b - dx > cx - 30]
                    ores0 = [o - dx for o in ores0 if o - dx > cx - 30]
                    _, pits1, ores1 = scene()
                    return (len(pits0) == len(pits1) and len(ores0) == len(ores1) and
                            all(abs(a - c) < 5 and abs(b - d) < 5 for (a, b), (c, d) in zip(pits0, pits1)) and
                            all(abs(a - c) < 6 for a, c in zip(ores0, ores1)))

                # A click due before we'd finish planning again: if nothing new has come into
                # view since we planned it, don't replan, just make it.
                commit = g["commit"] if g["commit"] != "none" else None
                if commit and (commit[0] == cart.mode or commit[0] == "landing"):
                    left = commit[1] - time.perf_counter()
                    if -0.01 < left < max(1.5 * frame_dt, 0.03) and same_scene(commit[4]):
                        click_at(*commit[1:4])
                        continue
                if static:
                    continue

                ores = [(x, val) for x, val, _ in g["ores"].ores]
                plan = Plan(r["pits"], ores, v, cx)
                t_done = time.perf_counter()
                first = int(math.ceil((t_done - gt) / DT))
                committed = None
                if commit and (commit[0] == cart.mode or commit[0] == "landing"):
                    committed = int(round((commit[1] - gt) / DT))
                y_now = cart.y_at(gt) if cart.mode == "air" else None
                vy_now = cart.vy_at(gt) if cart.mode == "air" else None
                mode_tag = cart.mode
                if landing is None:
                    step, _, due, kind, label = decide(plan, cart.mode, y_now, vy_now, first, committed)
                    if step is None and cart.mode == "air" and cart.landing() - gt < 0.35:
                        arc = ArcTable(y_now, vy_now)
                        if plan.hit[arc.n_o] < 0:     # coming down onto the track, not an ore
                            landing = cart.landing() + LAND_JUMP_NAT
                if landing is not None:
                    # Jump as soon as it's down, without waiting to see it land.
                    k_land = int(math.ceil((landing - gt) / DT))
                    vals = plan.jump_values_after(k_land, first)
                    step, _, due = choose(vals, max(first, k_land), committed, late=True)
                    kind, label, mode_tag = "jump", "jump", "landing"
                if step is None:
                    g["commit"] = "none"
                    continue
                t_click = gt + step * DT
                g["commit"] = (mode_tag, t_click, kind, label, scene())
                left = t_click - time.perf_counter()
                if due or left < max(1.5 * frame_dt, 0.03):
                    click_at(t_click, kind, label)
        except KeyboardInterrupt:
            pass
    print("Bye!")


def main():
    ap = argparse.ArgumentParser(description="Idleon mining (mine cart) bot")
    ap.parse_args()
    run_bot()


if __name__ == "__main__":
    main()
