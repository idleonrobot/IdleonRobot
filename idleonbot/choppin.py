"""Choppin' minigame bot.

The minigame: a leaf slides left/right above a bar made of red, green and
yellow segments. Chop while the leaf is over green (+1) or yellow (+2, and
slows the leaf down). Hitting a wall speeds the leaf up, and the segments
reshuffle after every chop.

Usage:
    python -m idleonbot.choppin               # F10 to stop
    python -m idleonbot.choppin --target 100  # stop at 100 points
"""
import argparse
import collections
import math
import time
from dataclasses import dataclass, field

import numpy as np

NONE, RED, GREEN, YELLOW = 0, 1, 2, 3
ZONE_NAMES = {NONE: "none", RED: "red", GREEN: "green", YELLOW: "yellow"}

# Exact colours sampled from the game (top half of the bar; the bottom half is a darker shade).
BAR_COLORS = {RED: (229, 97, 97), GREEN: (122, 206, 39), YELLOW: (252, 255, 122)}
LEAF_COLORS = np.array([(122, 206, 39), (189, 236, 63), (61, 161, 33), (31, 89, 30)], np.int16)
COLOR_TOL = 12

# Geometry measured at the game's native window size, where the bar is 220 px wide.
# Everything is scaled by (actual bar width / 220) so a resized window still works.
NATIVE_BAR_WIDTH = 220
LEAF_ROWS = (-27, -3)      # leaf occupies rows -24..-5 relative to the bar's top row
LEAF_PAD_X = 6             # leaf can hang past the bar ends by half its width (~5 px)
LEAF_MIN_PIXELS = 50       # the full leaf is ~135 matching pixels

# Timing (s) and safety margins.
MARGIN = 2.0               # native px either side of the leaf that must be green/yellow
LEAD = (0.005, 0.045)      # window ahead (input lag) where the leaf must be over green/yellow
YELLOW_WAIT = 0.6          # skip a green chop if a hittable yellow is this close
CHOP_GAP = 0.25            # min time between chops
STALL = 0.035              # time without leaf movement that counts as a game hitch
RECOVER = 0.15             # hold off chopping this long after a hitch
HOLD = 0.025               # how long to hold Space
REARM = 0.3                # wait this long for the bar to reshuffle after a chop before retrying


def match_colors(img: np.ndarray, colors: np.ndarray, tol: int = COLOR_TOL) -> np.ndarray:
    """Boolean mask of pixels within `tol` (per channel) of any of `colors`."""
    diff = np.abs(img.astype(np.int16)[..., None, :] - colors)
    return (diff.max(-1) <= tol).any(-1)


def classify(img: np.ndarray) -> np.ndarray:
    """Zone code (NONE/RED/GREEN/YELLOW) per pixel."""
    zones = np.zeros(img.shape[:-1], np.uint8)
    for code, rgb in BAR_COLORS.items():
        zones[match_colors(img, np.array([rgb], np.int16))] = code
    return zones


def longest_run(mask: np.ndarray, max_gap: int = 0) -> tuple[int, int]:
    """(start, end_exclusive) of the longest run of True in a 1-D mask,
    bridging gaps of up to `max_gap` False values (e.g. blended pixels where
    two segments meet in a scaled window)."""
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    edges = np.flatnonzero(np.diff(padded))
    if len(edges) == 0:
        return 0, 0
    starts, ends = edges[::2], edges[1::2]
    keep = np.concatenate(([True], starts[1:] - ends[:-1] > max_gap))
    starts, ends = starts[keep], np.concatenate((ends[:-1][keep[1:]], ends[-1:]))
    i = np.argmax(ends - starts)
    return int(starts[i]), int(ends[i])


@dataclass
class Bar:
    """Bar position within the searched image (x1 exclusive)."""
    x0: int
    x1: int
    top: int
    band: int  # height of the bright top half, in rows

    @property
    def scale(self) -> float:
        return (self.x1 - self.x0) / NATIVE_BAR_WIDTH


def find_bar(img: np.ndarray) -> Bar | None:
    """Search a full game-window capture for the minigame bar."""
    zones = classify(img)
    is_bar = zones != NONE
    min_len = max(60, img.shape[1] // 10)
    for y in np.flatnonzero(is_bar.sum(axis=1) >= min_len):
        x0, x1 = longest_run(is_bar[y], max_gap=4)
        if x1 - x0 < min_len or len(np.unique(zones[y, x0:x1])) < 2:
            continue  # a plain strip of one colour is scenery, not the bar
        band = 1
        while y + band < img.shape[0] and is_bar[y + band, x0:x1].mean() > 0.9:
            band += 1
        if band < 3:
            continue
        # Take the extent from columns that are bar-coloured through most of the band;
        # this drops the frame's green vine decoration, which only touches the top rows.
        cols = is_bar[y:y + band].mean(axis=0) >= 0.8
        x0, x1 = longest_run(cols, max_gap=4)
        if x1 - x0 >= min_len:
            return Bar(x0, x1, int(y), band)
    return None


@dataclass
class Reading:
    bar_ok: bool
    zones: np.ndarray       # zone per bar column (index 0 = bar's left end)
    leaf_x: float | None    # leaf centre, in bar columns
    under_leaf: int | None  # zone under the whole leaf window, or None if mixed/red/unknown


class ChoppinReader:
    """Reads one small region of interest (bar + leaf track) per frame."""

    def __init__(self, bar: Bar, margin: float):
        self.bar = bar
        s = bar.scale
        pad = round(LEAF_PAD_X * s) + 2
        self.left = bar.x0 - pad
        self.top = bar.top + round(LEAF_ROWS[0] * s)
        self.width = bar.x1 - bar.x0 + 2 * pad
        self.height = bar.top + bar.band - self.top
        self.bar_row = bar.top + bar.band // 2 - self.top
        self.leaf_rows = slice(0, bar.top + round(LEAF_ROWS[1] * s) - self.top)
        self.bar_cols = slice(pad, pad + bar.x1 - bar.x0)
        self.pad = pad
        self.margin = margin * s
        self.leaf_min = LEAF_MIN_PIXELS * s * s

    def roi(self) -> tuple[int, int, int, int]:
        """(left, top, width, height) relative to the image find_bar searched."""
        return self.left, self.top, self.width, self.height

    def read(self, roi: np.ndarray) -> Reading:
        zones = classify(roi[self.bar_row, self.bar_cols])
        bar_ok = (zones != NONE).mean() > 0.9

        leaf = match_colors(roi[self.leaf_rows], LEAF_COLORS)
        xs = np.nonzero(leaf)[1]
        if len(xs) < self.leaf_min:
            return Reading(bar_ok, zones, None, None)
        leaf_x = xs.mean() - self.pad
        return Reading(bar_ok, zones, leaf_x, self.zone_over(zones, leaf_x, leaf_x))

    def zone_over(self, zones: np.ndarray, a: float, b: float) -> int | None:
        """The zone covering bar columns a..b (plus margin), if it's all green/yellow.
        A green/yellow straddle counts as green, the lower-scoring of the two."""
        lo = int(np.floor(min(a, b) - self.margin))
        hi = int(np.ceil(max(a, b) + self.margin))
        if lo < 0 or hi >= len(zones):
            return None
        under = zones[lo:hi + 1]
        if np.isin(under, (GREEN, YELLOW)).all():
            return int(under.min())
        return None


def yellow_coming(zones: np.ndarray, x: float, v: float, need: float, margin: float,
                  horizon: float, step: float = 0.002) -> float | None:
    """Seconds until the leaf (bouncing off the bar ends) reaches a yellow segment wide
    enough to chop safely at this speed (`need` px of travel must fit inside it), or None
    if there isn't one within `horizon` seconds."""
    width = len(zones) - 1
    yellow = (zones == YELLOW).astype(np.int8)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], yellow, [0]))))
    targets = [(a + margin, b - 1 - margin) for a, b in zip(edges[::2], edges[1::2])
               if (b - a) - 2 * margin >= need]
    if not targets:
        return None
    ts = np.arange(0, horizon, step)
    pos = np.mod(x + v * ts, 2 * width)
    pos = np.where(pos > width, 2 * width - pos, pos)  # reflect off the ends
    inside = np.zeros(len(ts), bool)
    for lo, hi in targets:
        inside |= (pos >= lo) & (pos <= hi)
    hit = np.flatnonzero(inside)
    return float(ts[hit[0]]) if len(hit) else None


def leaf_velocity(history: collections.deque, now: float, window: float = 0.06) -> float | None:
    """Leaf speed in px/s over roughly the last `window` seconds (None if unknown).
    `history` should hold only frames where the leaf moved, so repeated frames
    (capture faster than the game draws, or a game hitch) don't read as zero speed."""
    if not history or now - history[-1][0] > 0.05:
        return None
    t1, x1 = history[-1]
    for t0, x0 in history:
        if t1 - t0 <= window:
            break
    if t1 - t0 < 0.015:
        return None
    return (x1 - x0) / (t1 - t0)


@dataclass
class Game:
    """State for one play of the minigame."""
    score: int = 0
    chops: int = 0
    last_press_t: float = float("-inf")
    last_press_layout: np.ndarray | None = None
    pending: tuple | None = None  # (zone, leaf_x, velocity) of a chop not yet confirmed
    history: collections.deque = field(default_factory=lambda: collections.deque(maxlen=32))
    stalled: bool = False
    hold_until: float = 0.0       # no chopping until then (recovering from a game hitch)
    speeds: collections.deque = field(default_factory=collections.deque)  # (t, |v|) recent


def run_bot(args):
    from . import inputs
    from .capture import ScreenGrabber
    from .hotkeys import VK_F10, HotkeyPoller
    from .window import GameWindow

    win = GameWindow()
    hk = HotkeyPoller(VK_F10)

    print("Choppin' bot running: open the minigame (F10 stops it).")
    print(f"Chopping with Space; stopping at {args.target} pts.")

    reader = None           # set once the bar is found
    game = None
    origin = (0, 0)         # client-area origin the reader's coords are relative to
    next_search = 0.0
    lost_since = None
    release_at = None       # when to release the currently held key

    def release():
        nonlocal release_at
        if release_at is not None:
            inputs.key_up()
            release_at = None

    def end_game():
        msg = f"Game over: {game.score} pts from {game.chops} chops."
        if game.pending:
            zone, x, v = game.pending
            msg += (f"\n  The last chop (leaf x={x:.1f}, {v:+.0f} px/s, looked {ZONE_NAMES[zone]}) "
                    f"was never confirmed - it probably landed on red.")
        print(msg)

    with ScreenGrabber() as grabber:
        try:
            while True:
                now = time.perf_counter()
                if release_at is not None and now >= release_at:
                    release()
                if hk.pressed(VK_F10):
                    break

                # Only ever look at (or type into) the game when it's the active window.
                if not win.is_foreground() or win.is_minimized():
                    release()
                    time.sleep(0.03)
                    continue

                cl, ct, cw, ch = win.client_rect()
                if reader is None:
                    if now < next_search:
                        time.sleep(0.01)
                        continue
                    next_search = now + 0.5
                    bar = find_bar(grabber.grab(cl, ct, cw, ch))
                    if bar is None:
                        continue
                    reader = ChoppinReader(bar, MARGIN)
                    game = Game()
                    origin = (cl, ct)
                    lost_since = None
                    print(f"Found bar: {bar.x1 - bar.x0} px wide (scale {bar.scale:.2f})")
                    continue
                if (cl, ct) != origin:  # window moved
                    reader = game = None
                    continue

                l, t, w, h = reader.roi()
                roi = grabber.grab(cl + l, ct + t, w, h)
                r = reader.read(roi)

                if not r.bar_ok:
                    lost_since = lost_since or now
                    if now - lost_since > 0.5:
                        end_game()
                        reader = game = None
                    continue
                lost_since = None

                if r.leaf_x is not None and (not game.history or r.leaf_x != game.history[-1][1]):
                    game.history.append((now, r.leaf_x))
                v = leaf_velocity(game.history, now)
                if v is not None:
                    # When the game stutters, the leaf's on-screen motion is jerky and a short
                    # measurement can read far too slow (the leaf never really slows down -
                    # wall bounces only speed it up). Use at least its typical recent speed.
                    game.speeds.append((now, abs(v)))
                    while now - game.speeds[0][0] > 0.5:
                        game.speeds.popleft()
                    v = math.copysign(max(abs(v), float(np.percentile([sp for _, sp in game.speeds], 75))), v)

                # The leaf never pauses mid-bar. If it hasn't moved for longer than a normal
                # frame gap, the game has hitched: its real leaf keeps going while the screen
                # is frozen. Don't chop until it's been moving smoothly again for a bit.
                if game.history and now - game.history[-1][0] > STALL:
                    if not game.stalled:
                        print(f"  (game hitched at leaf x={game.history[-1][1]:.1f} - holding off)")
                    game.stalled = True
                    game.hold_until = now + RECOVER
                else:
                    game.stalled = False

                # A chop only counts once the bar reshuffles; until then, don't chop again
                # (we'd be acting on a stale frame). Give up waiting after REARM seconds.
                if game.pending and not np.array_equal(r.zones, game.last_press_layout):
                    zone, x, v0 = game.pending
                    game.pending = None
                    pts = 2 if zone == YELLOW else 1
                    game.score += pts
                    print(f"chop {game.chops:3d}: {ZONE_NAMES[zone]:6s} +{pts} -> {game.score:3d} pts"
                          f"   (leaf x={x:6.1f}, {v0:+5.0f} px/s)")
                    if game.score >= args.target:
                        print(f"Reached {args.target} points - stopping.")
                        break
                elif game.pending and now - game.last_press_t > REARM:
                    print("  (last chop didn't seem to register)")
                    game.pending = None

                if (game.pending is None and release_at is None and r.leaf_x is not None
                        and v is not None and now >= game.hold_until
                        and now - game.last_press_t >= CHOP_GAP):
                    # The game registers the chop a little after the frame we're looking at
                    # (capture + input lag, ~5-45 ms). Chop only if everywhere the leaf could be
                    # by then, at its current speed, is green/yellow.
                    lo, hi = LEAD
                    zone = reader.zone_over(r.zones, r.leaf_x + v * lo, r.leaf_x + v * hi)
                    if zone == GREEN:
                        # Yellow is worth double and slows the leaf: skip this green if a
                        # yellow we can safely hit is coming up soon. "Safely" = its interior
                        # is longer than the leaf travels during the input-lag window plus a frame.
                        need = abs(v) * (hi - lo + 0.02)
                        if yellow_coming(r.zones, r.leaf_x, v, need, reader.margin, YELLOW_WAIT):
                            zone = None
                    if zone:
                        inputs.key_down()
                        release_at = now + HOLD
                        game.chops += 1
                        game.pending = (zone, r.leaf_x, v)
                        game.last_press_layout, game.last_press_t = r.zones.copy(), now

        except KeyboardInterrupt:
            pass
        finally:
            release()
    print("Bye!")


def main():
    ap = argparse.ArgumentParser(description="Idleon Choppin' minigame bot")
    ap.add_argument("--target", type=int, default=150, help="stop chopping at this score (default 150)")
    args = ap.parse_args()
    run_bot(args)


if __name__ == "__main__":
    main()
