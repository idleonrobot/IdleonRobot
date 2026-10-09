"""Throwy Darts bot.

The minigame: the player's arm sweeps the dart up and down; clicking throws
it. The dart flies in a parabola to a striped board on the right, and the red
stripe in the middle is the bullseye (+5). The player moves after every hit,
wind changes every 6 hits, and the sweep speeds up over the game.

How the bot aims (all measured from a recording of real throws):
  * The sweep is a clean sine wave (within ~1 degree), so the arm angle can be
    predicted a fraction of a second ahead.
  * The dart leaves the hand ~20-35 ms after the click (less when the arm is
    rising than when it's lowering), from ~11 px above the
    in-hand dart, at the hand angle minus ~3 degrees.
  * The flight is an exact parabola in screen space:
        y = y0 + tan(th)*dx + K/cos(th)^2 * dx^2
    with K = 5.9e-4 without wind. Wind changes K.
  * The stripe it scores is the parabola's height at x = 920.
So each frame it fits the sweep, predicts where a release at every moment of
the next swing would land, and clicks so the dart leaves in the middle of the
longest stretch that lands on the bullseye (or the best stripe it can reach). After every throw it
fits the actual flight and updates its click->release delays and the K for the
current wind; these are saved to darts_calibration.json between runs.

Usage:
    python -m idleonbot.darts               # F10 to stop; stops by itself after 9 bullseyes in a row
"""
import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .digits import native, read_number

# Geometry at the game's native client size (960 px wide); scaled for other sizes.
NATIVE_W = 960
PLAY_TOP = 90            # playfield starts below the HUD
PLAY_RIGHT = 830         # darts stuck in the board sit right of this
BOARD_X = 920            # x where the flight path's height decides the stripe
WIND_BOX = (480, 40, 606, 82)
TROPHY_BOX = (0, 530, 60, 572)  # yellow darts icon, bottom-left

FLIGHT_COLORS = np.array([(255, 224, 67), (215, 148, 13)], np.int16)
SHAFT_COLORS = np.array([(214, 214, 214), (173, 173, 173), (80, 80, 80)], np.int16)
COLOR_TOL = 8

K_NO_WIND = 5.9e-4
LAUNCH_DY = -11.0        # launch point relative to the in-hand dart centre
LAUNCH_DANGLE = -3.0     # launch angle relative to the in-hand dart angle
# Click -> release, in seconds of screen time, learned from live throws; it depends on
# which way the arm is moving at release.
DEFAULT_DELAYS = {"lowering": 0.0253, "raising": 0.0204}
# K for each wind seen so far (keyed by a hash of the wind indicator), learned from live
# throws; a new wind starts at K_NO_WIND. darts_calibration.json refines these as it plays.
KNOWN_WINDS = {
    "7a3606b3": 5.932e-4, "1ec4eb49": 5.527e-4, "791ec5bf": 5.500e-4, "ca3bbb56": 5.383e-4,
    "889c7811": 6.045e-4, "8c04ee60": 5.421e-4, "bdf0a151": 5.676e-4, "65c10744": 5.082e-4,
    "43c963f9": 5.172e-4, "bb7f6d37": 6.752e-4, "1fc733f5": 7.128e-4, "836e2e2e": 6.166e-4,
    "4fa04acd": 6.177e-4, "c6c85f1f": 5.255e-4, "4582d19e": 5.195e-4, "184f6da4": 5.225e-4,
    "bfb9b190": 5.231e-4, "96f890b4": 4.557e-4, "53aa51ab": 4.460e-4, "a3d5153c": 6.318e-4,
    "441f3c0c": 6.254e-4, "0c06efab": 5.266e-4, "1a8d7b8c": 5.292e-4, "4f04f9c3": 5.525e-4,
    "be011c9c": 5.515e-4,
}

# A wind not learned yet: guess its K from the indicator's arrow. In the eight winds
# measured with their indicator (3 and 4 mph, arrows tilted -43 to +42 degrees), an arrow
# tilted up lowered K and one tilted down raised it; the effect builds up over the first
# ~6 degrees of tilt and then stays the same, at either speed (fits them to ~0.15e-4).
WIND_K0, WIND_DK, WIND_TILT = 5.61e-4, 0.424e-4, 6.0

CALIBRATION_FILE = Path("darts_calibration.json")


def match(img: np.ndarray, colors: np.ndarray) -> np.ndarray:
    diff = np.abs(img.astype(np.int16)[..., None, :] - colors)
    return (diff.max(-1) <= COLOR_TOL).any(-1)


@dataclass
class Dart:
    x: float      # centre of the dart's flight + shaft pixels, in native px
    y: float
    angle: float  # degrees, positive = pointing down


def find_dart(img: np.ndarray, s: float, ox: float = 0, oy: float = 0) -> Dart | None:
    """Find the leftmost dart in `img` (a capture whose top-left is native (ox, oy))."""
    fl = match(img, FLIGHT_COLORS)
    ys, xs = np.nonzero(fl)
    if len(xs) < 8 * s * s:
        return None
    left = xs < xs.min() + 25 * s
    xs, ys = xs[left], ys[left]
    fx, fy = xs.mean(), ys.mean()
    # Only look for the shaft near the flights (cheap, and ignores other darts).
    x0, y0 = max(0, int(fx - 20 * s)), max(0, int(fy - 70 * s))
    win = img[y0:int(fy + 70 * s), x0:int(fx + 70 * s)]
    sy, sx = np.nonzero(match(win, SHAFT_COLORS))
    sx, sy = sx + x0, sy + y0
    d = np.hypot(sx - fx, sy - fy)
    near = (d > 6 * s) & (d < 60 * s) & (sx > fx - 5 * s)
    if near.sum() < 5 * s * s:
        return None
    angle = math.degrees(math.atan2(sy[near].mean() - fy, sx[near].mean() - fx))
    n = len(xs) + near.sum()
    cx = (xs.sum() + sx[near].sum()) / n
    cy = (ys.sum() + sy[near].sum()) / n
    return Dart(ox + cx / s, oy + cy / s, angle)


class Sweep:
    """Sine-wave model of the arm angle, plus the pivot the dart rotates around."""

    PERIODS = np.arange(1.2, 6.0, 0.01)

    def __init__(self):
        self.t, self.x, self.y, self.a = [], [], [], []
        self.params = None  # (w, s, c, centre)
        self.pivot = None   # (px, py, r)

    def add(self, t: float, d: Dart):
        self.t.append(t), self.x.append(d.x), self.y.append(d.y), self.a.append(d.angle)

    def span(self) -> float:
        return self.t[-1] - self.t[0] if self.t else 0.0

    def fit(self, window: float = 1.5) -> bool:
        t = np.array(self.t)
        keep = t >= t[-1] - window
        t, a = t[keep], np.array(self.a)[keep]
        if len(t) < 12 or t[-1] - t[0] < 0.4:
            return False
        tc = t - t[-1]
        w = 2 * np.pi / self.PERIODS[:, None]
        S, C = np.sin(w * tc), np.cos(w * tc)
        n = len(t)
        ata = np.empty((len(self.PERIODS), 3, 3))
        ata[:, 0, 0], ata[:, 1, 1], ata[:, 2, 2] = (S * S).sum(1), (C * C).sum(1), n
        ata[:, 0, 1] = ata[:, 1, 0] = (S * C).sum(1)
        ata[:, 0, 2] = ata[:, 2, 0] = S.sum(1)
        ata[:, 1, 2] = ata[:, 2, 1] = C.sum(1)
        aty = np.stack([(S * a).sum(1), (C * a).sum(1), np.full(len(self.PERIODS), a.sum())], 1)
        coef = np.linalg.solve(ata + np.eye(3) * 1e-9, aty[..., None])[..., 0]
        # RMS error: repeated game frames make the samples a staircase, so max error is too strict.
        resid = np.sqrt(((coef[:, :1] * S + coef[:, 1:2] * C + coef[:, 2:] - a) ** 2).mean(1))
        amp = np.hypot(coef[:, 0], coef[:, 1])
        # Sanity: sweeps are centred near -20 deg, amplitude 38-47 (grows over a game).
        ok = (amp > 20) & (amp < 80) & (coef[:, 2] > -40) & (coef[:, 2] < 0)
        if not ok.any():
            return False
        i = np.flatnonzero(ok)[np.argmin(resid[ok])]
        if resid[i] > 1.5:
            return False
        self.params = (float(w[i, 0]), *map(float, coef[i]), float(t[-1]))

        # Pivot: centre = pivot + r * (cos a, sin a), linear least squares.
        ar = np.radians(np.array(self.a)[keep])
        A = np.zeros((2 * n, 3))
        b = np.empty(2 * n)
        A[0::2, 0], A[0::2, 2], b[0::2] = 1, np.cos(ar), np.array(self.x)[keep]
        A[1::2, 1], A[1::2, 2], b[1::2] = 1, np.sin(ar), np.array(self.y)[keep]
        self.pivot = tuple(np.linalg.lstsq(A, b, rcond=None)[0])
        return True

    def angle(self, t):
        w, s, c, centre, t0 = self.params
        return s * np.sin(w * (t - t0)) + c * np.cos(w * (t - t0)) + centre

    @property
    def period(self) -> float:
        return 2 * np.pi / self.params[0]

    def rate(self, t):
        """Angular speed in deg/s (positive = arm lowering)."""
        w, s, c, _, t0 = self.params
        return s * w * np.cos(w * (t - t0)) - c * w * np.sin(w * (t - t0))

    def next_time(self, target: float, t_from: float, horizon: float) -> float | None:
        """Earliest time >= t_from when the sweep passes through `target` degrees."""
        ts = t_from + np.arange(0, horizon, 0.001)
        d = self.angle(ts) - target
        cross = np.flatnonzero(np.sign(d[:-1]) != np.sign(d[1:]))
        return float(ts[cross[0]]) if len(cross) else None


def launch(pivot, alpha: float) -> tuple[float, float, float]:
    """(x0, y0, theta) of the flight for a release at hand angle `alpha`."""
    px, py, r = pivot
    a = math.radians(alpha)
    return px + r * math.cos(a), py + r * math.sin(a) + LAUNCH_DY, a + math.radians(LAUNCH_DANGLE)


def landing_y(pivot, alpha, k: float):
    """Board height the throw lands at, for a release at hand angle(s) `alpha`."""
    px, py, r = pivot
    a = np.radians(alpha)
    x0, y0 = px + r * np.cos(a), py + r * np.sin(a) + LAUNCH_DY
    th = a + math.radians(LAUNCH_DANGLE)
    dx = BOARD_X - x0
    return y0 + np.tan(th) * dx + k / np.cos(th) ** 2 * dx * dx


# What to aim for, best first. Stay a couple of px inside each stripe (its edges are
# less certain than its middle).
AIM_ZONES = [("bullseye", [(307, 341)]), ("green", [(255, 301), (347, 393)]),
             ("tan", [(196, 249), (399, 452)])]
EDGE_MARGIN = 2


def plan_release(sweep: "Sweep", k: float, now: float, delays: dict,
                 lookback: float = 0.3, step: float = 0.002):
    """Choose when to release over the next sweep: the middle of the longest stretch of
    time in which every release lands in the best reachable stripe. The longer the
    stretch, the more timing error the throw can absorb. Returns
    (t_release, t_click, hand angle, stretch seconds, zone name, direction) or None."""
    ts = now - lookback + np.arange(0, sweep.period + lookback + 0.3, step)
    al = sweep.angle(ts)
    lowering = sweep.rate(ts) > 0
    click = ts - np.where(lowering, delays["lowering"], delays["raising"])
    y = landing_y(sweep.pivot, al, k)
    direct = al >= -50  # steeper than this is a lob; the model is untested there
    for name, bands in AIM_ZONES:
        inside = np.zeros(len(ts), bool)
        for lo, hi in bands:
            inside |= (y >= lo + EDGE_MARGIN) & (y <= hi - EDGE_MARGIN)
        inside &= direct
        edges = np.flatnonzero(np.diff(np.concatenate(([0], inside.astype(np.int8), [0]))))
        starts, ends = edges[::2], edges[1::2]
        mids = (starts + ends - 1) // 2
        # Complete stretches whose middle we can still click in time for.
        ok = (ends < len(ts)) & (click[mids] >= now)
        if not ok.any():
            continue
        idx = np.flatnonzero(ok)
        lengths = ends[idx] - starts[idx]
        # Prefer the soonest stretch unless a later one is clearly longer.
        best = idx[0] if lengths[0] >= 0.9 * lengths.max() else idx[np.argmax(lengths)]
        i = mids[best]
        return (float(ts[i]), float(click[i]), float(al[i]), float((ends[best] - starts[best]) * step),
                name, "lowering" if lowering[i] else "raising")
    return None


def stripe_name(y: float) -> str:
    """The stripe a dart landing at height y scores. A dart measured at y=306, 1 px into the
    5 px divider above the bullseye, counted as a bullseye, so each divider is split
    between the stripes either side of it."""
    for lo, hi, name in [(123, 190, "gray"), (196, 249, "tan"), (255, 301, "green"),
                         (307, 341, "BULLSEYE"), (347, 393, "green"), (399, 452, "tan"),
                         (458, 525, "gray")]:
        if lo - 2.5 <= y <= hi + 2.5:
            return name
    return "miss"


def fit_flight(points: list[tuple[float, float]], pivot, alpha_guess: float):
    """Fit an observed flight. Returns (release hand angle, K, landing y) or None."""
    p = np.array(points)
    if len(p) < 6 or p[-1, 0] - p[0, 0] < 150:
        return None
    c2, c1, c0 = np.polyfit(p[:, 0], p[:, 1], 2)
    x0, _, _ = launch(pivot, alpha_guess)
    th = math.atan(c1 + 2 * c2 * x0)
    return (math.degrees(th) - LAUNCH_DANGLE, c2 * math.cos(th) ** 2,
            float(np.polyval((c2, c1, c0), BOARD_X)))


def read_wind(img):
    """(mph, arrow direction in degrees: 0 = blowing right, + = up) from a native-scale grab
    of WIND_BOX; either is None if it can't be read."""
    img = img.astype(np.int16)
    blue = (img[..., 2] > 150) & (img[..., 2] > img[..., 0] + 40)   # the speed text and arrow
    mph = read_number(blue[:, :75])
    ys, xs = np.nonzero(blue[:, 75:])
    if len(xs) < 200:
        return mph, None
    pts = np.c_[xs - xs.mean(), ys - ys.mean()]
    axis = np.linalg.eigh(np.cov(pts.T))[1][:, 1]                    # the arrow's long axis
    if ((pts @ axis) ** 3).mean() > 0:      # the head is the heavy end: point the axis at it
        axis = -axis
    return mph, float(np.degrees(np.arctan2(-axis[1], axis[0])))


def guess_k(mph, angle):
    """K for a wind that hasn't been learned yet (see WIND_K0)."""
    if angle is None:
        return K_NO_WIND
    tilt = math.degrees(math.asin(math.sin(math.radians(angle))))   # up/down part, -90..90
    return WIND_K0 - WIND_DK * math.tanh(tilt / WIND_TILT)


class Calibration:
    """Click->release delays (per sweep direction) and per-wind K, refined after every throw and saved."""

    def __init__(self, path: Path = CALIBRATION_FILE):
        self.path = path
        self.delay = dict(DEFAULT_DELAYS)
        self.k = dict(KNOWN_WINDS)
        self.winds = {}         # what each wind's indicator showed: {"mph": .., "angle": ..}
        if path.exists():
            d = json.loads(path.read_text())
            if isinstance(d.get("delay"), dict):
                self.delay.update(d["delay"])
            self.k.update(d.get("k", {}))
            self.winds = d.get("winds", {})

    def k_for(self, wind: str) -> float:
        if wind in self.k:
            return self.k[wind]
        w = self.winds.get(wind, {})
        return guess_k(w.get("mph"), w.get("angle"))

    def note(self, wind: str, mph, angle):
        """Remember what a wind's indicator showed (so its effect can be modelled)."""
        if wind not in self.winds and mph is not None and angle is not None:
            self.winds[wind] = {"mph": mph, "angle": round(angle, 1)}
            self.save()

    def save(self):
        self.path.write_text(json.dumps({"delay": self.delay, "k": self.k, "winds": self.winds}, indent=1))

    def update(self, wind: str, k_obs: float, direction: str, delay_obs: float | None):
        self.k[wind] = k_obs if wind not in self.k else 0.5 * self.k[wind] + 0.5 * k_obs
        if delay_obs is not None:
            self.delay[direction] = 0.8 * self.delay[direction] + 0.2 * delay_obs
        self.save()


def run_bot(args):
    from . import inputs
    from .capture import ScreenGrabber
    from .hotkeys import VK_F10, HotkeyPoller
    from .window import GameWindow

    win = GameWindow()
    hk = HotkeyPoller(VK_F10)
    cal = Calibration()

    print("Darts bot running: open Throwy Darts (F10 stops it).")

    state = "search"
    sweep = Sweep()
    last_seen = None
    lost_since = None
    throw = None
    throws = 0
    bullseyes = 0                # in a row
    frame_dt = 0.015
    last_t = None

    def native_rect(cl, ct, s, x0, y0, x1, y1):
        """Screen rect for a native-coordinate box."""
        return cl + int(x0 * s), ct + int(y0 * s), int((x1 - x0) * s), int((y1 - y0) * s)

    def playfield(grabber, cl, ct, s, play_bottom):
        """The playfield, with the yellow darts icon (bottom left) blanked: it looks like a dart."""
        img = grabber.grab(*native_rect(cl, ct, s, 0, PLAY_TOP, PLAY_RIGHT, play_bottom)).copy()
        img[int((TROPHY_BOX[1] - PLAY_TOP) * s):, :int(TROPHY_BOX[2] * s)] = 0
        return img

    with ScreenGrabber() as grabber:
        try:
            while True:
                now = time.perf_counter()
                if hk.pressed(VK_F10):
                    break
                if not win.is_foreground() or win.is_minimized():
                    time.sleep(0.03)
                    continue

                cl, ct, cw, ch = win.client_rect()
                s = cw / NATIVE_W
                play_bottom = ch / s

                if state == "search":
                    # Find the dart in the player's hand: a dart that isn't moving between scans.
                    img = playfield(grabber, cl, ct, s, play_bottom)
                    d = find_dart(img, s, 0, PLAY_TOP)
                    # A dart that stays within 25 px for 0.15 s is in the hand (a flying one moves ~80 px).
                    if d and last_seen and math.hypot(d.x - last_seen[0].x, d.y - last_seen[0].y) >= 25:
                        last_seen = None
                    if d and last_seen and now - last_seen[1] >= 0.15:
                        state, sweep, lost_since, last_t = "aim", Sweep(), None, None
                        wimg = grabber.grab(*native_rect(cl, ct, s, *WIND_BOX)).copy()
                        wind = hashlib.md5(wimg.tobytes()).hexdigest()[:8]
                        mph, angle = read_wind(native(wimg, WIND_BOX[2] - WIND_BOX[0], WIND_BOX[3] - WIND_BOX[1], s))
                        cal.note(wind, mph, angle)
                        roi_c = (d.x, d.y)
                    else:
                        if d is None:
                            lost_since = lost_since or now
                            if now - lost_since > 5 and throws:
                                print("No dart for 5 s - game over. Stopping.")
                                break
                        else:
                            lost_since = None
                        if d and last_seen is None:
                            last_seen = (d, now)
                        time.sleep(0.03)
                    continue

                if state in ("aim", "released"):
                    x0, y0 = roi_c[0] - 60, roi_c[1] - 70
                    img = grabber.grab(*native_rect(cl, ct, s, x0, y0, roi_c[0] + 90, roi_c[1] + 70))
                    if last_t is not None:
                        frame_dt = 0.9 * frame_dt + 0.1 * min(now - last_t, 0.05)
                    last_t = now
                    d = find_dart(img, s, x0, y0)
                    # The dart centre swings ~25 px around the hand; further than 45 means it's gone.
                    moved = d is None or math.hypot(d.x - roi_c[0], d.y - roi_c[1]) > 45

                    if state == "released":
                        if moved or now - throw["click_t"] > 0.4:
                            state, throw["flight"], throw["left_hand"] = "flight", [], moved
                        continue
                    if moved:
                        state, last_seen = "search", None
                        continue

                    sweep.add(now, d)
                    if sweep.span() < 0.5 or not sweep.fit():
                        continue
                    plan = plan_release(sweep, cal.k_for(wind), now, cal.delay)
                    if plan is None:
                        continue
                    _, t_click, target, _, _, direction = plan
                    if t_click - now > frame_dt / 2:
                        continue

                    # Throw! Click in the playfield, then put the cursor back.
                    cursor = inputs.get_cursor()
                    cx, cy = cl + int(816 * s), ct + int(343 * s)
                    inputs.mouse_down(cx, cy)
                    inputs.mouse_up()
                    inputs.set_cursor(*cursor)
                    throws += 1
                    throw = {"click_t": now, "wind": wind, "direction": direction,
                             "target": target, "pivot": sweep.pivot}
                    state = "released"
                    continue

                if state == "flight":
                    img = playfield(grabber, cl, ct, s, play_bottom)
                    d = find_dart(img, s, 0, PLAY_TOP)
                    elapsed = now - throw["click_t"]
                    if d is not None and d.x > roi_c[0] - 20 and (not throw["flight"] or d.x >= throw["flight"][-1][0]):
                        throw["flight"].append((d.x, d.y))
                    done = elapsed > 1.8 or (throw["flight"] and throw["flight"][-1][0] > PLAY_RIGHT - 40)
                    if not done:
                        continue
                    fit = fit_flight(throw["flight"], throw["pivot"], throw["target"])
                    if fit:
                        alpha_obs, k_obs, _ = fit
                        # When did the sweep actually reach the release angle? -> real delay.
                        t_rel = sweep.next_time(alpha_obs, throw["click_t"], 0.35)
                        delay_obs = None if t_rel is None else t_rel - throw["click_t"]
                        # Only learn from throws that look like ours (not a stray or misread flight).
                        if (throw["left_hand"] and abs(alpha_obs - throw["target"]) < 8 and 2e-4 < k_obs < 9e-4
                                and delay_obs is not None and 0 < delay_obs < 0.25):
                            cal.update(throw["wind"], k_obs, throw["direction"], delay_obs)
                    state, last_seen = "search", None
                    # The trophy: a nine-dart finish (only bullseyes the bot saw land count).
                    bullseyes = bullseyes + 1 if fit and stripe_name(fit[2]) == "BULLSEYE" else 0
                    print(f"throw {throws}: " + ("didn't see it land" if not fit else "miss" if not bullseyes else
                                                 f"{bullseyes} bullseye{'s' if bullseyes > 1 else ''} in a row"))
                    if args.stop_at and bullseyes >= args.stop_at:
                        print(f"That's {bullseyes} bullseyes in a row - the Nine Dart Finish trophy! Stopping.")
                        print("If this trophy is worth as much as a supporter pack to you, please consider tipping here! Ko-fi.com/idlerobot")
                        break
        except KeyboardInterrupt:
            pass
    print("Bye!")


def main():
    ap = argparse.ArgumentParser(description="Idleon Throwy Darts bot")
    ap.add_argument("--stop-at", type=int, default=9,
                    help="stop after this many bullseyes in a row (default 9, the Nine Dart Finish "
                         "trophy; 0 = keep playing)")
    args = ap.parse_args()
    run_bot(args)


if __name__ == "__main__":
    main()
