"""Swishy Hoops bot.

The minigame: the player stands on a platform that bobs up and down; clicking
makes them jump and shoot. The throw itself never changes, so where the ball
goes depends only on the platform at the moment of the jump. A clean swish
scores double.

The model (measured from a recording of real shots):
  * The platform moves in a sine wave (period ~5 s, +-110 px).
  * The ball's flight is a parabola y = y200 - (U/VX)(x-200) + C(x-200)^2 with
    fixed horizontal speed VX = 390.5 px/s and curvature C = 22.8e-4, where
      - y200 = platform height 0.25 s after the click - 239.8     (rms 1.9 px)
      - U    = 187 + 0.505 * platform upward speed 0.55 s after the click
                                                                 (rms 2.8 px/s)
  * Swishes came down (ball centre at rim height) ~40-55% of the way from the
    front of the rim to the back; bounce-ins further back, misses at the front.
So each frame it fits the platform's sine, predicts where a click at every
moment over the next cycle would bring the ball down, and clicks in the middle
of the longest stretch that lands in the swish zone. A moving hoop is tracked
the same way and predicted to where it will be when the ball arrives.

Usage:
    python -m idleonbot.hoops           # F10 to stop
It stops once the score reaches 40 (the Baller trophy; --stop-at 0 to keep playing).
"""
import argparse
import collections
import math
import time

import numpy as np

from .digits import native, read_number

NATIVE_W = 960
PLAT_BOX = (0, 430, 150, 545)    # x0, x1, y0, y1: where the platform can be (it goes
                                 # elliptical later in a game, so allow sideways room)
HOOP_BOX = (300, 865, 90, 560)   # where the hoop can be (clear of the HUD icon at x~885)
PLAYER_X = 135                   # platform centre x the model below was measured at

PLAT_COLOR = np.array([(173, 113, 44)], np.int16)
RIM_COLOR = np.array([(254, 56, 7)], np.int16)
BALL_COLORS = np.array([(255, 168, 62), (255, 116, 19), (205, 73, 0)], np.int16)
COLOR_TOL = 10

VX = 390.5
C = 22.8e-4
JUMP_DT, RELEASE_DT = 0.25, 0.55
Y_OFFSET = -239.8
U0, U_INHERIT = 187.0, 0.505
# Once the platform moves sideways (elliptical stage), the whole flight shifts with the
# platform's x at release, but the ball does NOT pick up the platform's sideways speed
# (measured: ball vx stayed ~390 px/s while the platform moved at +-60 px/s).
X_INHERIT = 0.0
G = 2 * C * VX ** 2      # gravity, px/s^2
T_AT_200 = 0.70          # s after the click the ball passes x = 200
HELD_BALL_DY = -47       # held ball centre relative to the platform top

SWISH_ZONE = (0.33, 0.63)   # where on the rim (0 = front, 1 = back) swishes came down
TARGET_REL = 0.48
REL_BIAS = 0.05             # live shots came down ~0.05 further back than predicted
FALLBACK_HALF_WIDTH = 0.08  # when no swish is reachable, aim this close to the best we can do
SHOT_GAP = 2.8              # min s between shots
SWISH_PATIENCE = 20.0       # s to wait for a swish before taking a short shot at the front of the rim

# The "Score: N" counter, top left (white pixel font, 9 px tall, digits from x=49).
SCORE_BOX = (46, 110, 56, 69)    # x0, x1, y0, y1
SCORE_COLOR = np.array([(246, 249, 255)], np.int16)
TROPHY_SCORE = 40                # the Baller trophy: 40 points in one game


def read_score(img, s=1.0):
    """The game's score from a grab of SCORE_BOX (at scale s), or None if it can't be read."""
    m = match(native(img, SCORE_BOX[1] - SCORE_BOX[0], SCORE_BOX[3] - SCORE_BOX[2], s), SCORE_COLOR)
    return read_number(m, strict=True)


def match(img, colors):
    return (np.abs(img.astype(np.int16)[..., None, :] - colors).max(-1) <= COLOR_TOL).any(-1)


def fit_sine(t, y, periods, max_rms=3.0):
    """Robust sine fit: fit, drop points > 4 px off (game freezes), refit.
    Returns (w, s, c, centre, t0) or None."""
    t, y = np.asarray(t), np.asarray(y)
    t0 = t[-1]
    keep = np.ones(len(t), bool)
    for _ in range(2):
        tc, yk = t[keep] - t0, y[keep]
        if len(tc) < 10:
            return None
        w = 2 * np.pi / periods[:, None]
        S, Cc = np.sin(w * tc), np.cos(w * tc)
        n = len(tc)
        ata = np.empty((len(periods), 3, 3))
        ata[:, 0, 0], ata[:, 1, 1], ata[:, 2, 2] = (S * S).sum(1), (Cc * Cc).sum(1), n
        ata[:, 0, 1] = ata[:, 1, 0] = (S * Cc).sum(1)
        ata[:, 0, 2] = ata[:, 2, 0] = S.sum(1)
        ata[:, 1, 2] = ata[:, 2, 1] = Cc.sum(1)
        aty = np.stack([(S * yk).sum(1), (Cc * yk).sum(1), np.full(len(periods), yk.sum())], 1)
        coef = np.linalg.solve(ata + np.eye(3) * 1e-9, aty[..., None])[..., 0]
        rms = np.sqrt(((coef[:, :1] * S + coef[:, 1:2] * Cc + coef[:, 2:] - yk) ** 2).mean(1))
        i = int(np.argmin(rms))
        params = (float(w[i, 0]), *map(float, coef[i]), float(t0))
        keep = np.abs(sine_at(params, t) - y) < 4
    return params if rms[i] <= max_rms else None


def fit_growing_sine(t, y, w, max_rms=2.0):
    """Sine of known angular frequency w whose amplitude may be changing linearly (the
    platform's sideways swing ramps up over ~15 s). Fit, drop outliers, refit.
    Returns (w, coefs, t0) or None."""
    t, y = np.asarray(t), np.asarray(y)
    t0 = t[-1]
    keep = np.ones(len(t), bool)
    for _ in range(2):
        if keep.sum() < 10:
            return None
        tc = t[keep] - t0
        A = np.c_[np.sin(w * tc), np.cos(w * tc), tc * np.sin(w * tc), tc * np.cos(w * tc), np.ones_like(tc)]
        coef = np.linalg.lstsq(A, y[keep], rcond=None)[0]
        params = (float(w), coef.tolist(), float(t0))
        resid = growing_sine_at(params, t) - y
        rms = np.sqrt((resid[keep] ** 2).mean())
        keep = np.abs(resid) < 4
    return params if rms <= max_rms else None


def growing_sine_at(p, t):
    w, (a, b, c, d, e), t0 = p
    tc = np.asarray(t) - t0
    return (a + c * tc) * np.sin(w * tc) + (b + d * tc) * np.cos(w * tc) + e


def sine_at(p, t):
    w, s, c, centre, t0 = p
    return s * np.sin(w * (t - t0)) + c * np.cos(w * (t - t0)) + centre


def sine_rate(p, t):
    w, s, c, _, t0 = p
    return w * (s * np.cos(w * (t - t0)) - c * np.sin(w * (t - t0)))


class Hoop:
    """Tracks the rim; predicts it with a sine when it's moving."""

    PERIODS = np.arange(2.0, 12.0, 0.02)

    def __init__(self):
        self.samples = collections.deque(maxlen=400)  # (t, x0, x1, y)
        self.fx = self.fy = None

    def add(self, t, x0, x1, y):
        if self.samples and (abs(x0 - self.samples[-1][1]) > 15 or abs(y - self.samples[-1][3]) > 15):
            self.samples.clear()  # the hoop jumped to a new spot
        self.samples.append((t, x0, x1, y))

    def fit(self):
        s = np.array(self.samples)
        recent = s[s[:, 0] > s[-1, 0] - 4]
        self.moving_x = np.ptp(recent[:, 1]) > 2
        self.moving_y = np.ptp(recent[:, 3]) > 2
        self.width = float(np.median(recent[:, 2] - recent[:, 1]))
        # A moving hoop needs a good stretch of history (it cycles every ~4-5 s) before its
        # motion can be extrapolated ~2 s ahead; a short fit can go badly wrong.
        if (self.moving_x or self.moving_y) and s[-1, 0] - s[0, 0] < 2.5:
            self.fx = self.fy = None
            return False
        self.fx = fit_sine(recent[:, 0], recent[:, 1], self.PERIODS) if self.moving_x else None
        self.fy = fit_sine(recent[:, 0], recent[:, 3], self.PERIODS) if self.moving_y else None
        return not ((self.moving_x and self.fx is None) or (self.moving_y and self.fy is None))

    def at(self, t):
        """(front x, y) of the rim at time(s) t."""
        last = self.samples[-1]
        x = sine_at(self.fx, t) if self.fx else np.full_like(np.asarray(t, float), last[1])
        y = sine_at(self.fy, t) if self.fy else np.full_like(np.asarray(t, float), last[3])
        return x, y


class Platform:
    """Sine model of the platform: vertical always, sideways once it goes elliptical."""

    PERIODS = np.arange(3.0, 8.0, 0.02)

    def __init__(self):
        self.samples = collections.deque(maxlen=300)  # (t, x, y)
        self.fx = self.fy = None
        self.x_const = PLAYER_X
        self.sideways = False

    def add(self, t, x, y):
        self.samples.append((t, x, y))

    def span(self) -> float:
        return self.samples[-1][0] - self.samples[0][0] if self.samples else 0.0

    def fit(self) -> bool:
        p = np.array(self.samples)
        self.fy = fit_sine(p[:, 0], p[:, 2], self.PERIODS)
        recent = p[p[:, 0] > p[-1, 0] - 4]
        self.sideways = np.ptp(recent[:, 1]) > 4
        if self.sideways:
            # The sideways swing grows over ~15 s when it starts, so fit only the last few
            # seconds, sharing the vertical motion's period (it's one ellipse). If that
            # doesn't fit cleanly (e.g. right as it starts), don't trust it yet.
            self.fx = None
            if self.fy is not None:
                last = p[p[:, 0] > p[-1, 0] - 3.5]
                self.fx = fit_growing_sine(last[:, 0], last[:, 1], self.fy[0])
        else:
            self.fx, self.x_const = None, float(np.median(recent[:, 1]))
        return self.fy is not None and (self.fx is not None or not self.sideways)

    @property
    def period(self) -> float:
        return 2 * np.pi / self.fy[0]

    def y(self, t):
        return sine_at(self.fy, t)

    def vy(self, t):
        return sine_rate(self.fy, t)

    def x(self, t):
        return growing_sine_at(self.fx, t) if self.fx else self.x_const

    def vx(self, t):
        return (growing_sine_at(self.fx, t + 0.005) - growing_sine_at(self.fx, t - 0.005)) / 0.01 if self.fx else 0.0


def predict_rel(plat: Platform, hoop: Hoop, t_click):
    """Where on the rim (0 front .. 1 back) a click at time(s) t_click comes down."""
    y_ref = plat.y(t_click + JUMP_DT) + Y_OFFSET             # ball height at x_ref
    u = U0 + U_INHERIT * -plat.vy(t_click + RELEASE_DT)      # upward speed at x_ref
    x_ref = 200 + plat.x(t_click + RELEASE_DT) - PLAYER_X    # the path shifts with the platform
    vx = VX + X_INHERIT * plat.vx(t_click + RELEASE_DT)
    c = G / (2 * vx ** 2)
    rim_x, rim_y = hoop.at(t_click + 2.0)
    for _ in range(2):  # the arrival time depends on where it comes down; iterate
        b = -u / vx
        disc = b * b - 4 * c * (y_ref - rim_y)
        du = (-b + np.sqrt(np.where(disc >= 0, disc, np.nan))) / (2 * c)
        t_arrive = t_click + T_AT_200 + np.nan_to_num(du, nan=500) / vx
        rim_x, rim_y = hoop.at(t_arrive)
    return (x_ref + du - rim_x) / hoop.width + REL_BIAS


def plan_shot(plat: Platform, hoop, now, step=0.01, lookback=1.5):
    """The middle of the longest stretch of click times that land in the swish zone,
    or, if no swish is reachable, that land as close to it as possible.
    Returns (t_click, rel, stretch seconds, kind, (reachable min, max)) or None."""
    # With a moving hoop, the platform and hoop are on different cycles, so a swish may
    # only line up after more than one platform cycle: look two cycles ahead for one.
    cycles = 2 if (hoop.fx or hoop.fy) else 1
    ts = now - lookback + np.arange(0, lookback + cycles * plat.period + 0.5, step)
    rel = predict_rel(plat, hoop, ts)
    if not np.isfinite(rel).any():
        return None
    plan = _best_stretch(ts, rel, SWISH_ZONE, now, step, target=TARGET_REL)
    one = ts <= now + plat.period + 0.5
    reach = (float(np.nanmin(rel[one])), float(np.nanmax(rel[one])))
    if plan:
        return (*plan, "swish", reach)
    # No swish: take the closest shot within the next cycle rather than waiting longer.
    ts, rel = ts[one], rel[one]
    closest = float(rel[np.nanargmin(np.abs(rel - TARGET_REL))])
    zone = (closest - FALLBACK_HALF_WIDTH, closest + FALLBACK_HALF_WIDTH)
    plan = _best_stretch(ts, rel, zone, now, step)
    return (*plan, "closest", reach) if plan else None


def _best_stretch(ts, rel, zone, now, step, target=None, edge_s=0.12, good=0.08):
    """Pick when to shoot among the stretches of click times that land in `zone`.

    Within a stretch, the best moment is the one still ahead of us whose landing is
    closest to `target`, keeping min(edge_s, a quarter of the stretch) clear of both ends
    for timing slack. When the landing sweeps across the zone that's its middle; when it
    hovers near one side (platform near the top/bottom of its swing) it's the most central
    point it reaches. Take the soonest stretch whose best moment lands within `good` of the
    target, else whichever gets closest. (Preferring the soonest matters: always holding
    out for a slightly better stretch one cycle later means never shooting.)
    The grid starts before `now` so a stretch we're already in keeps its true shape."""
    target = (zone[0] + zone[1]) / 2 if target is None else target
    inside = (rel >= zone[0]) & (rel <= zone[1])
    edges = np.flatnonzero(np.diff(np.concatenate(([0], inside.astype(np.int8), [0]))))
    now_i = int(np.searchsorted(ts, now))
    cands = []  # (index, distance from target, stretch length)
    for a, b in zip(edges[::2], edges[1::2]):
        if a == 0 or b >= len(ts):
            continue  # cut off by the grid: we can't see its real extent
        pad = min((b - a) // 4, int(round(edge_s / step)))
        lo, hi = max(a + pad, now_i), b - pad
        if lo >= hi:
            continue  # its usable part has already gone by
        j = lo + int(np.argmin(np.abs(rel[lo:hi] - target)))
        cands.append((j, abs(rel[j] - target), b - a))
    if not cands:
        return None
    good_ones = [c for c in cands if c[1] <= good]
    j, _, length = good_ones[0] if good_ones else min(cands, key=lambda c: c[1])
    return float(ts[j]), float(rel[j]), float(length * step)


def run_bot(args):
    from . import inputs
    from .capture import ScreenGrabber
    from .hotkeys import VK_F10, HotkeyPoller
    from .window import GameWindow

    win = GameWindow()
    hk = HotkeyPoller(VK_F10)

    print("Hoops bot running: open Swishy Hoops (F10 stops it).")
    plat = Platform()
    hoop = Hoop()
    told_sideways = False
    last_shot = float("-inf")
    shot = None
    shots = 0
    frame_dt, last_t = 0.02, None
    no_flight = 0                # consecutive shots whose ball never showed up near the hoop
    no_swish_since = None        # when we started finding no reachable swish for this hoop
    fitted_at = float("-inf")    # refitting every frame is slow; the motion doesn't change fast
    fits_ok = False
    why, why_since, warned = "", 0.0, False
    score_read_at, at_target = float("-inf"), 0

    def native_rect(cl, ct, s, x0, x1, y0, y1):
        return cl + int(x0 * s), ct + int(y0 * s), int((x1 - x0) * s), int((y1 - y0) * s)

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
                if last_t is not None:
                    frame_dt = 0.9 * frame_dt + 0.1 * min(now - last_t, 0.2)
                last_t = now

                # The trophy: stop once the game's score counter shows it (read twice to be sure).
                if args.stop_at and now - score_read_at > 0.25:
                    score_read_at = now
                    v = read_score(grabber.grab(*native_rect(cl, ct, s, *SCORE_BOX)), s)
                    at_target = at_target + 1 if v is not None and v >= args.stop_at else 0
                    if at_target >= 2:
                        print(f"Score {v} - that's the Baller trophy ({args.stop_at} points)! Stopping.")
                        print("If this trophy is worth as much as a supporter pack to you, please consider tipping here! Ko-fi.com/idlerobot")
                        break

                # Platform: top rows of the platform's brown.
                pimg = grabber.grab(*native_rect(cl, ct, s, *PLAT_BOX))
                pm = match(pimg, PLAT_COLOR)
                rows = np.flatnonzero(pm.sum(1) > 40 * s)
                plat_y = plat_x = None
                if len(rows):
                    plat_y = PLAT_BOX[2] + rows[0] / s
                    plat_x = PLAT_BOX[0] + np.nonzero(pm[rows[0]:rows[0] + int(12 * s)])[1].mean() / s
                    plat.add(now, plat_x, plat_y)
                # Ready = holding a ball on the platform.
                ready = False
                if plat_y is not None:
                    hy = int((plat_y + HELD_BALL_DY - PLAT_BOX[2]) * s)
                    hx = int((plat_x - PLAT_BOX[0]) * s)
                    r = int(25 * s)
                    ready = match(pimg[max(0, hy - r):hy + r, max(0, hx - r):hx + r], BALL_COLORS).sum() > 250 * s * s

                # Hoop: the row with the most rim colour.
                himg = grabber.grab(*native_rect(cl, ct, s, *HOOP_BOX))
                hm = match(himg, RIM_COLOR)
                ys, xs = np.nonzero(hm)
                if len(xs) > 30 * s:
                    row = np.bincount(ys).argmax()
                    rx = xs[ys == row]
                    hoop.add(now, HOOP_BOX[0] + rx.min() / s, HOOP_BOX[0] + rx.max() / s, HOOP_BOX[2] + row / s)

                # Follow the last shot's ball while it's clearly above the rim, then fit its arc
                # to see where it actually came down (a swish vanishes into the net).
                if shot and "actual_rel" not in shot:
                    if now - shot["t"] < 2.4:
                        bys, bxs = np.nonzero(match(himg, BALL_COLORS))
                        rim_now = hoop.samples[-1] if hoop.samples else None
                        if len(bxs) > 600 * s * s and rim_now:
                            bx, by = HOOP_BOX[0] + bxs.mean() / s, HOOP_BOX[2] + bys.mean() / s
                            if shot["pts"] and bx < shot["pts"][-1][0] - 2:
                                shot["bounced"] = True  # coming back off the rim/backboard
                            if by < rim_now[3] - 25 and bx < rim_now[1] and not shot.get("bounced"):
                                shot["pts"].append((bx, by, rim_now[1], rim_now[2], rim_now[3]))
                    else:
                        pts = np.array(shot["pts"])
                        if len(pts) >= 4 and np.ptp(pts[:, 0]) > 60:
                            rx0, rx1, ry = pts[-1, 2:]
                            if len(pts) >= 6 and np.ptp(pts[:, 0]) > 150:
                                cc, bb, aa = np.polyfit(pts[:, 0] - 200, pts[:, 1], 2)
                            else:  # too few points for a free fit: use the known curvature
                                cc = C
                                A = np.c_[pts[:, 0] - 200, np.ones(len(pts))]
                                bb, aa = np.linalg.lstsq(A, pts[:, 1] - C * (pts[:, 0] - 200) ** 2, rcond=None)[0]
                            disc = bb * bb - 4 * cc * (aa - ry)
                            xc = 200 + (-bb + math.sqrt(max(disc, 0))) / (2 * cc)
                            shot["actual_rel"] = float((xc - rx0) / (rx1 - rx0))
                            print(f"          came down at {shot['actual_rel']:.2f} of the rim "
                                  f"(aimed {shot['rel']:.2f})")
                        else:
                            shot["actual_rel"] = None
                        # Three shots in a row that never reached the hoop: the game has ended.
                        no_flight = 0 if shot["pts"] else no_flight + 1
                        if no_flight >= 3:
                            print("Last three shots never reached the hoop - game over. Stopping.")
                            break

                def wait(reason):
                    nonlocal why, why_since, warned
                    if reason.split(" (")[0] != why.split(" (")[0]:
                        why, why_since, warned = reason, now, False
                    elif not warned and now - why_since > 8:
                        print(f"  (not shooting for 8 s: {reason})")
                        warned = True

                if now - last_shot < SHOT_GAP:
                    wait("waiting for the last shot to finish")
                    continue
                if not ready:
                    wait("no ball in hand")
                    continue
                if len(plat.samples) < 20 or plat.span() < 1.5:
                    wait("watching the platform")
                    continue
                if not hoop.samples or now - hoop.samples[0][0] < 0.5:
                    wait("watching the hoop")
                    continue
                if now - fitted_at > 0.15:
                    fitted_at = now
                    fits_ok = plat.fit() and hoop.fit()
                    if not fits_ok:
                        fit_problem = ("can't fit the platform's motion" if plat.fy is None or
                                       (plat.sideways and plat.fx is None) else "hoop is moving; learning its motion")
                if not fits_ok:
                    wait(fit_problem)
                    continue
                if plat.sideways and not told_sideways:
                    print("  (platform is moving sideways now)")
                    told_sideways = True
                plan = plan_shot(plat, hoop, now)
                if plan is None:
                    wait("no click time lands anywhere near the hoop")
                    continue
                t_click, rel, window, kind, reach = plan
                if kind == "swish":
                    no_swish_since = None
                else:
                    no_swish_since = no_swish_since or now
                    # Short of the swish zone means clipping the front of the rim, which has
                    # always missed; long shots often bounce in off the back. When only short
                    # shots are possible, wait for the hoop's and platform's cycles to drift
                    # into a better alignment (they can move nearly in step for a while).
                    if rel < SWISH_ZONE[0] and now - no_swish_since < SWISH_PATIENCE:
                        wait(f"no swish reachable yet, only short shots ({reach[1]:.2f} max); "
                             f"waiting for the hoop to line up")
                        continue
                wait(f"waiting to shoot (in {t_click - now:.1f} s)")
                if t_click - now > frame_dt:
                    continue

                cursor = inputs.get_cursor()
                inputs.mouse_down(cl + int(498 * s), ct + int(301 * s))
                inputs.mouse_up()
                inputs.set_cursor(*cursor)
                shots += 1
                last_shot = now
                no_swish_since = None
                rim = hoop.samples[-1]
                shot = {"t": now, "rel": rel, "pts": []}
                if kind == "swish":
                    print(f"shot {shots:2d}: aiming {rel:.2f} of the way into the rim "
                          f"({window * 1000:.0f} ms window{', moving hoop' if hoop.fx or hoop.fy else ''})")
                else:
                    print(f"shot {shots:2d}: no swish reachable (rim at x={rim[1]:.0f}, y={rim[3]:.0f}; "
                          f"reachable {reach[0]:.2f} to {reach[1]:.2f}) - going for {rel:.2f}")
        except KeyboardInterrupt:
            pass
    print("Bye!")


def main():
    ap = argparse.ArgumentParser(description="Idleon Swishy Hoops bot")
    ap.add_argument("--stop-at", type=int, default=TROPHY_SCORE,
                    help=f"stop once the score reaches this (default {TROPHY_SCORE}, the Baller trophy; "
                         f"0 = keep playing)")
    run_bot(ap.parse_args())


if __name__ == "__main__":
    main()
