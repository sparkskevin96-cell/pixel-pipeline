"""Premium spell / hit effects: big, many-frame, hand-crafted-looking pixel FX (CPU, no AI).

Kevin (2026-10-13): "a bit like SNES but as if 2D consoles never stopped... not limited
palettes, bigger palettes and great FXs". So: crisp readable shapes, 30-60 colour hue-shifted
ramps (fire: white-yellow core -> orange -> red -> deep purple smoke), 12-32 frames with
anticipation -> impact -> dissipation timing (non-uniform frame durations), smears, glow maps.

    render_spell("fire_burst", theme="ice", size="large")   -> effects.Effect with
        .frames, .glow (emissive map per frame), .durations (ms), .tags [(name, from, to, loop)],
        .blend ("add" = draw additively in the engine)

Drawing engine: fxcore.py. effects.render_effect / make_effect call this for SPELL_KINDS.
"""
from __future__ import annotations

import math

import numpy as np

from .fxcore import (
    SIZES, Ctx, Particle, back_out, bolt_points, clamp, draw_bolt, draw_spark, ease_in,
    ease_out, finish, flame_field, loop_beats, phase_t, puff, spell_ramps, timeline,
)

SPELL_THEMES = ("fire", "ice", "lightning", "holy", "poison", "dark", "arcane", "nature", "water",
                "steel", "project")


# =========================================================================== the effects
# Each: fx(ctx, n) -> (beats, draw(cv, beat), loops {tag: bool}). Sizes are in output px,
# scaled from ctx.S so every size reads the same.

def _fire_burst(c: Ctx, n):
    """Embers gather and flicker, a white flash, flame tongues erupt and lick upward,
    then break off into rising purple smoke and drifting embers."""
    beats = timeline([("gather", 4, 80, 50), ("flash", 1, 40, 40), ("burst", 7, 40, 55),
                      ("fade", 12, 60, 90)], n)
    S, rng = c.S, c.rng
    cx, gy = c.w / 2, c.h * 0.88
    t0 = phase_t(beats, "flash")
    noise = c.noise(4, 3)
    sn = c.noise(5, 2)
    K = 7
    tongues = []
    for k in range(K):
        f = k / (K - 1) - 0.5
        tongues.append(dict(x=cx + f * S * 0.34 + rng.uniform(-1, 1) * S * 0.02,
                            hgt=S * (0.72 - 0.75 * f * f + rng.uniform(-0.05, 0.05)),
                            wid=S * (0.15 + rng.uniform(0, 0.03) - 0.08 * abs(f)),
                            lean=f * 0.5 + rng.uniform(-0.08, 0.08),
                            delay=abs(f) * 0.09 + rng.uniform(0, 0.03)))
    gather = [Particle(0, 0, 0, 0, ang=a) for a in rng.uniform(0, 2 * math.pi, 7)]
    embers = [Particle(cx + rng.uniform(-0.15, 0.15) * S, gy - rng.uniform(0.05, 0.3) * S,
                       rng.uniform(-0.6, 0.6) * S, -rng.uniform(0.9, 1.9) * S,
                       t0 + rng.uniform(0.0, 0.25), rng.uniform(0.5, 0.9), 2.2, 0, -0.15 * S,
                       size=float(rng.choice([0.8, 1.1, 1.4])), heat=1.0, ph=rng.uniform(0, 6))
              for _ in range(16)]
    smokes = [dict(x=cx + rng.uniform(-0.22, 0.22) * S, y=gy - rng.uniform(0.3, 0.5) * S,
                   r=rng.uniform(0.07, 0.11) * S, vy=-rng.uniform(0.18, 0.3) * S,
                   vx=rng.uniform(-0.12, 0.12) * S, born=rng.uniform(0.15, 0.4)) for _ in range(8)]
    end = beats[-1].t

    def draw(cv, b):
        t = b.t
        if b.phase == "gather":
            q = b.p
            for p in gather:            # sparks spiral in
                a = p.ang + q * 1.6
                rr = S * 0.42 * (1 - ease_in(q, 1.6)) + 2
                x, y = cx + math.cos(a) * rr, gy - S * 0.1 + math.sin(a) * rr * 0.55
                a2, rr2 = a - 0.35, rr + S * 0.07
                cv.stroke("accent", [(cx + math.cos(a2) * rr2, gy - S * 0.1 + math.sin(a2) * rr2 * 0.55),
                                     (x, y)], [0.3, 0.9], [0.4, 0.95])
            r = S * (0.03 + 0.05 * ease_in(q)) * (1 + 0.15 * (b.k % 2))
            cv.ellipse("main", cx, gy - S * 0.08, r * 1.3, r, heat=lambda rho, X, Y: 1 - 0.55 * rho)
            cv.ring("back", cx, gy, S * 0.16 * (0.4 + q), 1.2, heat=0.45, squash=0.3)
            return
        if b.phase == "flash":
            fy = gy - S * 0.2
            cv.ellipse("main", cx, fy, S * 0.24, S * 0.26, heat=lambda rho, X, Y: 1.05 - 0.4 * rho ** 3)
            for k in range(8):
                a = k * math.pi / 4 + 0.2
                L = S * (0.5 if k % 2 == 0 else 0.32)
                cv.stroke("main", [(cx, fy), (cx + math.cos(a) * L, fy + math.sin(a) * L)],
                          [S * 0.05, 0.4], [1.0, 0.75])
            cv.ring("back", cx, gy, S * 0.38, 2.0, heat=0.8, squash=0.28)
            return
        q = (t - t0) / (end - t0)          # 0 at the flash .. 1 at the end
        gl = 1 - ease_out(q / 0.75, 1.5)
        if gl > 0.1:
            cv.ellipse("back", cx, gy, S * 0.36 * (0.7 + 0.3 * gl), S * 0.08,
                       heat=lambda rho, X, Y: 0.15 + 0.55 * gl * (1 - rho))
        grow0, die0 = ease_out(q / 0.12, 2.5), ease_in((q - 0.1) / 0.4, 1.2)
        if grow0 > 0 and die0 < 1:         # the fire's body: one wide flame under the tongues
            bh, bwid = S * 0.42 * grow0 * (1 - 0.6 * die0), S * 0.24 * (1 - 0.5 * die0)

            def body(X, Y, bh=bh, bwid=bwid, die0=die0):
                d, v = flame_field(X, Y, cx, gy, bh, bwid, t, noise, turb=0.5)
                d = d - die0 * 0.7 * noise(X / S * 3 + 0.3, Y / S * 3 + t * 2)
                return d > 0, (0.14 + 1.0 * np.clip(d, 0, 1) ** 1.1 + 0.1 * (1 - v) - 0.3 * v) * (1 - 0.45 * die0)
            cv.field("main", (cx - bwid * 2, gy - bh - 2, cx + bwid * 2, gy + bwid), body)
        for tg in tongues:
            qq = q - tg["delay"]
            grow = ease_out(qq / 0.16, 2.5)
            die = ease_in((qq - 0.22) / 0.5, 1.4)
            if grow <= 0 or die >= 1:
                continue
            hgt = tg["hgt"] * grow * (1 - 0.55 * die)
            wid = tg["wid"] * (0.55 + 0.45 * grow) * (1 - 0.75 * die)
            by = gy - S * 0.45 * die ** 1.2           # the tongue rises off its base as it dies
            heat_k = 1.0 - 0.45 * die
            box = (tg["x"] - wid * 3 - hgt * 0.6, by - hgt - 2, tg["x"] + wid * 3 + hgt * 0.6, by + wid + 2)

            def fn(X, Y, tg=tg, hgt=hgt, wid=wid, by=by, heat_k=heat_k, die=die):
                d, v = flame_field(X, Y, tg["x"], by, hgt, wid, t, noise, lean=tg["lean"], sway=0.5)
                d = d - die * 0.6 * noise(X / S * 3, Y / S * 3 + t * 2)
                return d > 0, (0.12 + 0.95 * np.clip(d, 0, 1) ** 1.1 + 0.12 * (1 - v) - 0.3 * v) * heat_k
            cv.field("main", box, fn)
        for sm in smokes:
            a = (q - sm["born"]) / 0.62
            if not 0 < a < 1:
                continue
            x = sm["x"] + sm["vx"] * a
            y = sm["y"] + sm["vy"] * a * 1.3 - S * 0.06 * a * a
            puff(cv, x, y, sm["r"] * (0.6 + 1.0 * ease_out(a)), 0.78 - 0.55 * a, sn, t,
                 erosion=ease_in(a, 1.6) * 0.9, lit=(0.0, 1.0),
                 seed=int(sm["r"] * 1000))
        for p in embers:
            draw_spark(cv, p, t, 0.03, heat=0.75 + 0.25 * math.sin(p.ph + t * 30))
    return beats, draw, {"": False}


def _arc_points(cx, cy, r, a0, a1, n=24, squash=1.0):
    return [(cx + math.cos(a) * r, cy + math.sin(a) * r * squash) for a in np.linspace(a0, a1, n)]


def _sword_combo(c: Ctx, n):
    """1-3 swings: a glint, a fast smear crescent (white leading edge, colour body, thin
    tail), follow-through, then it frays into streaks and sparkles. opts hits=1..3."""
    hits = int(c.opts.get("hits") or 3)
    if not 1 <= hits <= 3:
        raise ValueError("hits must be 1, 2 or 3")
    S, rng = c.S, c.rng
    cx, cy = c.w / 2, c.h / 2
    # (start angle, end angle, radius share, squash, thickness share, centre offset)
    swings = [(-2.25, 0.95, 0.4, 1.0, 0.17, (0.04, -0.02)),
              (2.55, -0.55, 0.38, 0.8, 0.15, (-0.03, 0.04)),
              (-3.6, 2.6, 0.44, 0.42, 0.2, (0.0, 0.05))][:hits]
    per = [("glint", 1, 70, 70), ("smear", 3, 30, 40), ("follow", 2, 50, 60), ("fray", 3, 60, 90)]
    n_each = max(4, n // hits) if n else None
    beats = []
    for k in range(hits):
        beats += timeline(per, n_each, tag=f"swing{k + 1}", start=len(beats))
    sparks = {k: [Particle(0, 0, *(np.array([math.cos(a), math.sin(a)]) * rng.uniform(0.6, 1.4) * S),
                           0, rng.uniform(0.18, 0.3), 5.0, 0, 0.6 * S,
                           size=float(rng.choice([0.8, 1.2])), heat=1.0)
                  for a in rng.uniform(0, 2 * math.pi, 11)] for k in range(hits)}
    fray = c.noise(6, 2)
    follow_t = {f"swing{k + 1}": next(b.t for b in beats if b.tag == f"swing{k + 1}" and b.phase == "follow")
                for k in range(hits)}

    def draw(cv, b):
        k = int(b.tag[-1]) - 1
        a0, a1, rs, sq, th, (ox, oy) = swings[k]
        R, T = S * rs, S * th
        X0, Y0 = cx + ox * S, cy + oy * S
        sgn = 1 if a1 > a0 else -1
        span = abs(a1 - a0)
        if b.phase == "glint":
            pts = _arc_points(X0, Y0, R, a0, a0 + sgn * 0.35, 6, sq)
            cv.stroke("main", pts, [0.4, 0.8, 0.8, 0.8, 0.6, 0.3], 0.5)
            cv.star("accent", *pts[-1], S * 0.07, 1.0)
            return
        if b.phase == "smear":
            head = ease_out((b.k + 1) / b.count, 1.6)
            tail, thick, fade = max(0.0, head - 0.95), 0.75 + 0.25 * b.p, 0.0
        elif b.phase == "follow":
            head = 1.0 + 0.04 * (b.k + 1)
            tail, thick, fade = 0.25 + 0.3 * (b.k + 1) / b.count, 1.0 - 0.2 * b.p, 0.1 * b.p
        else:
            head = 1.08
            tail = 0.5 + 0.42 * ease_out((b.k + 1) / b.count)
            thick, fade = 0.8 - 0.35 * b.p, 0.15 + 0.55 * (b.k + 1) / b.count
        ha, ta = a0 + sgn * span * head, a0 + sgn * span * tail
        lo, hi = min(ha, ta), max(ha, ta)
        if hi - lo < 0.02:
            return
        Tk = T * thick
        box = (X0 - R - 2, Y0 - (R + 2) * sq - 2, X0 + R + 2, Y0 + (R + 2) * sq + 2)

        def fn(X, Y):
            dx, dy = X - X0, (Y - Y0) / sq
            r = np.hypot(dx, dy)
            ang = lo + np.mod(np.arctan2(dy, dx) - lo, 2 * math.pi)
            s = (ang - ta) / (ha - ta)                    # 0 tail .. 1 head
            prof = np.clip(s, 0, 1) ** 0.8 * np.clip((1.04 - s) / 0.14, 0, 1) ** 0.5
            wdt = Tk * prof
            depth = R - r                                  # 0 at the outer (leading) edge
            inside = (ang <= hi) & (depth >= -0.6) & (depth <= wdt)
            u = np.clip(depth / np.maximum(wdt, 0.5), 0, 1)
            # flat bands like a hand-drawn smear: white leading edge, light body, darker inside
            edge = 1.0 / np.maximum(wdt, 1.0)
            ht = np.where(u < np.maximum(0.16, 1.4 * edge), 1.0,
                          np.where(u < 0.5, 0.74, np.where(u < 0.8, 0.52, 0.34)))
            ht = ht - 0.32 * (1 - np.clip(s * 1.6, 0, 1)) - 0.3 * fade
            if fade > 0:                                   # fray into streaks along the arc
                nz = fray(r / S * 3.0, ang / (2 * math.pi) * 0.5)
                inside &= nz + 0.5 * (1 - u) > 0.3 + 0.6 * fade
            return inside, ht
        cv.field("main", box, fn)
        if b.phase != "fray" or b.k == 0:          # a thin second arc inside (motion line)
            pts = _arc_points(X0, Y0, R - Tk * 1.25, ta + (ha - ta) * 0.35, ha, 16, sq)
            s = np.linspace(0, 1, len(pts))
            cv.stroke("main", pts, np.maximum(0.25, 0.75 * np.sin(np.pi * s)), 0.5 - 0.3 * fade)
        hx, hy = X0 + math.cos(ha) * R, Y0 + math.sin(ha) * R * sq
        if b.phase == "follow" and b.k == 0:
            cv.star("accent", hx, hy, S * 0.11, 1.0, rot=0.3)
        if b.phase in ("follow", "fray"):
            tt = b.t - follow_t[b.tag]
            for p in sparks[k]:
                p.x, p.y = hx, hy
                draw_spark(cv, p, tt, 0.03)
            if b.phase == "fray":                   # sparkles left along the arc
                for j in range(4):
                    aa = a0 + sgn * span * (0.45 + 0.17 * j)
                    on = (b.k + j) % 2 == 0
                    rr = R - Tk * 0.3
                    cv.star("accent", X0 + math.cos(aa) * rr, Y0 + math.sin(aa) * rr * sq,
                            (2.2 if on else 1.2) * S / 64 * (1 - 0.5 * b.p), 0.95 if on else 0.7)
    return beats, draw, {f"swing{k + 1}": False for k in range(hits)}


def _lightning_bolt(c: Ctx, n):
    """Charge crackles at the top, the bolt strikes with branches, flickers between
    shapes, slams a flash + sparks into the ground, then fades violet in broken pieces."""
    beats = timeline([("charge", 3, 80, 60), ("strike", 1, 40, 40), ("flicker", 4, 40, 50),
                      ("impact", 2, 50, 60), ("after", 8, 60, 100)], n)
    S, rng, w, h = c.S, c.rng, c.w, c.h
    top, gy = h * 0.04, h * 0.9
    bw = max(1.0, w * 0.032)
    shapes = []
    for _ in range(3):
        x0, x1 = w / 2 + rng.uniform(-0.12, 0.12) * w, w / 2 + rng.uniform(-0.05, 0.05) * w
        main = bolt_points(rng, x0, top, x1, gy, 4, 0.16, 0.62)
        branches = []
        for _ in range(3):
            j = int(rng.integers(len(main) // 6, len(main) * 2 // 3))
            bx, by = main[j]
            side = rng.choice([-1, 1])
            L = h * rng.uniform(0.15, 0.3)
            branches.append(bolt_points(rng, bx, by, bx + side * L * rng.uniform(0.5, 0.9),
                                        by + L * rng.uniform(0.5, 0.9), 3, 0.18, 0.5))
        shapes.append((main, branches))
    t_hit = phase_t(beats, "strike")
    sparks = [Particle(w / 2, gy, math.cos(a) * v * S, math.sin(a) * v * S, t_hit + 0.02,
                       rng.uniform(0.3, 0.55), 3.0, 0, 1.6 * S, size=float(rng.choice([0.8, 1.2, 1.5])))
              for a, v in zip(rng.uniform(-math.pi * 0.95, -math.pi * 0.05, 14), rng.uniform(0.8, 2.0, 14))]
    crackle = [(rng.uniform(0.3, 0.7) * w, top + rng.uniform(0, 0.12) * h, rng.uniform(-1, 1))
               for _ in range(5)]
    fray = c.noise(8, 1)

    def draw(cv, b):
        if b.phase == "charge":
            for j, (x, y, dx) in enumerate(crackle):
                if (j + b.k) % 2 == 0:
                    pts = bolt_points(np.random.default_rng(b.k * 7 + j), x, y,
                                      x + dx * w * 0.15, y + h * 0.06 * (1 + b.p), 3, 0.3)
                    cv.stroke("main", pts, 0.55, 0.7 + 0.2 * b.p)
            cv.ellipse("back", w / 2, top + h * 0.04, w * 0.3 * (0.5 + 0.5 * b.p), h * 0.04,
                       heat=lambda rho, X, Y: 0.35 * (1 - rho) + 0.1)
            if b.k == b.count - 1:      # a faint leader stroke
                draw_bolt(cv, shapes[0][0], bw * 0.4, 0.55, part=0.35, glow=False)
            return
        idx = {"strike": 0, "flicker": [1, 0, 2, 0][min(b.k, 3)], "impact": 2}.get(b.phase, 0)
        main, branches = shapes[idx]
        if b.phase in ("strike", "flicker", "impact"):
            dim = b.phase == "flicker" and b.k == 2      # one dim beat in the flicker
            s = 0.55 if dim else 1.0
            draw_bolt(cv, main, bw * (0.6 if dim else 1.0), s)
            for br in branches:
                draw_bolt(cv, br, bw * 0.55, s * 0.85)
            fl = 1.0 if b.phase != "impact" else 1 - 0.4 * b.p
            cv.ellipse("main", main[-1][0], gy, w * 0.2 * fl, h * 0.03 * fl,
                       heat=lambda rho, X, Y: 1.0 - 0.5 * rho)
            cv.ellipse("back", main[-1][0], gy, w * 0.36 * fl, h * 0.05 * fl,
                       heat=lambda rho, X, Y: 0.55 - 0.4 * rho)
            cv.ring("back", main[-1][0], gy, w * (0.25 + 0.35 * (b.t - t_hit) / 0.3), 1.5,
                    heat=0.75, squash=0.3)
        else:                                            # afterglow: thin violet, breaking up
            q = b.p
            cv.stroke("main", main, max(0.5, bw * 0.55 * (1 - q)), 0.55 - 0.4 * q)
            cv.field("main", (0, 0, w, h), lambda X, Y: (fray(X / w, Y / h) < 0.2 + 0.7 * q, 0),
                     mode="erase")
            if q < 0.6:
                cv.ring("back", main[-1][0], gy, w * 0.2 * (1 - 0.4 * q), 1.4 * (1 - q), heat=0.45 - 0.3 * q,
                        squash=0.3)
        for p in sparks:
            draw_spark(cv, p, b.t, 0.03)
    return beats, draw, {"": False}


def _blast(cv, x, y, R, q, t, noise, S, cool_at=0.55):
    """A fireball blast at (x, y) growing to R: white core, orange body, red rim, then it
    cools, rises and is eaten from inside into a breaking ring of flame."""
    grow = ease_out(q / 0.22, 2.2)
    if grow <= 0:
        return
    r = R * (0.25 + 0.75 * grow)
    cy = y - S * 0.12 * ease_in(q, 1.2)
    cool = ease_in(q / cool_at, 1.3)
    hole = ease_in((q - 0.18) / 0.5, 1.3)
    if hole >= 1:
        return

    lobes = [(math.cos(a) * 0.5, math.sin(a) * 0.45, 0.58) for a in np.arange(7) * 0.9 + 0.3]

    def fn(X, Y):
        dx, dy = (X - x) / r, (Y - cy) / r
        rho = np.sqrt(dx * dx + dy * dy) / 0.85
        for ox, oy, k in lobes:                         # billowy lobes round the core
            rho = np.minimum(rho, np.hypot(dx - ox, dy - oy) / k)
        n = noise(X / (S * 0.55), Y / (S * 0.55) + t * 1.5)
        d = 1 - rho + (n - 0.5) * 0.55
        n2 = noise(X / (S * 1.6) + 0.37, Y / (S * 1.6) + t * 0.5)
        inner = np.sqrt(dx * dx + dy * dy) + (n2 - 0.5) * 0.5 < hole * 1.3
        ht = (0.12 + 1.1 * np.clip(d, 0, 1) ** 0.9 - 0.15 * dy) * (1 - 0.55 * cool) + 0.12 * (1 - cool)
        return (d > 0) & ~(inner & (hole > 0)), ht
    cv.field("main", (x - r * 1.4, cy - r * 1.4, x + r * 1.4, cy + r * 1.4), fn)


def _fireball(c: Ctx, n):
    """A fireball flying right (loop "fly": hot round head with a bright leading edge, a
    streaming flame tail, shed embers and smoke) and its "impact" blast at the same spot."""
    S, rng, w, h = c.S, c.rng, c.w, c.h
    nf = 8 if not n else max(4, min(12, n // 3))
    fly = loop_beats(nf, 60, "fly")
    imp = timeline([("flash", 1, 40, 40), ("blast", 5, 40, 50), ("fade", 12, 60, 90)],
                   (n - nf) if n else None, tag="impact", start=nf)
    beats = fly + imp
    ax, ay = w * 0.66, h * 0.5
    r0 = S * 0.2
    noise, sn = c.noise(4, 2), c.noise(5, 2)
    sheds = [(rng.uniform(0, 1), rng.uniform(-1, 1), float(rng.choice([0.8, 1.2]))) for _ in range(7)]
    wisps = [(k / 3 + rng.uniform(0, 0.1), rng.uniform(-0.5, 0.5)) for k in range(3)]
    t_imp = phase_t(beats, "flash")
    embers = [Particle(ax, ay, math.cos(a) * v * S, math.sin(a) * v * S, t_imp, rng.uniform(0.35, 0.7),
                       3.5, 0, 0.5 * S, size=float(rng.choice([0.8, 1.2, 1.6])), heat=1.0)
              for a, v in zip(rng.uniform(0, 2 * math.pi, 18), rng.uniform(1.0, 2.6, 18))]
    smokes = [(math.cos(a) * S * 0.18, math.sin(a) * S * 0.14, rng.uniform(0.09, 0.13) * S, k)
              for k, a in enumerate(rng.uniform(0, 2 * math.pi, 7))]
    end = beats[-1].t

    def draw(cv, b):
        if b.tag == "fly":
            t = b.p
            for ph, side in wisps:           # smoke at the end of the tail
                a = (ph + t) % 1.0
                puff(cv, ax - S * (0.5 + 0.55 * a), ay + side * S * 0.1 - a * S * 0.06,
                     S * (0.05 + 0.06 * a), 0.6 - 0.4 * a, sn, a, erosion=ease_in(a, 1.5) * 0.9,
                     lit=(0.6, 0.0), seed=int(ph * 99), lobes=2)
            L, W = S * 0.95, r0 * 1.05

            def tail(X, Y):
                d, v = flame_field(Y, X, ay, ax + r0 * 0.4, L, W, t, noise, turb=0.45, speed=2)
                return d > 0, 0.1 + 0.95 * np.clip(d, 0, 1) ** 1.1 + 0.15 * (1 - v) - 0.35 * v
            cv.field("main", (ax - L - 2, ay - W * 2, ax + r0 * 2.5, ay + W * 2), tail)
            pul = 1 + 0.06 * math.sin(2 * math.pi * t * 2)
            cv.ellipse("main", ax, ay, r0 * 1.05 * pul, r0 * 0.95 * pul,
                       heat=lambda rho, X, Y: 0.9 - 0.62 * rho ** 1.5 + 0.28 * np.clip((X - ax) / r0, -1, 1))
            cv.ellipse("main", ax + r0 * 0.3, ay - r0 * 0.1, r0 * 0.42, r0 * 0.36, heat=1.0)
            for ph, side, sz in sheds:      # embers shed backwards, drifting off
                a = (ph + t) % 1.0
                x = ax - S * 0.2 - a * S * 0.9
                y = ay + side * S * (0.08 + 0.2 * a)
                cv.stroke("accent", [(x + S * 0.06, y), (x, y)], [0.3, sz * (1 - 0.5 * a)], [0.6, 1 - 0.5 * a])
            return
        t = b.t
        if b.phase == "flash":
            cv.ellipse("main", ax, ay, S * 0.3, heat=lambda rho, X, Y: 1.05 - 0.35 * rho ** 3)
            for k in range(8):
                a = k * math.pi / 4 + 0.4
                L = S * (0.5 if k % 2 == 0 else 0.34)
                cv.stroke("main", [(ax, ay), (ax + math.cos(a) * L, ay + math.sin(a) * L)], [S * 0.05, 0.4],
                          [1.0, 0.7])
            return
        q = (t - t_imp) / (end - t_imp)
        if q < 0.45:
            cv.ring("back", ax, ay, S * 0.48 * ease_out(q / 0.45), 2.2 * (1 - q / 0.45) + 0.6,
                    heat=0.85 - q)
        for dx, dy, r, k in smokes:
            a = (q - 0.12) / 0.88
            if 0 < a < 1:
                puff(cv, ax + dx * (1 + a), ay + dy * (1 + a) - S * 0.25 * a, r * (0.7 + 0.8 * a),
                     0.75 - 0.5 * a, sn, t, erosion=ease_in(a, 1.4) * 0.95, lit=(0, 1), seed=k)
        _blast(cv, ax, ay, S * 0.3, q, t, noise, S)
        for p in embers:
            draw_spark(cv, p, t, 0.035)
    return beats, draw, {"fly": True, "impact": False}


def _big_explosion(c: Ctx, n):
    """Flash, a fireball punches out with a ground shock ring and flying debris (rocks with
    fire trails turning to smoke), rises and cools into a billowing smoke column that
    breaks up; embers drift."""
    beats = timeline([("flash", 1, 40, 40), ("blast", 6, 40, 50), ("billow", 10, 55, 70),
                      ("fade", 15, 70, 100)], n)
    S, rng, w, h = c.S, c.rng, c.w, c.h
    cx, gy = w / 2, h * 0.84
    by = gy - S * 0.2
    noise, sn = c.noise(4, 3), c.noise(5, 2)
    t0 = phase_t(beats, "flash")
    end = beats[-1].t
    debris = []
    for k in range(9):
        a = -math.pi / 2 + rng.uniform(-1.25, 1.25)
        v = rng.uniform(1.0, 1.9) * S
        sides = int(rng.integers(4, 6))
        shape = [(math.cos(2 * math.pi * j / sides + rng.uniform(-0.3, 0.3)) * rng.uniform(0.7, 1.0),
                  math.sin(2 * math.pi * j / sides + rng.uniform(-0.3, 0.3)) * rng.uniform(0.7, 1.0))
                 for j in range(sides)]
        debris.append(Particle(cx, by, math.cos(a) * v, math.sin(a) * v, t0, rng.uniform(0.7, 1.0), 1.2,
                               0, 2.6 * S, size=rng.uniform(1.8, 3.2) * S / 64, heat=0.8, shape=shape,
                               spin=rng.uniform(-14, 14)))
    embers = [Particle(cx + rng.uniform(-0.1, 0.1) * S, by, rng.uniform(-1.2, 1.2) * S,
                       -rng.uniform(0.6, 2.0) * S, t0 + rng.uniform(0, 0.3), rng.uniform(0.6, 1.1), 2.5,
                       0, -0.1 * S, size=float(rng.choice([0.8, 1.1, 1.5])), ph=rng.uniform(0, 6))
              for _ in range(18)]
    smokes = []
    for k in range(13):
        a = rng.uniform(0, 2 * math.pi)
        smokes.append(dict(dx=math.cos(a) * S * rng.uniform(0.1, 0.24), dy=math.sin(a) * S * rng.uniform(0.06, 0.16),
                           r=S * rng.uniform(0.08, 0.13), born=rng.uniform(0.1, 0.3), rise=rng.uniform(0.3, 0.6),
                           k=k))
    for k in range(4):          # the stem
        smokes.append(dict(dx=rng.uniform(-0.04, 0.04) * S, dy=S * (0.1 + 0.06 * k), r=S * 0.07,
                           born=0.2 + 0.03 * k, rise=0.15, k=20 + k))

    def draw(cv, b):
        t = b.t
        if b.phase == "flash":
            cv.ellipse("main", cx, by, S * 0.3, S * 0.28, heat=lambda rho, X, Y: 1.05 - 0.3 * rho ** 3)
            for k in range(10):
                a = k * math.pi / 5 + 0.15
                L = S * (0.55 if k % 2 == 0 else 0.36)
                cv.stroke("main", [(cx, by), (cx + math.cos(a) * L, by + math.sin(a) * L)], [S * 0.045, 0.4],
                          [1.0, 0.7])
            cv.ellipse("back", cx, gy, S * 0.45, S * 0.07, heat=0.8)
            return
        q = (t - t0) / (end - t0)
        if q < 0.35:                 # shock ring on the ground, and a dust skirt
            qq = q / 0.35
            cv.ring("back", cx, gy, S * 0.5 * ease_out(qq), 2.5 * (1 - qq) + 0.7, heat=0.9 - 0.6 * qq, squash=0.28)
        for sm in smokes:
            a = (q - sm["born"]) / (1 - sm["born"])
            if not 0 < a < 1:
                continue
            x = cx + sm["dx"] * (1 + 0.8 * a)
            y = by + sm["dy"] - S * sm["rise"] * ease_out(a, 1.5)
            warm = clamp(1 - a / 0.35)          # lit by the fire below at first
            puff(cv, x, y, sm["r"] * (0.7 + 0.9 * ease_out(a)), 0.5 + 0.35 * warm - 0.35 * a, sn, t,
                 erosion=clamp((a - 0.1) / 0.6) * 1.3, lit=(0.0, 1.0), seed=sm["k"])
        _blast(cv, cx, by, S * 0.32, q * 1.25, t, noise, S, cool_at=0.6)
        for p in debris:
            a = p.age(t)
            if not 0 <= a <= 1:
                continue
            x, y = p.pos(t)
            if y > gy + 2:
                continue
            ang = p.spin * (t - t0)
            ca, sa = math.cos(ang), math.sin(ang)
            sz = p.size
            pts = [(x + (u * ca - v * sa) * sz, y + (u * sa + v * ca) * sz) for u, v in p.shape]
            hot = clamp(1 - a / 0.4)
            px, py = p.pos(max(t0, t - 0.06))
            if hot > 0:                            # fire trail, then a smoke trail
                cv.stroke("main", [(px, py), (x, y)], [0.4, sz * 0.9], [0.3 * hot, 0.75 * hot])
            elif a < 0.85:
                cv.stroke("smoke", [(px, py), (x, y)], [0.3, sz * 0.7], [0.3, 0.45])
            cv.poly("smoke", pts, 0.18 + 0.35 * hot)
            cv.poly("smoke", [pts[0], pts[1], (x, y)], 0.55 + 0.4 * hot)   # lit face
        for p in embers:
            draw_spark(cv, p, t, 0.03, heat=0.75 + 0.25 * math.sin(p.ph + t * 28))
    return beats, draw, {"": False}


def _shard_poly(bx, by, ang, L, W):
    """Crystal on the ground: (lit facet, shadow facet, lit edge line) in output px."""
    ca, sa = math.sin(ang), -math.cos(ang)            # up-axis rotated by ang (0 = straight up)
    px, py = math.cos(ang), math.sin(ang)             # across

    def P(u, v):
        return (bx + ca * v + px * u, by + sa * v + py * u)
    bl, br = P(-W / 2, 0), P(W / 2, 0)
    ml, mr = P(-W / 2, L * 0.72), P(W / 2, L * 0.7)
    tip = P(-W * 0.08, L)
    rb, rt = P(-W * 0.08, 0), P(-W * 0.08, L * 0.78)
    return [bl, ml, tip, rt, rb], [rb, rt, tip, mr, br], [bl, ml, tip]


def _ice_shard(c: Ctx, n):
    """Frost creeps on the ground, crystal shards burst up (overshoot), glints run up their
    edges, then they shatter into tumbling fragments, sparkles and frost mist."""
    beats = timeline([("frost", 3, 70, 60), ("erupt", 4, 30, 45), ("hold", 6, 60, 60),
                      ("shatter", 1, 40, 40), ("break", 10, 55, 90)], n)
    S, rng, w, h = c.S, c.rng, c.w, c.h
    cx, gy = w / 2, h * 0.86
    sn = c.noise(5, 2)
    shards = []
    for k in range(6):
        f = (k + 0.5) / 6 - 0.5
        shards.append(dict(x=cx + f * S * 0.62 + rng.uniform(-0.03, 0.03) * S, a=f * 1.1 + rng.uniform(-0.12, 0.12),
                           L=S * (0.62 - 0.9 * f * f + rng.uniform(-0.06, 0.04)), W=S * rng.uniform(0.11, 0.15),
                           delay=rng.uniform(0, 0.4)))
    shards.sort(key=lambda d: -d["L"])            # tall ones behind
    t_sh = phase_t(beats, "shatter")
    frags = []
    for sd in shards:
        for j in range(4):
            v = sd["L"] * (0.25 + 0.2 * j)
            x = sd["x"] + math.sin(sd["a"]) * v
            y = gy - math.cos(sd["a"]) * v
            a = math.atan2(y - (gy + S * 0.1), x - cx) + rng.uniform(-0.4, 0.4)
            sp = rng.uniform(0.8, 1.8) * S
            sz = sd["W"] * rng.uniform(0.3, 0.5)
            frags.append(Particle(x, y, math.cos(a) * sp, math.sin(a) * sp, t_sh, rng.uniform(0.45, 0.75), 3.0,
                                  0, 2.2 * S, size=sz, spin=rng.uniform(-12, 12), heat=rng.uniform(0.55, 0.95)))
    mists = [dict(x=cx + f * S * 0.7 + rng.uniform(-0.05, 0.05) * S, r=S * rng.uniform(0.05, 0.1), k=k,
                  vx=f * rng.uniform(0.3, 0.6) * S, y=rng.uniform(0.04, 0.14) * S)
             for k, f in enumerate(np.linspace(-0.5, 0.5, 5))]
    end = beats[-1].t

    def crystal(cv, sd, grow):
        L = sd["L"] * grow
        if L < 1:
            return
        lit, shade, edge = _shard_poly(sd["x"], gy, sd["a"], L, sd["W"] * min(1.0, 0.5 + grow))
        cv.poly("main", shade, 0.42, mode="over")
        cv.poly("main", lit, 0.78, mode="over")
        cv.stroke("main", edge, 0.55, 0.97, mode="over")
        cv.stroke("main", [edge[0], edge[1]], 0.45, 0.88, mode="over")

    def draw(cv, b):
        t = b.t
        if b.phase in ("frost", "erupt", "hold"):
            fr = ease_out((t) / (t_sh + 0.01), 1.5)
            cv.ellipse("back", cx, gy, S * 0.44 * (0.3 + 0.7 * fr), S * 0.07 * (0.4 + 0.6 * fr),
                       heat=lambda rho, X, Y: 0.55 - 0.45 * rho)
        if b.phase == "frost":
            for j in range(5):
                a = j * 1.3 + b.k
                cv.star("accent", cx + math.cos(a) * S * 0.3 * (1 - b.p * 0.5), gy - S * 0.05 + math.sin(a) * S * 0.04,
                        1.6 * S / 64, 0.85)
            return
        if b.phase in ("erupt", "hold"):
            for sd in shards:
                e = (b.k + 1) / b.count if b.phase == "erupt" else 1.0
                g = back_out(clamp(e * 1.15 - sd["delay"] * 0.4), 2.2) if b.phase == "erupt" else 1.0
                crystal(cv, sd, max(0.0, g))
            if b.phase == "hold":                 # glints run up the lit edges
                for j, sd in enumerate(shards):
                    u = ((b.k + j * 2) % 6) / 5
                    if u < 0.95:
                        x = sd["x"] + math.sin(sd["a"]) * sd["L"] * u - math.cos(sd["a"]) * sd["W"] * 0.45
                        y = gy - math.cos(sd["a"]) * sd["L"] * u - math.sin(sd["a"]) * sd["W"] * 0.45
                        cv.star("accent", x, y, S * (0.05 + 0.04 * (j % 2)), 1.0)
            return
        if b.phase == "shatter":
            for sd in shards:
                crystal(cv, sd, 1.0)
            cv.ellipse("main", cx, gy - S * 0.28, S * 0.3, S * 0.3, heat=lambda rho, X, Y: 1.0 - 0.5 * rho, mode="max")
            return
        q = (t - t_sh) / (end - t_sh)
        for m in mists:
            a = clamp(q * 1.2)
            if a < 1:
                puff(cv, m["x"] + m["vx"] * a, gy - m["y"] - S * 0.12 * a, m["r"] * (0.8 + 0.9 * a),
                     0.75 - 0.4 * a, sn, t, erosion=0.15 + ease_in(a, 1.1), lit=(-0.5, -0.8), seed=m["k"],
                     squash=0.7)
        for p in frags:
            a = p.age(t)
            if not 0 <= a <= 1:
                continue
            x, y = p.pos(t)
            if y > gy + 1:
                continue
            ang = p.spin * (t - t_sh)
            sz = p.size * (1 - 0.6 * a)
            pts = [(x + math.cos(ang + k * 2.1) * sz, y + math.sin(ang + k * 2.1) * sz * 1.4) for k in range(3)]
            cv.poly("main", pts, p.heat * (1 - 0.3 * a))
            if (b.k + int(p.spin)) % 3 == 0:
                cv.star("accent", x, y, sz * 1.2, 1.0)
        for j in range(6):                         # sparkles left hanging in the air
            if (b.k + j) % 3:
                a = j * 2.4
                cv.star("accent", cx + math.cos(a) * S * 0.3, gy - S * (0.15 + 0.07 * j) - q * S * 0.1,
                        S * 0.045 * (1 - q), 0.9)
    return beats, draw, {"": False}


def _frost_nova(c: Ctx, n):
    """Cold gathers, a flash, a ring of frost races out over the ground raising crystal spikes
    and leaving a frosted floor, then the spikes burst into sparkles and mist rolls away."""
    beats = timeline([("charge", 3, 70, 60), ("flash", 1, 40, 40), ("spread", 8, 40, 55),
                      ("fade", 12, 60, 100)], n)
    S, rng, w, h = c.S, c.rng, c.w, c.h
    cx, cy, sq = w / 2, h * 0.62, 0.42
    Rmax = w * 0.44
    sn, crack = c.noise(5, 2), c.noise(6, 2)
    K = 16
    spikes = [(2 * math.pi * (k + rng.uniform(-0.25, 0.25)) / K, rng.uniform(0.8, 1.0), rng.uniform(0.7, 1.2))
              for k in range(K)]
    spikes.sort(key=lambda s: math.sin(s[0]))       # back ones first
    t0 = phase_t(beats, "flash")
    end = beats[-1].t
    mists = [(a, rng.uniform(0.8, 1.05), S * rng.uniform(0.07, 0.1), k) for k, a in enumerate(rng.uniform(0, 2 * math.pi, 10))]

    def draw(cv, b):
        t = b.t
        if b.phase == "charge":
            for j in range(8):
                a = j * math.pi / 4 + b.p
                rr = Rmax * 0.7 * (1 - ease_in(b.p))
                cv.stroke("accent", [(cx + math.cos(a) * (rr + 4), cy + math.sin(a) * (rr + 4) * sq),
                                     (cx + math.cos(a) * rr, cy + math.sin(a) * rr * sq)], [0.3, 0.9], [0.5, 0.95])
            cv.ellipse("main", cx, cy, S * 0.07 * (1 + b.p), S * 0.05 * (1 + b.p), heat=lambda rho, X, Y: 1 - 0.5 * rho)
            return
        if b.phase == "flash":
            cv.ellipse("main", cx, cy, Rmax * 0.3, Rmax * 0.3 * sq * 1.6, heat=lambda rho, X, Y: 1.05 - 0.4 * rho)
            cv.star("accent", cx, cy - S * 0.05, S * 0.3, 1.0, stretch=1.6)
            return
        q = (t - t0) / (end - t0)
        R = Rmax * ease_out(q / 0.42, 2.2)
        fade = ease_in((q - 0.4) / 0.6, 1.2)
        # frosted floor with cracks, fading out from the middle
        if fade < 1:
            def floor(X, Y):
                dx, dy = (X - cx) / max(R, 1), (Y - cy) / max(R * sq, 1)
                rho = np.sqrt(dx * dx + dy * dy)
                nz = crack(X / w * 2, Y / h * 2)
                cr = np.abs(nz - 0.5) < 0.03
                inside = (rho < 1) & (rho > max(0.62, fade * 1.05) + (nz - 0.5) * 0.3)
                return inside, np.where(cr, 0.6, 0.3 + 0.25 * rho) * (1 - 0.5 * fade)
            cv.field("back", (cx - R - 1, cy - R * sq - 1, cx + R + 1, cy + R * sq + 1), floor)
        if q < 0.55:                             # the racing ring: bright outer edge
            wd = S * 0.07 * (1 - q / 0.55) + 1.0
            cv.ring("main", cx, cy, R, wd, heat=1.0, squash=sq, inner_heat=0.45)
        for a, f, hk in spikes:                  # crystals raised where the ring passed
            rr = Rmax * f * 0.92
            if R < rr:
                continue
            pass_q = clamp((R - rr) / (Rmax * 0.15))
            g = back_out(pass_q, 2.0) * (1 - fade)
            if g <= 0.05:
                continue
            x, y = cx + math.cos(a) * rr, cy + math.sin(a) * rr * sq
            L = S * 0.2 * hk * g * (0.75 + 0.25 * math.sin(a))
            lean = math.cos(a) * 0.5
            lit, shade, edge = _shard_poly(x, y, lean, L, S * 0.07)
            back = math.sin(a) < 0
            cv.poly("main", shade, 0.32 if back else 0.42, mode="over")
            cv.poly("main", lit, 0.62 if back else 0.8, mode="over")
            cv.stroke("main", edge, 0.45, 0.95, mode="over")
        if fade > 0:
            for k, (a, f, hk) in enumerate(spikes):
                rr = Rmax * f * 0.92
                x, y = cx + math.cos(a) * rr, cy + math.sin(a) * rr * sq - S * 0.1 - S * 0.15 * fade
                if (b.k + k) % 2 == 0 and fade < 0.9:
                    cv.star("accent", x, y, S * 0.05 * (1 - fade) + 1, 0.95)
            for a, f, r, k in mists:
                x, y = cx + math.cos(a) * Rmax * f * (1 + 0.15 * fade), cy + math.sin(a) * Rmax * f * sq - S * 0.08 * fade
                puff(cv, x, y, r * (0.8 + 0.6 * fade), 0.7 - 0.4 * fade, sn, t, erosion=ease_in(fade, 1.2),
                     lit=(-0.4, -0.9), seed=k, squash=0.7)
    return beats, draw, {"": False}


def _chain_lightning(c: Ctx, n):
    """A bolt jumps target to target left -> right: each hop crackles (re-drawn every frame),
    flashes the target with a burst and sparks, then leaves a violet afterglow."""
    beats = timeline([("chain", 14, 40, 50), ("after", 10, 60, 100)], n)
    S, rng, w, h = c.S, c.rng, c.w, c.h
    nodes = [(w * 0.07, h * 0.55), (w * 0.37, h * 0.42), (w * 0.64, h * 0.6), (w * 0.93, h * 0.45)]
    N = len(beats)
    hop_at = [1 + int(k * (N * 0.5 - 1) / 3) for k in range(4)]     # frame the bolt reaches node k
    bw = max(0.8, h * 0.017)
    fray = c.noise(8, 1)
    sparks = {k: [Particle(*nodes[k], math.cos(a) * v * S, math.sin(a) * v * S, 0, rng.uniform(0.25, 0.45), 3.5,
                           0, 1.2 * S, size=float(rng.choice([0.8, 1.2])))
                  for a, v in zip(rng.uniform(0, 2 * math.pi, 8), rng.uniform(0.8, 1.8, 8))] for k in range(4)}
    tnode = {k: (beats[min(N - 1, hop_at[k])].t - beats[min(N - 1, hop_at[k])].ms / 2000) for k in range(4)}

    def path(k, i):
        (x0, y0), (x1, y1) = nodes[k], nodes[k + 1]
        return bolt_points(np.random.default_rng(1000 + i * 31 + k), x0, y0, x1, y1, 4, 0.2, 0.6)

    def draw(cv, b):
        i, t = b.i, b.t
        for k in range(4):
            age = i - hop_at[k]
            if age < 0:
                continue
            x, y = nodes[k]
            if age <= 2:
                rr = S * (0.16 + 0.06 * age)
                cv.star("main", x, y, rr * (1.4 - 0.3 * age), 1.0 - 0.2 * age, rot=0.3 * age)
                cv.ellipse("main", x, y, rr * 0.5, heat=lambda rho, X, Y: 1 - 0.4 * rho)
            if age < 10:
                cv.ring("back", x, y, S * (0.1 + 0.03 * age), 1.4 * (1 - age / 10) + 0.4, heat=0.75 - age * 0.06)
            for p in sparks[k]:
                p.t0 = tnode[k]
                draw_spark(cv, p, t, 0.03)
            if k < 3:
                age2 = i - hop_at[k]
                arrive = hop_at[k + 1]
                if age2 < 0:
                    continue
                if i < arrive:                  # leader racing to the next target
                    part = (i - hop_at[k] + 1) / max(1, arrive - hop_at[k])
                    draw_bolt(cv, path(k, i), bw * 0.8, 0.8, part=part, taper=0.2)
                elif i < arrive + 4:
                    dim = (i - arrive) == 2
                    draw_bolt(cv, path(k, i), bw * (0.6 if dim else 1.0), 0.6 if dim else 1.0, taper=0.0)
                elif i < arrive + 14:
                    q = (i - arrive - 4) / 10
                    pts = path(k, arrive + 3)
                    cv.stroke("main", pts, max(0.5, bw * 0.5 * (1 - q)), 0.5 - 0.35 * q)
        if b.phase == "after":
            q = b.p
            cv.field("main", (0, 0, w, h), lambda X, Y: (fray(X / w, Y / h) < 0.1 + 0.45 * q, 0), mode="erase")
    return beats, draw, {"": False}


def _hit_spark(c: Ctx, n):
    """Directional impact (opts angle: radians the hit travels, 0 = to the right): a
    white-hot star flash, a shock crescent, a cone of fast sparks that slow, drop and die."""
    ang = float(c.opts.get("angle") or 0.0)
    beats = timeline([("flash", 2, 40, 50), ("spray", 5, 35, 45), ("fade", 7, 55, 90)], n)
    S, rng, w, h = c.S, c.rng, c.w, c.h
    dx, dy = math.cos(ang), math.sin(ang)
    x0, y0 = w / 2 - dx * S * 0.16, h / 2 - dy * S * 0.16
    t0 = beats[1].t
    sparks = [Particle(x0, y0, math.cos(a) * v * S, math.sin(a) * v * S, beats[0].t, rng.uniform(0.45, 0.85), 4.0,
                       0, 1.2 * S, size=float(rng.choice([0.7, 0.9, 1.1])) * max(1.0, S / 64), heat=0.9)
              for a, v in zip(ang + rng.normal(0, 0.42, 16), rng.uniform(2.5, 6.5, 16))]
    bits = [Particle(x0, y0, math.cos(a) * v * S, math.sin(a) * v * S, beats[0].t, rng.uniform(0.35, 0.6), 4.0,
                     0, 1.5 * S, size=0.8, heat=0.8)
            for a, v in zip(ang + math.pi + rng.normal(0, 0.9, 5), rng.uniform(0.8, 1.6, 5))]
    end = beats[-1].t

    def draw(cv, b):
        t = b.t
        if b.phase == "flash":
            k = b.k
            g = 1.0 if k == 0 else 0.7
            spikes = [(ang + math.pi / 2, 0.48), (ang - math.pi / 2, 0.48), (ang, 0.4), (ang + math.pi, 0.22)]
            spikes += [(ang + d, 0.22) for d in (0.8, -0.8, 2.4, -2.4)]
            for a, L in spikes:
                cv.stroke("main", [(x0, y0), (x0 + math.cos(a) * L * S * g, y0 + math.sin(a) * L * S * g)],
                          [S * 0.06 * g, 0.35], [1.0, 0.7])
            cv.ellipse("main", x0, y0, S * 0.13 * g, heat=lambda rho, X, Y: 1.05 - 0.45 * rho ** 2)
            if k == 1:
                cv.ring("back", x0, y0, S * 0.24, 1.6, heat=0.85)
        q = (t - t0) / (end - t0)
        if b.phase != "flash" and q < 0.3:            # shock crescent facing the hit
            qq = q / 0.3
            keep = lambda a: np.cos(a - ang) > 0.25      # noqa: E731
            cv.ring("main", x0, y0, S * (0.2 + 0.28 * ease_out(qq)), S * 0.06 * (1 - qq) + 0.6, heat=0.95 - 0.5 * qq,
                    keep=keep, inner_heat=0.4)
        if b.phase == "spray" and b.k < 2:
            cv.ellipse("main", x0, y0, S * 0.07 * (1 - b.k * 0.4), heat=0.9)
        for p in sparks:
            draw_spark(cv, p, t, 0.05, streak=1.0)
        for p in bits:
            draw_spark(cv, p, t, 0.02, mat="accent", heat=0.6)
    return beats, draw, {"": False}


def _holy_light(c: Ctx, n):
    """A circle of light opens on the ground, a shimmering pillar shines down, halo rings
    rise and shrink, motes spiral up twinkling; it fades from the top down. loop=True:
    the shine part repeats."""
    loop = bool(c.opts.get("_loop"))
    S, rng, w, h = c.S, c.rng, c.w, c.h
    if loop:
        beats = loop_beats(n or 16, 70)
    else:
        beats = timeline([("open", 4, 70, 50), ("shine", 16, 60, 60), ("fade", 8, 70, 100)], n)
    cx, gy = w / 2, h * 0.9
    rx = S * 0.42
    motes = [(rng.uniform(0, 1), rng.uniform(0, 2 * math.pi), int(rng.choice([1, 1, 2])), float(rng.choice([1.0, 1.6, 2.2])))
             for _ in range(12)]
    end = beats[-1].t

    def draw(cv, b):
        if loop:
            env, grow, ph, q = 1.0, 1.0, b.p, 0.5
        else:
            q = b.t / end
            grow = ease_out(b.t / (phase_t(beats, "shine") + 0.05), 2.0)
            env = 1 - ease_in((b.t - phase_t(beats, "fade")) / (end - phase_t(beats, "fade")), 1.2) \
                if b.phase == "fade" else 1.0
            ph = b.t / 0.84                  # the shine cycle = 0.84 s, as in the loop
        # pillar: thin shimmering rays of light (no muddy column), eaten from the top as it fades
        he = 0.72 + 0.28 * env            # fading shrinks and eats it, never darkens it to mud
        grow = grow * env ** 0.6
        W = rx * 0.62 * grow * (0.6 + 0.4 * env)
        top = h * (1 - env) * 0.9 if not loop else 0.0
        if W > 0.8:
            for k, (off, wk, hk) in enumerate(((-0.75, 0.5, 0.62), (-0.4, 0.9, 0.78), (0.0, 1.5, 0.97),
                                               (0.42, 0.9, 0.78), (0.8, 0.5, 0.62))):
                pul = 0.55 + 0.45 * math.sin(2 * math.pi * (ph * 2 + k * 0.37))
                wd = max(0.6, wk * S / 30 * (0.6 + 0.6 * pul) * (0.5 + 0.5 * grow))
                x = cx + off * W

                def ray(X, Y, x=x, wd=wd, hk=hk):
                    v = np.clip((gy - Y) / max(gy - top, 1), 0, 1)
                    ww = wd * (1 - 0.65 * v)
                    u = np.abs(X - x) / np.maximum(ww, 0.3)
                    return (u <= 1) & (Y >= top) & (Y <= gy), (hk * he + 0.05) - 0.22 * v - 0.25 * u * u
                cv.field("back", (x - wd - 1, top, x + wd + 1, gy), ray)
        # ground circle
        cv.ellipse("back", cx, gy, rx * grow, rx * 0.28 * grow, heat=lambda rho, X, Y: (0.5 + 0.45 * (1 - rho)) * he)
        cv.ring("main", cx, gy, rx * grow * (0.97 + 0.03 * math.sin(ph * 2 * math.pi)), 1.2, heat=0.9 * he + 0.05,
                squash=0.28)
        # halo rings rising up the pillar
        for k in range(2):
            p = (ph + k * 0.5) % 1.0
            if not loop and b.phase == "open":
                continue
            y = gy - p * h * 0.75
            r = rx * (0.95 - 0.5 * p)
            if env < 0.35:
                continue
            hr = (0.98 - 0.3 * p) * he
            cv.ring("back", cx, y, r, 1.4, heat=hr * 0.7, squash=0.28)
            cv.ring("main", cx, y, r, 1.6, heat=hr, squash=0.28, inner_heat=0.65 * he,
                    keep=lambda a: np.sin(a) > -0.1)
        for t0, a0, spd, sz in motes:
            p = (t0 + ph * spd) % 1.0
            a = a0 + p * 4.0
            x = cx + math.cos(a) * rx * (0.9 - 0.5 * p)
            y = gy - p * h * 0.85 + math.sin(a) * rx * 0.2
            if y < (h * (1 - env) * 0.9 if not loop else -1):
                continue
            tw = 0.6 + 0.4 * math.sin(2 * math.pi * (p * 6))
            cv.star("accent", x, y, sz * S / 64 * tw * (1.3 - 0.5 * p) + 0.8, (1.0 - 0.3 * p) * he + 0.05)
    return beats, draw, {"": loop}


def _poison_cloud(c: Ctx, n):
    """A toxic puddle (loop): bubbles swell with a glossy highlight and pop into rings and
    droplets, purple vapour wisps curl up and thin away."""
    S, rng, w, h = c.S, c.rng, c.w, c.h
    beats = loop_beats(n or 20, 70)
    cx, gy = w / 2, h * 0.85
    sn = c.noise(5, 2)
    bubbles = [(rng.uniform(-0.32, 0.32) * w, k / 7 + rng.uniform(0, 0.06), rng.uniform(0.05, 0.11) * S,
                rng.uniform(0.02, 0.1) * h) for k in range(7)]
    wisps = [(k / 7 + rng.uniform(0, 0.08), rng.uniform(-0.32, 0.32) * w, k) for k in range(7)]

    def draw(cv, b):
        t = b.p
        for ph, x, k in wisps:              # each wisp: a short curling chain of puffs
            for j in range(3):
                a = (ph + t - j * 0.07) % 1.0
                curl = math.sin(a * 6 + k) * S * 0.07 * (0.3 + a)
                puff(cv, cx + x + curl, gy - S * 0.06 - a * h * 0.62, S * (0.035 + 0.06 * a) * (1 - 0.25 * j),
                     0.78 - 0.5 * a - 0.08 * j, sn, a, erosion=0.02 + ease_in(a, 1.2) * 0.85, lit=(-0.4, -0.8),
                     seed=k * 3 + j, lobes=3)
        cv.ellipse("main", cx, gy, w * 0.42, S * 0.08, heat=lambda rho, X, Y: 0.22 + 0.25 * (1 - rho) - 0.12 * (Y < gy))
        cv.ring("main", cx, gy, w * 0.4, 1.0, heat=0.62, squash=S * 0.08 / (w * 0.4), keep=lambda a: np.sin(a) < 0)
        for x, ph, r, rise in bubbles:
            a = (ph + t) % 1.0
            bx = cx + x
            if a < 0.8:                       # swell and rise
                g = ease_out(a / 0.8)
                rr = r * (0.4 + 0.6 * g)
                y = gy - S * 0.03 - rise * ease_in(a / 0.8, 1.3)
                cv.ellipse("main", bx, y, rr, heat=lambda rho, X, Y, y=y, rr=rr, bx=bx:
                           0.4 + 0.22 * np.clip(-(Y - y) / rr, -1, 1) - 0.12 * rho ** 4)
                cv.ring("main", bx, y, rr * 0.62, max(0.7, rr * 0.18), heat=0.85,
                        keep=lambda an: (an < -1.8) & (an > -3.0))
                cv.ellipse("accent", bx - rr * 0.38, y - rr * 0.42, max(0.6, rr * 0.16), heat=1.0, snap=rr < 3)
            else:                             # pop: a ring and droplets
                q = (a - 0.8) / 0.2
                y = gy - S * 0.03 - rise
                cv.ring("main", bx, y, r * (0.9 + 0.6 * q), max(0.6, 1.4 * (1 - q)), heat=0.85 - 0.4 * q, squash=0.7)
                for j in range(4):
                    aa = -math.pi / 2 + (j - 1.5) * 0.7
                    d = r * (1 + 2.2 * q)
                    cv.ellipse("main", bx + math.cos(aa) * d, y + math.sin(aa) * d + S * 0.25 * q * q, 0.8, heat=0.8,
                               snap=True)
    return beats, draw, {"": True}


def _dark_magic(c: Ctx, n):
    """Wisps spiral in, a dark orb forms with a hot violet rim and crackles, collapses, then
    bursts: tendrils lash out curling, a ring blows outward, all fades into dark smoke."""
    beats = timeline([("gather", 8, 60, 45), ("orb", 3, 50, 50), ("burst", 2, 40, 40), ("lash", 5, 45, 55),
                      ("fade", 8, 60, 100)], n)
    S, rng, w, h = c.S, c.rng, c.w, c.h
    cx, cy = w / 2, h / 2
    sn = c.noise(5, 2)
    K = 6
    wisp_a = [2 * math.pi * k / K + rng.uniform(-0.2, 0.2) for k in range(K)]
    lashes = [(2 * math.pi * k / 9 + rng.uniform(-0.2, 0.2), rng.uniform(0.75, 1.05), rng.choice([-1, 1]) * rng.uniform(0.8, 1.6))
              for k in range(9)]
    t_b = phase_t(beats, "burst")
    end = beats[-1].t
    motes = [Particle(cx, cy, math.cos(a) * v * S, math.sin(a) * v * S, t_b, rng.uniform(0.5, 0.9), 3.0, 0, -0.3 * S,
                      size=float(rng.choice([0.8, 1.2])), heat=0.9)
             for a, v in zip(rng.uniform(0, 2 * math.pi, 12), rng.uniform(0.5, 1.2, 12))]
    smokes = [(a, k) for k, a in enumerate(rng.uniform(0, 2 * math.pi, 7))]

    def orb(cv, r, crack_k):
        cv.ellipse("main", cx, cy, r * 1.25, heat=lambda rho, X, Y: 0.35 - 0.25 * rho)   # aura
        cv.ellipse("main", cx, cy, r, heat=lambda rho, X, Y: 0.04 + 0.95 * rho ** 4 + 0.15 * np.clip(-(X - cx + Y - cy) / r, 0, 1) * rho,
                   mode="over")
        for j in range(3):
            a = crack_k * 2.1 + j * 2.2
            pts = bolt_points(np.random.default_rng(crack_k * 5 + j), cx + math.cos(a) * r, cy + math.sin(a) * r,
                              cx + math.cos(a) * r * 1.7, cy + math.sin(a) * r * 1.7, 2, 0.3)
            cv.stroke("main", pts, 0.5, 0.9)

    def draw(cv, b):
        t = b.t
        if b.phase == "gather":
            q = b.p
            for k, a0 in enumerate(wisp_a):
                head_r = S * 0.46 * (1 - ease_in(q, 1.4)) + S * 0.05
                spin = a0 + q * 2.6
                pts, wd, ht = [], [], []
                for j in range(9):
                    pts.append((cx + math.cos(spin - j * 0.13) * (head_r + j * S * 0.025),
                                cy + math.sin(spin - j * 0.13) * (head_r + j * S * 0.025)))
                    wd.append(max(0.3, 1.8 * S / 96 * (1 - j / 9)))
                    ht.append(0.85 - 0.06 * j)
                cv.stroke("main", pts, wd, ht)
            if q > 0.4:
                orb(cv, S * 0.1 * ease_out((q - 0.4) / 0.6), b.k)
            return
        if b.phase == "orb":
            orb(cv, S * (0.12 + 0.03 * b.k), b.k + 9)
            return
        if b.phase == "burst":
            if b.k == 0:
                cv.ellipse("main", cx, cy, S * 0.07, heat=1.0)
                cv.star("accent", cx, cy, S * 0.35, 1.0, thin=0.35)
            else:
                cv.ring("main", cx, cy, S * 0.2, S * 0.05, heat=1.0, inner_heat=0.5)
                cv.ellipse("main", cx, cy, S * 0.12, heat=lambda rho, X, Y: 0.1 + 0.85 * rho ** 3)
            return
        q = (t - t_b) / (end - t_b)
        if q < 0.6:
            qq = q / 0.6
            cv.ring("main", cx, cy, S * (0.22 + 0.25 * ease_out(qq)), S * 0.04 * (1 - qq) + 0.6, heat=0.9 - 0.5 * qq,
                    inner_heat=0.3)
        for a, k in smokes:
            aa = clamp((q - 0.2) / 0.8)
            if 0 < aa < 1:
                puff(cv, cx + math.cos(a) * S * (0.2 + 0.15 * aa), cy + math.sin(a) * S * 0.2 - S * 0.18 * aa,
                     S * (0.06 + 0.04 * aa), 0.8 - 0.3 * aa, sn, t, erosion=0.2 + ease_in(aa, 1.0), lit=(-0.5, -0.8),
                     seed=k)
        for a, L, curl in lashes:
            g = ease_out(q / 0.25, 2.0)
            die = ease_in((q - 0.25) / 0.55, 1.2)
            if die >= 1:
                continue
            length = S * 0.46 * L * g
            pts, wd, ht = [], [], []
            for j in range(10):
                s = j / 9
                aa = a + curl * s * s * (0.6 + 0.8 * die)
                rr = S * 0.08 + length * s
                pts.append((cx + math.cos(aa) * rr, cy + math.sin(aa) * rr))
                wd.append(max(0.25, 2.4 * S / 96 * (1 - s) * (1 - 0.7 * die)))
                ht.append(0.95 - 0.5 * s - 0.4 * die)
            start = int(die * 9)
            cv.stroke("main", pts[start:], wd[start:], ht[start:])
        for p in motes:
            draw_spark(cv, p, t, 0.03)
    return beats, draw, {"": False}


def _buff_aura(c: Ctx, n):
    """Loop: a pulsing ring at the feet, flame-like aura tongues licking up around the body
    (behind and in front), sparks rising."""
    S, rng, w, h = c.S, c.rng, c.w, c.h
    beats = loop_beats(n or 16, 60)
    cx, gy = w / 2, h * 0.9
    rx = w * 0.4
    noise = c.noise(4, 2)
    K = 10
    tongues = [(2 * math.pi * (k + 0.5) / K + rng.uniform(-0.15, 0.15), rng.uniform(0.35, 0.6) * h, rng.uniform(0, 1))
               for k in range(K)]
    tongues.sort(key=lambda a: math.sin(a[0]))
    sparks = [(rng.uniform(0, 1), rng.uniform(-1, 1), float(rng.choice([1, 2])), rng.integers(1, 3)) for _ in range(9)]

    def draw(cv, b):
        t = b.p
        pulse = 0.5 + 0.5 * math.cos(2 * math.pi * t)
        cv.ring("back", cx, gy, rx * 0.85, rx * 0.25, heat=0.6 + 0.2 * pulse, squash=0.3, inner_heat=0.25)
        cv.ring("main", cx, gy, rx * (0.95 + 0.05 * pulse), 1.2, heat=0.9, squash=0.3)
        for a, hk, ph in tongues:
            x, y = cx + math.cos(a) * rx * 0.85, gy + math.sin(a) * rx * 0.3 * 0.85
            hh = hk * (0.65 + 0.35 * math.sin(2 * math.pi * (t + ph)))
            wid = w * 0.1 * (1 - 0.35 * abs(math.cos(a)))
            back = math.sin(a) < 0
            mat = "back" if back else "main"

            def fn(X, Y, x=x, y=y, hh=hh, wid=wid, back=back, ph=ph):
                d, v = flame_field(X + ph * 40, Y, x + ph * 40, y, hh, wid, t, noise, turb=0.5, speed=2,
                                   lean=-math.copysign(0.08, x - cx))
                return d > 0, (0.32 + 0.95 * np.clip(d, 0, 1) - 0.4 * v) * (0.8 if back else 1.0)
            cv.field(mat, (x - wid * 3, y - hh - 2, x + wid * 3, y + wid), fn)
        for ph, xs, sz, sp in sparks:
            p = (ph + t * sp) % 1.0
            x = cx + xs * rx * (1 - 0.3 * p)
            y = gy - p * h * 0.9
            cv.stroke("accent", [(x, y + sz * 2.5), (x, y)], [0.3, sz * 0.6], [0.5, 1.0 - 0.4 * p])
    return beats, draw, {"": True}


def _level_up_burst(c: Ctx, n):
    """Light gathers, an 8-point star flash, rings burst out and sparkles fly and twinkle,
    columns of light and chevrons rise, everything twinkles out."""
    beats = timeline([("gather", 5, 70, 45), ("flash", 1, 40, 40), ("burst", 6, 40, 55), ("rise", 10, 60, 60),
                      ("fade", 6, 70, 100)], n)
    S, rng, w, h = c.S, c.rng, c.w, c.h
    cx, cy, gy = w / 2, h * 0.58, h * 0.92
    t0 = phase_t(beats, "flash")
    end = beats[-1].t
    stars = [Particle(cx, cy, math.cos(a) * v * S, math.sin(a) * v * S * 0.9, t0, rng.uniform(0.7, 1.1), 4.0, 0, -0.15 * S,
                      size=float(rng.choice([2.5, 3.5, 4.5])) * S / 96, ph=rng.integers(0, 3))
             for a, v in zip(np.linspace(0, 2 * math.pi, 12, endpoint=False) + rng.uniform(-0.2, 0.2, 12),
                             rng.uniform(1.0, 1.8, 12))]
    cols = [(rng.uniform(-0.38, 0.38) * w, rng.uniform(0, 1), rng.uniform(0.15, 0.3) * h) for _ in range(6)]
    inward = rng.uniform(0, 2 * math.pi, 8)

    def draw(cv, b):
        t = b.t
        if b.phase == "gather":
            q = b.p
            for a in inward:
                rr = S * 0.48 * (1 - ease_in(q, 1.5)) + 2
                x, y = cx + math.cos(a) * rr, cy + math.sin(a) * rr
                x2, y2 = cx + math.cos(a) * (rr + S * 0.1), cy + math.sin(a) * (rr + S * 0.1)
                cv.stroke("accent", [(x2, y2), (x, y)], [0.3, 1.0], [0.5, 1.0])
            cv.ellipse("main", cx, cy, S * (0.04 + 0.06 * q), heat=lambda rho, X, Y: 1 - 0.5 * rho)
            cv.ellipse("back", cx, gy, w * 0.4 * q, w * 0.1 * q, heat=lambda rho, X, Y: 0.75 - 0.35 * rho)
            return
        if b.phase == "flash":
            cv.star("main", cx, cy, S * 0.48, 1.0, thin=0.36)
            cv.star("main", cx, cy, S * 0.3, 0.9, rot=math.pi / 4, thin=0.36)
            cv.ellipse("main", cx, cy, S * 0.12, heat=1.0)
            return
        q = (t - t0) / (end - t0)
        if q < 0.45:
            for k, dl in enumerate((0.0, 0.12)):
                qq = (q - dl) / 0.33
                if 0 < qq < 1:
                    cv.ring("main", cx, cy, S * 0.48 * ease_out(qq), S * 0.05 * (1 - qq) + 0.6, heat=0.95 - 0.5 * qq,
                            inner_heat=0.5 - 0.3 * qq)
        if b.phase == "burst" and b.k < 2:
            cv.star("main", cx, cy, S * (0.25 - 0.08 * b.k), 0.95, thin=0.38)
        fade = ease_in((q - 0.7) / 0.3)
        if b.phase in ("rise", "fade"):        # columns of light and chevrons rise
            for x, ph, L in cols:
                p = (ph + (t - t0) * 1.6) % 1.0
                y = gy - p * h * 0.95
                if fade < 1:
                    cv.stroke("back", [(cx + x, y + L * (1 - fade)), (cx + x, y)], [0.5, 0.6 * S / 96 + 0.3],
                              [0.55, 0.95])
            for k in range(2):
                p = clamp(((t - t0) * 1.4 - 0.25 - k * 0.3) % 1.2)
                if 0 < p < 1:
                    y = gy - S * 0.1 - p * h * 0.75
                    a = S * 0.14
                    a *= 1 - 0.6 * fade
                    if a > 1:
                        cv.stroke("main", [(cx - a, y + a * 0.8), (cx, y), (cx + a, y + a * 0.8)],
                                  [0.6, 1.4, 0.6], 0.95 - 0.3 * p)
        cv.ellipse("back", cx, gy, w * 0.4 * (1 - fade), w * 0.1 * (1 - fade), heat=lambda rho, X, Y: 0.5 + 0.4 * (1 - rho))
        for p in stars:
            a = p.age(t)
            if 0 <= a <= 1:
                x, y = p.pos(t)
                tw = [1.0, 0.55, 0.8][(b.i + p.ph) % 3]
                cv.star("accent", x, y, p.size * tw * (1.2 - 0.6 * a) + 0.5, 1.0 - 0.3 * a)
    return beats, draw, {"": False}


# --------------------------------------------------------------------------- registry
# kind: (label, (w, h) at medium, default frames, default theme, blend, outline, maker)
SPELLS = {
    "fire_burst":      ("Fire burst", (96, 96), 24, "fire", "add", "none", _fire_burst),
    "fireball":        ("Fireball + impact", (128, 80), 26, "fire", "add", "none", _fireball),
    "ice_shard":       ("Ice shards", (96, 96), 24, "ice", "mix", "none", _ice_shard),
    "frost_nova":      ("Frost nova", (128, 96), 24, "ice", "mix", "none", _frost_nova),
    "lightning_bolt":  ("Lightning bolt", (72, 128), 18, "lightning", "add", "none", _lightning_bolt),
    "chain_lightning": ("Chain lightning", (192, 96), 24, "lightning", "add", "none", _chain_lightning),
    "sword_combo":     ("Sword combo (1-3 hits)", (96, 96), 27, "steel", "add", "none", _sword_combo),
    "hit_spark":       ("Hit sparks (directional)", (64, 64), 14, "fire", "add", "none", _hit_spark),
    "holy_light":      ("Heal / holy light", (64, 112), 28, "holy", "add", "none", _holy_light),
    "poison_cloud":    ("Poison bubbles (loop)", (96, 80), 20, "poison", "mix", "none", _poison_cloud),
    "dark_magic":      ("Dark magic burst", (96, 96), 26, "dark", "mix", "none", _dark_magic),
    "big_explosion":   ("Explosion + debris + smoke", (128, 128), 32, "fire", "mix", "none", _big_explosion),
    "buff_aura":       ("Buff aura (loop)", (64, 96), 16, "arcane", "add", "none", _buff_aura),
    "level_up_burst":  ("Level-up sparkle burst", (96, 112), 28, "holy", "add", "none", _level_up_burst),
}
LOOPABLE = ("holy_light",)              # one-shot by default, loop=True gives a loop version
SPELL_KINDS = tuple(SPELLS)


def spell_size(kind: str, size=None, style=None) -> tuple[int, int]:
    """(w, h): size = (w, h), a number or small/medium/large/huge for the longer side;
    default: medium scaled by the project's person height (64 = medium)."""
    bw, bh = SPELLS[kind][1]
    if isinstance(size, (tuple, list)):
        return max(8, int(size[0])), max(8, int(size[1]))
    if isinstance(size, str):
        if size not in SIZES:
            raise ValueError(f"Unknown size '{size}' (use {', '.join(SIZES)} or a number)")
        size = SIZES[size]
    if size:
        k = int(size) / max(bw, bh)
    else:
        ph = getattr(style, "person_height", 64) if style is not None else 64
        k = ph / 64
    return max(8, round(bw * k)), max(8, round(bh * k))


def render_spell(kind: str, *, theme: str | None = None, colours=None, size=None,
                 frames: int | None = None, loop: bool | None = None, seed: int = 0,
                 style=None, palette: np.ndarray | None = None, outline: str | None = None,
                 **opts):
    """Draw a premium effect -> effects.Effect (frames, glow, durations, tags, blend).
    theme "project" = the effect's own theme snapped to the project palette (or `palette`).
    outline: none | dark | rim (default: the effect's). opts: hits (sword_combo), angle..."""
    from .effects import Effect, project_palette
    if kind not in SPELLS:
        raise ValueError(f"Unknown effect '{kind}' (use {', '.join(SPELL_KINDS)})")
    label, _, n0, theme0, blend, outline0, maker = SPELLS[kind]
    if frames and not 4 <= int(frames) <= 48:
        raise ValueError("frames must be between 4 and 48")
    w, h = spell_size(kind, size, style)
    use = theme or theme0
    pal = palette
    if use == "project":
        pal = palette if palette is not None else project_palette(style)
        use = theme0
    ramps = spell_ramps(None if colours else use, colours, pal)
    if kind in LOOPABLE:
        opts["_loop"] = bool(loop)
    rng = np.random.default_rng(seed)
    ctx = Ctx(w, h, rng, opts)
    beats, draw, loops = maker(ctx, int(frames) if frames else None)
    if loop is not None and len(loops) == 1:
        loops = {k: bool(loop) for k in loops}
    outs, glows, cols = [], [], []
    for b in beats:
        cv = ctx.canvas()
        draw(cv, b)
        rgba, glow, used = finish(cv, ramps, outline or outline0)
        outs.append(rgba)
        glows.append(glow)
        cols.append(used)
    tags, start = [], 0
    for i, b in enumerate(beats):
        if i == len(beats) - 1 or beats[i + 1].tag != b.tag:
            tags.append((b.tag or kind, start, i, bool(loops.get(b.tag, False))))
            start = i + 1
    palette_used = np.unique(np.concatenate([c for c in cols if len(c)]), axis=0)
    durs = [b.ms for b in beats]
    fx = Effect(kind, outs, palette_used, durs[0], len(tags) == 1 and tags[0][3],
                None if colours else (theme or theme0), ramps["main"])
    fx.durations, fx.glow, fx.blend, fx.tags = durs, glows, blend, tags
    return fx


# --------------------------------------------------------------------------- previews

def contact_sheet(effects_list, out, scale: int = 2, backgrounds=((24, 22, 36), (226, 222, 210)),
                  max_frames: int = 32, label: bool = True):
    """Every effect a row, frames across, on a dark and a light strip -> PNG."""
    from PIL import Image, ImageDraw
    rows = []
    for fx in effects_list:
        w, h = fx.size
        nf = min(len(fx.frames), max_frames)
        for bg in backgrounds:
            strip = Image.new("RGB", (w * nf, h), bg)
            for i in range(nf):
                im = Image.fromarray(fx.frames[i], "RGBA")
                strip.paste(im, (i * w, 0), im)
            rows.append((fx, strip.resize((strip.width * scale, strip.height * scale), Image.NEAREST)))
    left = 130 if label else 0
    W = max(r.width for _, r in rows) + left
    H = sum(r.height + 2 for _, r in rows)
    sheet = Image.new("RGB", (W, H), (12, 12, 18))
    d = ImageDraw.Draw(sheet)
    y = 0
    for j, (fx, r) in enumerate(rows):
        sheet.paste(r, (left, y))
        if label and j % len(backgrounds) == 0:
            d.text((6, y + 6), f"{fx.kind}\n{fx.theme or ''}\n{len(fx.frames)}f {sum(fx.durations)}ms\n"
                               f"{len(fx.palette)} colours", fill=(220, 220, 230))
        y += r.height + 2
    sheet.save(out)
    return out


def preview_gif(effects_list, out, scale: int = 2, columns: int = 5, tick_ms: int = 40,
                ground=(24, 22, 36), glow: bool = True, pause_ms: int = 400):
    """Every effect in a grid, all playing at their real frame timing (one-shots pause, then
    replay), with bloom from their glow maps -> animated GIF."""
    from PIL import Image
    from .fxcore import composite
    cell = max(max(fx.size) for fx in effects_list) * scale
    rows = -(-len(effects_list) // columns)
    cycles = [sum(fx.frame_durations()) + (0 if fx.loop else pause_ms) for fx in effects_list]
    total = max(cycles)
    images = []
    for t in range(0, total, tick_ms):
        canvas = Image.new("RGB", (columns * cell, rows * cell), ground)
        for k, fx in enumerate(effects_list):
            tt, acc, frame = t % cycles[k], 0, None
            for i, ms in enumerate(fx.frame_durations()):
                if tt < acc + ms:
                    frame = i
                    break
                acc += ms
            w, h = fx.size
            if frame is None:
                continue
            g = fx.glow[frame] if glow and fx.glow is not None else None
            im = Image.fromarray(composite(fx.frames[frame], ground, g)[..., :3])
            im = im.resize((w * scale, h * scale), Image.NEAREST)
            canvas.paste(im, ((k % columns) * cell + (cell - im.width) // 2,
                              (k // columns) * cell + (cell - im.height) // 2))
        images.append(canvas.quantize(256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
    images[0].save(out, save_all=True, append_images=images[1:], duration=tick_ms, loop=0, optimize=False)
    return out
