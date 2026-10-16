"""Structured progress for long jobs: how far (overall and the current stage), what it's doing,
about how long is left, and a Stop.

One `Tracker` per job. It is ALSO the job's callback, so every pipeline keeps calling what it
already calls:

    say("Drawing it facing SE (… 3 of 16, about 4 min left)…")   # text: stage words, "n of N",
                                                                   # "about X min left" are read
    progress("shape", "Building the 3D shape…")                    # model3d's (step, text)
    progress(3, 10)                                                # (done, total)
    progress(0.4)                                                  # a fraction

Pipelines that know more report it directly: `t.stage(key)`, `t.step(done, total)`,
`t.fraction(f)`. The GPU pictures inside a job report ComfyUI's sampling steps through the
thread's current tracker (`using(t)` / `current()`; generate.ComfyClient.wait calls
`gpu_sampling`), so a single 25-step picture moves the bar too.

A plan (list of `Stage`s, weights in expected minutes) turns stages into one overall bar;
ETAs come from the plan scaled by how fast the finished stages really went, the live rate of
"n of N" steps, and "about X min left" hints. Without a plan: "n of N" and the GPU steps, and
an `estimate` (seconds) if one is given. Bars never go backwards.

Stop: `t.stop()` from any thread. The next text the job says raises `Stopped` (stop_raises),
GPU waits raise it too (and ComfyUI is interrupted): cooperative, at the next message.

No Qt here: ui/common.py turns updates into progress bars.
"""
from __future__ import annotations

import math
import re
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Callable


class Stopped(Exception):
    """The user pressed Stop; what the job finished so far is kept where the job keeps it."""


@dataclass
class Stage:
    key: str
    label: str
    minutes: float = 1.0                  # expected time (relative weights are what matter)
    match: str = ""                       # regex (case-insensitive) on said text -> this stage
    pictures: int = 0                     # GPU calls expected in it (0 = unknown: time-based)

    def matches(self, text: str) -> bool:
        if text.strip().lower() == self.key.lower():
            return True
        return bool(self.match) and re.search(self.match, text, re.I) is not None


@dataclass
class Update:
    """What a progress bar shows. fraction None = unknown (a busy bar)."""
    title: str = ""
    text: str = ""
    fraction: float | None = None          # overall 0..1
    stage: str = ""                        # current stage label
    stage_index: int = 0                   # 1-based; 0 = no stages
    stage_count: int = 0
    stage_fraction: float | None = None    # 0..1 of the current stage
    step: int | None = None                # "n of N" in the stage: done so far
    total: int | None = None
    sampling: tuple[int, int] | None = None  # the GPU picture's (step, steps)
    eta: float | None = None               # seconds left (overall)
    elapsed: float = 0.0
    done: bool = False
    stopping: bool = False

    def eta_text(self) -> str:
        return eta_text(self.eta)

    def percent(self) -> int | None:
        return None if self.fraction is None else int(round(100 * self.fraction))


def eta_text(seconds: float | None) -> str:
    """'about 4 min left' / 'about 40 s left' / '' (unknown)."""
    if seconds is None or not math.isfinite(seconds):
        return ""
    if seconds < 1:
        return "almost done"
    if seconds < 55:
        return f"about {max(5, int(round(seconds / 5)) * 5)} s left"
    m = seconds / 60
    if m < 1.5:
        return "about 1 min left"
    if m < 90:
        return f"about {int(round(m))} min left"
    return f"about {m / 60:.1f} h left"


# --------------------------------------------------------------------------- text reading

_OF = re.compile(r"(?<![\w.])(\d+)\s*(?:of|/)\s*(\d+)\b(\s+[\w ]{0,24}?\b(?:done|finished|made|bent))?",
                 re.I)
_LEFT = re.compile(r"about\s+(\d+(?:\.\d+)?)\s*(h|hours?|min|minutes?|m|s|sec|seconds?)\s+left", re.I)
_NOT_PROGRESS = re.compile(r"^\s*(reusing|kept|keeping|skipp|warning|colours? per)|failure \d+ of|"
                           r"try \d+ of|attempt \d+ of", re.I)


def read_steps(text: str) -> tuple[int, int] | None:
    """'Drawing … 3 of 16 …' -> (2, 16) (the 3rd is under way); '… 3 of 16 directions done' ->
    (3, 16). None when the text has no count (or the count isn't progress: 'failure 1 of 3')."""
    if _NOT_PROGRESS.search(text):
        return None
    for m in _OF.finditer(text):
        n, total = int(m.group(1)), int(m.group(2))
        if total <= 0 or n > total or total > 100000:
            continue
        done = n if m.group(3) else max(0, n - 1)
        return done, total
    return None


def read_left(text: str) -> float | None:
    """'about 4 min left' -> 240.0 seconds."""
    m = _LEFT.search(text)
    if not m:
        return None
    v, unit = float(m.group(1)), m.group(2).lower()
    if unit.startswith("h"):
        return v * 3600
    if unit.startswith("m"):
        return v * 60
    return v


# --------------------------------------------------------------------------- plans

def character_plan(*, model: bool = True, drawn: bool = True, preview: bool = False,
                   directions: int = 16, approve: bool = False, views: bool = True) -> list[Stage]:
    """Stages of making a library character (makeany.GPU minutes): picture -> 3D shape ->
    (views/colours) -> rig -> front drawing [-> review] -> each direction -> bending -> sheets.
    model False = from a model already made; approve = finishing a front-first draft."""
    try:
        from .makeany import GPU
    except Exception:  # noqa: BLE001 - the table is only for the weights
        GPU = {"sdxl": 0.25, "model3d": 2.5, "drawn_dir": 0.75, "bend": 1.8}
    st: list[Stage] = []
    if model:
        st += [Stage("picture", "Front picture", GPU.get("sdxl", 0.25),
                     r"front picture|picture of|from a picture", pictures=1),
               Stage("cutout", "Cutting out", 0.05, r"cutting out"),
               Stage("shape", "3D shape", max(0.5, GPU.get("model3d", 2.5) - 0.8),
                     r"3d shape|hunyuan|new \w+ model of", pictures=1)]
        if views:
            st.append(Stage("views", "Back and sides", 0.6,
                            r"drawing the (back|left|right|side)|side of it|^views$"))
        st.append(Stage("colour", "Painting the model", 0.1, r"painting the model|own colours|^colour$"))
    if not approve:
        st.append(Stage("rig", "Skeleton + animations", 0.3,
                        r"skeleton|project's clips|swinging|capes, hair"))
    if drawn:
        n = max(1, int(directions))
        per = GPU.get("drawn_dir", 0.75)
        if preview:
            st += [Stage("front", "Front drawing", per * 1.5 + 0.3,
                         r"drawing it facing|keeping out|colours per part|painting the back|"
                         r"reading the skeleton"),
                   Stage("preview", "Idle preview", 0.15, r"short .* preview|waiting for your ok")]
        else:
            draw = per * (n - 1 if approve else n) + (0 if approve else 0.3)
            st += [Stage("draw", f"Drawing {n} directions", draw,
                         r"drawing it facing|mirrored from|keeping out|colours per part|"
                         r"painting the back|reading the skeleton"),
                   Stage("bend", "Bending the drawings", GPU.get("bend", 1.8),
                         r"bending")]
    else:
        st.append(Stage("render", f"Rendering {directions} directions", 1.8,
                        r"rendering every animation"))
    return st


def model_plan(views: bool = True) -> list[Stage]:
    """prompt_to_model alone (Characters page Make, props' 3D shape)."""
    return character_plan(model=True, drawn=False, views=views)[:-2]


def variations_plan(n: int, per_minutes: float, what: str = "variation", setup: float = 0.1,
                    finish: float = 0.1) -> list[Stage]:
    """n GPU pictures one after another, then the CPU finish (items, props, tiles, cards)."""
    return [Stage("setup", "Getting ready", setup, r"starting comfyui|loading|warming"),
            Stage("make", f"{n} {what}{'s' if n != 1 else ''}", max(0.05, per_minutes * n),
                  r"drawing|painting|making|picture|icon|swatch|art", pictures=1),
            Stage("finish", "Finishing", finish, r"pixel|convert|clean|tile|saving|writ|normal|sheet")]


def game_model_plan(shape: bool = False, engines: int = 1) -> list[Stage]:
    """Library -> Export 3D model: (GPU shape of a drawing) -> read -> remesh + LODs -> bake ->
    write -> previews -> export per engine (gamemodel.build_game_model's messages)."""
    st = [Stage("shape", "3D shape from the drawing", 1.5,
                r"3d shape from the drawing|hunyuan", pictures=1)] if shape else []
    return st + [
        Stage("read", "Reading the model", 0.05, r"reading the model|up to date"),
        Stage("remesh", "Remesh + fewer triangles", 0.45, r"remesh|cleaning"),
        Stage("bake", "Baking textures", 0.35, r"baking"),
        Stage("write", "Writing files", 0.05, r"writing files"),
        Stage("preview", "Previews", 0.12, r"rendering previews"),
        Stage("export", f"Export ({engines} engine{'s' if engines != 1 else ''})", 0.05 * engines,
              r"export"),
    ]


# --------------------------------------------------------------------------- the tracker

_local = threading.local()


def current() -> "Tracker | None":
    """The tracker of the job running on this thread (None outside a tracked job)."""
    stack = getattr(_local, "stack", None)
    return stack[-1] if stack else None


class using:
    """with using(tracker): ... -> current() is it on this thread (GPU steps report into it)."""

    def __init__(self, tracker: "Tracker | None"):
        self.tracker = tracker

    def __enter__(self):
        if self.tracker is not None:
            _local.__dict__.setdefault("stack", []).append(self.tracker)
        return self.tracker

    def __exit__(self, *exc):
        if self.tracker is not None:
            stack = getattr(_local, "stack", [])
            if stack and stack[-1] is self.tracker:
                stack.pop()


def gpu_sampling(value: int | None, maximum: int | None, picture: int | None = None,
                 pictures: int | None = None) -> None:
    """Called while ComfyUI makes a picture: its sampler step (None = no websocket news)."""
    t = current()
    if t is not None:
        t.sampling(value, maximum, picture, pictures)


def gpu_pictures(i: int, n: int) -> None:
    """Picture i (0-based) of n in one generate call is next."""
    t = current()
    if t is not None:
        t.pictures(i, n)


def stop_requested() -> bool:
    """Stop pressed on a job that stops mid-way (a batch's "after this one" doesn't)."""
    t = current()
    return t is not None and t.stopped and t.stop_raises


def check_stop() -> None:
    t = current()
    if t is not None:
        t.check()


def wrap(callback) -> Callable:
    """Any old-style callback (text) / (step, text) / None -> one accepting every form (the
    first argument's text goes to old callbacks)."""
    if callback is None:
        return lambda *a, **k: None
    if isinstance(callback, Tracker):
        return callback

    def call(*args, **kw):
        texts = [a for a in args if isinstance(a, str)]
        if texts:
            return callback(texts[-1])
    return call


@dataclass
class _StageState:
    started: float | None = None
    ended: float | None = None
    skipped: bool = False
    frac: float = 0.0
    step: tuple[int, int] | None = None
    step_at: float | None = None
    step_times: list = field(default_factory=list)
    hint: tuple[float, float] | None = None   # (seconds left, when said)
    sample: tuple[int, int] | None = None
    sample_at: float | None = None
    picture: tuple[int, int] | None = None
    calls: int = 0                            # GPU generate calls started in this stage


class Tracker:
    """Progress of one job. Call it like any progress callback (see the module doc); read
    `snapshot()`; `on_update(Update)` is called (throttled) from the job's thread."""

    def __init__(self, plan: list[Stage] | None = None, *, estimate: float | None = None,
                 on_update: Callable[[Update], None] | None = None, title: str = "",
                 stop_raises: bool = True, clock: Callable[[], float] = time.monotonic,
                 min_interval: float = 0.1, parent: "Tracker | None" = None,
                 span: tuple[float, float] = (0.0, 1.0)):
        if not plan:
            plan = [Stage("work", title or "Working", (estimate or 60.0) / 60.0)]
            self._planned = estimate is not None
        else:
            self._planned = True
        self.plan = list(plan)
        self.title = title
        self.on_update = on_update
        self.stop_raises = stop_raises
        self.clock = clock
        self.min_interval = min_interval
        self.parent = parent
        self.span = span
        self._lock = threading.RLock()
        self._stop = threading.Event() if parent is None else parent._stop
        self.t0 = clock()
        self.text = ""
        self.cur = 0
        self.states = [_StageState() for _ in self.plan]
        self.states[0].started = self.t0
        self._best = 0.0                     # never go backwards
        self._best_stage = 0.0
        self._last_emit = -1e9
        self.done = False
        self.history: list[Update] = []      # tests / debugging (kept short)
        self.children = 0
        self._confirmed = len(self.plan) == 1    # the first stage really happened (said / GPU)

    # ---- the callback forms
    def __call__(self, *args, **kw):
        if len(args) == 1 and isinstance(args[0], str):
            self.say(args[0])
        elif len(args) == 2 and isinstance(args[0], str) and isinstance(args[1], str):
            self.stage(args[0], args[1])
        elif len(args) == 2 and all(isinstance(a, (int, float)) for a in args):
            self.step(args[0], args[1])
        elif len(args) == 1 and isinstance(args[0], (int, float)):
            self.fraction(float(args[0]))
        elif len(args) == 2 and isinstance(args[1], str):       # (anything, text)
            self.say(args[1])
        elif args and isinstance(args[-1], str):
            self.say(args[-1])
        else:
            self._emit()

    # ---- reporting
    def say(self, text: str):
        """A status message: picks the stage it names, reads 'n of N' and 'about X min left'."""
        self.check()
        text = str(text)
        with self._lock:
            self.text = text
            if self.cur == 0 and not self._confirmed and self.plan[0].matches(text):
                self._confirmed = True
            self._match_stage(text)
            st = self.states[self.cur]
            steps = read_steps(text)
            if steps is not None:
                self._set_step(st, *steps)
            left = read_left(text)
            if left is not None:
                st.hint = (left, self.clock())
        self._emit(force=True)

    def stage(self, key: str, text: str | None = None):
        """Move to the stage with this key (or label); unknown keys just show the text."""
        self.check()
        with self._lock:
            for j in range(self.cur, len(self.plan)):
                s = self.plan[j]
                if key.lower() in (s.key.lower(), s.label.lower()):
                    if j == 0:
                        self._confirmed = True
                    self._enter(j)
                    break
            if text is not None:
                self.text = text
                st = self.states[self.cur]
                steps = read_steps(text)
                if steps is not None:
                    self._set_step(st, *steps)
                left = read_left(text)
                if left is not None:
                    st.hint = (left, self.clock())
        self._emit(force=True)

    def step(self, done: float, total: float, text: str | None = None):
        """done of total in the current stage."""
        self.check()
        with self._lock:
            if total > 0:
                self._set_step(self.states[self.cur], int(done), int(total))
            if text is not None:
                self.text = text
        self._emit(force=True)

    def fraction(self, f: float, text: str | None = None):
        """The current stage is f (0..1) done."""
        self.check()
        with self._lock:
            st = self.states[self.cur]
            st.frac = max(st.frac, min(1.0, max(0.0, float(f))))
            if text is not None:
                self.text = text
        self._emit()

    def hint(self, seconds_left: float):
        with self._lock:
            self.states[self.cur].hint = (float(seconds_left), self.clock())
        self._emit()

    def pictures(self, i: int, n: int):
        """Picture i (0-based) of n of one GPU call is next (i == 0: a new call)."""
        with self._lock:
            st = self.states[self.cur]
            if int(i) == 0:
                st.calls += 1
            st.picture = (int(i), int(n))
            st.sample, st.sample_at = None, None
        self._emit()

    def sampling(self, value, maximum, picture=None, pictures=None):
        """The GPU picture's sampler step (value None = still working, nothing known)."""
        with self._lock:
            st = self.states[self.cur]
            if picture is not None and pictures:
                st.picture = (int(picture), int(pictures))
            if value is not None and maximum:
                self._confirmed = True
                st.sample = (int(value), int(maximum))
                st.sample_at = self.clock()
        self._emit()

    def child(self, k: int, n: int, plan: list[Stage] | None = None, *, estimate=None,
              title: str = "") -> "Tracker":
        """Job k (0-based) of n inside this tracker's current stage (a batch): the child's
        overall progress is this stage's ((k + child) / n). Shares Stop."""
        with self._lock:
            self._set_step(self.states[self.cur], k, n)
            self.children += 1
        return Tracker(plan, estimate=estimate, title=title, stop_raises=self.stop_raises,
                       clock=self.clock, min_interval=0.0, parent=self, span=(k / n, (k + 1) / n))

    # ---- stop
    def stop(self):
        self._stop.set()
        hook = getattr(self, "on_stop", None)     # a job that stops its own way (training)
        if hook is not None:
            hook()
        self._emit(force=True)

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def check(self):
        if self.stop_raises and self._stop.is_set():
            raise Stopped("Stopped (what was finished is kept)")

    def finish(self, text: str | None = None):
        with self._lock:
            now = self.clock()
            for j in range(self.cur, len(self.plan)):
                if self.states[j].started is None:
                    self.states[j].skipped = True
                self.states[j].ended = self.states[j].ended or now
                self.states[j].frac = 1.0
            self.done = True
            if text:
                self.text = text
        self._emit(force=True)

    # ---- internals
    def _match_stage(self, text: str):
        for j in range(self.cur + 1, len(self.plan)):
            if self.plan[j].matches(text):
                self._enter(j)
                return

    def _enter(self, j: int):
        if j == self.cur:
            return
        now = self.clock()
        old = self.states[self.cur]
        old.ended, old.frac = now, 1.0
        if self.cur == 0 and not self._confirmed:     # the job started past it
            old.skipped = True
        self._confirmed = True
        for i in range(self.cur + 1, j):
            self.states[i].skipped = True
        self.cur = j
        self.states[j].started = now
        self._best_stage = 0.0

    def _set_step(self, st: _StageState, done: int, total: int):
        now = self.clock()
        self._confirmed = True
        if st.step is not None and st.step[1] == total and done > st.step[0] and st.step_at is not None:
            st.step_times.append((now - st.step_at) / (done - st.step[0]))
        if st.step is None or done != st.step[0] or total != st.step[1]:
            st.step_at = now
            st.sample, st.sample_at = None, None
            st.picture = None
            st.calls = 0
        st.step = (done, total)

    def _weights(self):
        return [0.0 if s.skipped else max(1e-6, p.minutes)
                for p, s in zip(self.plan, self.states)]

    def _sub(self, st: _StageState, expected_s: float, now: float) -> float | None:
        """0..1 within the current piece of work (one step / the whole stage) from the GPU steps."""
        if st.sample is not None:
            v, m = st.sample
            f = min(1.0, v / m)
            if st.picture is not None and st.picture[1] > 1:
                i, n = st.picture
                f = (min(i, n - 1) + f) / n
            return f
        if st.picture is not None and st.picture[1] > 1:
            return min(st.picture[0], st.picture[1]) / st.picture[1]
        return None

    def _stage_fraction(self, now: float, ratio: float) -> float:
        j = self.cur
        st = self.states[j]
        if self.done:
            return 1.0
        t_in = now - (st.started or self.t0)
        expected = max(1.0, self.plan[j].minutes * 60 * ratio)
        if st.step is not None:
            done, total = st.step
            per = (sorted(st.step_times)[len(st.step_times) // 2] if st.step_times
                   else (t_in / done if done else expected / total))
            sub = self._sub(st, per, now)
            if sub is None:
                sub = _crawl((now - (st.step_at or now)) / max(1.0, per))
            f = (done + min(0.97, sub)) / total
        else:
            sub = self._sub(st, expected, now)
            want = self.plan[j].pictures
            if want and st.calls:                 # k-th of the stage's expected GPU calls
                f = (min(st.calls - 1, want - 1) + (sub if sub is not None else 0.0)) / want
                if self._planned:
                    f = max(f, 0.5 * _crawl(t_in / expected))
            elif sub is not None and st.calls <= 1:
                f = sub if not self._planned else max(sub, _crawl(t_in / expected) * 0.5)
            elif self._planned:
                f = _crawl(t_in / expected)
            else:
                f = 0.0
        return max(st.frac, min(1.0, f))

    def _ratio(self, now: float) -> float:
        """How much slower (>1) or faster the finished stages went than planned."""
        planned = actual = 0.0
        for p, s in zip(self.plan, self.states):
            if s.started is not None and s.ended is not None and not s.skipped:
                planned += p.minutes * 60
                actual += s.ended - s.started
        if planned < 15 or actual <= 0:
            return 1.0
        r = actual / planned
        w = min(1.0, planned / 120)              # trust it more as more is finished
        return min(4.0, max(0.25, 1 + (r - 1) * w))

    def snapshot(self) -> Update:
        with self._lock:
            now = self.clock()
            ratio = self._ratio(now)
            j = self.cur
            st = self.states[j]
            sf = self._stage_fraction(now, ratio)
            self._best_stage = max(self._best_stage, sf)
            sf = self._best_stage
            w = self._weights()
            total = sum(w) or 1.0
            before = sum(w[:j])
            known = self._planned or st.step is not None or st.sample is not None or self.done
            frac = (before + w[j] * sf) / total if known else None
            if frac is not None:
                frac = max(self._best, min(1.0, frac))
                self._best = frac
            elif self._best > 0:
                frac = self._best
            # time left
            eta = None
            t_in = now - (st.started or self.t0)
            if self.done:
                eta = 0.0
            elif known:
                later = sum(w[j + 1:]) * 60 * ratio
                cur_left = None
                if st.hint is not None:
                    cur_left = max(0.0, st.hint[0] - (now - st.hint[1]))
                    if sf > 0.05 and t_in > 10:      # blend with the live rate
                        live = t_in / sf * (1 - sf)
                        cur_left = 0.6 * cur_left + 0.4 * live
                elif st.step is not None and st.step[0] >= 1 and sf > 0:
                    cur_left = t_in / sf * (1 - sf)
                elif sf > 0.15 and t_in > 5 and not self._planned:
                    cur_left = t_in / sf * (1 - sf)
                if cur_left is None:
                    if self._planned:
                        cur_left = max(0.0, w[j] * 60 * ratio * (1 - sf))
                    else:
                        cur_left = None
                if cur_left is not None:
                    eta = cur_left + (later if self._planned else 0.0)
            label = self.plan[j].label if (len(self.plan) > 1 or self._planned) else ""
            sample = st.sample
            stepped = st.step
            return Update(
                title=self.title, text=self.text, fraction=frac, stage=label,
                stage_index=(sum(1 for s in self.states[:j + 1] if not s.skipped)
                             if len(self.plan) > 1 else 0),
                stage_count=sum(1 for s in self.states if not s.skipped) if len(self.plan) > 1 else 0,
                stage_fraction=sf if known else None,
                step=stepped[0] if stepped else None, total=stepped[1] if stepped else None,
                sampling=sample, eta=eta, elapsed=now - self.t0, done=self.done,
                stopping=self._stop.is_set())

    def _emit(self, force: bool = False):
        if self.parent is not None:
            self._to_parent()
            return
        now = self.clock()
        if not force and now - self._last_emit < self.min_interval:
            return
        self._last_emit = now
        u = self.snapshot()
        self.history.append(u)
        if len(self.history) > 2000:
            del self.history[:1000]
        if self.on_update is not None:
            self.on_update(u)

    def _to_parent(self):
        """A child's progress = the parent's current stage fraction; its stage + text show."""
        u = self.snapshot()
        p = self.parent
        a, b = self.span
        f = u.fraction if u.fraction is not None else 0.0
        with p._lock:
            st = p.states[p.cur]
            st.frac = max(st.frac, a + (b - a) * f)
            if u.text:
                p.text = u.text
            p._child = u
        p._emit()


def _crawl(x: float) -> float:
    """Time-based guess: linear to 0.85 at the expected time, then creeping toward 0.97."""
    x = max(0.0, x)
    if x <= 1.0:
        return 0.85 * x
    return 0.85 + 0.12 * (1 - math.exp(-(x - 1.0)))


def child_update(t: Tracker) -> Update | None:
    """The latest update of a batch's current child job (the 'current job' bar)."""
    return getattr(t, "_child", None)


def snapshot_with_child(t: Tracker) -> tuple[Update, Update | None]:
    return t.snapshot(), child_update(t)


def merge(u: Update, **kw) -> Update:
    return replace(u, **kw)
