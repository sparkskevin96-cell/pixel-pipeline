"""Shared pieces of the desktop app's tool tabs: background jobs, pixel-art previews and the
small sprite options box (type, retro system, size, colours)."""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QBoxLayout, QComboBox, QFormLayout, QGridLayout, QHBoxLayout, QLabel,
    QMessageBox, QProgressBar, QPushButton, QScrollArea, QSizePolicy, QSpinBox, QToolButton,
    QVBoxLayout, QWidget,
)

from ..settings import SUBJECTS, Settings, settings_for
from ..systems import SYSTEMS


def checkerboard(rgba: np.ndarray, cell: int = 8) -> np.ndarray:
    """Composite RGBA over a grey checkerboard so transparency is visible."""
    h, w = rgba.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    bg = np.where(((yy // cell + xx // cell) % 2 == 0)[..., None], 205.0, 165.0)
    a = rgba[..., 3:4].astype(np.float32) / 255
    rgb = rgba[..., :3].astype(np.float32) * a + bg * (1 - a)
    return np.concatenate([np.rint(rgb).astype(np.uint8), np.full((h, w, 1), 255, np.uint8)], -1)


def to_pixmap(rgba: np.ndarray) -> QPixmap:
    rgba = np.ascontiguousarray(rgba)
    h, w = rgba.shape[:2]
    return QPixmap.fromImage(QImage(rgba.data, w, h, w * 4, QImage.Format_RGBA8888).copy())


def grid(images: list[np.ndarray], columns: int, gap: int = 2) -> np.ndarray:
    """Same-size RGBA images side by side, wrapping after `columns`, transparent gaps."""
    h = max(i.shape[0] for i in images)
    w = max(i.shape[1] for i in images)
    cols = max(1, min(columns, len(images)))
    rows = math.ceil(len(images) / cols)
    out = np.zeros((rows * h + (rows - 1) * gap, cols * w + (cols - 1) * gap, 4), np.uint8)
    for k, im in enumerate(images):
        r, c = divmod(k, cols)
        y, x = r * (h + gap) + h - im.shape[0], c * (w + gap) + (w - im.shape[1]) // 2
        out[y:y + im.shape[0], x:x + im.shape[1]] = im
    return out


class PixelView(QLabel):
    """Pixel art enlarged by a whole number to fill the space (crisp), or shrunk smoothly when
    it's bigger than the space (maps), on a checkerboard."""

    def __init__(self, placeholder: str = "", minimum: tuple[int, int] = (320, 280)):
        super().__init__(placeholder)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(*minimum)
        # Ignored: the picture never makes the label (and the window) grow, so a resize can't
        # feed back into a bigger picture and another resize.
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setStyleSheet("background:#1c2030;color:#9aa3b5;border-radius:6px;")
        self.setWordWrap(True)
        self.rgba: np.ndarray | None = None

    def show_rgba(self, rgba: np.ndarray | None):
        self.rgba = rgba
        if rgba is None:
            self.clear()
            return
        h, w = rgba.shape[:2]
        aw, ah = max(1, self.width() - 8), max(1, self.height() - 8)
        k = min(aw / w, ah / h)
        if k >= 1:
            k = max(1, min(16, int(k)))
            big = np.repeat(np.repeat(rgba, k, 0), k, 1)
            self.setPixmap(to_pixmap(checkerboard(big, cell=max(4, k * 2))))
        else:
            pix = to_pixmap(checkerboard(rgba, cell=8))
            self.setPixmap(pix.scaled(int(w * k), int(h * k), Qt.KeepAspectRatio,
                                      Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.rgba is not None:
            self.show_rgba(self.rgba)


class _Worker(QObject):
    progress = Signal(str)
    update = Signal(object)              # progress.Update (from the job's Tracker)
    done = Signal(object)
    failed = Signal(str)
    stopped = Signal(str)

    def __init__(self, job, tracker=None):
        super().__init__()
        self.job = job
        self.tracker = tracker

    def run(self):
        from .. import progress
        try:
            if self.tracker is None:
                self.done.emit(self.job(self.progress.emit))
                return
            with progress.using(self.tracker):
                result = self.job(self.tracker)
            self.tracker.finish()
            self.done.emit(result)
        except progress.Stopped as e:
            self.stopped.emit(str(e) or "Stopped")
        except Exception as e:  # noqa: BLE001 - shown to the user
            if self.tracker is not None and self.tracker.stopped and is_cancel(e):
                self.stopped.emit(str(e) or "Stopped")
            else:
                self.failed.emit(str(e) or type(e).__name__)


def is_cancel(e: BaseException) -> bool:
    """A pipeline's own 'you stopped me' exception (drawnrig.Cancelled, progress.Stopped…)."""
    return type(e).__name__ in ("Cancelled", "Stopped") or str(e).startswith("Stopped")


# --------------------------------------------------------------------------- progress bars

class ProgressHub(QObject):
    """Every running job's latest progress, for the compact bar in the main window's status bar
    (visible from any tab). changed(update or None): None = nothing running."""
    changed = Signal(object)

    def __init__(self):
        super().__init__()
        self.jobs: dict[int, tuple[object, object]] = {}     # id -> (Update, tracker)
        self.last: int | None = None

    def job_update(self, jid: int, update, tracker=None):
        self.jobs[jid] = (update, tracker)
        self.last = jid
        self.changed.emit(update)

    def job_end(self, jid: int):
        self.jobs.pop(jid, None)
        if self.last == jid or self.last not in self.jobs:
            self.last = next(iter(self.jobs), None)
        self.changed.emit(self.jobs[self.last][0] if self.last is not None else None)

    def current(self):
        return None if self.last is None else self.jobs[self.last][0]

    def stop_all(self):
        for _, t in list(self.jobs.values()):
            if t is not None:
                t.stop()


_hub: ProgressHub | None = None


def hub() -> ProgressHub:
    global _hub
    if _hub is None:
        _hub = ProgressHub()
    return _hub


def _bar(height: int = 16) -> QProgressBar:
    bar = QProgressBar()
    bar.setRange(0, 1000)
    bar.setTextVisible(True)
    bar.setMaximumHeight(height)
    return bar


def _set_bar(bar: QProgressBar, fraction, fmt: str):
    if fraction is None:
        bar.setRange(0, 0)               # busy: nothing known yet
        bar.setFormat(fmt.replace("%p%", "").strip(" ·") or "working…")
        return
    if bar.maximum() == 0:
        bar.setRange(0, 1000)
    f = max(0.0, min(1.0, float(fraction)))
    bar.setValue(int(round(1000 * f)))
    bar.setFormat(fmt.replace("%p%", f"{int(100 * f)}%"))


def _clock(seconds: float) -> str:
    s = int(max(0, seconds))
    return f"{s // 60}:{s % 60:02d}"


class ProgressPanel(QWidget):
    """A job's progress where the work is: what it's doing, the overall bar (with the stage),
    the current stage's bar (in a batch: the current job's) with the GPU picture's sampler step,
    about how long is left, and Stop. Hidden when nothing runs. A timer re-reads the tracker so
    time-based estimates and the countdown move between messages."""

    def __init__(self, parent=None, stop: bool = True):
        super().__init__(parent)
        self.setObjectName("progressPanel")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("#progressPanel{background:#20263a;border-radius:6px;}"
                           "#progressPanel QLabel{color:#d8deea;}")
        g = QGridLayout(self)
        g.setContentsMargins(8, 6, 8, 6)
        g.setVerticalSpacing(3)
        self.title = QLabel("")
        self.title.setStyleSheet("font-weight:600;")
        self.text = QLabel("")
        self.text.setStyleSheet("color:#9aa3b5;")
        self.text.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.overall = _bar(18)
        self.stage_label = QLabel("")
        self.stage_label.setStyleSheet("color:#9aa3b5;")
        self.stage = _bar(13)
        self.eta = QLabel("")
        self.eta.setStyleSheet("font-weight:600;")
        self.eta.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setToolTip("Stop after the step in progress (what's finished is kept)")
        self.stop_btn.clicked.connect(self._stop_clicked)
        self.stop_btn.setVisible(stop)
        g.addWidget(self.title, 0, 0)
        g.addWidget(self.text, 0, 1, 1, 2)
        g.addWidget(self.overall, 1, 0, 1, 2)
        g.addWidget(self.eta, 1, 2)
        g.addWidget(self.stage_label, 2, 0)
        g.addWidget(self.stage, 2, 1)
        g.addWidget(self.stop_btn, 2, 2)
        g.setColumnStretch(1, 1)
        g.setColumnMinimumWidth(0, 120)
        g.setColumnMinimumWidth(2, 110)
        self.tracker = None
        self.on_stop = None
        self.last = None
        self.updates = 0
        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self.refresh)
        self.setVisible(False)

    def start(self, tracker, title: str = "", on_stop=None):
        self.tracker, self.on_stop = tracker, on_stop
        self.title.setText(title or getattr(tracker, "title", "") or "Working")
        self.stop_btn.setEnabled(True)
        self.stop_btn.setText("Stop")
        self.setVisible(True)
        self.show_update(tracker.snapshot())
        self._timer.start()

    def refresh(self):
        if self.tracker is not None:
            self.show_update(self.tracker.snapshot())

    def show_update(self, u):
        from ..progress import child_update
        self.last = u
        self.updates += 1
        if u.text:
            self.text.setText(u.text)
            self.text.setToolTip(u.text)
        sub = child_update(self.tracker) if self.tracker is not None else None
        stage_txt = u.stage
        if u.stage_count > 1:
            stage_txt = f"{u.stage_index} of {u.stage_count}: {u.stage}"
        _set_bar(self.overall, u.fraction, "%p%" + (f" · {stage_txt}" if stage_txt else ""))
        eta = u.eta_text()
        self.eta.setText(eta or _clock(u.elapsed))
        self.eta.setToolTip(f"{_clock(u.elapsed)} so far")
        if sub is not None:                      # a batch: the current job's own bar
            k = (u.step or 0) + 1
            self.stage_label.setText(f"Job {min(k, u.total)} of {u.total}" if u.total else "This job")
            label = sub.stage or sub.title
            fmt = "%p%" + (f" · {label}" if label else "")
            if sub.sampling:
                fmt += f" · GPU step {sub.sampling[0]}/{sub.sampling[1]}"
            _set_bar(self.stage, sub.fraction, fmt)
        else:
            if u.total:
                self.stage_label.setText(f"{min(u.step + 1, u.total)} of {u.total}")
            else:
                self.stage_label.setText("This step")
            fmt = "%p%"
            if u.sampling:
                fmt += f" · GPU step {u.sampling[0]}/{u.sampling[1]}"
            _set_bar(self.stage, u.stage_fraction, fmt)
        if u.stopping and not u.done:
            self.stop_btn.setEnabled(False)
            self.stop_btn.setText("Stopping…")

    def _stop_clicked(self, _checked=False):
        self.stop()

    def stop(self):
        if self.tracker is not None:
            self.tracker.stop()
        if self.on_stop is not None:
            self.on_stop()
        self.stop_btn.setEnabled(False)
        self.stop_btn.setText("Stopping…")

    def end(self):
        self._timer.stop()
        self.tracker = None
        self.on_stop = None
        self.setVisible(False)


class GlobalProgress(QWidget):
    """The compact bar in the main window's status bar: the running job from any tab (+ how
    many more run), ■ stops them. The % also goes in front of the window title."""

    def __init__(self, window=None):
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 6, 0)
        self.label = QLabel("")
        self.label.setStyleSheet("color:#9aa3b5;")
        self.bar = _bar(14)
        self.bar.setFixedWidth(180)
        self.stop_btn = QToolButton()
        self.stop_btn.setText("■")
        self.stop_btn.setToolTip("Stop the running job(s) after the step in progress")
        self.stop_btn.clicked.connect(self._stop_all)
        for w in (self.label, self.bar, self.stop_btn):
            lay.addWidget(w)
        self.window = window
        self.base_title = window.windowTitle() if window is not None else ""
        self._active = False
        hub().changed.connect(self.show_update)
        self.setVisible(False)

    def _stop_all(self, _checked=False):
        hub().stop_all()

    def show_update(self, u):
        if u is None:
            self.setVisible(False)
            if self.window is not None and self.base_title and self._active:
                self.window.setWindowTitle(self.base_title)
            self._active = False
            return
        if self.window is not None and not self._active:
            self.base_title = self.window.windowTitle()
        self._active = True
        self.setVisible(True)
        n = len(hub().jobs)
        what = u.title or u.stage or "Working"
        eta = u.eta_text()
        self.label.setText(what + (f" (+{n - 1} more)" if n > 1 else "") + (f" · {eta}" if eta else ""))
        self.label.setToolTip(u.text)
        _set_bar(self.bar, u.fraction, "%p%")
        if self.window is not None and self.base_title:
            pct = u.percent()
            self.window.setWindowTitle((f"{pct}% · " if pct is not None else "… ") + self.base_title)


_job_ids = iter(range(1, 10**9))


class JobRunner:
    """Runs job(progress) off the UI thread with a progress.Tracker as its callback (it takes
    text like the old say(), (step, text), (done, total) and fractions), feeding a ProgressPanel
    and the global hub; Stop stops the job at its next message / GPU wait. sync=True runs it
    right here (tests), the panel updated the same way."""

    def __init__(self, owner: QWidget, panel: ProgressPanel | None, say, on_failed,
                 on_stopped=None, busy=None):
        self.owner, self.panel, self.say = owner, panel, say
        self.on_failed, self.on_stopped, self.busy = on_failed, on_stopped, busy
        self.thread = self.worker = None
        self.tracker = None
        self._last_text = ""

    def running(self) -> bool:
        return self.thread is not None and self.thread.isRunning()

    def start(self, job, on_done, *, sync: bool = False, plan=None, estimate=None,
              title: str = "", on_stop=None, stop_raises: bool = True):
        from .. import progress
        jid = next(_job_ids)
        tracker = progress.Tracker(plan, estimate=estimate, title=title, stop_raises=stop_raises)
        self.tracker = tracker
        self._last_text = ""
        stopped_cb = self.on_stopped or self.on_failed

        def update(u):
            if u.text and u.text != self._last_text and self.say is not None:
                self._last_text = u.text
                self.say(u.text)
            if self.panel is not None and self.panel.tracker is tracker:
                self.panel.show_update(u)
            if jid in hub().jobs or not u.done:
                hub().job_update(jid, u, tracker)

        if self.panel is not None:
            self.panel.start(tracker, title, on_stop)
        if sync:
            tracker.on_update = update
            try:
                with progress.using(tracker):
                    result = job(tracker)
                tracker.finish()
            except progress.Stopped as e:
                self._end(jid)
                stopped_cb(str(e) or "Stopped")
                return
            except BaseException as e:
                self._end(jid)
                if tracker.stopped and is_cancel(e):
                    stopped_cb(str(e) or "Stopped")
                    return
                raise
            self._end(jid)
            on_done(result)
            return
        if self.busy:
            self.busy(True)
        thread, worker = QThread(self.owner), _Worker(job, tracker)
        tracker.on_update = worker.update.emit
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.update.connect(update)
        worker.done.connect(on_done)
        worker.failed.connect(self.on_failed)
        worker.stopped.connect(self.on_stopped or self.on_failed)
        for sig in (worker.done, worker.failed, worker.stopped):
            sig.connect(thread.quit)
        busy = self.busy
        thread.finished.connect(lambda: self._end(jid))
        if busy:
            thread.finished.connect(lambda: busy(False))
        thread.finished.connect(worker.deleteLater)
        self.thread, self.worker = thread, worker
        thread.start()

    def _end(self, jid: int):
        hub().job_end(jid)
        if self.panel is not None and (self.panel.tracker is None or self.panel.tracker is self.tracker):
            self.panel.end()
        # drop every reference back to the page: a page <-> runner cycle left for Python's GC
        # can be collected on another thread (a server's), and Qt aborts deleting widgets there
        if self.tracker is not None:
            self.tracker.on_update = None
        self.owner = self.say = self.busy = None
        self.on_failed = self.on_stopped = None
        self.panel = None

    def stop(self):
        if self.tracker is not None:
            self.tracker.stop()


def attach_panel(widget: QWidget, panel: QWidget) -> bool:
    """Put a progress panel at the bottom of a page: its vertical layout, else the first
    vertical column of a side-by-side layout (the controls)."""
    lay = widget.layout()
    if lay is None:
        return False
    if isinstance(lay, QVBoxLayout):
        lay.addWidget(panel)
        return True
    if isinstance(lay, QFormLayout):
        lay.addRow(panel)
        return True
    if isinstance(lay, QGridLayout):
        lay.addWidget(panel, lay.rowCount(), 0, 1, max(1, lay.columnCount()))
        return True
    if isinstance(lay, QBoxLayout):
        for i in range(lay.count()):
            item = lay.itemAt(i)
            sub = item.layout()
            if isinstance(sub, QVBoxLayout):
                sub.addWidget(panel)
                return True
            if isinstance(sub, QFormLayout):
                sub.addRow(panel)
                return True
            w = item.widget()
            if w is None:
                continue
            if isinstance(w, QScrollArea) and w.widget() is not None:
                w = w.widget()
            if isinstance(w.layout(), QVBoxLayout):
                w.layout().addWidget(panel)
                return True
            if isinstance(w.layout(), QFormLayout):
                w.layout().addRow(panel)
                return True
    return False


class ToolTab(QWidget):
    """A tab whose slow work (rendering, AI) runs off the UI thread, with a progress panel
    (bars, about how long is left, Stop) at the bottom of the page while it runs."""
    title = "pixel-pipeline"

    def __init__(self, status=None):
        super().__init__()
        self.sync = False                 # tests: run jobs right here and let errors raise
        self._status = status or (lambda text: None)
        self.busy_widgets: list[QWidget] = []
        self._thread = self._worker = None
        self._on_done = None
        self._panel: ProgressPanel | None = None
        self._runner: JobRunner | None = None
        self.on_stop = None               # the page's own Stop hook (e.g. a cancel Event)

    def say(self, text: str):
        self._status(text)

    @property
    def progress_panel(self) -> ProgressPanel:
        """The page's progress panel (made and put at the bottom of the page on first use; a
        page may place its own first with set_progress_panel)."""
        if self._panel is None:
            self._panel = ProgressPanel()
            if not attach_panel(self, self._panel):
                self._panel.setParent(self)
        return self._panel

    def set_progress_panel(self, panel: ProgressPanel):
        self._panel = panel

    def running(self) -> bool:
        return self._runner is not None and self._runner.running()

    def run(self, job, on_done, what: str = "Working…", *, plan=None, estimate=None,
            title: str | None = None, stop_raises: bool = True):
        """job(progress) -> result off the UI thread, then on_done(result) back on it.
        progress is a progress.Tracker: call it with text (as before), (step, text),
        (done, total) or a fraction. plan: progress.Stage list; estimate: seconds.
        stop_raises False: Stop only sets the flag (the job stops itself, e.g. a batch after
        the current one)."""
        if self.running():
            self.say("Still working on the last one…")
            return
        self.say(what)
        self._runner = JobRunner(self, self.progress_panel, self.say, self._job_failed,
                                 self._job_stopped, busy=self._busy)
        self._on_done = on_done
        self._runner.start(job, on_done if self.sync else self._job_done, sync=self.sync,
                           plan=plan, estimate=estimate,
                           title=title or what.rstrip("…. ") or self.title,
                           on_stop=self._stop_hook, stop_raises=stop_raises)
        self._thread, self._worker = self._runner.thread, self._runner.worker

    def stop(self):
        """Stop the running job at its next step (what's finished is kept)."""
        if self._runner is not None:
            self._runner.stop()
        self._stop_hook()

    def _stop_hook(self):
        if self.on_stop is not None:
            self.on_stop()

    @Slot(object)
    def _job_done(self, result):
        self._on_done(result)

    @Slot(str)
    def _job_failed(self, message: str):
        self.say("Stopped: " + message.splitlines()[0])
        QMessageBox.warning(self, self.title, message)

    @Slot(str)
    def _job_stopped(self, message: str):
        first = message.splitlines()[0] if message else "Stopped"
        self.say(first if first.startswith("Stopped") else "Stopped: " + first)

    def _busy(self, busy: bool):
        for w in self.busy_widgets:
            w.setEnabled(not busy)
        if busy:
            QApplication.setOverrideCursor(Qt.BusyCursor)
        else:
            QApplication.restoreOverrideCursor()

    def warn(self, message: str):
        if self.sync:
            raise RuntimeError(message)
        QMessageBox.warning(self, self.title, message)


def spin(lo: int, hi: int, value: int, special: str | None = None) -> QSpinBox:
    box = QSpinBox()
    box.setRange(lo, hi)
    box.setValue(value)
    if special:
        box.setSpecialValueText(special)
    return box


class SpriteOptions(QWidget):
    """Type, retro system, size and colours -> Settings (the rest from the type's preset)."""

    def __init__(self, subject: str = "person", width: int = 48, height: int = 48,
                 colors: int = 24, subjects=SUBJECTS):
        super().__init__()
        self.subject = QComboBox()
        self.subject.addItems(list(subjects))
        self.subject.setCurrentText(subject)
        self.system = QComboBox()
        self.system.addItem("Off (free colours)", "off")
        for key, sysm in SYSTEMS.items():
            self.system.addItem(sysm.name, key)
        self.width_ = spin(4, 512, width)
        self.height_ = spin(4, 512, height)
        self.colors = spin(2, 64, colors)
        form = QFormLayout(self)
        form.setContentsMargins(0, 0, 0, 0)
        form.addRow("Type", self.subject)
        form.addRow("Retro system", self.system)
        form.addRow("Width", self.width_)
        form.addRow("Height", self.height_)
        form.addRow("Colours", self.colors)
        self.system.currentIndexChanged.connect(self._system_changed)

    def _system_changed(self, *_):
        free = (self.system.currentData() or "off") == "off"
        for w in (self.width_, self.height_, self.colors):
            w.setEnabled(free)
        tip = "" if free else "Set by the retro system (readable look)"
        for w in (self.width_, self.height_, self.colors):
            w.setToolTip(tip)

    def settings(self, **extra) -> Settings:
        system = self.system.currentData() or "off"
        if system != "off":
            return settings_for(self.subject.currentText(), system=system, **extra)
        return settings_for(self.subject.currentText(), width=self.width_.value(),
                            height=self.height_.value(), colors=self.colors.value(), **extra)
