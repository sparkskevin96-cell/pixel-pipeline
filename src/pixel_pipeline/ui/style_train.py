"""Train style dialog: tick the project's approved assets, see the count and a time estimate,
Start / Stop (resumable), live GPU temperature + steps + loss, then a before/after test (same
prompt and seed without / with the new style LoRA). The GPU work is styletrain's."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QObject, QSize, Qt, QThread, Signal, Slot
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMessageBox, QProgressBar, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from .. import assets as assets_mod
from .. import styletrain as stt
from .common import PixelView, checkerboard, to_pixmap


def _thumb_icon(rgba: np.ndarray, px: int = 72) -> QIcon:
    h, w = rgba.shape[:2]
    k = max(1, px // max(h, w))
    big = np.repeat(np.repeat(rgba, k, 0), k, 1)
    return QIcon(to_pixmap(checkerboard(big, cell=max(4, k * 2))))


def _fmt_minutes(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    m = seconds / 60
    return f"{m:.0f} min" if m >= 1 else f"{seconds:.0f} s"


class _Job(QObject):
    progress = Signal(object)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        try:
            self.done.emit(self.fn())
        except Exception as e:  # noqa: BLE001 - shown to the user
            self.failed.emit(str(e) or type(e).__name__)


class StyleTrainDialog(QDialog):
    def __init__(self, project: assets_mod.Project | None = None, parent: QWidget | None = None,
                 *, backend=None, gpu=None, cfg: stt.TrainConfig | None = None):
        super().__init__(parent)
        self.project = project or assets_mod.current_project()
        self.backend = backend                     # None = ComfyUI (tests: FakeBackend)
        self.gpu = gpu or stt.read_gpu
        self.cfg = cfg or stt.TrainConfig()
        self.sync = False                          # tests: run jobs right here
        self.trainer: stt.StyleTrainer | None = None
        self.samples: list[stt.Sample] = []
        self.last: stt.Progress | None = None
        self.result_state: stt.TrainState | None = None
        self._thread = self._job = None
        self.setWindowTitle(f"Train style: {self.project.name}")
        self.resize(760, 680)

        lay = QVBoxLayout(self)
        intro = QLabel("Teach the AI this project's look from finished assets, so new characters, "
                       "items and props match. 20 or more pictures work best. The GPU rests "
                       "between chunks and waits when it's hot; Stop keeps the progress.")
        intro.setWordWrap(True)
        lay.addWidget(intro)

        form = QFormLayout()
        self.trigger = QLineEdit(self.cfg.trigger or stt.trigger_word(self.project.name))
        self.trigger.setToolTip("A made-up word that means 'this project's look' in prompts.")
        form.addRow("Style word", self.trigger)
        self.approved = QCheckBox(f"Only assets tagged '{stt.APPROVED_TAG}' (all if none are)")
        self.approved.setChecked(True)
        self.approved.toggled.connect(self.reload)
        form.addRow("", self.approved)
        self.res = QComboBox()
        self.res.addItems(["1024", "768"])
        self.res.setCurrentText(str(self.cfg.resolution))
        self.res.setToolTip("768 if 1024 runs out of memory or is very slow.")
        self.res.currentTextChanged.connect(self.update_estimate)
        self.steps = QSpinBox()
        self.steps.setRange(0, 5000)
        self.steps.setSingleStep(150)
        self.steps.setSpecialValueText("auto")
        self.steps.setValue(self.cfg.steps)
        self.steps.valueChanged.connect(self.update_estimate)
        row = QHBoxLayout()
        row.addWidget(QLabel("Picture size"))
        row.addWidget(self.res)
        row.addWidget(QLabel("Steps"))
        row.addWidget(self.steps)
        row.addStretch(1)
        form.addRow("", row)
        lay.addLayout(form)

        self.list = QListWidget()
        self.list.setViewMode(QListWidget.IconMode)
        self.list.setIconSize(QSize(72, 72))
        self.list.setGridSize(QSize(104, 112))
        self.list.setResizeMode(QListWidget.Adjust)
        self.list.setWordWrap(True)
        self.list.itemChanged.connect(self.update_estimate)
        lay.addWidget(self.list, 1)
        sel = QHBoxLayout()
        b_all, b_none = QPushButton("Tick all"), QPushButton("Tick none")
        b_all.clicked.connect(self.tick_all)
        b_none.clicked.connect(self.tick_none)
        sel.addWidget(b_all)
        sel.addWidget(b_none)
        self.info = QLabel()
        sel.addWidget(self.info, 1)
        lay.addLayout(sel)

        run = QHBoxLayout()
        self.start_btn, self.stop_btn = QPushButton("Start"), QPushButton("Stop")
        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)
        self.stop_btn.setEnabled(False)
        run.addWidget(self.start_btn)
        run.addWidget(self.stop_btn)
        self.bar = QProgressBar()
        run.addWidget(self.bar, 1)
        self.temp = QLabel("GPU –")
        run.addWidget(self.temp)
        lay.addLayout(run)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)

        test = QHBoxLayout()
        self.prompt = QLineEdit("a knight with a red cape")
        self.test_btn = QPushButton("Before / after")
        self.test_btn.setToolTip("Same prompt and seed, without and with the trained style (GPU).")
        self.test_btn.clicked.connect(self.before_after)
        test.addWidget(QLabel("Test prompt"))
        test.addWidget(self.prompt, 1)
        test.addWidget(self.test_btn)
        lay.addLayout(test)
        pics = QHBoxLayout()
        self.before_view, self.after_view = PixelView("Before", (220, 180)), PixelView("After", (220, 180))
        pics.addWidget(self.before_view)
        pics.addWidget(self.after_view)
        lay.addLayout(pics)

        self.reload()
        self._show_record()

    # -- assets
    def reload(self, *_):
        self.samples = stt.collect(self.project, tag=stt.APPROVED_TAG if self.approved.isChecked()
                                   else None, trigger=self.trigger.text().strip() or None)
        self.list.blockSignals(True)
        self.list.clear()
        for s in self.samples:
            item = QListWidgetItem(_thumb_icon(s.rgba), s.asset.name)
            item.setToolTip(f"{s.asset.kind}: {s.caption}")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            item.setData(Qt.UserRole, s.key)
            self.list.addItem(item)
        self.list.blockSignals(False)
        self.update_estimate()

    def _set_all(self, state):
        self.list.blockSignals(True)
        for i in range(self.list.count()):
            self.list.item(i).setCheckState(state)
        self.list.blockSignals(False)
        self.update_estimate()

    def tick_all(self):
        self._set_all(Qt.Checked)

    def tick_none(self):
        self._set_all(Qt.Unchecked)

    def chosen(self) -> list[stt.Sample]:
        keys = {self.list.item(i).data(Qt.UserRole) for i in range(self.list.count())
                if self.list.item(i).checkState() == Qt.Checked}
        trig = self.trigger.text().strip() or stt.trigger_word(self.project.name)
        out = []
        for s in self.samples:
            if s.key in keys:
                s.caption = stt.caption(s.asset, trig)
                out.append(s)
        return out

    def config(self) -> stt.TrainConfig:
        from dataclasses import replace
        return replace(self.cfg, resolution=int(self.res.currentText()), steps=self.steps.value(),
                       trigger=self.trigger.text().strip())

    def update_estimate(self, *_):
        n = len(self.chosen())
        cfg = self.config()
        plan = stt.StyleTrainer(self.project, cfg, backend=self.backend or _NoBackend()).plan(
            [None] * n)
        text = f"{n} pictures · {plan['steps']} steps · about {plan['minutes']:.0f} min"
        if plan["done"]:
            text += f" (resumes at step {plan['done']})"
        if plan["few"] and n:
            text += f" · fewer than {stt.MIN_IMAGES}: the look may not stick"
        if plan["over_budget"]:
            text += f" · stops at {cfg.max_minutes:.0f} min (Train more continues)"
        self.info.setText(text)
        self.start_btn.setEnabled(n > 0 and self._thread is None)
        self.plan = plan
        return plan

    # -- training
    def start(self):
        samples = self.chosen()
        if not samples:
            return
        self.trainer = stt.StyleTrainer(self.project, self.config(), backend=self.backend,
                                        gpu=self.gpu, progress=self._progress_any_thread)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status.setText("Starting…")
        self._launch(lambda: self.trainer.run(samples), self._trained)

    def stop(self):
        if self.trainer is not None:
            self.trainer.stop()
            self.status.setText("Stopping after ComfyUI lets go of the GPU…")

    def _launch(self, fn, on_done):
        if self.sync:
            try:
                on_done(fn())
            finally:
                self._idle()
            return
        thread, job = QThread(self), _Job(fn)
        job.moveToThread(thread)
        thread.started.connect(job.run)
        job.done.connect(on_done)
        job.failed.connect(self._failed)
        job.progress.connect(self.on_progress)
        for sig in (job.done, job.failed):
            sig.connect(thread.quit)
        thread.finished.connect(self._idle)
        thread.finished.connect(job.deleteLater)
        self._thread, self._job = thread, job
        thread.start()

    def _progress_any_thread(self, p: stt.Progress):
        if self._job is not None and not self.sync:
            self._job.progress.emit(p)       # queued to the UI thread
        else:
            self.on_progress(p)

    @Slot(object)
    def on_progress(self, p: stt.Progress):
        self.last = p
        if p.total:
            self.bar.setRange(0, p.total)
            self.bar.setValue(min(p.step, p.total))
        self.temp.setText(f"GPU {p.temp:.0f} °C" if p.temp is not None else "GPU –")
        bits = [p.message] if p.message else []
        if p.loss is not None:
            bits.append(f"loss {p.loss:.4f}")
        if p.eta is not None and p.phase not in ("done", "stopped", "error"):
            bits.append(f"about {_fmt_minutes(p.eta)} left")
        self.status.setText(" · ".join(bits))

    @Slot(object)
    def _trained(self, st: stt.TrainState):
        self.result_state = st
        if st.status == "done":
            self.status.setText(f"Done: {st.result} ({st.done_steps} steps). Its style word is "
                                f"'{st.trigger}'. Try Before / after.")
        else:
            self.status.setText(f"Stopped at step {st.done_steps} of {st.total_steps}: "
                                f"{st.notes[-1] if st.notes else ''}. Start resumes.")
        self.update_estimate()

    @Slot(str)
    def _failed(self, message: str):
        self.status.setText("Stopped: " + message.splitlines()[0])
        QMessageBox.warning(self, "Train style", message)

    @Slot()
    def _idle(self):
        self._thread = self._job = None
        self.stop_btn.setEnabled(False)
        self.update_estimate()

    def _show_record(self):
        rec = stt.project_style_lora(self.project)
        self.test_btn.setEnabled(rec is not None)
        if rec:
            self.status.setText(f"This project's style: {rec['lora']} (word '{rec['trigger']}'). "
                                "Start again to train more.")

    # -- before / after
    def before_after(self):
        prompt = self.prompt.text().strip()
        if not prompt:
            return
        self.test_btn.setEnabled(False)
        self.status.setText("Drawing the same picture without and with the style…")
        self._launch(lambda: stt.before_after(self.project, prompt), self._show_pair)

    @Slot(object)
    def _show_pair(self, pair):
        before, after = pair
        self.before_view.show_rgba(np.asarray(before.convert("RGBA")))
        self.after_view.show_rgba(np.asarray(after.convert("RGBA")))
        self.test_btn.setEnabled(True)
        self.status.setText("Left: without the style. Right: with it.")

    def closeEvent(self, event):
        if self._thread is not None and self._thread.isRunning():
            if self.trainer is not None:
                self.trainer.stop()      # the run keeps its last checkpoint; Start resumes
            self.status.setText("Stopping… close this window again once it says Stopped.")
            event.ignore()
            return
        super().closeEvent(event)


class _NoBackend:
    """plan() never touches the backend; this keeps it from creating a ComfyBackend."""

    def lora_dir(self):
        raise RuntimeError("not for training")
