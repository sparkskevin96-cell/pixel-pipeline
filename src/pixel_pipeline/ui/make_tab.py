"""Make: one "Describe it" box for everything. Words -> the router (makeany.py) picks the
pipeline and fills in its settings -> the plan is shown (editable, with "it could also be…")
-> Make it runs it off the UI thread -> variations to pick where there are some -> saved in the
project, then its own page opens for tweaking. A list (one per line) goes through the paced
creation queue (jobqueue) with a summary at the end.

Drawn characters stop after their front drawing (plan "review"): the review panel
(character_review.py) shows it here; Looks good draws the rest and opens it. A batch with "Stop
for my OK on each character" collects those drafts in a list to review one by one."""
from __future__ import annotations

import threading

import numpy as np
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QPlainTextEdit, QPushButton, QSpinBox, QStackedWidget,
    QVBoxLayout, QWidget,
)

from .. import makeany as M
from .common import PixelView, ToolTab, checkerboard, spin, to_pixmap

EXAMPLES = ("a wolf", "rusty iron sword", "tavern with a thatched roof", "cobblestone road tiles",
            "ice nova spell", "goblin portrait, angry", "fire card", "wooden UI panel",
            "a forest map with a river")


def _rgba(v) -> np.ndarray | None:
    if isinstance(v, np.ndarray):
        return v
    try:
        from PIL import Image
        return np.asarray(Image.open(v).convert("RGBA"))
    except Exception:  # noqa: BLE001 - a missing picture just shows nothing
        return None


def _icon(v, px: int = 72) -> QIcon:
    rgba = _rgba(v)
    if rgba is None:
        return QIcon()
    h, w = rgba.shape[:2]
    k = max(1, px // max(h, w))
    big = np.repeat(np.repeat(rgba, k, 0), k, 1) if k > 1 else rgba
    pix = to_pixmap(checkerboard(big, cell=max(4, k * 2)))
    return QIcon(pix.scaled(px, px, Qt.KeepAspectRatio, Qt.FastTransformation))


class MakeTab(ToolTab):
    """Describe it -> plan -> make -> pick -> open its page."""
    title = "Make"
    changed = Signal()                  # something was saved in the project

    def __init__(self, status=None, open_result=None, client=None):
        super().__init__(status)
        self.client = client             # tests: a fake ComfyUI
        self.makers: dict | None = None  # tests: fake pipelines {kind: fn(plan, project, say, client)}
        self.open_result = open_result or (lambda made: None)
        self.guesses: list[M.Guess] = []
        self.plan: M.Plan | None = None
        self.made: M.Made | None = None
        self.batch_jobs: list = []
        self._stop = threading.Event()
        self._filling = False
        self.inputs: dict[str, QWidget] = {}
        self._build()

    # ------------------------------------------------------------------ layout
    def _build(self):
        self.describe = QLineEdit()
        self.describe.setPlaceholderText("Describe it… e.g. " + ", ".join(EXAMPLES[:5]))
        self.describe.setStyleSheet("font-size:16px;padding:8px;")
        self.describe.returnPressed.connect(self.plan_text)
        self.plan_btn = QPushButton("Plan it")
        self.plan_btn.clicked.connect(lambda: self.plan_text())
        top = QHBoxLayout()
        top.addWidget(self.describe, 1)
        top.addWidget(self.plan_btn)
        examples = QHBoxLayout()
        examples.addWidget(QLabel("Try:"))
        for e in EXAMPLES[:6]:
            b = QPushButton(e)
            b.setFlat(True)
            b.setStyleSheet("color:#7fb2ff;text-align:left;padding:0 4px;")
            b.clicked.connect(lambda _=False, t=e: self.plan_text(t))
            examples.addWidget(b)
        examples.addStretch(1)

        self.summary = QLabel("Type what you want and press Enter: I'll show the plan before "
                              "anything runs.")
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("font-size:14px;font-weight:600;padding:6px 2px;")
        self.why = QLabel("")
        self.why.setStyleSheet("color:#9aa3b5;")
        self.alt_row = QHBoxLayout()
        self.alt_label = QLabel("It could also be:")
        self.alt_row.addWidget(self.alt_label)
        self.alt_pool: list[QPushButton] = []          # reused: no widgets made and dropped
        for i in range(4):
            b = QPushButton()
            b.clicked.connect(lambda _=False, i=i: self._alt_clicked(i))
            b.setVisible(False)
            self.alt_row.addWidget(b)
            self.alt_pool.append(b)
        self.alt_buttons: list[QPushButton] = []
        self._alt_kinds: list[str] = []
        self.alt_row.addStretch(1)
        self.alt_label.setVisible(False)

        self.kind = QComboBox()
        for k in M.KINDS:
            self.kind.addItem(M.LABELS[k], k)
        self.kind.currentIndexChanged.connect(self._kind_changed)
        self.name = QLineEdit()
        self.prompt = QLineEdit()
        self.prompt.setToolTip("The words your GPU gets (the house style is added to them)")
        for w in (self.name, self.prompt):
            w.textEdited.connect(self._fields_changed)
        self.details = QGroupBox("The plan (change anything)")
        self.form = QFormLayout(self.details)
        self.form.addRow("Make", self.kind)
        self.form.addRow("Name", self.name)
        self.form.addRow("Words", self.prompt)
        self.details.setEnabled(False)

        self.make_btn = QPushButton("Make it")
        self.make_btn.setStyleSheet("font-weight:600;font-size:14px;padding:6px 18px;")
        self.make_btn.setEnabled(False)
        self.make_btn.clicked.connect(self.make)

        self.variations = QListWidget()
        self.variations.setViewMode(QListWidget.IconMode)
        self.variations.setIconSize(QSize(72, 72))
        self.variations.setFixedHeight(110)
        self.variations.setMovement(QListWidget.Static)
        self.variations.currentRowChanged.connect(self._variation_shown)
        self.view = PixelView("What you make shows here", minimum=(260, 200))
        self.keep_btn = QPushButton("Use this one and open it")
        self.keep_btn.setEnabled(False)
        self.keep_btn.clicked.connect(lambda: self.keep_chosen())
        self.open_btn = QPushButton("Open it to tweak")
        self.open_btn.setEnabled(False)
        self.open_btn.clicked.connect(self.open_made)
        self.picks = QGroupBox("Pick one")
        pv = QVBoxLayout(self.picks)
        pv.addWidget(self.variations)
        row = QHBoxLayout()
        row.addWidget(self.keep_btn)
        row.addWidget(self.open_btn)
        row.addStretch(1)
        pv.addLayout(row)

        left = QVBoxLayout()
        left.addLayout(top)
        left.addLayout(examples)
        left.addWidget(self.summary)
        left.addWidget(self.why)
        left.addLayout(self.alt_row)
        mid = QHBoxLayout()
        col = QVBoxLayout()
        col.addWidget(self.details)
        mk = QHBoxLayout()
        mk.addWidget(self.make_btn)
        mk.addStretch(1)
        col.addLayout(mk)
        col.addStretch(1)
        mid.addLayout(col, 2)
        from .character_review import CharacterReview
        self.review = CharacterReview(self._review_run, self.say)
        self.review.approved.connect(self._review_approved)
        self.review.changed.connect(lambda ch: self.changed.emit())
        self.review.closed.connect(self._review_closed)
        self.view_stack = QStackedWidget()
        self.view_stack.addWidget(self.view)
        self.view_stack.addWidget(self.review)
        mid.addWidget(self.view_stack, 3)
        left.addLayout(mid, 1)
        left.addWidget(self.picks)

        # many at once
        self.batch = QPlainTextEdit()
        self.batch.setPlaceholderText("One per line:\na wolf\nrusty iron sword\nice nova spell\n…")
        self.batch.textChanged.connect(self._batch_changed)
        self.batch_plan = QLabel("")
        self.batch_plan.setWordWrap(True)
        self.rest = spin(0, 600, 20)
        self.rest.setSuffix(" s")
        self.rest.setToolTip("Rest between generations so the laptop never runs flat out for hours")
        self.batch_review = QCheckBox("Stop for my OK on each character")
        self.batch_review.setChecked(True)
        self.batch_review.setToolTip("Drawn characters: only the front is drawn, then they wait in "
                                     "the list below (and the Character Library) for your OK. "
                                     "Items and props are made as usual. Off: every character all "
                                     "the way (16 directions) with nobody looking.")
        self.waiting = QListWidget()
        self.waiting.setViewMode(QListWidget.IconMode)
        self.waiting.setIconSize(QSize(56, 56))
        self.waiting.setGridSize(QSize(84, 86))
        self.waiting.setMovement(QListWidget.Static)
        self.waiting.setFixedHeight(100)
        self.waiting.itemClicked.connect(lambda item: self.review_character(item.data(Qt.UserRole)))
        self.waiting_box = QGroupBox("Waiting for your OK (click one)")
        QVBoxLayout(self.waiting_box).addWidget(self.waiting)
        self.waiting_box.setVisible(False)
        self.queue_btn = QPushButton("Queue them all")
        self.queue_btn.clicked.connect(self.queue_batch)
        self.stop_btn = QPushButton("Stop after this one")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_batch)
        self.report = QPlainTextEdit()
        self.report.setReadOnly(True)
        self.report.setPlaceholderText("A summary shows here when the list is done.")
        many = QGroupBox("Many at once")
        mv = QVBoxLayout(many)
        mv.addWidget(self.batch, 2)
        mv.addWidget(self.batch_plan)
        f = QFormLayout()
        f.addRow("Rest between", self.rest)
        f.addRow(self.batch_review)
        mv.addLayout(f)
        b = QHBoxLayout()
        b.addWidget(self.queue_btn)
        b.addWidget(self.stop_btn)
        mv.addLayout(b)
        mv.addWidget(self.report, 1)
        mv.addWidget(self.waiting_box)
        many.setMaximumWidth(380)

        outer = QHBoxLayout(self)
        lw = QWidget()
        lw.setLayout(left)
        outer.addWidget(lw, 1)
        outer.addWidget(many)
        self.busy_widgets = [self.make_btn, self.plan_btn, self.queue_btn, self.keep_btn]
        self._show_picks([])

    # ------------------------------------------------------------------ planning
    def plan_text(self, text: str | None = None):
        """Route the words; show the best plan and the alternatives."""
        if text is not None:
            self.describe.setText(text)
        text = self.describe.text().strip()
        if not text:
            self.say("Describe what to make first, e.g. “a wolf”.")
            return None
        self.guesses = M.classify(text)
        self.made = None                       # a new request: the last result's picks go
        self._show_picks([])
        self.view_stack.setCurrentWidget(self.view)
        self.view.show_rgba(None)
        self.view.setText("What you make shows here")
        self.set_plan(self.guesses[0].plan, self.guesses[0].plan.why)
        self._alternatives()
        return self.plan

    def _alternatives(self):
        current = self.plan.kind if self.plan else None
        others = [g for g in self.guesses if g.kind != current][:len(self.alt_pool)]
        self._alt_kinds = [g.kind for g in others]
        for i, b in enumerate(self.alt_pool):
            if i < len(others):
                b.setText(f"{others[i].plan.label}: {others[i].plan.name}")
                b.setToolTip(others[i].plan.summary())
            b.setVisible(i < len(others))
        self.alt_buttons = self.alt_pool[:len(others)]
        self.alt_label.setVisible(bool(others))

    def _alt_clicked(self, i: int):
        if i < len(self._alt_kinds):
            self.use_kind(self._alt_kinds[i])

    def alternatives(self) -> list[str]:
        """The kinds offered under "It could also be…" (best first)."""
        return [g.kind for g in self.guesses if self.plan is None or g.kind != self.plan.kind]

    def use_kind(self, kind: str):
        """Make it as another kind (an alternative, or the Make box)."""
        g = next((g for g in self.guesses if g.kind == kind), None)
        plan = g.plan if g else M.plan_for(kind, self.describe.text().strip() or kind)
        self.set_plan(M.Plan.from_dict(plan.to_dict()), plan.why or "you picked it")
        self._alternatives()

    def set_plan(self, plan: M.Plan, why: str = ""):
        self.plan = plan
        self._filling = True
        try:
            self.kind.setCurrentIndex(M.KINDS.index(plan.kind))
            self.name.setText(plan.name)
            self.prompt.setText(plan.prompt)
            self._build_fields(plan)
        finally:
            self._filling = False
        self.details.setEnabled(True)
        self.make_btn.setEnabled(True)
        self.why.setText(f"Why: {why}" if why else "")
        self._show_summary()

    def _build_fields(self, plan: M.Plan):
        while self.form.rowCount() > 3:
            self.form.removeRow(3)
        self.inputs = {}
        self.prompt.setEnabled(plan.kind not in ("ui", "map", "effect"))
        for key, label, spec in M.fields_for(plan.kind):
            value = plan.params.get(key)
            if spec is bool:
                w = QCheckBox()
                w.setChecked(bool(value))
                w.toggled.connect(self._fields_changed)
            elif isinstance(spec, tuple):
                w = QComboBox()
                for c in spec:
                    w.addItem(str(c) if c != "" else "(default)", c)
                i = w.findData(value)
                if i < 0 and value not in (None, ""):
                    w.addItem(str(value), value)
                    i = w.count() - 1
                w.setCurrentIndex(max(0, i))
                w.currentIndexChanged.connect(self._fields_changed)
            elif isinstance(spec, str) and spec.startswith(("int:", "float:")):
                typ, lo, hi = spec.split(":")
                w = QSpinBox() if typ == "int" else QDoubleSpinBox()
                if typ == "float":
                    w.setDecimals(1)
                    w.setSingleStep(0.1)
                    w.setRange(float(lo), float(hi))
                    w.setValue(float(value or 0))
                else:
                    w.setRange(int(lo), int(hi))
                    w.setValue(int(value or 0))
                w.valueChanged.connect(self._fields_changed)
            else:
                w = QLineEdit(str(value or ""))
                w.textEdited.connect(self._fields_changed)
            self.inputs[key] = w
            self.form.addRow(label, w)

    def current_plan(self) -> M.Plan | None:
        """The plan as the fields say now."""
        if self.plan is None:
            return None
        p = dict(self.plan.params)
        for key, w in self.inputs.items():
            if isinstance(w, QCheckBox):
                p[key] = w.isChecked()
            elif isinstance(w, QComboBox):
                p[key] = w.currentData()
            elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
                p[key] = w.value()
            else:
                p[key] = w.text().strip()
        if self.plan.kind == "prop":
            p["building"] = float(p.get("height_people", 1)) >= 3.0
        return M.Plan(self.plan.kind, self.name.text().strip() or self.plan.name,
                      self.prompt.text().strip() or self.plan.prompt, p, self.plan.text, self.plan.why)

    def _fields_changed(self, *_):
        if not self._filling:
            self.plan = self.current_plan()
            self._show_summary()

    def _kind_changed(self, *_):
        if not self._filling:
            self.use_kind(self.kind.currentData())

    def _show_summary(self):
        if self.plan is not None:
            self.summary.setText(self.plan.summary())

    # ------------------------------------------------------------------ making
    def project(self):
        from ..assets import current_project
        return current_project()

    def make(self):
        plan = self.current_plan() or self.plan_text()
        if plan is None:
            return
        self.plan = plan
        project, client, makers = self.project(), self.client, self.makers
        self.made = None
        self._show_picks([])
        stages, estimate = M.progress_plan(plan)
        self.run(lambda say: M.run_plan(plan, project, say, client=client, makers=makers),
                 self._made, f"Making {plan.label.lower()} “{plan.name}”…", plan=stages,
                 estimate=estimate, title=f"{plan.label} “{plan.name}”")

    def _made(self, made: M.Made):
        self.made = made
        if made.saved:
            self.changed.emit()
        if made.notice:                     # e.g. the GPU failed mid-drawing: never silent
            from PySide6.QtWidgets import QMessageBox
            box = QMessageBox(QMessageBox.Warning, "Not quite as planned", made.notice,
                              QMessageBox.Ok, self)
            box.setModal(False)
            box.show()
            self._notice_box = box
        took = f" in {made.seconds / 60:.1f} min" if made.seconds >= 60 else ""
        if made.waiting:                    # a drawn character's front, for the OK
            self._show_picks([])
            self.review_character(made.asset)
            self.say(f"{made.name}: the front is drawn{took}. Looks good? Then the other "
                     "directions are drawn.")
            return
        if len(made.variations) > 1:
            self._show_picks(made.variations, made.chosen)
            self.say(f"{made.plan.name}: made{took}. Pick the one you like, then “Use this one and "
                     "open it”.")
            return
        self._show_picks([])
        if not made.saved:
            M.keep(made, 1)
            self.changed.emit()
        self.say(f"{made.name}: saved{took} -> {made.folder}")
        self._preview_made()
        self.open_made()

    def _show_picks(self, variations, chosen: int = 1):
        self.variations.blockSignals(True)
        self.variations.clear()
        for i, v in enumerate(variations, 1):
            self.variations.addItem(QListWidgetItem(_icon(v), str(i)))
        self.variations.blockSignals(False)
        self.picks.setVisible(bool(variations))
        self.keep_btn.setEnabled(bool(variations))
        self.open_btn.setEnabled(self.made is not None and self.made.saved)
        if variations:
            self.variations.setCurrentRow(max(0, chosen - 1))
            self._variation_shown(max(0, chosen - 1))

    def _variation_shown(self, row: int):
        if self.made is not None and 0 <= row < len(self.made.variations):
            self.view.show_rgba(_rgba(self.made.variations[row]))

    def _preview_made(self):
        if self.made is None or self.made.folder is None:
            return
        for name in ("thumb.png", "preview.png", "icon.png", "sheet.png"):
            p = self.made.folder / name
            if p.exists():
                self.view.show_rgba(_rgba(p))
                return

    def keep_chosen(self, index: int | None = None):
        """Use variation `index` (1-based; default the one picked in the list), save, open."""
        if self.made is None:
            return None
        index = index or (self.variations.currentRow() + 1) or 1
        M.keep(self.made, index)
        self.changed.emit()
        self.say(f"{self.made.name}: number {index} -> {self.made.folder}")
        self.open_made()
        return self.made

    def open_made(self):
        if self.made is not None and self.made.saved:
            self.open_btn.setEnabled(True)
            self.open_result(self.made)

    # ------------------------------------------------------------------ front first, then OK
    def _review_run(self, job, on_done, **progress):
        self.review.sync = self.sync
        self.run(job, on_done, "Working on the drawing…", **progress)

    def review_character(self, ch):
        """Show a draft character's front drawing for the OK (here, beside the plan)."""
        self.review.sync = self.sync
        self.review.set_character(ch)
        self.view_stack.setCurrentWidget(self.review)

    def _review_approved(self, ch):
        self._drop_waiting(ch)
        self.changed.emit()
        self.view_stack.setCurrentWidget(self.view)
        self.say(f"{ch.name}: every direction drawn -> {ch.folder}")
        if self.made is not None and getattr(self.made.asset, "folder", None) == ch.folder:
            self.made.asset = ch
        else:
            self.made = M.Made(M.Plan("character", ch.name, ch.prompt, {"body": ch.kind}), ch)
        self.open_made()

    def _review_closed(self, ch):
        if ch is None:                       # thrown away
            if self.made is not None and self.made.plan.kind == "character":
                self.made = None
            self._show_picks([])
            self._refresh_waiting()
        self.changed.emit()
        self.view_stack.setCurrentWidget(self.view)
        self.view.show_rgba(None)
        self.view.setText("What you make shows here")

    def _drop_waiting(self, ch):
        for i in range(self.waiting.count() - 1, -1, -1):
            if self.waiting.item(i).data(Qt.UserRole).folder == ch.folder:
                self.waiting.takeItem(i)
        self.waiting_box.setVisible(self.waiting.count() > 0)

    def _refresh_waiting(self, chars=None):
        """The batch's characters still waiting for an OK."""
        from ..library import in_preview
        if chars is None:
            chars = [self.waiting.item(i).data(Qt.UserRole) for i in range(self.waiting.count())]
        self.waiting.clear()
        seen = set()
        for ch in chars:
            if (ch.folder in seen or not (ch.folder / "character.json").exists()
                    or not in_preview(ch)):
                continue
            seen.add(ch.folder)
            item = QListWidgetItem(_icon(ch.thumb, 56), ch.name)
            item.setData(Qt.UserRole, ch)
            item.setToolTip(f"{ch.name}: the front is drawn, waiting for your OK")
            self.waiting.addItem(item)
        self.waiting_box.setVisible(self.waiting.count() > 0)

    def waiting_characters(self) -> list:
        return [self.waiting.item(i).data(Qt.UserRole) for i in range(self.waiting.count())]

    # ------------------------------------------------------------------ many at once
    def batch_plans(self) -> list[M.Plan]:
        plans = [M.route(t) for t in M.parse_batch(self.batch.toPlainText())]
        for p in plans:                      # the batch's choice for every drawn character
            if p.kind == "character":
                p.params["review"] = self.batch_review.isChecked()
        return plans

    def _batch_changed(self):
        plans = self.batch_plans()
        if not plans:
            self.batch_plan.setText("")
            return
        gpu = sum(p.gpu_minutes() for p in plans)
        rest = self.rest.value() * max(0, len(plans) - 1) / 60
        kinds: dict[str, int] = {}
        for p in plans:
            kinds[M.LABELS[p.kind]] = kinds.get(M.LABELS[p.kind], 0) + 1
        what = ", ".join(f"{n} {k.lower()}" for k, n in kinds.items())
        self.batch_plan.setText(f"{len(plans)} to make ({what}): about {round(gpu)} min of GPU "
                                f"+ {round(rest)} min resting.")

    def queue_batch(self):
        """Every line -> its plan -> the project's creation queue, worked through with a rest
        between each; the summary shows when it's done (or stopped)."""
        from .. import jobqueue
        plans = self.batch_plans()
        if not plans:
            self.say("Write one thing per line first.")
            return
        project = self.project()
        q = jobqueue.Queue(project)
        jobs = M.queue_plans(project, plans, queue=q)
        self.batch_jobs = jobs
        ids = {j.id for j in jobs}
        makers, pause, stop = self.makers, float(self.rest.value()), self._stop
        stop.clear()

        from .. import progress
        total = sum(max(1, int(getattr(j, "variations", 1) or 1)) for j in jobs)
        count = [0]

        def worker(name, prompt, options, proj):
            # job k of N on the batch bar; this job's own stages on the second bar
            batch = progress.current()
            if batch is None:
                return M.queue_worker(name, prompt, options, proj, makers=makers)
            p = M.Plan.from_dict(options["plan"]) if "plan" in options else M.route(prompt or name)
            stages, estimate = M.progress_plan(p)
            child = batch.child(min(count[0], total - 1), total, stages, estimate=estimate,
                                title=name or p.name)
            count[0] += 1
            with progress.using(child):
                made = M.queue_worker(name, prompt, options, proj, makers=makers)
            child.finish()
            return made

        def job(say):
            return jobqueue.run_queue(project, {"make": worker}, pause, stop, queue=q, job_ids=ids,
                                      on_update=lambda j: say(j.summary()))
        self.on_stop = stop.set            # the panel's Stop: after the current one, like the button
        self.stop_btn.setEnabled(True)
        self.report.setPlainText(f"Queued {len(jobs)}; working through them…")
        gpu = sum(p.gpu_minutes() + M.cpu_minutes(p) for p in plans) * 60 + pause * max(0, len(plans) - 1)
        self.run(job, self._batch_done, f"Making {len(jobs)} things, one at a time…",
                 plan=[progress.Stage("batch", f"{len(jobs)} things", max(0.1, gpu / 60))],
                 title=f"Batch of {len(jobs)}", stop_raises=False)

    def stop_batch(self):
        self._stop.set()
        self.say("Stopping after the current one…")

    def _batch_done(self, jobs):
        from ..library import from_folder
        self.on_stop = None
        self.stop_btn.setEnabled(False)
        text = M.batch_summary(self.batch_jobs or jobs)
        self.report.setPlainText(text)
        waiting = []
        for j in self.batch_jobs or jobs:
            for r in j.results:
                if r.get("waiting") and r.get("folder"):
                    ch = from_folder(r["folder"])
                    if ch is not None:
                        waiting.append(ch)
        self._refresh_waiting(self.waiting_characters() + waiting)
        self.changed.emit()
        self.say(text.splitlines()[0])
