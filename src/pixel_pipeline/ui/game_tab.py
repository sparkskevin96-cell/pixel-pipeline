"""The Game tab: pick a blue team and a red team from your Character Library, press Make game,
then Play. Writes the HD-2D tactics project (hd2d.py: real 3D, fake 3D and isometric 2D looks of
the harbour map) to ~/Pixel Pipeline/Games/<name> and opens it in Godot when Godot is installed.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from PySide6.QtCore import QSize, Qt, QUrl
from PySide6.QtGui import QDesktopServices, QIcon, QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QPushButton, QVBoxLayout,
)

from .common import ToolTab

MAX_TEAM = 6


def find_godot() -> str | None:
    """Godot 4 on this PC: $GODOT_BIN, on PATH, or installed by winget."""
    env = os.environ.get("GODOT_BIN")
    if env and Path(env).exists():
        return env
    for name in ("godot", "godot4", "Godot"):
        hit = shutil.which(name)
        if hit:
            return hit
    winget = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    if winget.exists():
        found = sorted(winget.glob("GodotEngine.GodotEngine*/Godot_v4*_win64.exe"))
        found = [p for p in found if "console" not in p.name]
        if found:
            return str(found[-1])
    return None


def games_root() -> Path:
    env = os.environ.get("PIXEL_GAMES")
    return Path(env) if env else Path.home() / "Pixel Pipeline" / "Games"


class GameTab(ToolTab):
    """Two teams from the library -> a Godot tactics game you can play."""
    title = "Game"

    def __init__(self, status=None):
        super().__init__(status)
        self.out: Path | None = None
        self.blue = self._team_list()
        self.red = self._team_list()
        teams = QHBoxLayout()
        for title, lst, colour in (("Blue team", self.blue, "#5b8cff"), ("Red team", self.red, "#ff6a5a")):
            box = QGroupBox(f"{title} (tick up to {MAX_TEAM})")
            box.setStyleSheet(f"QGroupBox {{ color: {colour}; font-weight: 600; }}")
            lay = QVBoxLayout(box)
            lay.addWidget(lst)
            teams.addWidget(box, 1)
        self.name = QLineEdit("Harbour Skirmish")
        self.map_pick = QComboBox()
        self.map_pick.setToolTip("Any map from your Map Library (2 World), or the built-in harbour")
        self.light = QComboBox()
        self.light.addItems(["Sunset", "Night", "Dawn", "Noon"])
        self.light.setToolTip("Time of day the game starts at: sun or moon, torches, glow "
                              "(press L in the game to change it)")
        self.iso = QCheckBox("Also the isometric 2D look")
        self.iso.setChecked(True)
        self.make_btn = QPushButton("Make game")
        self.make_btn.setStyleSheet("font-weight:600;")
        self.make_btn.clicked.connect(self.make)
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.play)
        self.folder_btn = QPushButton("Open the project folder")
        self.folder_btn.clicked.connect(self.show_folder)
        for b in (self.play_btn, self.folder_btn):
            b.setEnabled(False)
        self.busy_widgets = [self.make_btn, self.play_btn]
        self.note = QLabel("Pick a map (yours from 2 World, or the harbour), tick each side. "
                           "Real 3D (lit diorama, tilt-shift), fake 3D (pixel camera) and "
                           "isometric 2D, all playable: click a unit, a purple cell to move, a red "
                           "enemy to attack. Stats come from each character's class.")
        self.note.setWordWrap(True)
        self.godot_label = QLabel()
        self.godot_label.setWordWrap(True)
        row = QHBoxLayout()
        for w in (QLabel("Map"), self.map_pick, QLabel("Light"), self.light, QLabel("Game name"), self.name,
                  self.iso, self.make_btn,
                  self.play_btn, self.folder_btn):
            row.addWidget(w, 1 if w is self.name else 0)
        lay = QVBoxLayout(self)
        lay.addWidget(self.note)
        lay.addLayout(teams, 1)
        lay.addLayout(row)
        lay.addWidget(self.godot_label)
        self.refresh()

    def _team_list(self) -> QListWidget:
        lst = QListWidget()
        lst.setViewMode(QListWidget.IconMode)
        lst.setIconSize(QSize(64, 64))
        lst.setGridSize(QSize(92, 100))
        lst.setResizeMode(QListWidget.Adjust)
        lst.setMovement(QListWidget.Static)
        lst.itemChanged.connect(lambda _item: self._count())
        return lst

    def refresh(self):
        """Fill both teams from the library, keeping what was ticked (new characters unticked)."""
        from ..library import in_preview, list_characters
        ticked = {lst: {i.data(Qt.UserRole).folder for i in self._items(lst)
                        if i.checkState() == Qt.Checked} for lst in (self.blue, self.red)}
        every = list_characters()
        chars = [c for c in every if not in_preview(c)]   # drafts have no sprites yet
        self.drafts = len(every) - len(chars)
        first_time = not any(ticked.values())
        for k, lst in enumerate((self.blue, self.red)):
            lst.blockSignals(True)
            lst.clear()
            for j, ch in enumerate(sorted(chars, key=lambda c: c.created)):
                item = QListWidgetItem(ch.name)
                if ch.thumb.exists():
                    item.setIcon(QIcon(QPixmap(str(ch.thumb))))
                item.setData(Qt.UserRole, ch)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                half = (len(chars) + 1) // 2                  # at first: first half blue, rest red
                first = j < min(half, MAX_TEAM) if k == 0 else half <= j < half + MAX_TEAM
                on = ch.folder in ticked[lst] if not first_time else first
                item.setCheckState(Qt.Checked if on else Qt.Unchecked)
                item.setToolTip(f"{ch.stats_class}: HP %d ATK %d DEF %d MOV %d JMP %d RNG %d" % ch.stats())
                lst.addItem(item)
            lst.blockSignals(False)
        self._count()
        from ..maplib import list_maps
        current = self.map_pick.currentText()
        self.map_pick.blockSignals(True)
        self.map_pick.clear()
        self.map_pick.addItem("Harbour (built-in)", None)
        for asset in list_maps():
            self.map_pick.addItem(asset.name, asset)
        if self.map_pick.findText(current) >= 0:
            self.map_pick.setCurrentText(current)
        self.map_pick.blockSignals(False)
        godot = find_godot()
        self.godot_label.setText(f"Godot: {godot}" if godot else
                                 "Godot 4 isn't installed, so Play can't open the game. Get it with: "
                                 "winget install GodotEngine.GodotEngine (or set GODOT_BIN).")

    @staticmethod
    def _items(lst):
        return [lst.item(i) for i in range(lst.count())]

    def team(self, lst) -> list:
        return [i.data(Qt.UserRole) for i in self._items(lst) if i.checkState() == Qt.Checked]

    def _count(self):
        b, r = len(self.team(self.blue)), len(self.team(self.red))
        self.make_btn.setEnabled(1 <= b <= MAX_TEAM and 1 <= r <= MAX_TEAM)
        drafts = getattr(self, "drafts", 0)
        self.say(f"Blue {b}, red {r}." + ("" if self.make_btn.isEnabled() else
                                          f" Tick 1 to {MAX_TEAM} characters on each side.")
                 + (f" ({drafts} draft character(s) waiting for your OK on the Characters page "
                    "aren't listed: they have no sprites yet.)" if drafts else ""))

    def make(self):
        from ..diorama import harbour, place_units
        from ..hd2d import make_hd2d
        from ..library import slug
        blue, red = self.team(self.blue), self.team(self.red)
        out = games_root() / slug(self.name.text() or "game")
        iso = self.iso.isChecked()
        map_asset = self.map_pick.currentData()
        light = self.light.currentText().lower()

        def job(say):
            say("Writing the game…")
            sheets, stats = {}, {}
            for ch in blue + red:
                sheets[slug(ch.name)] = ch.sheet
                stats[slug(ch.name)] = ch.stats()
            blue_names, red_names = [slug(c.name) for c in blue], [slug(c.name) for c in red]
            if map_asset is not None:
                from ..maplib import load_diorama
                d = load_diorama(map_asset, blue=blue_names, red=red_names)
            else:
                d = place_units(harbour(), blue_names, red_names)
            for u in d.units:
                u.hp, u.attack, u.defence, u.move, u.jump, u.reach = stats[u.name]
            make_hd2d(out, sheets, d, name=self.name.text() or "Tactics", iso=iso, lighting=light)
            from ..assets import current_project
            from ..maplib import install_board_extras
            install_board_extras(out, current_project().style.tile)   # trees, houses, walls in 3D
            godot = find_godot()
            if godot and iso:
                say("Letting Godot import the pictures (once)…")
                subprocess.run([godot, "--headless", "--path", str(out), "--import"],
                               capture_output=True, timeout=600)
            return out

        def done(path):
            self.out = path
            for b in (self.play_btn, self.folder_btn):
                b.setEnabled(True)
            self.play_btn.setEnabled(find_godot() is not None)
            self.say(f"Game ready in {path}. Press Play (or open project.godot in Godot and press F5).")
        self.run(job, done, "Making the game…")

    def play(self):
        godot = find_godot()
        if not (godot and self.out):
            return
        subprocess.Popen([godot, "--path", str(self.out)], cwd=str(self.out))
        self.say("Starting the game in Godot… Esc goes back to its menu.")

    def show_folder(self):
        if self.out:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.out)))
