"""The desktop app's install and update windows; updater.py does the work."""
from __future__ import annotations

import threading

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication, QDialog, QDialogButtonBox, QLabel, QLineEdit, QMessageBox, QProgressDialog,
    QVBoxLayout,
)

from . import updater

TITLE = "Pixel Pipeline"


def offer_install(parent=None) -> bool:
    """A downloaded copy offers to install itself, or opens the installed copy when that one is
    as new. True: another copy was started and this one should close."""
    if not updater.can_update():
        return False
    here, build = updater.app_dir(), updater.current_build()
    plan = updater.install_plan(here, build)
    if plan == "run":
        return False
    if plan == "open":
        try:
            updater.launch(updater.install_dir() / updater.EXE_NAME)
            return True
        except OSError:
            return False
    box = QMessageBox(QMessageBox.Question, TITLE,
                      "Install Pixel Pipeline on this PC?" if plan == "install" else
                      "Update the installed Pixel Pipeline to this download?", parent=parent)
    box.setInformativeText("It goes in your user folder (no admin needed), gets a desktop icon and "
                           "a Start menu entry, and updates itself from then on. After that you "
                           "can delete the folder you downloaded.")
    yes = box.addButton("Install", QMessageBox.AcceptRole)
    box.addButton("Just run this copy", QMessageBox.RejectRole)
    box.exec()
    if box.clickedButton() is not yes:
        return False
    QApplication.setOverrideCursor(Qt.BusyCursor)
    try:
        exe = updater.install(here, shortcuts=False)
        try:
            updater.make_shortcuts(exe)
        except updater.UpdateError as e:   # installed anyway; the Start menu search finds it
            QMessageBox.warning(parent, TITLE, f"Installed, but {e}")
    except updater.UpdateError as e:
        QMessageBox.warning(parent, TITLE, str(e))
        return False
    finally:
        QApplication.restoreOverrideCursor()
    updater.launch(exe)
    return True


class TokenDialog(QDialog):
    """A read-only GitHub token, so the app can see new builds while the repo is private."""

    def __init__(self, parent=None, why: str = ""):
        super().__init__(parent)
        self.setWindowTitle("GitHub token for updates")
        text = QLabel(
            (f"<p>{why}</p>" if why else "") +
            "<p>The app's code is in a private GitHub repo, so to see new builds it needs a "
            "read-only token from your GitHub account:</p><ol>"
            f'<li>Open <a href="{updater.TOKEN_URL}">GitHub → new token</a> (sign in if asked).</li>'
            "<li>Token name: <b>Pixel Pipeline updates</b>. Expiration: the longest it offers.</li>"
            "<li>Repository access: <b>Only select repositories</b> → <b>pixel-pipeline</b>.</li>"
            "<li>Permissions → Repository permissions → <b>Contents</b>: <b>Read-only</b>.</li>"
            "<li><b>Generate token</b>, copy it and paste it below.</li></ol>"
            "<p>It's saved in your user folder (.pixel-pipeline/keys.json) and only sent to GitHub.</p>")
        text.setTextFormat(Qt.RichText)
        text.setOpenExternalLinks(True)
        text.setWordWrap(True)
        self.edit = QLineEdit(updater.github_token())
        self.edit.setEchoMode(QLineEdit.Password)
        self.edit.setPlaceholderText("github_pat_…")
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        for w in (text, self.edit, buttons):
            lay.addWidget(w)
        self.resize(520, 0)

    def save(self):
        from ..online import load_keys, save_keys
        keys = load_keys(env=False)
        keys.pop("github", None)
        if self.edit.text().strip():
            keys["github"] = self.edit.text().strip()
        save_keys(keys)
        self.accept()


class _Cancelled(Exception):
    pass


class Updates(QObject):
    """Help menu entries, the check a few seconds after the app opens, and the update itself."""
    _checked = Signal(object)          # Release, or (message, needs_token)
    _progress = Signal(int, int)
    _fetched = Signal(object)          # unpacked folder, error message, or None (cancelled)

    def __init__(self, win):
        super().__init__(win)
        self.win = win
        self.loud = self.busy = self.cancelled = False
        self.dialog: QProgressDialog | None = None
        self._checked.connect(self._on_checked)
        self._progress.connect(self._on_progress)
        self._fetched.connect(self._on_fetched)

    def add_menu(self, menubar):
        menu = menubar.addMenu("&Help")
        for text, slot in (("Check for &updates…", lambda: self.check(loud=True)),
                           ("Updates &token (GitHub)…", lambda: self.ask_token()),
                           ("&About Pixel Pipeline", self.about)):
            action = QAction(text, self.win)
            action.triggered.connect(slot)
            menu.addAction(action)

    def say(self, text: str, ms: int = 15000):
        self.win.statusBar().showMessage(text, ms)

    def started(self, flags=()):
        """Right after the window opens: report a finished update, then check for a newer one."""
        build = updater.current_build()
        if "--updated" in flags and build:
            self.say(f"Updated to {build.label}.")
            threading.Thread(target=updater.cleanup, daemon=True).start()
        elif "--update-failed" in flags:
            from .crash import LOG_DIR
            QMessageBox.warning(self.win, TITLE, "The update didn't finish, so this is still the "
                                f"old version. Details: {LOG_DIR / 'update.log'}")
        if updater.can_update():
            QTimer.singleShot(3000, lambda: self.check(loud=False))

    def about(self):
        build = updater.current_build()
        QMessageBox.information(self.win, TITLE, f"Pixel Pipeline, {build.label}." if build else
                                "Pixel Pipeline, running from source (no version number).")

    def ask_token(self, why: str = "") -> bool:
        if TokenDialog(self.win, why).exec() != QDialog.Accepted:
            return False
        self.check(loud=True)
        return True

    # ---- checking -----------------------------------------------------------------------
    def check(self, loud: bool = True):
        if not updater.can_update():
            if loud:
                QMessageBox.information(self.win, TITLE, "Updates are for the installed Windows "
                                        "app. This copy runs from source: git pull updates it.")
            return
        if self.busy:
            return
        self.busy, self.loud = True, loud
        token = updater.github_token()

        def work():
            try:
                self._checked.emit(updater.latest(token))
            except updater.NeedsToken as e:
                self._checked.emit((str(e), True))
            except Exception as e:
                self._checked.emit((str(e) or type(e).__name__, False))
        threading.Thread(target=work, daemon=True).start()

    def _on_checked(self, result):
        self.busy = False
        if isinstance(result, tuple):
            message, needs_token = result
            if self.loud and needs_token:
                self.ask_token(message)
            elif self.loud:
                QMessageBox.warning(self.win, TITLE, message)
            elif needs_token:
                self.say("Updates: " + message, 30000)
            return
        current = updater.current_build()
        if not updater.is_newer(result, current):
            if self.loud:
                QMessageBox.information(self.win, TITLE, f"You have the newest: {current.label}.")
            return
        box = QMessageBox(QMessageBox.Question, TITLE, "A new Pixel Pipeline is ready.", parent=self.win)
        box.setInformativeText(f"New: {result.build.label}\nYours: {current.label}\n\n"
                               "Update now? The app closes and opens again by itself in about a "
                               "minute. Save your work first.")
        yes = box.addButton("Update now", QMessageBox.AcceptRole)
        box.addButton("Later", QMessageBox.RejectRole)
        box.exec()
        if box.clickedButton() is yes:
            self.fetch(result)

    # ---- updating -----------------------------------------------------------------------
    def fetch(self, release):
        self.cancelled = False
        self.dialog = QProgressDialog("Downloading the update…", "Cancel", 0, 100, self.win)
        self.dialog.setWindowTitle(TITLE)
        self.dialog.setWindowModality(Qt.WindowModal)
        self.dialog.setMinimumDuration(0)
        self.dialog.canceled.connect(lambda: setattr(self, "cancelled", True))
        self.dialog.setValue(0)
        token = updater.github_token()

        def progress(done, total):
            if self.cancelled:
                raise _Cancelled
            self._progress.emit(done, total)

        def work():
            try:
                zip_path = updater.download(release, token, progress=progress)
                self._fetched.emit(updater.unpack(zip_path))
            except _Cancelled:
                self._fetched.emit(None)
            except Exception as e:
                self._fetched.emit(str(e) or type(e).__name__)
        threading.Thread(target=work, daemon=True).start()

    def _on_progress(self, done: int, total: int):
        if self.dialog is None:
            return
        if total:
            self.dialog.setValue(min(99, int(100 * done / total)))
            self.dialog.setLabelText(f"Downloading the update… {done >> 20} of {total >> 20} MB")
        else:
            self.dialog.setMaximum(0)

    def _on_fetched(self, result):
        if self.dialog is not None:
            self.dialog.close()
            self.dialog = None
        if result is None:
            return
        if isinstance(result, str):
            QMessageBox.warning(self.win, TITLE, f"The update didn't download: {result}")
            return
        try:
            updater.start_update(result)
        except OSError as e:
            QMessageBox.warning(self.win, TITLE, f"Couldn't start the update: {e}")
            return
        QApplication.quit()
