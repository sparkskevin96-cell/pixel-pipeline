"""Keeps the Windows app current. No Qt here (ui/update_ui.py has the windows and buttons).

Install: the first run of a downloaded copy offers to copy itself to
%LOCALAPPDATA%\\Programs\\PixelPipeline (no admin needed) with a desktop icon and a Start menu
entry. A downloaded copy that is older than the installed one just opens the installed one.

Update: CI writes build.json (build number, commit, date) into the app and uploads the same
file as version.json next to PixelPipeline-windows.zip on the windows-latest release. The app
compares the two, downloads the zip, unpacks it into %LOCALAPPDATA%\\PixelPipeline\\update,
starts a small PowerShell script and quits. Windows locks a running exe, so the script waits
for the app to close, copies the new build over it and starts it again with --updated.

While the GitHub repo is private, GitHub only shows the release to someone signed in, so the
app needs a read-only token (saved with the AI keys as "github"). A public repo needs none.
"""
from __future__ import annotations

import http.client
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

REPO = "sparkskevin96-cell/pixel-pipeline"
TAG = "windows-latest"
API = "https://api.github.com"
ZIP_NAME = "PixelPipeline-windows.zip"
VERSION_NAME = "version.json"
EXE_NAME = "PixelPipeline.exe"
BUILD_FILE = Path(__file__).with_name("build.json")
BUILD_IN_APP = Path("_internal", "pixel_pipeline", "ui", "build.json")   # inside an app folder
TOKEN_URL = "https://github.com/settings/personal-access-tokens/new"
FLAGS = ("--updated", "--update-failed")          # what the update script starts the app with
SHORTCUT = "Pixel Pipeline.lnk"
NO_WINDOW = 0x08000000                            # CREATE_NO_WINDOW


class UpdateError(Exception):
    pass


class NeedsToken(UpdateError):
    """The repo is private and there's no token, or the token can't see it."""


def _local() -> Path:
    return Path(os.environ.get("LOCALAPPDATA") or Path.home() / ".pixel-pipeline")


def install_dir() -> Path:
    return _local() / "Programs" / "PixelPipeline"


def update_dir() -> Path:
    return _local() / "PixelPipeline" / "update"


@dataclass
class Build:
    number: int
    sha: str = ""
    date: str = ""

    @property
    def label(self) -> str:
        return f"build {self.number} ({self.date[:10] or '?'}, {self.sha[:7] or '?'})"


@dataclass
class Release:
    build: Build
    zip_url: str
    zip_size: int


def read_build(path: Path) -> Build | None:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        return Build(int(data["number"]), str(data.get("sha", "")), str(data.get("date", "")))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def current_build() -> Build | None:
    """This app's build; None when running from source (git pull updates that instead)."""
    return read_build(BUILD_FILE)


def app_dir() -> Path:
    return Path(sys.executable).parent


def can_update() -> bool:
    return os.name == "nt" and getattr(sys, "frozen", False) and current_build() is not None


def github_token(keys: dict | None = None) -> str:
    from ..online import load_keys
    return (keys if keys is not None else load_keys()).get("github", "")


# ---- asking GitHub -----------------------------------------------------------------------
def _open(url: str, token: str, accept: str = "application/vnd.github+json", timeout: float = 30):
    req = urllib.request.Request(url, headers={"Accept": accept, "User-Agent": "pixel-pipeline",
                                               "X-GitHub-Api-Version": "2022-11-28"})
    if token:
        # Unredirected: GitHub sends downloads on to a storage server that refuses a second login.
        req.add_unredirected_header("Authorization", f"Bearer {token}")
    return urllib.request.urlopen(req, timeout=timeout)


def _asset_url(asset: dict, token: str) -> str:
    # With a token the API address (works for private repos), else the plain download link.
    return asset["url"] if token else asset["browser_download_url"]


def latest(token: str = "", *, repo: str = REPO, tag: str = TAG, api: str = API) -> Release:
    """The newest build on GitHub."""
    try:
        with _open(f"{api}/repos/{repo}/releases/tags/{tag}", token) as r:
            release = json.load(r)
        assets = {a["name"]: a for a in release.get("assets", [])}
        if ZIP_NAME not in assets or VERSION_NAME not in assets:
            raise UpdateError("The newest release has no app build yet; try again in a few minutes.")
        with _open(_asset_url(assets[VERSION_NAME], token), token, "application/octet-stream") as r:
            info = json.loads(r.read().decode("utf-8-sig"))
    except urllib.error.HTTPError as e:
        if e.code == 403 and e.headers.get("X-RateLimit-Remaining") == "0":
            raise UpdateError("GitHub is limiting checks right now; try again in an hour.") from e
        if e.code == 401:
            raise NeedsToken("GitHub refused the token (expired or mistyped?). "
                             "Make a new one: Help → Updates token…") from e
        if e.code in (403, 404):
            # GitHub answers 404 for a private repo to someone not signed in.
            raise NeedsToken("GitHub can't show the updates: " + (
                "the token has no access to the pixel-pipeline repo." if token else
                "the repo is private, so the app needs a GitHub token (Help → Updates token…).")) from e
        raise UpdateError(f"GitHub answered {e.code} {e.reason}") from e
    except (urllib.error.URLError, OSError) as e:
        raise UpdateError(f"Can't reach GitHub ({getattr(e, 'reason', e)}). Online?") from e
    zip_asset = assets[ZIP_NAME]
    build = Build(int(info["number"]), str(info.get("sha", "")), str(info.get("date", "")))
    return Release(build, _asset_url(zip_asset, token), int(zip_asset.get("size") or 0))


def is_newer(release: Release, build: Build | None) -> bool:
    return build is not None and release.build.number > build.number


def download(release: Release, token: str = "", dest: Path | None = None, progress=None) -> Path:
    """The zip, into the update folder. progress(done_bytes, total_bytes)."""
    dest = Path(dest or update_dir())
    dest.mkdir(parents=True, exist_ok=True)
    path, part = dest / ZIP_NAME, dest / (ZIP_NAME + ".part")
    done = total = 0
    try:
        with _open(release.zip_url, token, "application/octet-stream", timeout=60) as r, \
                part.open("wb") as f:
            total = int(r.headers.get("Content-Length") or 0) or release.zip_size
            while chunk := r.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)
    except (urllib.error.URLError, OSError, http.client.HTTPException) as e:
        raise UpdateError(f"The download stopped ({getattr(e, 'reason', e)}). Try again.") from e
    if total and done != total:
        raise UpdateError("The download was cut off. Try again.")
    part.replace(path)
    return path


def unpack(zip_path: Path, dest: Path | None = None) -> Path:
    """Unzip; the folder that holds PixelPipeline.exe."""
    dest = Path(dest or Path(zip_path).parent / "new")
    shutil.rmtree(dest, ignore_errors=True)
    try:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(dest)
    except (zipfile.BadZipFile, OSError) as e:
        raise UpdateError(f"The download is damaged ({e}). Try again.") from e
    found = sorted(dest.rglob(EXE_NAME), key=lambda p: len(p.parts))
    if not found:
        raise UpdateError("The download has no PixelPipeline.exe in it.")
    return found[0].parent


# ---- replacing the app ---------------------------------------------------------------------
def _ps(text) -> str:
    return "'" + str(text).replace("'", "''") + "'"


def update_script(new: Path, app: Path, pid: int, relaunch: list[str], log: Path) -> str:
    """PowerShell: wait for the app (pid) to close, copy `new` over `app`, start relaunch + a flag.
    _internal (the libraries) is mirrored so no stale old library is left behind; the rest of
    the folder is only added to, never cleared."""
    args = {flag: subprocess.list2cmdline([*relaunch[1:], flag]) for flag in FLAGS}
    return f"""# Pixel Pipeline update (written by the app; safe to delete)
$ErrorActionPreference = 'Continue'
$new = {_ps(new)}; $app = {_ps(app)}; $log = {_ps(log)}
Wait-Process -Id {int(pid)} -Timeout 120 -ErrorAction SilentlyContinue
Start-Sleep -Milliseconds 300
$ok = $false
for ($try = 0; $try -lt 3 -and -not $ok; $try++) {{
  if ($try) {{ Start-Sleep -Seconds 2 }}
  robocopy (Join-Path $new '_internal') (Join-Path $app '_internal') /MIR /R:5 /W:1 /NP /NJH /NJS /LOG+:$log | Out-Null
  $a = $LASTEXITCODE
  robocopy $new $app /E /XD (Join-Path $new '_internal') /R:5 /W:1 /NP /NJH /NJS /LOG+:$log | Out-Null
  $ok = ($a -lt 8) -and ($LASTEXITCODE -lt 8)
}}
$flag = if ($ok) {{ {_ps(args[FLAGS[0]])} }} else {{ {_ps(args[FLAGS[1]])} }}
Start-Process -FilePath {_ps(relaunch[0])} -WorkingDirectory $app -ArgumentList $flag
"""


def start_update(new: Path, app: Path | None = None, relaunch: list[str] | None = None,
                 pid: int | None = None) -> Path:
    """Start the script that swaps the app once this process ends. The caller then quits."""
    from .crash import LOG_DIR
    app = Path(app or app_dir())
    relaunch = relaunch or [str(app / EXE_NAME)]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    script = Path(new).parent / "update.ps1"
    script.write_text(update_script(Path(new), app, pid or os.getpid(), relaunch,
                                    LOG_DIR / "update.log"), encoding="utf-8-sig")
    subprocess.Popen(["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                      "-WindowStyle", "Hidden", "-File", str(script)],
                     creationflags=NO_WINDOW | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
                     close_fds=True)
    return script


def cleanup() -> None:
    """After an update: the downloaded zip and unpacked copy aren't needed any more."""
    shutil.rmtree(update_dir(), ignore_errors=True)


# ---- installing -----------------------------------------------------------------------------
def _same(a: Path, b: Path) -> bool:
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def install_plan(here: Path, build: Build | None, target: Path | None = None) -> str:
    """What to do when this copy (app folder `here`) starts:
    'run' (it is the installed copy, or there's nothing to compare), 'install' (none installed),
    'upgrade' (the installed copy is older) or 'open' (the installed copy is as new or newer)."""
    target = Path(target or install_dir())
    if build is None or _same(here, target):
        return "run"
    installed = read_build(target / BUILD_IN_APP) if (target / EXE_NAME).exists() else None
    if installed is None:
        return "install"
    return "upgrade" if installed.number < build.number else "open"


def install(here: Path, target: Path | None = None, shortcuts: bool = True) -> Path:
    """Copy this app folder to the install folder (+ desktop and Start menu icons). Returns the
    installed exe. The copy goes next to it first and is swapped in by renaming, so an install
    that fails (say the installed app is open: Windows won't move it) leaves the old one intact."""
    target = Path(target or install_dir())
    new, old = target.with_name(target.name + ".new"), target.with_name(target.name + ".old")
    try:
        for d in (new, old):
            shutil.rmtree(d, ignore_errors=True)
        shutil.copytree(here, new)
        if target.exists():
            target.rename(old)
        new.rename(target)
    except OSError as e:
        shutil.rmtree(new, ignore_errors=True)
        raise UpdateError(f"Couldn't install to {target}: {e}\n"
                          "If Pixel Pipeline is open in another window, close it and try again.") from e
    shutil.rmtree(old, ignore_errors=True)
    exe = target / EXE_NAME
    if shortcuts:
        make_shortcuts(exe)
    return exe


def make_shortcuts(exe: Path, folders: list[Path] | None = None) -> list[Path]:
    """'Pixel Pipeline' icons on the desktop and in the Start menu (or in `folders`)."""
    where = (", ".join(_ps(f) for f in folders) if folders else
             "[Environment]::GetFolderPath('Desktop'), [Environment]::GetFolderPath('Programs')")
    script = f"""$shell = New-Object -ComObject WScript.Shell
foreach ($dir in @({where})) {{
  $path = Join-Path $dir {_ps(SHORTCUT)}
  $link = $shell.CreateShortcut($path)
  $link.TargetPath = {_ps(exe)}
  $link.WorkingDirectory = {_ps(Path(exe).parent)}
  $link.IconLocation = {_ps(str(exe) + ",0")}
  $link.Description = 'Photos and prompts to pixel-art sprites'
  $link.Save()
  Write-Output $path
}}"""
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                             capture_output=True, text=True, timeout=60, creationflags=NO_WINDOW)
    except (OSError, subprocess.SubprocessError) as e:
        raise UpdateError(f"Couldn't make the desktop icon: {e}") from e
    if out.returncode != 0:
        raise UpdateError(f"Couldn't make the desktop icon: {out.stderr.strip()[:300]}")
    return [Path(line) for line in out.stdout.splitlines() if line.strip()]


def launch(exe: Path, args: list[str] | None = None) -> None:
    subprocess.Popen([str(exe), *(args or [])], cwd=str(Path(exe).parent), close_fds=True,
                     creationflags=getattr(subprocess, "DETACHED_PROCESS", 0))
