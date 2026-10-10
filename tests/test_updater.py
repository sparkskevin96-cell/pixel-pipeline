"""Install and self-update of the Windows app (ui/updater.py) against a fake GitHub."""
import json
import os
import subprocess
import sys
import threading
import time
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from pixel_pipeline.ui import updater
from pixel_pipeline.ui.updater import Build, NeedsToken, Release, UpdateError

windows = pytest.mark.skipif(os.name != "nt", reason="Windows only (PowerShell, robocopy)")
REPO, TOKEN = "kev/pixel-pipeline", "github_pat_test"


class FakeGitHub(BaseHTTPRequestHandler):
    """The release API, asset downloads that redirect to a storage server (which, like GitHub's,
    refuses a request that still carries the token), and plain public download links."""
    private = True
    files: dict = {}
    seen: list = []

    def log_message(self, *args):
        pass

    def _send(self, code, body=b"", headers=None):
        self.send_response(code)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        auth = self.headers.get("Authorization", "")
        self.seen.append((self.path, auth))
        base = f"http://{self.headers['Host']}"
        if self.path.startswith("/storage/"):
            if auth:
                return self._send(400, b"Only one auth mechanism allowed")
            return self._send(200, self.files[self.path.rsplit("/", 1)[1]])
        if self.path == "/cut":                    # the connection drops halfway
            self.send_response(200)
            self.send_header("Content-Length", "1000")
            self.end_headers()
            return self.wfile.write(b"x" * 10)
        if self.path.startswith("/download/") and not self.private:
            return self._send(200, self.files[self.path.rsplit("/", 1)[1]])
        if auth and auth != f"Bearer {TOKEN}":
            return self._send(401, b'{"message": "Bad credentials"}')
        if self.private and not auth:
            return self._send(404, b'{"message": "Not Found"}')
        if self.path == f"/repos/{REPO}/releases/tags/windows-latest":
            assets = [{"name": n, "size": len(b), "url": f"{base}/api/assets/{n}",
                       "browser_download_url": f"{base}/download/{n}"} for n, b in self.files.items()]
            return self._send(200, json.dumps({"assets": assets}).encode())
        if self.path.startswith("/api/assets/"):
            assert self.headers["Accept"] == "application/octet-stream"
            return self._send(302, headers={"Location": f"{base}/storage/{self.path.rsplit('/', 1)[1]}"})
        self._send(404)


def _app_zip(path: Path, number: int) -> bytes:
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("PixelPipeline/PixelPipeline.exe", b"MZ fake")
        z.writestr("PixelPipeline/_internal/pixel_pipeline/ui/build.json", json.dumps({"number": number}))
    return path.read_bytes()


@pytest.fixture
def github(tmp_path):
    version = {"number": 7, "sha": "abc1234def", "date": "2026-10-10T12:00:00Z"}
    FakeGitHub.files = {"version.json": json.dumps(version).encode(),
                        "PixelPipeline-windows.zip": _app_zip(tmp_path / "src.zip", 7)}
    FakeGitHub.seen = []
    FakeGitHub.private = True
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeGitHub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_private_repo_needs_a_token_and_the_token_never_reaches_storage(github, tmp_path):
    with pytest.raises(NeedsToken, match="private"):
        updater.latest("", repo=REPO, api=github)
    with pytest.raises(NeedsToken, match="refused the token"):
        updater.latest("wrong", repo=REPO, api=github)
    rel = updater.latest(TOKEN, repo=REPO, api=github)
    assert rel.build == Build(7, "abc1234def", "2026-10-10T12:00:00Z")
    assert rel.build.label == "build 7 (2026-10-10, abc1234)"
    assert updater.is_newer(rel, Build(6)) and not updater.is_newer(rel, Build(7))
    assert not updater.is_newer(rel, None)                       # from source: never
    steps = []
    zip_path = updater.download(rel, TOKEN, tmp_path / "update", progress=lambda d, t: steps.append((d, t)))
    assert zip_path.read_bytes() == FakeGitHub.files["PixelPipeline-windows.zip"]
    assert steps[-1][0] == steps[-1][1] == rel.zip_size
    assert all(not auth for path, auth in FakeGitHub.seen if path.startswith("/storage/"))
    new = updater.unpack(zip_path)
    assert new == tmp_path / "update" / "new" / "PixelPipeline" and (new / "PixelPipeline.exe").exists()
    assert updater.read_build(new / updater.BUILD_IN_APP).number == 7


def test_public_repo_needs_no_token(github, tmp_path):
    FakeGitHub.private = False
    rel = updater.latest("", repo=REPO, api=github)
    assert rel.build.number == 7 and "/download/" in rel.zip_url
    assert updater.download(rel, "", tmp_path).stat().st_size == rel.zip_size


def test_errors_are_plain_words(github, tmp_path):
    with pytest.raises(UpdateError, match="Can't reach GitHub"):
        updater.latest("", repo=REPO, api="http://127.0.0.1:9")
    FakeGitHub.files.pop("version.json")
    with pytest.raises(UpdateError, match="no app build yet"):
        updater.latest(TOKEN, repo=REPO, api=github)
    cut = Release(Build(8), f"{github}/cut", 1000)
    with pytest.raises(UpdateError, match="cut off|stopped"):
        updater.download(cut, "", tmp_path)
    (tmp_path / "bad.zip").write_bytes(b"not a zip")
    with pytest.raises(UpdateError, match="damaged"):
        updater.unpack(tmp_path / "bad.zip")


def test_build_files(tmp_path, monkeypatch):
    (tmp_path / "build.json").write_text('﻿{"number": 12, "sha": "f00", "date": "x"}', encoding="utf-8")
    assert updater.read_build(tmp_path / "build.json") == Build(12, "f00", "x")    # BOM from PowerShell ok
    assert updater.read_build(tmp_path / "missing.json") is None
    monkeypatch.setattr(updater, "BUILD_FILE", tmp_path / "missing.json")
    assert updater.current_build() is None and not updater.can_update()          # running from source


def _fake_app(folder: Path, number: int | None, extra: str = "") -> Path:
    (folder / "_internal" / "pixel_pipeline" / "ui").mkdir(parents=True)
    (folder / "PixelPipeline.exe").write_bytes(b"MZ" + extra.encode())
    if number is not None:
        (folder / updater.BUILD_IN_APP).write_text(json.dumps({"number": number}))
    return folder


def test_install_plan_and_install(tmp_path):
    here = _fake_app(tmp_path / "Downloads" / "PixelPipeline", 5)
    target = tmp_path / "Programs" / "PixelPipeline"
    assert updater.install_plan(here, Build(5), target) == "install"
    assert updater.install_plan(target, Build(5), target) == "run"            # this is the installed one
    assert updater.install_plan(here, None, target) == "run"                  # from source
    exe = updater.install(here, target, shortcuts=False)
    assert exe == target / "PixelPipeline.exe" and exe.exists()
    assert updater.install_plan(here, Build(5), target) == "open"             # same build: use that one
    assert updater.install_plan(here, Build(4), target) == "open"             # an old download
    assert updater.install_plan(here, Build(6), target) == "upgrade"
    (target / "_internal" / "stale.dll").write_text("old")
    newer = _fake_app(tmp_path / "New" / "PixelPipeline", 6, "new")
    updater.install(newer, target, shortcuts=False)
    assert updater.read_build(target / updater.BUILD_IN_APP).number == 6
    assert not (target / "_internal" / "stale.dll").exists()
    assert not any(target.with_name(f"PixelPipeline.{x}").exists() for x in ("new", "old"))


def test_update_script_text(tmp_path):
    script = updater.update_script(tmp_path / "new", tmp_path / "it's here", 4242,
                                   [str(tmp_path / "PixelPipeline.exe")], tmp_path / "update.log")
    assert "Wait-Process -Id 4242" in script and "it''s here" in script        # PowerShell quoting
    assert "'--updated'" in script and "'--update-failed'" in script
    assert "/MIR" in script.split("robocopy (Join-Path $new '_internal')")[1].splitlines()[0]


@windows
def test_update_swaps_files_after_the_app_closes(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    app = _fake_app(tmp_path / "App", 1)
    (app / "_internal" / "stale.dll").write_text("old")
    (app / "my notes.txt").write_text("keep")
    new = _fake_app(tmp_path / "update" / "new" / "PixelPipeline", 2, "new")
    app_proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(3)"])
    marker = tmp_path / "restarted.txt"
    restart = tmp_path / "restart.py"
    restart.write_text(f"import sys; open(r'{marker}', 'w').write(' '.join(sys.argv[1:]))")
    updater.start_update(new, app, [sys.executable, str(restart)], pid=app_proc.pid)
    time.sleep(1)
    assert updater.read_build(app / updater.BUILD_IN_APP).number == 1          # waits for the app
    app_proc.wait()
    for _ in range(60):
        if marker.exists():
            break
        time.sleep(1)
    assert marker.read_text() == "--updated"
    assert updater.read_build(app / updater.BUILD_IN_APP).number == 2
    assert (app / "PixelPipeline.exe").read_bytes() == b"MZnew"
    assert not (app / "_internal" / "stale.dll").exists()                     # libraries mirrored
    assert (app / "my notes.txt").read_text() == "keep"                       # the rest left alone


@windows
def test_shortcuts(tmp_path):
    exe = _fake_app(tmp_path / "App", 1) / "PixelPipeline.exe"
    folders = [tmp_path / "Desktop", tmp_path / "Programs"]
    for d in folders:
        d.mkdir()
    links = updater.make_shortcuts(exe, folders)
    assert [p.name for p in links] == ["Pixel Pipeline.lnk"] * 2 and all(p.exists() for p in links)
