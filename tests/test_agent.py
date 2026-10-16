"""The asset studio for Claude (agent.py, agentjobs.py, gameexport.py, the studio MCP tools and the
CLI's --json): procedural makers for real, GPU makers against the fake ComfyUI, the job lifecycle
(queue, one GPU job at a time, cancel, persisted records, interrupted -> resume, the GPU lock),
front-first characters, exports into a temp Godot / Unity project with the manifest and
incremental re-export, and the Godot import check when GODOT_BIN is set."""
import base64
import io
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pixel_pipeline import agent as G
from pixel_pipeline import agentjobs as J
from pixel_pipeline import assets as A
from pixel_pipeline import mcp_server as S
from test_generate import comfy, white_key  # noqa: F401  (pytest fixtures)
from test_preview import robot_lib  # noqa: F401  (fixture: drawn robot, fake ComfyUI)

GODOT = os.environ.get("GODOT_BIN") or shutil.which("godot")


@pytest.fixture(autouse=True)
def fresh_jobs(monkeypatch):
    monkeypatch.setenv("PIXEL_AGENT_PAUSE", "0")
    J.reset_manager()
    yield
    m = J.manager()
    for t in list(m.trackers.values()):
        t.stop()
    J.reset_manager()


def _json(blocks) -> dict:
    return json.loads(blocks[0])


def _images(blocks) -> list[Image.Image]:
    out = []
    for b in blocks[1:]:
        c = b.to_image_content()
        out.append(Image.open(io.BytesIO(base64.b64decode(c.data))))
    return out


def _project(name="Studio", type="topdown-rpg"):
    return G.create_project(name, type=type)


def _sheet_character(name: str = "Hero", dirs=("S", "E", "N", "W"), acts=("idle", "walk")):
    """A library character from a synthetic sheet (no GPU, no rig): 2 clips x 4 directions."""
    from pixel_pipeline import library
    from pixel_pipeline.export import write_animation
    frames, names = [], []
    for a in acts:
        for d in dirs:
            for i in range(4):
                f = np.zeros((32, 24, 4), np.uint8)
                f[8 + (i % 2):30, 6:18] = (200, 40 + 40 * dirs.index(d), 60, 255)
                f[30:32, 4:20] = (40, 40, 40, 255)
                frames.append(f)
                names.append(f"{a}_{d} {i}")
    tmp = A.current_project().folder / "_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    write_animation(frames, tmp / "hero.png", 120, None, 8, names=names)
    ch = library.import_sheet(name, tmp / "hero.png")      # the library (PIXEL_LIBRARY in tests)
    shutil.rmtree(tmp)
    return ch


# --------------------------------------------------------------------------- tools + projects

def test_studio_tools_are_registered_next_to_the_old_ones():
    import asyncio
    names = [t.name for t in asyncio.run(S.server.list_tools())]
    assert len(names) == len(set(names))
    old = {"list_palettes", "convert_photos", "make_sprite_sheet", "make_tileset", "video_to_animation",
           "turnaround", "prompt_to_model", "team_colours", "portrait", "map_tiles", "online_sprites",
           "godot_demo", "idle_animation", "parallax_layers", "generate_sprite", "draw_sprite",
           "preview_image"}
    new = {"list_projects", "create_project", "switch_project", "get_house_style", "set_house_style",
           "lock_palette", "list_assets", "get_asset", "delete_asset", "tag_asset", "list_collections",
           "what_can_be_made", "plan_asset", "make", "make_character", "make_item", "make_prop",
           "make_terrain_tiles", "make_effect", "make_atmosphere", "make_portrait", "make_card",
           "make_ui_kit", "make_map", "pick_variation", "character_review", "character_approve",
           "character_redraw_front", "character_change_words", "character_pick_try", "character_finish",
           "character_redraw", "list_motions", "add_animation", "add_mocap", "add_springs",
           "reskin_character", "restyle_project", "game_model", "map_info", "map_edit",
           "export_to_game", "verify_godot_import", "job_status", "list_jobs", "cancel_job", "resume_jobs"}
    assert old <= set(names) and new <= set(names)


def test_projects_and_house_style():
    d = _json(S.create_project("My Game", type="side-scroller", engine="unity"))
    assert d["current"] and d["style"]["directions"] == 2 and d["style"]["engine"] == "unity"
    assert _json(S.list_projects())["current"] == "My Game"
    st = _json(S.set_house_style({"style_prompt": "bold outlines, warm palette", "shading": "flat",
                                  "tile": 24}))
    assert st["style"]["style_prompt"] == "bold outlines, warm palette"
    assert st["style"]["shading"] == "flat" and st["style"]["light"] == 0.0 and st["style"]["tile"] == 24
    got = _json(S.get_house_style())
    assert "bold outlines" in got["style"]["_words"]["sprite"]
    with pytest.raises(Exception, match="Unknown style field"):
        S.set_house_style({"nonsense": 1})
    G.create_project("Other", switch=False)
    assert A.current_project().name == "My Game"
    assert _json(S.switch_project("other"))["current"] == "Other"


def test_plan_is_a_dry_run_with_minutes():
    _project()
    d = _json(S.plan_asset("a knight in a red cape"))
    p = d["plan"]
    assert p["kind"] == "character" and p["uses_gpu"] and p["gpu_minutes"] > 2
    assert p["after_ok_gpu_minutes"] > 0 and "front first" in p["summary"]
    assert any(f["key"] == "review" for f in d["fields"])
    d = _json(S.make("ice nova spell", dry_run=True))
    assert d["plan"]["kind"] == "effect" and not d["plan"]["uses_gpu"]
    assert not J.manager().jobs                                   # nothing ran
    assert _json(S.plan_asset("stone", kind="tileset"))["plan"]["kind"] == "tileset"


# --------------------------------------------------------------------------- procedural makers + library

def test_cpu_makers_finish_at_once_with_previews_and_library_tools():
    _project()
    blocks = S.make("ice nova spell")
    d = _json(blocks)
    assert d["status"] == "done" and not d["gpu"]
    a = d["result"]["asset"]
    assert a["id"] == "effect/ice-nova" and a["kind"] == "effect" and Path(a["files"]["sheet"]).exists()
    assert Path(a["previews"]["preview"]).exists() and Path(a["previews"]["animation"]).exists()
    assert "quality" in a and a["quality"]["frames"] > 4
    imgs = _images(blocks)
    assert imgs and max(imgs[0].size) <= 1024                   # Claude sees the frames
    for words, kind in (("wooden UI panel", "ui"), ("rain at night", "weather"),
                        ("animated lagoon water", "water")):
        assert _json(S.make(words))["result"]["asset"]["kind"] == kind
    ui = _json(S.make_ui_kit("stone"))["result"]["asset"]
    assert ui["id"] == "ui/stone-ui-kit"
    m = _json(S.make_map("village", cols=12, rows=10))["result"]["asset"]
    assert m["kind"] == "map"
    lst = _json(S.list_assets())
    assert lst["count"] == 6 and {r["kind"] for r in lst["assets"]} >= {"effect", "ui", "map", "water"}
    assert [r["id"] for r in _json(S.list_assets(kind="effect"))["assets"]] == ["effect/ice-nova"]
    t = _json(S.tag_asset("effect/ice-nova", add_tags=["approved", "magic"], add_collections=["starter"]))
    assert "approved" in t["tags"] and t["collections"] == ["starter"]
    assert _json(S.list_collections())["collections"] == {"starter": ["effect/ice-nova"]}
    assert _json(S.list_assets(query="magic"))["count"] == 1
    got = _json(S.get_asset("Ice nova"))                         # by name too
    assert got["id"] == "effect/ice-nova" and got["animations"]
    with pytest.raises(Exception, match="No asset"):
        S.get_asset("nothing like it")
    gone = _json(S.delete_asset("ui/stone-ui-kit"))
    assert Path(gone["moved_to"]).exists() and ".trash" in gone["moved_to"]
    assert _json(S.list_assets(kind="ui"))["count"] == 1


def test_map_info_and_edits():
    _project()
    m = _json(S.make_map("flat", cols=12, rows=10))["result"]["asset"]
    info = _json(S.map_info(m["id"]))
    assert info["cols"] == 12 and len(info["terrain"]) == 10 and len(info["terrain"][0]) == 12
    water = next(k for k, v in info["legend"].items() if v == "water")
    d = _json(S.map_edit(m["id"], [{"op": "rect", "col": 0, "row": 0, "col2": 3, "row2": 1, "terrain": "water"},
                                  {"op": "prop", "col": 6, "row": 5, "kind": "tree"},
                                  {"op": "spawn", "col": 8, "row": 8, "team": 1},
                                  {"op": "height", "col": 10, "row": 2, "level": 5}]))
    assert d["edits"] == 4 and d["terrain"][0][:4] == water * 4 and d["terrain"][1][:4] == water * 4
    assert {"kind": "tree", "col": 6, "row": 5} in d["props"]
    assert {"team": 1, "col": 8, "row": 8} in d["spawns"] and d["height"][2][10] == "5"
    with pytest.raises(Exception, match="Unknown map edit"):
        S.map_edit(m["id"], [{"op": "explode"}])


# --------------------------------------------------------------------------- jobs

@J.register("test_work")
def _test_work(args, tracker):
    n = int(args.get("steps", 5))
    for i in range(n):
        tracker.say(f"Working {i + 1} of {n}…")
        if args.get("hold"):
            args["hold"].wait(5)
        time.sleep(float(args.get("dt", 0.01)))
    if args.get("fail"):
        raise ValueError("it broke")
    return {"made": n}


def test_jobs_queue_one_gpu_job_at_a_time_cancel_and_persist(tmp_path):
    proj = A.create_project("Jobs")
    m = J.manager()
    hold = threading.Event()
    first = m.start("test_work", {"steps": 3, "hold": hold}, project=proj, gpu=True, title="first")
    second = m.start("test_work", {"steps": 2}, project=proj, gpu=True, title="second")
    third = m.start("test_work", {"steps": 2}, project=proj, gpu=True, title="third")
    time.sleep(0.3)
    assert m.get(first.id).status == "running" and m.get(second.id).status == "queued"
    assert G.job_view(second.id)["queue_position"] == 1
    assert J.lock_holder()["pid"] == os.getpid()                 # the GPU lock is held
    assert G.cancel_job(third.id)["status"] == "cancelled"        # queued: dropped at once
    cpu = m.start("test_work", {"steps": 2}, project=proj, gpu=False, wait_s=5)
    assert cpu.status == "done"                                   # CPU jobs don't queue behind GPU
    hold.set()
    m.wait(second.id, 10)
    assert m.get(first.id).status == "done" and m.get(second.id).status == "done"
    assert m.get(first.id).result == {"made": 3} and m.get(first.id).percent == 100
    assert J.lock_holder() is None
    rec = json.loads((proj.folder / "agent_jobs" / f"{first.id}.json").read_text())
    assert rec["status"] == "done" and rec["log"][-1].startswith("Working 3 of 3")
    # cancel a running job: Stopped at its next message
    hold2 = threading.Event()
    slow = m.start("test_work", {"steps": 50, "dt": 0.05, "hold": hold2}, project=proj, gpu=True)
    time.sleep(0.2)
    m.cancel(slow.id)
    hold2.set()
    assert m.wait(slow.id, 10).status == "cancelled"
    bad = m.start("test_work", {"fail": True}, project=proj, gpu=False, wait_s=5)
    d = G.job_status(bad.id)
    assert d["status"] == "failed" and "it broke" in d["error"]
    listed = G.list_jobs(project="Jobs")
    assert len(listed["jobs"]) == 6 and listed["gpu_lock"] == "free"


def test_interrupted_jobs_are_found_and_resumed(tmp_path):
    proj = A.create_project("Crash")
    rec = J.JobRecord("deadbeef01", "test_work", {"steps": 2}, "from a crashed server", proj.name,
                      str(proj.folder), True, "running", time.time() - 60, started=time.time() - 50,
                      pid=999999)                                # a process that's gone
    J.manager()._save(rec)
    J.reset_manager()                                             # a new server process
    jobs = G.list_jobs()
    assert jobs["interrupted"] == 1 and jobs["jobs"][0]["status"] == "interrupted"
    assert G.job_status("deadbeef01")["hint"].startswith("the process running it stopped")
    d = G.resume_jobs()
    assert d["resumed"] == ["deadbeef01"]
    assert J.manager().wait("deadbeef01", 10).status == "done"
    assert J.manager().get("deadbeef01").attempts == 1


def test_gpu_jobs_wait_for_the_lock_held_by_another_process():
    proj = A.create_project("Lock")
    lock = J.lock_path()
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(json.dumps({"pid": os.getppid(), "what": "another agent", "t": time.time()}))
    job = J.manager().start("test_work", {"steps": 1}, project=proj, gpu=True)
    time.sleep(0.6)
    rec = J.manager().get(job.id)
    assert rec.status == "running" and "Waiting for the GPU (another agent" in rec.text
    lock.unlink()
    assert J.manager().wait(job.id, 10).status == "done"
    assert J.manager().get(job.id).gpu_wait_s > 0.3


# --------------------------------------------------------------------------- GPU makers (fake ComfyUI)

def test_gpu_makers_through_jobs_and_pick_variation(comfy, monkeypatch):  # noqa: F811
    pytest.importorskip("trimesh")
    fake, client = comfy
    _project()
    J.manager().client = client
    d = _json(S.make_item("rusty iron sword", wait_s=120))
    assert d["status"] == "done" and d["gpu"], d
    a = d["result"]["asset"]
    assert a["kind"] == "item" and len(a["variations"]) == 3 and a["chosen"] == 1
    assert Path(a["previews"]["variations"]).exists()
    assert "subject_check" in a["quality"]
    p = _json(S.pick_variation(2, ref=a["id"]))
    assert p["picked"] == 2 and p["asset"]["chosen"] == 2
    t = _json(S.make_terrain_tiles("cobblestone road tiles", count=2, wait_s=120))
    assert t["status"] == "done", t
    first = t["result"]["asset"]
    assert first["kind"] == "tileset" and t["result"]["variations"]["count"] == 2
    assert Path(first["files"]["tileset_godot"]).exists()
    J.reset_manager()                                         # a restart: the pick still works
    J.manager().client = client
    p = _json(S.pick_variation(2, job=t["job"]))
    assert p["picked"] == 2 and p["asset"]["kind"] == "tileset"
    assert not Path(first["folder"]).exists()                    # the first set went to .trash
    assert _json(S.list_assets(kind="tileset"))["count"] == 1


# --------------------------------------------------------------------------- characters: front first

def test_front_first_review_and_approve_through_jobs(robot_lib):
    from pixel_pipeline import library
    fake, model = robot_lib
    ch = library.add_character("Bot", model, prompt="a robot", preview=True)
    rid = "character/" + ch.folder.name
    blocks = S.character_review(rid)
    d = _json(blocks)
    assert d["state"].startswith("draft") and d["review"]["direction"] == "S"
    assert len(d["review"]["idle_frames"]) == 8 and d["review"]["tries"][0]["current"]
    assert Path(d["previews"]["preview"]).exists() and _images(blocks)
    with pytest.raises(Exception, match="waiting for your OK"):
        S.add_animation(rid, ["jump"])
    from pixel_pipeline.gameexport import export_to_game
    game = Path(A.current_project().folder).parent / "drafts_game"
    game.mkdir()
    (game / "project.godot").write_text("config_version=5\n")
    rep = export_to_game([G.resolve(rid)], "godot", game)
    assert rep["skipped"][0]["why"].startswith("draft")
    r = _json(S.character_redraw_front(rid, wait_s=120))
    assert r["status"] == "done" and len(r["result"]["asset"]["review"]["tries"]) == 2
    assert _json(S.character_pick_try(rid, 1))["review"]["tries"][0]["current"]
    job = _json(S.character_approve(rid, wait_s=300))
    assert job["status"] == "done", job
    a = job["result"]["asset"]
    assert a["state"] == "ready (drawn)" and a["directions"] and "drawn" in a["quality"]
    assert a["quality"]["drawn"]["directions"] == 4
    with pytest.raises(Exception, match="isn't waiting"):
        S.character_approve(rid)


# --------------------------------------------------------------------------- exports into a game

def _godot_project(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "project.godot").write_text('config_version=5\n\n[application]\n\nconfig/name="T"\n',
                                        encoding="utf-8")
    return root


def test_export_into_godot_writes_manifest_and_is_incremental(tmp_path):
    _project()
    ch = _sheet_character()
    for words in ("ice nova spell", "wooden UI panel", "a forest map with a river", "animated lagoon water"):
        S.make(words)
    game = _godot_project(tmp_path / "game")
    d = _json(S.export_to_game(str(game), engine="godot", subdir="art"))
    assert d["status"] == "done", d
    rep = d["result"]
    assert rep["counts"]["exported"] == 5 and not rep["skipped"]
    man = json.loads((game / "art" / "pixel_pipeline_manifest.json").read_text())
    hero = man["assets"]["character/" + ch.folder.name]
    assert hero["frame"] == {"w": 24, "h": 32} and set(hero["animations"]) == {"idle", "walk"}
    assert hero["animations"]["walk"]["directions"] == ["S", "E", "N", "W"]
    assert hero["animations"]["idle"]["frames"] == 4 and hero["animations"]["idle"]["ms"] == 120
    assert hero["anchor"]["y"] == 32 and hero["res"]["sprite_frames"].startswith("res://art/characters/")
    assert man["by_name"]["hero"] == hero["id"]
    tex = game / "art" / hero["files"]["texture"]
    assert tex.exists() and "compress/mode=0" in tex.with_name(tex.name + ".import").read_text()
    assert "mipmaps/generate=false" in tex.with_name(tex.name + ".import").read_text()
    assert "default_texture_filter=0" in (game / "project.godot").read_text()
    mp = next(e for e in man["assets"].values() if e["kind"] == "map")
    folder = game / "art" / mp["folder"]
    assert not (folder / "project.godot").exists()                # embedded, not a nested project
    assert all("res://art/maps/" in ln for f in folder.glob("*.tscn") for ln in f.read_text().splitlines()
               if 'path="res://' in ln)
    ui = next(e for e in man["assets"].values() if e["kind"] == "ui")
    assert ui["files"]["theme"].endswith("theme.tres")
    assert not (game / "art" / ui["folder"] / "fonts" / "pixelkit_1x.png.import").exists()   # font pages: Godot's
    # again: nothing changed -> nothing written
    rep2 = _json(S.export_to_game(str(game), engine="godot", subdir="art"))["result"]
    assert rep2["counts"] == {"exported": 0, "unchanged": 5, "skipped": 0, "in_manifest": 5}
    # change one asset: only it is exported again
    time.sleep(0.05)
    sheet = ch.folder / "sheet.png"
    img = Image.open(sheet).convert("RGBA")
    img.putpixel((0, 0), (255, 0, 0, 255))
    img.save(sheet)
    rep3 = _json(S.export_to_game(str(game), engine="godot", subdir="art"))["result"]
    assert [e["id"] for e in rep3["exported"]] == [hero["id"]] and rep3["counts"]["unchanged"] == 4
    with pytest.raises(Exception, match="not a Godot project"):
        S.export_to_game(str(tmp_path), engine="godot")


def test_export_into_unity(tmp_path):
    _project()
    ch = _sheet_character("Mage")
    S.make("ice nova spell")
    game = tmp_path / "UnityGame"
    (game / "Assets").mkdir(parents=True)
    rep = _json(S.export_to_game(str(game), engine="unity"))["result"]
    assert rep["subdir"] == "Assets/PixelPipeline" and rep["counts"]["exported"] == 2
    base = game / "Assets" / "PixelPipeline"
    man = json.loads((base / "pixel_pipeline_manifest.json").read_text())
    assert (base / "pixel_pipeline_manifest.json.meta").exists()
    mage = man["assets"]["character/" + ch.folder.name]
    png = base / mage["files"]["texture"]
    meta = png.with_name(png.name + ".meta").read_text()
    assert "filterMode: 0" in meta and "spriteMode: 2" in meta
    assert "res" not in mage and mage["engine"] == "unity"


@pytest.mark.skipif(not GODOT, reason="set GODOT_BIN to import the exported assets in Godot")
def test_exported_set_imports_in_godot(tmp_path):
    _project()
    _sheet_character()
    for words in ("ice nova spell", "wooden UI panel", "a forest map with a river", "animated lagoon water",
                  "rain at night", "sunset mountains background"):
        S.make(words)
    game = _godot_project(tmp_path / "game")
    d = _json(S.export_to_game(str(game), engine="godot", verify=True, wait_s=600))
    assert d["status"] == "done", d
    v = d["result"]["verify"]
    assert v["ok"], v
    res = _json(S.verify_godot_import(str(game), wait_s=600))["result"]
    assert res["ok"] and len(res["assets"]) == 7
    hero = next(r for k, r in res["assets"].items() if k.startswith("character/"))
    assert any(s.startswith("sprite_frames: SpriteFrames anims=8") for s in hero["loaded"])


# --------------------------------------------------------------------------- CLI --json

def test_cli_json(tmp_path, capsys):
    from pixel_pipeline.cli import main
    assert main(["project", "create", "CliGame", "--type", "topdown-rpg", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["project"] == "CliGame"
    assert main(["make", "a knight in a red cape", "--dry-run", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["plan"]["kind"] == "character"
    assert main(["make", "ice nova spell", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["status"] == "done" and d["result"]["asset"]["id"] == "effect/ice-nova"
    assert main(["library", "search", "nova", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["assets"][0]["id"] == "effect/ice-nova"
    assert main(["library", "tag", "effect/ice-nova", "--add", "approved", "--json"]) == 0
    assert "approved" in json.loads(capsys.readouterr().out)["tags"]
    assert main(["jobs", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["jobs"][0]["status"] == "done"
    assert main(["project", "style", "--set", "tile=24", "--set", 'style_prompt="warm"', "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["style"]["tile"] == 24
    game = _godot_project(tmp_path / "g")
    assert main(["export", "--godot", str(game), "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["status"] == "done" and d["result"]["counts"]["exported"] == 1
    assert main(["library", "show", "nope", "--json"]) == 2
    assert "error" in json.loads(capsys.readouterr().out)


def test_cli_entry_point_runs(tmp_path):
    exe = Path(os.sys.executable).with_name("pixel-pipeline.exe")
    if not exe.exists():
        pytest.skip("console script not installed")
    env = {**os.environ, "PIXEL_HOME": str(tmp_path / "home")}
    r = subprocess.run([str(exe), "make", "fireball spell", "--dry-run", "--json"], capture_output=True,
                       text=True, env=env, timeout=120)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["plan"]["kind"] == "effect"


def test_studio_tools_over_stdio(tmp_path):
    """The real server process: a CPU make returns the JSON + preview images to the client."""
    import asyncio
    import sys
    from mcp import Client, StdioServerParameters, stdio_client
    src = Path(__file__).resolve().parents[1] / "src"
    env = {**os.environ, "PYTHONPATH": str(src), "PIXEL_HOME": str(tmp_path / "home"),
           "COMFYUI_URL": "http://127.0.0.1:9", "PIXEL_WORKERS": "1"}
    env.pop("PIXEL_LIBRARY", None)

    async def run():
        params = StdioServerParameters(command=sys.executable, args=["-m", "pixel_pipeline.mcp_server"],
                                       cwd=str(tmp_path), env=env)
        async with Client(stdio_client(params), read_timeout_seconds=120) as client:
            r = await client.call_tool("create_project", {"name": "Wire", "type": "topdown-rpg"})
            assert not r.is_error, r
            r = await client.call_tool("make", {"description": "ice nova spell"})
            assert not r.is_error, r
            text = "\n".join(b.text for b in r.content if b.type == "text")
            d = json.loads(text)
            assert d["status"] == "done" and d["result"]["asset"]["id"] == "effect/ice-nova"
            assert any(b.type == "image" for b in r.content)
            r = await client.call_tool("list_jobs", {})
            assert json.loads(r.content[0].text)["jobs"][0]["status"] == "done"
    asyncio.run(run())
