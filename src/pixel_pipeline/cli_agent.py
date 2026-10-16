"""CLI for the asset studio (the same operations as the MCP tools, agent.py), --json everywhere.

    pixel-pipeline make "a dwarf warrior with a big axe" --json      # waits, prints the result
    pixel-pipeline make "rusty iron sword" --dry-run --json          # the plan + GPU minutes
    pixel-pipeline character approve character/dwarf --json
    pixel-pipeline jobs [--status running] / jobs status ID --wait 60 / jobs cancel ID / jobs resume
    pixel-pipeline library search sword --kind item --json / library show item/rusty-iron-sword
    pixel-pipeline project list / project create MyGame --type topdown-rpg --engine godot
    pixel-pipeline project style --set style_prompt="16-bit SNES JRPG" --set shading=cel
    pixel-pipeline export --godot C:/Games/MyGame --subdir art --verify --json

GPU jobs started here run in THIS process (one at a time, the cross-process GPU lock keeps the
MCP server's jobs and these apart): the command waits for them (Ctrl+C cancels; what is finished
is kept). --no-wait queues them in the record only when another process will run them, so it is
rarely what you want; an interrupted job resumes with `jobs resume`.
"""
from __future__ import annotations

import json
import sys
import time


def _out(a, d: dict, human=None) -> int:
    if getattr(a, "json", False) or human is None:
        print(json.dumps(d, indent=1, default=str))
    else:
        human(d)
    bad = d.get("status") in ("failed", "cancelled") if isinstance(d, dict) else False
    return 1 if bad else 0


def _kv(pairs) -> dict:
    out = {}
    for p in pairs or []:
        k, sep, v = p.partition("=")
        if not sep:
            raise ValueError(f"Expected key=value, got {p!r}")
        try:
            out[k.strip()] = json.loads(v)
        except ValueError:
            out[k.strip()] = v
    return out


def _follow(job: dict, quiet: bool = False) -> dict:
    """Wait for a job, printing progress to stderr; Ctrl+C cancels it."""
    from . import agent as G
    jid = job.get("job")
    if not jid:
        return job
    last = None
    try:
        while True:
            d = G.job_status(jid, wait_s=2.0)
            if d["status"] not in ("queued", "running"):
                return d
            line = f"[{d.get('percent') or 0:>3}%] {d.get('stage', '')} {d.get('doing', '')} {d.get('eta', '')}".strip()
            if line != last and not quiet:
                print(line, file=sys.stderr, flush=True)
                last = line
    except KeyboardInterrupt:
        print("Stopping (what is finished is kept)…", file=sys.stderr, flush=True)
        G.cancel_job(jid)
        return G.job_status(jid, wait_s=120)


def _human_job(d: dict) -> None:
    print(f"{d.get('title', d.get('op'))}: {d['status']}"
          + (f" ({d['minutes']} min)" if d.get("minutes") is not None else ""))
    if d.get("error"):
        print("error:", d["error"])
    res = d.get("result") or {}
    a = res.get("asset") if isinstance(res, dict) else None
    if a:
        print(f"  {a['id']}  {a['name']}  [{a['state']}]  {a['folder']}")
        for k, v in (a.get("previews") or {}).items():
            print(f"  preview {k}: {v}")
        for w in (a.get("quality") or {}).get("warnings", []):
            print("  warning:", w)
    if res.get("next"):
        print("  next:", res["next"])


def _cmd_make(a) -> int:
    from . import agent as G
    params = _kv(a.param)
    if a.dry_run:
        return _out(a, G.plan(a.description, a.kind, params, a.name, a.project),
                    lambda d: print(d["plan"]["summary"]))
    job = G.make(a.description, a.kind, params, a.name, wait_s=0, project=a.project)
    if a.no_wait:
        return _out(a, job, lambda d: print(f"job {d['job']}: {d['status']} — {d['plan']['summary']}"))
    if not a.json:
        print(job["plan"]["summary"], file=sys.stderr)
    return _out(a, _follow(job, a.json), _human_job)


def _cmd_character(a) -> int:
    from . import agent as G
    act = a.action
    if act == "review":
        return _out(a, G.review(a.ref, a.project), None)
    if act == "pick":
        return _out(a, G.pick_try(a.ref, int(a.value), a.project), None)
    if act == "approve":
        job = G.approve(a.ref, 0, a.project)
    elif act == "redraw-front":
        job = G.redraw_front(a.ref, int(a.value) if a.value else None, 0, a.project)
    elif act == "words":
        job = G.change_words(a.ref, a.value or "", a.new_model, 0, a.project)
    elif act == "finish":
        job = G.finish(a.ref, 0, a.project)
    elif act == "redraw":
        job = G.redraw_directions(a.ref, (a.value or "").replace(",", " ").split(), 0, a.project)
    else:
        raise ValueError(f"Unknown character action {act}")
    return _out(a, _follow(job, a.json), _human_job)


def _cmd_jobs(a) -> int:
    from . import agent as G
    act = a.action or "list"
    if act == "list":
        return _out(a, G.list_jobs(a.status, a.project, a.limit), lambda d: [
            print(f"{j['job']}  {j['status']:<11} {str(j.get('percent') or ''):>4}  {j['title']}")
            for j in d["jobs"]])
    if act == "status":
        return _out(a, G.job_status(a.ids[0], a.wait), _human_job)
    if act == "cancel":
        return _out(a, G.cancel_job(a.ids[0]), _human_job)
    if act == "resume":
        d = G.resume_jobs(a.ids or None)
        if not d["resumed"]:
            return _out(a, d, lambda x: print("nothing to resume"))
        res = [_follow(j, a.json) for j in d["jobs"]]
        return _out(a, {"resumed": d["resumed"], "jobs": res}, lambda x: [_human_job(j) for j in x["jobs"]])
    raise ValueError(f"Unknown jobs action {act}")


def _cmd_library(a) -> int:
    from . import agent as G
    act = a.action
    if act in ("search", "list"):
        return _out(a, G.list_assets(a.kind, " ".join(a.words), a.tag, a.collection, a.project, a.limit),
                    lambda d: [print(f"{r['id']:<44} {r['name']}" + (f"  [{r['state']}]" if r.get("state") else ""))
                               for r in d["assets"]])
    if act == "show":
        return _out(a, G.get_asset(a.words[0], a.project), None)
    if act == "delete":
        return _out(a, G.delete_asset(a.words[0], a.project), None)
    if act == "tag":
        return _out(a, G.tag_asset(a.words[0], a.add or (), a.remove or (), a.add_collection or (),
                                   a.remove_collection or (), a.project), None)
    if act == "collections":
        return _out(a, G.collections(a.project), None)
    if act == "pick":
        return _out(a, G.pick_variation(a.words[0] if a.words else None, int(a.index), a.project, a.job), None)
    raise ValueError(f"Unknown library action {act}")


def _cmd_project(a) -> int:
    from . import agent as G
    act = a.action or "list"
    if act == "list":
        return _out(a, G.projects(), lambda d: [print(("* " if p["current"] else "  ") + p["name"]
                                                      + f"  ({p['type']}, {p['engine']}, {p['assets']} assets)")
                                                for p in d["projects"]])
    if act == "create":
        return _out(a, G.create_project(a.name, a.type, a.size, a.engine, not a.no_switch), None)
    if act == "switch":
        return _out(a, G.switch_project(a.name), None)
    if act == "style":
        if a.set:
            return _out(a, G.set_style(_kv(a.set), a.name), None)
        return _out(a, G.get_style(a.name), None)
    raise ValueError(f"Unknown project action {act}")


def _cmd_export(a) -> int:
    from . import agent as G
    engine, path = ("godot", a.godot) if a.godot else ("unity", a.unity)
    if not path:
        raise ValueError("Give --godot <project folder> or --unity <project folder>")
    job = G.export(a.assets or None, engine, path, a.subdir, a.collection, a.kind or None, a.force,
                   a.models, a.build_models, a.verify, a.preset, 0, a.project)
    d = _follow(job, a.json)

    def human(x):
        r = x.get("result") or {}
        if x["status"] != "done":
            return _human_job(x)
        print(f"{r['counts']} -> {r['manifest']}")
        for s in r.get("skipped", []):
            print(f"  skipped {s['id']}: {s['why']}")
        for w in r.get("warnings", []):
            print("  warning:", w)
        if "verify" in r:
            print("  Godot import:", "ok" if r["verify"]["ok"] else r["verify"])
    return _out(a, d, human)


def add_parsers(sub) -> None:
    def common(p, project=True):
        p.add_argument("--json", action="store_true", help="print JSON (for scripts / Claude)")
        if project:
            p.add_argument("--project", default=None, help="project name (default: the current one)")

    p = sub.add_parser("make", help="make anything from words in the project's house style (asset studio)")
    p.add_argument("description")
    p.add_argument("--kind", default=None, help="force the kind (character, item, prop, tileset, effect, "
                                                "portrait, card, ui, map, weather, water, background)")
    p.add_argument("--param", action="append", metavar="KEY=VALUE", help="a plan setting (repeat)")
    p.add_argument("--name", default=None)
    p.add_argument("--dry-run", action="store_true", help="only the plan and GPU minutes")
    p.add_argument("--no-wait", action="store_true", help="queue and return (another process must run it)")
    common(p)
    p.set_defaults(func=_cmd_make)

    p = sub.add_parser("character", help="front-first review: review / approve / redraw-front / words / pick / finish / redraw")
    p.add_argument("action", choices=["review", "approve", "redraw-front", "words", "pick", "finish", "redraw"])
    p.add_argument("ref", help="character id or name")
    p.add_argument("value", nargs="?", default=None, help="words: the new words; pick: try number; "
                                                          "redraw-front: seed; redraw: directions 'NE,NNE'")
    p.add_argument("--new-model", action="store_true", help="words: a new 3D shape too")
    common(p)
    p.set_defaults(func=_cmd_character)

    p = sub.add_parser("jobs", help="background jobs: list / status ID / cancel ID / resume [IDs]")
    p.add_argument("action", nargs="?", default="list", choices=["list", "status", "cancel", "resume"])
    p.add_argument("ids", nargs="*")
    p.add_argument("--status", default=None)
    p.add_argument("--wait", type=float, default=0.0, help="status: wait up to this many seconds")
    p.add_argument("--limit", type=int, default=20)
    common(p)
    p.set_defaults(func=_cmd_jobs)

    p = sub.add_parser("library", help="assets: search [words] / show ID / tag ID / delete ID / collections / pick ID --index N")
    p.add_argument("action", choices=["search", "list", "show", "tag", "delete", "collections", "pick"])
    p.add_argument("words", nargs="*")
    p.add_argument("--kind", default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--collection", default=None)
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--add", action="append", help="tag: tag to add")
    p.add_argument("--remove", action="append", help="tag: tag to remove")
    p.add_argument("--add-collection", action="append")
    p.add_argument("--remove-collection", action="append")
    p.add_argument("--index", type=int, default=1, help="pick: variation number")
    p.add_argument("--job", default=None, help="pick: the job that made a tileset")
    common(p)
    p.set_defaults(func=_cmd_library)

    p = sub.add_parser("project", help="projects: list / create NAME / switch NAME / style [NAME] [--set k=v]")
    p.add_argument("action", nargs="?", default="list", choices=["list", "create", "switch", "style"])
    p.add_argument("name", nargs="?", default=None)
    p.add_argument("--type", default="custom")
    p.add_argument("--size", default="normal")
    p.add_argument("--engine", default="godot")
    p.add_argument("--no-switch", action="store_true")
    p.add_argument("--set", action="append", metavar="FIELD=VALUE", help="style: change a field (repeat)")
    common(p, project=False)
    p.set_defaults(func=_cmd_project)

    p = sub.add_parser("export", help="write assets into a Godot/Unity project + pixel_pipeline_manifest.json")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--godot", metavar="PROJECT", help="Godot project folder (has project.godot)")
    g.add_argument("--unity", metavar="PROJECT", help="Unity project folder (has Assets/)")
    p.add_argument("--subdir", default="", help="folder inside the project (Godot default art, Unity Assets/PixelPipeline)")
    p.add_argument("--assets", nargs="*", help="asset ids (default: every asset)")
    p.add_argument("--collection", default=None)
    p.add_argument("--kind", action="append")
    p.add_argument("--force", action="store_true", help="re-export unchanged assets too")
    p.add_argument("--models", action="store_true", help="also game-ready 3D models")
    p.add_argument("--build-models", action="store_true", help="build missing 3D models (slow)")
    p.add_argument("--verify", action="store_true", help="Godot: import headless + load everything")
    p.add_argument("--preset", default=None)
    common(p)
    p.set_defaults(func=_cmd_export)
