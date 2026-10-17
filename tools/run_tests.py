"""Run the test suite as several pytest processes at once (no pytest-xdist needed).

    python tools/run_tests.py                 # DEFAULT_JOBS processes, the whole suite
    python tools/run_tests.py --jobs 6 -- -x  # anything after -- goes to every pytest
    python tools/run_tests.py --split 2 --part 1 -- -v   # CI: run only group 1 of 2, in this process

Test files are spread over the groups by their measured times (tests/file_times.json, longest
first onto the emptiest group); a file missing from it counts DEFAULT_S. Every tests/test_*.py
lands in exactly one group, so nothing is skipped. Each process gets its own --basetemp; the
tests already keep their state in per-test temp folders (conftest: PIXEL_HOME, PIXEL_LIBRARY,
PIXEL_CACHE, the GPU lock under PIXEL_HOME) and the fake servers take free ports, so processes
don't share anything. --update-times rewrites tests/file_times.json from this run's results.
Every pytest runs at idle priority (conftest; children like Godot inherit it), and the default
is few processes, so the laptop stays usable while the tests go.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
TIMES = TESTS / "file_times.json"
DEFAULT_S = 5.0
# at most half the P-cores (Core Ultra 7 155H: 6 P-cores -> 3); 2 on a 4-thread CI runner
DEFAULT_JOBS = max(1, min(3, (os.cpu_count() or 2) // 4))


# children start at idle priority on Windows (conftest lowers it too; this covers startup)
LOW = getattr(subprocess, "IDLE_PRIORITY_CLASS", 0)


def test_files() -> list[str]:
    return sorted(p.name for p in TESTS.glob("test_*.py"))


def file_times() -> dict[str, float]:
    try:
        return {str(k): float(v) for k, v in json.loads(TIMES.read_text()).items()}
    except (OSError, ValueError, AttributeError):
        return {}


def groups(n: int, files: list[str] | None = None, times: dict[str, float] | None = None
           ) -> list[list[str]]:
    """`files` in `n` groups of about equal time (longest first onto the lightest group;
    ties by name, so every machine makes the same groups)."""
    files = test_files() if files is None else files
    times = file_times() if times is None else times
    out: list[list[str]] = [[] for _ in range(max(1, n))]
    load = [0.0] * len(out)
    for f in sorted(files, key=lambda f: (-times.get(f, DEFAULT_S), f)):
        i = min(range(len(out)), key=lambda k: (load[k], k))
        out[i].append(f)
        load[i] += times.get(f, DEFAULT_S)
    return [sorted(g) for g in out]


def junit_file_times(xml: Path) -> dict[str, float]:
    per: dict[str, float] = {}
    for tc in ET.parse(xml).iter("testcase"):
        name = (tc.get("classname") or "").split(".")[-1] + ".py"
        per[name] = per.get(name, 0.0) + float(tc.get("time") or 0)
    return per


def _utf8_out():
    """Test output can hold any character; a cp1252 console can't print them all."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def main(argv: list[str] | None = None) -> int:
    _utf8_out()
    argv = sys.argv[1:] if argv is None else argv
    extra: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jobs", type=int, default=DEFAULT_JOBS,
                    help=f"pytest processes at once (default {DEFAULT_JOBS})")
    ap.add_argument("--split", type=int, help="CI: number of groups")
    ap.add_argument("--part", type=int, help="CI: which group to run here (1-based)")
    ap.add_argument("--list", action="store_true", help="print the groups and exit")
    ap.add_argument("--update-times", action="store_true", help="save this run's file times")
    a = ap.parse_args(argv)
    os.chdir(ROOT)
    base = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider"]

    if a.split:
        part = groups(a.split)[(a.part or 1) - 1]
        print(f"group {a.part or 1} of {a.split}: {' '.join(part)}", flush=True)
        return subprocess.call(base + extra + [f"tests/{f}" for f in part])

    gs = [g for g in groups(a.jobs) if g]
    if a.list:
        times = file_times()
        for i, g in enumerate(gs, 1):
            print(f"{i}: ~{sum(times.get(f, DEFAULT_S) for f in g):.0f} s  {' '.join(g)}")
        return 0
    tmp = Path(tempfile.mkdtemp(prefix="pp-tests-"))
    t0 = time.monotonic()
    procs = []
    for i, g in enumerate(gs, 1):
        log = open(tmp / f"group{i}.log", "w", encoding="utf-8", errors="replace")
        cmd = base + ["-q", "--basetemp", str(tmp / f"base{i}"), "--junitxml",
                      str(tmp / f"group{i}.xml")] + extra + [f"tests/{f}" for f in g]
        procs.append((i, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                          creationflags=LOW), log))
    print(f"{len(procs)} pytest processes, logs in {tmp}", flush=True)
    failed = []
    for i, p, log in procs:
        code = p.wait()
        log.close()
        text = (tmp / f"group{i}.log").read_text(encoding="utf-8", errors="replace")
        last = [ln for ln in text.splitlines() if ln.strip()][-1:] or ["(no output)"]
        print(f"group {i}: exit {code}, {last[0]}  ({time.monotonic() - t0:.0f} s)", flush=True)
        if code not in (0, 5):          # 5 = nothing collected
            failed.append(i)
            print("\n".join(text.splitlines()[-60:]))
    if a.update_times:
        new = file_times()
        for i, _, _ in procs:
            if (tmp / f"group{i}.xml").exists():
                new.update({k: round(v, 1) for k, v in junit_file_times(tmp / f"group{i}.xml").items()})
        TIMES.write_text(json.dumps(dict(sorted(new.items())), indent=1) + "\n")
    print(f"{'FAILED groups ' + str(failed) if failed else 'all groups passed'} in "
          f"{time.monotonic() - t0:.0f} s", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
