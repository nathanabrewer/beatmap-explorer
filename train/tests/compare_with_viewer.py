"""Check that train/charts.py reads maps exactly as the viewer does.

    python3 -I train/tests/compare_with_viewer.py [--index N] [MAP_FOLDER ...]

For each difficulty of each map folder (and of the first N extracted songs in
index.json), both readers run on the same files; every object's time (to
1 microsecond), beat, place, colour, direction and rotation must agree.
Needs Godot (found like run.sh does: $GODOT or "godot" in config.json).
"""
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))
import charts  # noqa: E402


def godot():
    if os.environ.get("GODOT"):
        return os.environ["GODOT"]
    cfg = os.path.join(ROOT, "config.json")
    if os.path.exists(cfg):
        g = json.load(open(cfg)).get("godot", "")
        if g:
            return os.path.expanduser(g)
    return "godot"


def viewer(d, song):
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "dump.json")
        subprocess.run([godot(), "--headless", "--path", os.path.join(ROOT, "viewer"), "--script", os.path.join(HERE, "viewer_dump.gd"), "--",
                        d["beatmap"], d["lightshow"] or "", song["audioData"] or "", str(song["bpm"] or 120.0), out],
                       capture_output=True, check=True, timeout=120)
        return json.load(open(out))


def same(a, b):
    if len(a) != len(b):
        return f"{len(a)} vs {len(b)}"
    # Times are rounded before sorting: two readers can differ in the last
    # digits of a float, which would otherwise reorder simultaneous rows.
    key = lambda row: [round(float(v), 6) for v in row]
    for i, (x, y) in enumerate(zip(sorted(a, key=key), sorted(b, key=key))):
        for p, q in zip(x, y):
            if abs(float(p) - float(q)) > 1e-6:
                return f"row {i}: {x} vs {y}"
    return ""


def compare(song, label):
    problems = 0
    for d in song["difficulties"]:
        if "chart" not in d:
            print(f"  skip {label} {d['characteristic']} {d['name']}: {d.get('error')}")
            continue
        c, v = d["chart"], viewer(d, song)
        checks = {"notes": (c["notes"], v["notes"]), "bombs": (c["bombs"], v["bombs"]), "walls": (c["walls"], v["walls"]),
                  "arcs": ([r[:4] for r in c["arcs"]], v["arcs"]), "chains": ([r[:4] + [r[11]] for r in c["chains"]], v["chains"]),
                  "lights": (c["lights"], v["lights"]), "boxes": (c["boxes"], v["boxes"])}
        bad = {k: same(*pair) for k, pair in checks.items()}
        bad = {k: m for k, m in bad.items() if m}
        tag = f"{label} {d['characteristic']} {d['name']} (v{c['version']})"
        counts = ", ".join(f"{len(c[k])} {k}" for k in ("notes", "bombs", "walls", "arcs", "chains", "lights", "boxes") if c[k])
        if bad:
            problems += 1
            print(f"  DIFF {tag}: {bad}")
        else:
            print(f"  same {tag}: {counts}")
    return problems


def main():
    args = sys.argv[1:]
    n_index = 0
    if args[:1] == ["--index"]:
        n_index, args = int(args[1]), args[2:]
    problems = 0
    for folder in args:
        song = charts.read_folder(folder)
        problems += compare(song, os.path.basename(folder.rstrip("/")) or folder)
    if n_index:
        index = json.load(open(os.path.join(ROOT, "index.json")))
        for entry in index["songs"][:n_index]:
            problems += compare(charts.read_extracted(entry), entry["id"])
    print("all difficulties agree" if not problems else f"{problems} difficulties differ")
    sys.exit(1 if problems else 0)


main()
