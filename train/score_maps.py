"""Score any map (folder or zip) against the real map of the same song.

    python3 -I train/score_maps.py --reference library/train/official/of-SONG --map path/to/map.zip \
        [--diff Expert --diff ExpertPlus] [--label mapgen]

Uses the same numbers as evaluate.py (timing and colour F1, place, cut,
density, swing errors), so a model's maps, mapgen's maps or anyone's can be
compared on the same songs. Prints one line per difficulty and a JSON line.
"""
import argparse
import gzip
import json
import os
import sys
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import charts  # noqa: E402
import evaluate as E  # noqa: E402
import prepare  # noqa: E402


def read_any(path):
    if path.lower().endswith(".zip"):
        # Charts are read into memory, so the unpacked copy (song audio
        # included, which may be private) is removed straight away.
        with tempfile.TemporaryDirectory(prefix="bx-score-") as tmp:
            prepare.safe_unzip(path, tmp)
            return charts.read_folder(prepare.map_root(tmp))
    return charts.read_folder(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reference", required=True, help="prepared song path without extension")
    ap.add_argument("--map", required=True)
    ap.add_argument("--diff", action="append")
    ap.add_argument("--char", default="Standard")
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    ref = json.load(gzip.open(a.reference + ".json.gz", "rt"))
    gen = read_any(a.map)
    rows = []
    for name in a.diff or ["Expert", "ExpertPlus"]:
        r = next((d for d in ref["difficulties"] if d.get("chart") and d["name"] == name and d["characteristic"] == a.char), None)
        g = next((d for d in gen["difficulties"] if d.get("chart") and d["name"] == name and d["characteristic"] == a.char), None)
        if not r or not g:
            print(f"{name}: missing in {'reference' if not r else 'map'}")
            continue
        real = [(n[0], n[4], n[2], n[3], n[5]) for n in r["chart"]["notes"]]
        made = [(n[0], n[4], n[2], n[3], n[5]) for n in g["chart"]["notes"]]
        s = E.score(real, made, ref["seconds"])
        s.update({"song": ref["title"], "difficulty": name, "label": a.label, "njs": g["njs"], "njsReal": r["njs"]})
        rows.append(s)
        print(f"{a.label or 'map':<10} {ref['title'][:26]:<26} {name:<10} timing F1 {s['timingF1']:.2f}  colour F1 {s['colourF1']:.2f}  "
              f"place {s['place']:.2f}  cut {s['cut']:.2f}  density {s['density']:.2f}  swing errors {s['swingErrors']:.1f}")
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
