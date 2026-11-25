"""Score a model against real maps it never saw.

    python3 -I train/evaluate.py --ckpt runs/NAME/best.pt --data library/train/official [--holdout 0.1] \
        [--limit 20] [--diff Expert --diff ExpertPlus] [--out runs/NAME/eval.json]

For each held-out song (the same split as training) and difficulty, the model
maps the song and the result is compared with the real map:
    timing F1        notes at the right moments (within 30 ms), colour ignored
    colour F1        the same, per hand
    place            of the notes at the right moment and hand, the share on the right lane and row
    cut              of those, the share with the right cut direction
    density          generated notes per second / real notes per second
    swing errors     per 100 notes: a hand swinging the same way twice within 0.5 s
The same numbers for the real maps' own swing errors show what is normal.
"""
import argparse
import gzip
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import data as D  # noqa: E402
import generate as G  # noqa: E402
import prepare as P  # noqa: E402
from model import MapModel  # noqa: E402

TOL = 0.03


def match(ref, gen, tol=TOL):
    """Greedy one-to-one matching of two sorted time lists -> pairs (i, j)."""
    pairs, j = [], 0
    used = set()
    for i, t in enumerate(ref):
        while j < len(gen) and gen[j] < t - tol:
            j += 1
        k = j
        while k < len(gen) and gen[k] <= t + tol:
            if k not in used:
                pairs.append((i, k))
                used.add(k)
                break
            k += 1
    return pairs


def f1(n_ref, n_gen, n_match):
    p = n_match / n_gen if n_gen else 0.0
    r = n_match / n_ref if n_ref else 0.0
    return (2 * p * r / (p + r) if p + r else 0.0), p, r


def swing_errors(notes):
    """notes: [(t, c, d)] -> errors per 100 notes."""
    last, errors = {}, 0
    for t, c, d in sorted(notes):
        s = G.note_swing(d)
        if s is None:
            continue
        if c in last and t - last[c][0] < 0.5 and last[c][1] == s:
            errors += 1
        last[c] = (t, s)
    return 100.0 * errors / max(len(notes), 1)


def score(ref_notes, gen_notes, seconds):
    rt = sorted({round(n[0], 3) for n in ref_notes})
    gt = sorted({round(n[0], 3) for n in gen_notes})
    timing = f1(len(rt), len(gt), len(match(rt, gt)))
    colour_hits = colour_ref = colour_gen = place_ok = cut_ok = 0
    for c in (0, 1):
        r = sorted((n for n in ref_notes if n[1] == c), key=lambda n: n[0])
        g = sorted((n for n in gen_notes if n[1] == c), key=lambda n: n[0])
        pairs = match([n[0] for n in r], [n[0] for n in g])
        colour_hits += len(pairs)
        colour_ref += len(r)
        colour_gen += len(g)
        for i, j in pairs:
            if (r[i][2], r[i][3]) == (g[j][2], g[j][3]):
                place_ok += 1
            if r[i][4] == g[j][4]:
                cut_ok += 1
    colour = f1(colour_ref, colour_gen, colour_hits)
    return {"timingF1": timing[0], "timingPrecision": timing[1], "timingRecall": timing[2], "colourF1": colour[0],
            "place": place_ok / max(colour_hits, 1), "cut": cut_ok / max(colour_hits, 1),
            "density": (len(gen_notes) / max(len(ref_notes), 1)),
            "swingErrors": swing_errors([(n[0], n[1], n[4]) for n in gen_notes]),
            "realSwingErrors": swing_errors([(n[0], n[1], n[4]) for n in ref_notes]),
            "notesReal": len(ref_notes), "notesGenerated": len(gen_notes), "seconds": seconds}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", action="append", required=True)
    ap.add_argument("--holdout", type=float, default=0.02)
    ap.add_argument("--test-songs", help="score these songs (a file of titles); with --holdout 0, only these")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--diff", action="append")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--no-rules", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--keep-objects", action="store_true", help="also save each generated map's objects (for analysis.py)")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location="cpu")
    model = MapModel(ck["config"]["size"])
    model.load_state_dict(ck["model"])
    model.to(dev).eval()
    test = D.Windows(a.data, "test", a.holdout, test_songs=D.read_test_songs(a.test_songs))
    wanted = {G.NAMES.index(d) for d in a.diff} if a.diff else None
    items = [it for it in test.items if (wanted is None or it[1]["rank"] in wanted) and it[1]["char"] == "Standard"][: a.limit]
    results = []
    for folder, m, *_ in items:
        song = json.load(gzip.open(os.path.join(folder, m["key"] + ".json.gz"), "rt"))
        ref = next(d for d in song["difficulties"] if d.get("chart") and d["rank"] == m["rank"] and d["characteristic"] == m["char"])
        mel = P.mel_float(P.read_mel(os.path.join(folder, m["key"] + ".mel.npy")))
        t0 = time.time()
        njs, objs = G.generate_difficulty(model, mel, m["rank"], m["char"], dev, a.temperature, not a.no_rules, 1)
        seconds = time.time() - t0
        gen = [(o["tick"] / 100.0, o["c"], o["x"], o["y"], o["d"]) for o in objs if o["kind"] == "note"]
        real = [(n[0], n[4], n[2], n[3], n[5]) for n in ref["chart"]["notes"]]
        s = score(real, gen, song["seconds"])
        s.update({"song": m["title"], "key": m["key"], "difficulty": G.NAMES[m["rank"]], "njsPredicted": njs, "njsReal": ref["njs"],
                  "generateSeconds": round(seconds, 2), "songSeconds": song["seconds"], "device": dev.type})
        if a.keep_objects:
            s["objects"] = objs
        results.append(s)
        if a.out:  # saved after every map, so a run cut short still leaves its scores
            json.dump({"ckpt": a.ckpt, "step": ck.get("step"), "results": results}, open(a.out, "w"), indent=1)
        print(f"{m['title'][:28]:<28} {G.NAMES[m['rank']]:<10} timing F1 {s['timingF1']:.2f}  colour F1 {s['colourF1']:.2f}  "
              f"place {s['place']:.2f}  cut {s['cut']:.2f}  density {s['density']:.2f}  swing errors {s['swingErrors']:.1f} "
              f"(real {s['realSwingErrors']:.1f})  speed {njs} (real {ref['njs']:g})", flush=True)
    if results:
        keys = ["timingF1", "colourF1", "place", "cut", "density", "swingErrors", "realSwingErrors"]
        mean = {k: float(np.mean([r[k] for r in results])) for k in keys}
        print("mean: " + "  ".join(f"{k} {v:.2f}" for k, v in mean.items()))
        if a.out:
            json.dump({"ckpt": a.ckpt, "step": ck.get("step"), "mean": mean, "results": results}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
