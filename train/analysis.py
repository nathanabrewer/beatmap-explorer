"""Build the map maker comparison page: every method's maps of the test songs
against the real maps, scored the same way, as one self-contained HTML file.

    python3 -I train/analysis.py SPEC.json OUT.html

SPEC (JSON; paths may start with ~ or be relative to the project):
    title, subtitle            page heading
    reference                  prepared reference folder (prepare.py output)
    testSongs                  the test song list (data.read_test_songs)
    difficulties               e.g. ["Expert", "ExpertPlus"]
    methods: [{id, name, slot (1-8, its fixed colour), note?, and one of:
        eval     evaluate.py --keep-objects output (generated objects)
        maps     a map zip or folder per song, "{id}" = the song key without "of-"
        from + snapMs   another method's maps with notes moved to the nearest
                        audio onset within snapMs (as a clean-up step)
      }]  A method whose files are missing is listed as pending.
    runs: [{name, slot, log (train.py log.jsonl), trainLog (stdout, optional)}]
    speed: [{label, value, unit, note}]  measured generation speeds
    notes: ["...", ...]          caveats shown on the page
    public                     true: a page to publish. No audio paths, no audio.json,
                               no flagging (the page says so); credits from "credits"
    index                      optional: mapkit's index.json, for each song's audio
                               file (the virtual player plays it; the page must
                               stay on the machine that holds the files)

Everything is computed here with evaluate.score, so all methods are measured
identically: scores, the timing error of each note against the nearest real
note, the best fixed time shift (a lag in a method's clock shows up there),
onset-snapped variants, note density over time, and the notes themselves with
whether each one landed within 30 ms of a real note.
"""
import glob
import gzip
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import data as D  # noqa: E402
import evaluate as E  # noqa: E402
import prepare as P  # noqa: E402
import score_maps as S  # noqa: E402

OFFSET_BINS = list(range(-150, 151, 10))
SHIFTS = list(range(-100, 101, 5))
DENSITY_BIN = 2.0
WINDOW = 0.03


def path(p):
    p = os.path.expanduser(p)
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def onsets_for(base, cache={}):
    if base not in cache:
        import librosa
        mel = P.mel_float(P.read_mel(base + ".mel.npy"))
        env = librosa.onset.onset_strength(S=(mel.T * 4.343).astype(np.float32), sr=P.SAMPLE_RATE, hop_length=P.HOP)
        cache[base] = librosa.onset.onset_detect(onset_envelope=env, sr=P.SAMPLE_RATE, hop_length=P.HOP, units="time")
    return cache[base]


def snap(notes, onsets, window):
    if len(onsets) == 0:
        return notes
    t = np.array([n[0] for n in notes])
    j = np.clip(np.searchsorted(onsets, t), 1, len(onsets) - 1)
    near = np.where(np.abs(onsets[j] - t) < np.abs(onsets[j - 1] - t), onsets[j], onsets[j - 1])
    return [((float(o),) + tuple(n[1:])) if abs(o - n[0]) <= window else n for n, o in zip(notes, near)]


def nearest_offsets(real_times, gen_times):
    """Milliseconds from each generated note to the nearest real one (+ = the real note is later)."""
    r = np.array(real_times)
    g = np.array(gen_times)
    if len(r) < 2 or len(g) == 0:
        return np.zeros(0)
    j = np.clip(np.searchsorted(r, g), 1, len(r) - 1)
    near = np.where(np.abs(r[j] - g) < np.abs(r[j - 1] - g), r[j], r[j - 1])
    return (near - g) * 1000.0


def density(times, seconds):
    n = int(math.ceil(seconds / DENSITY_BIN))
    h = np.histogram(times, bins=n, range=(0, n * DENSITY_BIN))[0] / DENSITY_BIN
    return [round(float(v), 2) for v in h]


def matched_flags(real_times, gen_times):
    r = sorted(set(round(t, 3) for t in real_times))
    hit = set()
    for i, j in E.match(r, sorted(set(round(t, 3) for t in gen_times))):
        hit.add(j)
    gs = sorted(set(round(t, 3) for t in gen_times))
    ok = {gs[j] for j in hit}
    return [1 if round(t, 3) in ok else 0 for t in gen_times]


def best_shift(real, gen, seconds):
    best = (-1.0, 0)
    for s in SHIFTS:
        f = E.score(real, [(n[0] + s / 1000.0,) + tuple(n[1:]) for n in gen], seconds)["timingF1"]
        if f > best[0]:
            best = (f, s)
    return best


def map_row(method, key, diff, real, gen, seconds):
    sc = E.score(real, gen, seconds)
    gen = sorted(gen)
    gt = [n[0] for n in gen]
    off = nearest_offsets(sorted(n[0] for n in real), gt)
    hist = np.histogram(off, bins=OFFSET_BINS)[0]
    shift_f1, shift = best_shift(real, gen, seconds)
    flags = matched_flags([n[0] for n in real], gt)
    keep = ("timingF1", "timingPrecision", "timingRecall", "colourF1", "place", "cut", "density", "swingErrors")
    return {"m": method, "s": key, "d": diff, "sc": {k: round(float(sc[k]), 4) for k in keep},
            "off": [int(v) for v in hist], "offMedian": round(float(np.median(off)), 1) if len(off) else None,
            "shift": shift, "shiftF1": round(shift_f1, 4), "dens": density(gt, seconds),
            "notes": [[int(round(n[0] * 100)), int(n[1]), int(n[2]), int(n[3]), int(n[4]), f] for n, f in zip(gen, flags)]}


def read_eval(p):
    """{(song key, difficulty): notes} from evaluate.py --keep-objects (or the snap test) output."""
    out = {}
    for r in json.load(open(p)).get("results", []):
        if "objects" not in r:
            continue
        notes = [(o["tick"] / 100.0, o["c"], o["x"], o["y"], o["d"]) for o in r["objects"] if o["kind"] == "note"]
        out[(r.get("key") or r["song"], r["difficulty"])] = (notes, r.get("generateSeconds"))
    return out


def read_maps(pattern, key, diffs):
    p = path(pattern.replace("{id}", key[3:] if key.startswith("of-") else key))
    if not os.path.exists(p):
        return {}
    song = S.read_any(p)
    out = {}
    for d in song["difficulties"]:
        if d.get("chart") and d["characteristic"] == "Standard" and d["name"] in diffs:
            out[(key, d["name"])] = ([(n[0], n[4], n[2], n[3], n[5]) for n in d["chart"]["notes"]], None)
    return out


def read_run(run):
    rows = []
    log = path(run["log"])
    if os.path.exists(log):
        rows = [json.loads(line) for line in open(log) if line.strip()]
    batch, rate = None, []
    tl = path(run["trainLog"]) if run.get("trainLog") else None
    if tl and os.path.exists(tl):
        for line in open(tl):
            if line.startswith("batch ") and line.split()[1].isdigit():
                batch = int(line.split()[1])
            elif line.startswith("step ") and "windows/s" in line:
                parts = line.split()
                rate.append((int(parts[1]), float(parts[-2])))
    if not rows:
        return {"name": run["name"], "slot": run["slot"], "points": [], "batch": batch, "pending": True}
    # Minutes of training at each evaluation: from the clock, with the time
    # before the first evaluation estimated from the steps per second.
    t0 = rows[0]["time"]
    if len(rows) > 1:
        per_step = (rows[-1]["time"] - t0) / max(rows[-1]["step"] - rows[0]["step"], 1)
    else:
        per_step = 0.0
    start = t0 - rows[0]["step"] * per_step
    pts = [{"step": r["step"], "minutes": round((r["time"] - start) / 60.0, 2), "heldout": round(r["heldout"], 4),
            "njs": round(r.get("njsAccuracy", 0.0), 3), "examples": (r["step"] * batch) if batch else None} for r in rows]
    return {"name": run["name"], "slot": run["slot"], "points": pts, "batch": batch,
            "rate": rate[-1][1] if rate else None, "pending": False}


def main():
    spec_path, out_path = sys.argv[1], sys.argv[2]
    spec = json.load(open(spec_path))
    ref_dir = path(spec["reference"])
    test = D.read_test_songs(path(spec["testSongs"]))
    diffs = spec.get("difficulties", ["Expert", "ExpertPlus"])
    songs, reals = [], {}
    for f in sorted(glob.glob(os.path.join(ref_dir, "*.json.gz"))):
        s = json.load(gzip.open(f, "rt"))
        if D.title_key(s.get("title", "")) not in test:
            continue
        key = os.path.basename(f)[:-len(".json.gz")]
        songs.append({"key": key, "title": s["title"], "artist": s.get("artist", ""), "seconds": s["seconds"], "mapper": s.get("mapper", "")})
        for d in s["difficulties"]:
            if d.get("chart") and d["characteristic"] == "Standard" and d["name"] in diffs:
                notes = sorted((n[0], n[4], n[2], n[3], n[5]) for n in d["chart"]["notes"])
                reals[(key, d["name"])] = notes
    if spec.get("index") and not spec.get("public") and os.path.exists(path(spec["index"])):
        audio = {e.get("id"): e.get("audio", "") for e in json.load(open(path(spec["index"]))).get("songs", [])}
        for s in songs:
            a = audio.get(s["key"][3:] if s["key"].startswith("of-") else s["key"], "")
            if a and os.path.exists(a):
                s["audio"] = "file://" + a
    by_key = {s["key"]: s for s in songs}
    title_to_key = {s["title"]: s["key"] for s in songs}
    methods, rows, gen_sources = [], [], {}
    for m in spec["methods"]:
        src = {}
        if m.get("eval") and os.path.exists(path(m["eval"])):
            for (k, d), v in read_eval(path(m["eval"])).items():
                k = title_to_key.get(k, k)
                if k in by_key:
                    src[(k, d)] = v
        elif m.get("maps"):
            for s in songs:
                src.update(read_maps(m["maps"], s["key"], diffs))
        elif m.get("from") in gen_sources:
            for (k, d), (notes, secs) in gen_sources[m["from"]].items():
                src[(k, d)] = (snap(notes, onsets_for(os.path.join(ref_dir, k)), m.get("snapMs", 80) / 1000.0), secs)
        gen_sources[m["id"]] = src
        count = 0
        for (k, d), (notes, secs) in sorted(src.items()):
            if (k, d) not in reals or not notes:
                continue
            row = map_row(m["id"], k, d, reals[(k, d)], notes, by_key[k]["seconds"])
            row["genSeconds"] = secs
            rows.append(row)
            count += 1
        methods.append({k: m[k] for k in ("id", "name", "slot", "note", "from", "about", "external", "tileNote") if k in m} | {"maps": count, "pending": count == 0})
        print(f"{m['name']}: {count} maps", flush=True)
    # Onset snapping at several windows, for each method that writes its own maps.
    snapping = []
    for m in spec["methods"]:
        if m.get("from") or not gen_sources.get(m["id"]):
            continue
        for (k, d), (notes, _) in sorted(gen_sources[m["id"]].items()):
            if (k, d) not in reals:
                continue
            real, secs = reals[(k, d)], by_key[k]["seconds"]
            entry = {"m": m["id"], "s": k, "d": d}
            for w in (0, 30, 50, 80):
                g = snap(notes, onsets_for(os.path.join(ref_dir, k)), w / 1000.0) if w else notes
                sc = E.score(real, g, secs)
                entry[f"t{w}"] = round(sc["timingF1"], 4)
                entry[f"c{w}"] = round(sc["colourF1"], 4)
            snapping.append(entry)
    real_out = {f"{k}|{d}": {"notes": [[int(round(n[0] * 100)), int(n[1]), int(n[2]), int(n[3]), int(n[4])] for n in notes],
                             "dens": density([n[0] for n in notes], by_key[k]["seconds"]),
                             "swingErrors": round(E.swing_errors([(n[0], n[1], n[4]) for n in notes]), 2)}
                for (k, d), notes in reals.items()}
    page = {"title": spec.get("title", "Map makers compared"), "subtitle": spec.get("subtitle", ""), "built": spec.get("built", ""),
            "difficulties": diffs, "songs": songs, "methods": methods, "rows": rows, "real": real_out, "snapping": snapping,
            "runs": [read_run(r) for r in spec.get("runs", [])], "speed": spec.get("speed", []), "notes": spec.get("notes", []),
            "realAbout": spec.get("realAbout", {}), "public": bool(spec.get("public")), "credits": spec.get("credits", []), "songsLabel": spec.get("songsLabel", ""),
            "offsetBins": OFFSET_BINS, "densityBin": DENSITY_BIN, "shifts": SHIFTS}
    blob = json.dumps(page, separators=(",", ":")).replace("</", "<\\/")
    html = open(os.path.join(HERE, "analysis.html")).read().replace('"__DATA__"', blob)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    open(out_path, "w").write(html)
    # For report_server.py: the song files it may serve, by song key.
    audio = {s["key"]: s["audio"][len("file://"):] for s in songs if s.get("audio")}
    if audio and not spec.get("public"):
        json.dump(audio, open(os.path.join(os.path.dirname(os.path.abspath(out_path)), "audio.json"), "w"), indent=1)
    print(f"wrote {out_path}: {len(songs)} songs, {len(rows)} maps, {len(html) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
