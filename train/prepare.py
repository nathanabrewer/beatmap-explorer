"""Turn maps into training material: charts plus the song's spectrogram.

    python3 -I train/prepare.py beatsaver [--manifest M] [--out DIR] [--workers 4] [--limit N]
    python3 -I train/prepare.py official  [--index index.json] [--out DIR]
    python3 -I train/prepare.py folder MAP_FOLDER... [--out DIR]

For each song it writes, into --out:
    <key>.json.gz   the song (title, artist, mapper, BPM, source) and every
                    difficulty with its chart, as read by train/charts.py
    <key>.mel.npy   log-mel spectrogram, shape (frames, 96): 24 kHz mono,
                    100 frames per second (10 ms), 20 Hz to 12 kHz; stored as
                    8-bit steps of 0.094 (about 0.4 dB) from MEL_LO to MEL_HI,
                    half the size of float16, so the whole BeatSaver list
                    (about 1,660 hours) fits a 60-100 GB disk. Read it with
                    read_mel(), which also reads older float16 files.
BeatSaver maps are downloaded one at a time per worker into a fresh temporary
folder, unpacked (names that would leave the folder are refused), processed
and deleted, so only the compact result stays on disk. Downloads are treated
as untrusted data: nothing in them is ever run. Resumable; failures are listed
in <out>/failed.jsonl.

Official levels (`official`) come from mapkit's index.json and are for your
own machine only: never share or upload that output.
"""
import argparse
import gzip
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.request
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import charts  # noqa: E402

SAMPLE_RATE = 24000
HOP = 240            # 10 ms
N_FFT = 2048
N_MELS = 96
FMIN, FMAX = 20, 12000
MAX_UNPACKED = 300 * 1024 * 1024
USER_AGENT = "beatmap-explorer-dataset/0.1"
# The stored log-mel range. Silence is log(1e-6) = -13.8 and loud masters
# peak near 9.3, so the 8-bit steps cover every value with a little room.
MEL_LO, MEL_HI = -14.0, 10.0
MEL_STEP = (MEL_HI - MEL_LO) / 255


def quantize(feats):
    import numpy as np
    q = np.rint((np.asarray(feats, dtype=np.float32) - MEL_LO) / MEL_STEP)
    return np.clip(q, 0, 255).astype(np.uint8)


def mel_float(m):
    """A stored spectrogram (or a slice of one) as float32 log-mel values."""
    import numpy as np
    if m.dtype == np.uint8:
        return m.astype(np.float32) * np.float32(MEL_STEP) + np.float32(MEL_LO)
    return np.asarray(m, dtype=np.float32)


def read_mel(path, mmap=False):
    """The stored array as it is on disk (uint8 or float16); pass slices of it
    through mel_float(). With mmap, nothing is read until it is sliced."""
    import numpy as np
    return np.load(path, mmap_mode="r" if mmap else None)


def mel(audio_path):
    import librosa
    import numpy as np
    y, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    m = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0)
    return np.log(m + 1e-6).T.astype(np.float16), len(y) / SAMPLE_RATE


def save(out, key, song, audio_path):
    import numpy as np
    feats, seconds = mel(audio_path)
    song = dict(song)
    song["key"] = key
    song["seconds"] = round(seconds, 3)
    song["features"] = {"sampleRate": SAMPLE_RATE, "hop": HOP, "nFft": N_FFT, "nMels": N_MELS, "fmin": FMIN, "fmax": FMAX, "frames": int(feats.shape[0]),
                        "stored": {"dtype": "uint8", "lo": MEL_LO, "hi": MEL_HI}}
    for d in song["difficulties"]:
        for k in ("beatmap", "lightshow"):
            d[k] = os.path.basename(d.get(k) or "")
    for k in ("audio", "audioData", "folder"):
        song.pop(k, None)
    np.save(os.path.join(out, key + ".mel.npy"), quantize(feats))
    tmp = os.path.join(out, key + ".json.gz.tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(song, f, separators=(",", ":"))
    os.replace(tmp, os.path.join(out, key + ".json.gz"))
    return sum(1 for d in song["difficulties"] if "chart" in d)


def done(out, key):
    return os.path.exists(os.path.join(out, key + ".json.gz")) and os.path.exists(os.path.join(out, key + ".mel.npy"))


def safe_unzip(zip_path, dest):
    with zipfile.ZipFile(zip_path) as z:
        total = 0
        for info in z.infolist():
            name = info.filename
            parts = name.replace("\\", "/").split("/")
            if name.startswith(("/", "\\")) or ".." in parts or ":" in name:
                raise ValueError(f"unsafe name in zip: {name!r}")
            if info.is_dir():
                continue
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"symlink in zip: {name!r}")
            total += info.file_size
            if total > MAX_UNPACKED:
                raise ValueError("zip unpacks to more than 300 MB")
            target = os.path.join(dest, *parts)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with z.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)


def map_root(folder):
    """The folder holding Info.dat (zips sometimes nest one level)."""
    if charts.find_file(folder, "Info.dat"):
        return folder
    for d in sorted(os.listdir(folder)):
        p = os.path.join(folder, d)
        if os.path.isdir(p) and charts.find_file(p, "Info.dat"):
            return p
    raise ValueError("no Info.dat in the zip")


def beatsaver_one(rec, out):
    key = "bs-" + rec["id"]
    if done(out, key):
        return key, "already", 0
    with tempfile.TemporaryDirectory(prefix="bx-map-") as tmp:
        zpath = os.path.join(tmp, "map.zip")
        req = urllib.request.Request(rec["url"], headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=120) as r, open(zpath, "wb") as f:
            shutil.copyfileobj(r, f)
        unpacked = os.path.join(tmp, "map")
        os.makedirs(unpacked)
        safe_unzip(zpath, unpacked)
        os.remove(zpath)
        song = charts.read_folder(map_root(unpacked))
        if not song["audio"]:
            raise ValueError("song file missing")
        wanted = {(d["char"], d["diff"]) for d in rec["diffs"]}
        song["difficulties"] = [d for d in song["difficulties"] if (d["characteristic"], d["name"]) in wanted]
        song.update({"source": "beatsaver", "id": rec["id"], "hash": rec["hash"], "tiers": rec["tiers"], "score": rec["score"],
                     "chromaDiffs": [[d["char"], d["diff"]] for d in rec["diffs"] if d.get("chroma")]})
        n = save(out, key, song, song["audio"])
    return key, "ok", n


def disk_free_gb(path):
    return shutil.disk_usage(path).free / 1e9


def run_pool(jobs, out, workers, max_minutes=0, min_free_gb=0):
    """Run the jobs with at most 2 x workers in flight; with max_minutes,
    stop starting new ones at the deadline and finish those under way. With
    min_free_gb, also stop when the disk gets that full: with the list sorted
    best first, a smaller disk then simply holds fewer, still the best, maps,
    and training keeps room for its cache and checkpoints."""
    failed = open(os.path.join(out, "failed.jsonl"), "a")
    ok = skipped = bad = charts_done = 0
    start = time.time()
    deadline = start + max_minutes * 60 if max_minutes else None
    pending = iter(jobs)
    futures = {}
    i = 0
    disk_full = False
    with ProcessPoolExecutor(max_workers=workers) as pool:
        def room():
            nonlocal disk_full
            if min_free_gb and not disk_full and disk_free_gb(out) < min_free_gb:
                disk_full = True
            return not disk_full

        def top_up():
            while len(futures) < workers * 2 and (deadline is None or time.time() < deadline) and room():
                job = next(pending, None)
                if job is None:
                    return
                fn, arg, label = job
                futures[pool.submit(fn, arg, out)] = label
        top_up()
        while futures:
            fut = next(as_completed(futures))
            label = futures.pop(fut)
            i += 1
            try:
                key, state, n = fut.result()
                if state == "already":
                    skipped += 1
                else:
                    ok += 1
                    charts_done += n
            except Exception as error:  # noqa: BLE001 - one bad map never stops the run
                bad += 1
                failed.write(json.dumps({"map": label, "error": f"{type(error).__name__}: {error}"[:300]}) + "\n")
                failed.flush()
            if i % 25 == 0 or not futures:
                rate = i / max(time.time() - start, 1e-6)
                print(f"{i}/{len(jobs)}: {ok} prepared ({charts_done} charts), {skipped} already there, {bad} failed; {rate:.2f} songs/s", flush=True)
            top_up()
    if disk_full:
        print(f"stopped with {disk_free_gb(out):.1f} GB left on the disk after {i} of {len(jobs)} songs", flush=True)
    elif deadline is not None and time.time() >= deadline:
        print(f"stopped at the {max_minutes} minute limit after {i} of {len(jobs)} songs", flush=True)
    failed.close()
    return ok, skipped, bad


def official_one(entry, out):
    key = "of-" + entry["id"]
    if done(out, key):
        return key, "already", 0
    song = charts.read_extracted(entry)
    if not song["audio"] or not os.path.isfile(song["audio"]):
        raise ValueError("song file missing")
    audio = song["audio"]
    return key, "ok", save(out, key, song, audio)


def folder_one(folder, out):
    key = "map-" + "".join(c if c.isalnum() else "_" for c in os.path.basename(folder.rstrip("/")))[:80]
    if done(out, key):
        return key, "already", 0
    song = charts.read_folder(folder)
    if not song["audio"]:
        raise ValueError("song file missing")
    return key, "ok", save(out, key, song, song["audio"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("source", choices=["beatsaver", "official", "folder"])
    ap.add_argument("folders", nargs="*")
    ap.add_argument("--manifest", default=os.path.join(ROOT, "library", "train", "manifest.jsonl"))
    ap.add_argument("--index", default=os.path.join(ROOT, "index.json"))
    ap.add_argument("--out")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-minutes", type=float, default=0, help="stop starting new songs after this long")
    ap.add_argument("--min-free-gb", type=float, default=0, help="stop starting new songs when the disk has less free than this")
    ap.add_argument("--best-first", action="store_true", help="ranked, then curated, then highest rated first")
    a = ap.parse_args()
    out = a.out or os.path.join(ROOT, "library", "train", {"beatsaver": "beatsaver", "official": "official", "folder": "maps"}[a.source])
    os.makedirs(out, exist_ok=True)
    if a.source == "beatsaver":
        recs = [json.loads(line) for line in open(a.manifest)]
        if a.best_first:
            recs.sort(key=lambda r: (not (r.get("rankedSS") or r.get("rankedBL")), not r.get("curated"), -(r.get("score") or 0), r["id"]))
        jobs = [(beatsaver_one, r, r["id"]) for r in recs]
    elif a.source == "official":
        jobs = [(official_one, e, e["id"]) for e in json.load(open(a.index))["songs"]]
    else:
        jobs = [(folder_one, f, f) for f in a.folders]
    if a.limit:
        jobs = jobs[:a.limit]
    ok, skipped, bad = run_pool(jobs, out, a.workers, a.max_minutes, a.min_free_gb)
    print(f"done: {ok} prepared, {skipped} already there, {bad} failed -> {out}")


if __name__ == "__main__":
    main()
