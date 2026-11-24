"""Turn maps into training material: charts plus the song's spectrogram.

    python3 -I train/prepare.py beatsaver [--manifest M] [--out DIR] [--workers 4] [--limit N]
    python3 -I train/prepare.py official  [--index index.json] [--out DIR]
    python3 -I train/prepare.py folder MAP_FOLDER... [--out DIR]

For each song it writes, into --out:
    <key>.json.gz   the song (title, artist, mapper, BPM, source) and every
                    difficulty with its chart, as read by train/charts.py
    <key>.mel.npy   log-mel spectrogram, float16, shape (frames, 96): 24 kHz
                    mono, 100 frames per second (10 ms), 20 Hz to 12 kHz
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
    song["features"] = {"sampleRate": SAMPLE_RATE, "hop": HOP, "nFft": N_FFT, "nMels": N_MELS, "fmin": FMIN, "fmax": FMAX, "frames": int(feats.shape[0])}
    for d in song["difficulties"]:
        for k in ("beatmap", "lightshow"):
            d[k] = os.path.basename(d.get(k) or "")
    for k in ("audio", "audioData", "folder"):
        song.pop(k, None)
    np.save(os.path.join(out, key + ".mel.npy"), feats)
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


def run_pool(jobs, out, workers):
    failed = open(os.path.join(out, "failed.jsonl"), "a")
    ok = skipped = bad = charts_done = 0
    start = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, arg, out): label for fn, arg, label in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            label = futures[fut]
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
            if i % 25 == 0 or i == len(futures):
                rate = i / max(time.time() - start, 1e-6)
                print(f"{i}/{len(futures)}: {ok} prepared ({charts_done} charts), {skipped} already there, {bad} failed; {rate:.2f} songs/s", flush=True)
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
    a = ap.parse_args()
    out = a.out or os.path.join(ROOT, "library", "train", {"beatsaver": "beatsaver", "official": "official", "folder": "maps"}[a.source])
    os.makedirs(out, exist_ok=True)
    if a.source == "beatsaver":
        recs = [json.loads(line) for line in open(a.manifest)]
        jobs = [(beatsaver_one, r, r["id"]) for r in recs]
    elif a.source == "official":
        jobs = [(official_one, e, e["id"]) for e in json.load(open(a.index))["songs"]]
    else:
        jobs = [(folder_one, f, f) for f in a.folders]
    if a.limit:
        jobs = jobs[:a.limit]
    ok, skipped, bad = run_pool(jobs, out, a.workers)
    print(f"done: {ok} prepared, {skipped} already there, {bad} failed -> {out}")


if __name__ == "__main__":
    main()
