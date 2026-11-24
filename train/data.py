"""Training windows: a spectrogram slice and its map tokens.

Charts are compiled once per prepared folder into `charts.cache.npz`
(numpy arrays, about 11 bytes per object), so tens of thousands of charts
fit in memory. Each example is a random WINDOW-second window of a random
chart; windows start up to CONTEXT seconds before the song so the opening
is learned too.

Songs are split by a hash of their title and artist (not by map), so the
held-out set has songs the model never heard; `holdout` is the fraction kept
out (2 % by default). A list of test songs (read_test_songs) is held out in
every source on top of that, matched by title alone: the same song often has
maps in several sources with the artist written differently, and a map of a
test song from any source must never be trained on.
"""
import glob
import gzip
import hashlib
import json
import os
import random
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import prepare as P
import tokens as T

SILENCE = float(np.log(1e-6))
FRAMES = int(T.WINDOW * T.TICKS)
CTX = int(T.CONTEXT * T.TICKS)


def title_key(title):
    """A song title reduced to its words: case, punctuation and bracketed
    parts like (feat. X) or [Chroma] ignored."""
    t = re.sub(r"\(.*?\)|\[.*?\]", " ", str(title).lower())
    return " ".join(w for w in re.findall(r"[a-z0-9$]+", t) if w not in ("the", "a", "and", "feat", "ft"))


def read_test_songs(path):
    """Test songs, one title per line (anything after a | is a note)."""
    if not path:
        return None
    keys = {title_key(line.split("|")[0]) for line in open(path) if line.strip() and not line.startswith("#")}
    return keys - {""}


def song_split(song, holdout, test_songs=None):
    if test_songs and title_key(song.get("title", "")) in test_songs:
        return "test"
    key = f"{song.get('title', '').strip().lower()}|{song.get('artist', '').strip().lower()}"
    h = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if h < holdout else "train"


def _compile_file(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        song = json.load(f)
    rows = []
    for d in song["difficulties"]:
        chart = d.get("chart")
        if not chart or d["characteristic"] not in T.CHARACTERISTICS:
            continue
        ev = T.chart_events(chart)
        if len(ev) < 16:
            continue
        ticks = np.array([e[0] for e in ev], dtype=np.int32)
        toks = np.full((len(ev), 3), -1, dtype=np.int16)
        for i, e in enumerate(ev):
            toks[i, :len(e[2])] = e[2]
        rots = np.array([T.rot_bucket(e[3]) for e in ev], dtype=np.int8)
        rows.append({"rank": int(d["rank"]), "char": d["characteristic"], "njs": float(d["njs"]), "ticks": ticks, "toks": toks, "rots": rots})
    return {"key": song["key"], "title": song.get("title", ""), "artist": song.get("artist", ""), "source": song.get("source", ""),
            "frames": song["features"]["frames"], "charts": rows}


def compile_dir(folder, workers=8):
    """Build (or reuse) folder/charts.cache.npz from its *.json.gz songs."""
    cache = os.path.join(folder, "charts.cache.npz")
    files = sorted(glob.glob(os.path.join(folder, "*.json.gz")))
    if os.path.exists(cache) and os.path.getmtime(cache) >= max((os.path.getmtime(f) for f in files), default=0):
        return cache
    with ProcessPoolExecutor(max_workers=workers) as pool:
        songs = list(pool.map(_compile_file, files, chunksize=16))
    meta, ticks, toks, rots, offsets = [], [], [], [], [0]
    for s in songs:
        for c in s["charts"]:
            meta.append({"key": s["key"], "title": s["title"], "artist": s["artist"], "source": s["source"], "frames": s["frames"],
                         "rank": c["rank"], "char": c["char"], "njs": c["njs"]})
            ticks.append(c["ticks"])
            toks.append(c["toks"])
            rots.append(c["rots"])
            offsets.append(offsets[-1] + len(c["ticks"]))
    np.savez(cache, meta=json.dumps(meta), offsets=np.array(offsets, dtype=np.int64),
             ticks=np.concatenate(ticks) if ticks else np.zeros(0, np.int32),
             toks=np.concatenate(toks) if toks else np.zeros((0, 3), np.int16),
             rots=np.concatenate(rots) if rots else np.zeros(0, np.int8))
    return cache


class Windows:
    """Random training windows over the charts of one or more prepared folders."""

    def __init__(self, folders, split="train", holdout=0.02, ranks=None, chars=None, seed=0, test_songs=None, drop=None):
        self.items = []  # (folder, meta, ticks, toks, rots)
        self.mels = {}
        for folder in folders:
            z = np.load(compile_dir(folder))
            # Read each array once: every z[...] access decompresses the whole
            # array again, and per-chart slices of separate copies kept them all.
            meta, off = json.loads(str(z["meta"])), z["offsets"]
            all_ticks, all_toks, all_rots = z["ticks"], z["toks"], z["rots"]
            for i, m in enumerate(meta):
                if drop and title_key(m.get("title", "")) in drop:
                    continue
                if song_split(m, holdout, test_songs) != split:
                    continue
                if (ranks and m["rank"] not in ranks) or (chars and m["char"] not in chars):
                    continue
                a, b = off[i], off[i + 1]
                self.items.append((folder, m, all_ticks[a:b], all_toks[a:b], all_rots[a:b]))
        self.weights = np.array([max(m["frames"], 1) for _, m, *_ in self.items], dtype=np.float64)
        self.cumulative = np.cumsum(self.weights) / max(self.weights.sum(), 1e-9)
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.items)

    def mel(self, folder, key):
        path = os.path.join(folder, key + ".mel.npy")
        if path not in self.mels:
            # Each memory map holds a file descriptor; containers often allow
            # only 1024, so keep a few hundred open at most.
            if len(self.mels) > 256:
                self.mels.clear()
            self.mels[path] = P.read_mel(path, mmap=True)
        return self.mels[path]

    def sample(self, rng=None):
        rng = rng or self.rng
        i = int(np.searchsorted(self.cumulative, rng.random()))
        return self.window(min(i, len(self.items) - 1), None, rng)

    def window(self, i, start=None, rng=None):
        folder, m, ticks, toks, rots = self.items[i]
        last = int(ticks[-1]) if len(ticks) else m["frames"]
        if start is None:
            start = (rng or self.rng).randint(-CTX, max(last - CTX, -CTX + 1))
        lo, hi = np.searchsorted(ticks, start), np.searchsorted(ticks, start + FRAMES)
        events = [(int(ticks[j]), 0, tuple(int(t) for t in toks[j] if t >= 0), float(rots[j]) * 15.0) for j in range(lo, hi)]
        prev_rot = float(rots[lo - 1]) * 15.0 if lo > 0 else 0.0
        if not events:
            # A quiet stretch: the model learns to write nothing here.
            toks_out, sep = T.header(m["rank"], m["char"], m["njs"], prev_rot) + [T.SEP, T.EOS], 5
        else:
            toks_out, sep = T.encode_window(events, start, m["rank"], m["char"], m["njs"])
        mel = self.mel(folder, m["key"])
        x = np.full((FRAMES, mel.shape[1]), SILENCE, dtype=np.float32)
        a, b = max(start, 0), min(start + FRAMES, mel.shape[0])
        if b > a:
            x[a - start:b - start] = P.mel_float(mel[a:b])
        return x, toks_out, sep


def collate(batch, max_len=1024):
    """Pad a list of (mel, tokens, sep) into model inputs, targets and a loss
    mask: the note speed token and everything after SEP are learned."""
    import torch
    n = min(max(len(t) for _, t, _ in batch), max_len + 1)
    mels = torch.tensor(np.stack([m for m, _, _ in batch]))
    inp = torch.full((len(batch), n - 1), T.PAD, dtype=torch.long)
    tgt = torch.full((len(batch), n - 1), T.PAD, dtype=torch.long)
    mask = torch.zeros((len(batch), n - 1), dtype=torch.bool)
    for i, (_, toks, sep) in enumerate(batch):
        toks = toks[:n]
        L = len(toks) - 1
        inp[i, :L] = torch.tensor(toks[:-1])
        tgt[i, :L] = torch.tensor(toks[1:])
        mask[i, 2] = True                    # predicting NJS_ from BOS DIFF CHAR
        mask[i, sep:L] = True                # predicting everything after SEP
    return mels, inp, tgt, mask
