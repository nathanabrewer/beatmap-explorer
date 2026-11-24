"""Pipeline checks on a prepared folder: token round trip, chart cache,
train/held-out split by song, batch shapes.

    python3 -I train/tests/test_pipeline.py library/train/official
"""
import glob
import gzip
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data as D  # noqa: E402
import tokens as T  # noqa: E402


def round_trip(folder):
    bad = checked = 0
    for path in sorted(glob.glob(os.path.join(folder, "*.json.gz")))[:20]:
        song = json.load(gzip.open(path, "rt"))
        for d in song["difficulties"]:
            if "chart" not in d:
                continue
            ev = T.chart_events(d["chart"])
            if not ev:
                continue
            start = ev[len(ev) // 2][0] - 300
            toks, sep = T.encode_window(ev, start, d["rank"], d["characteristic"], d["njs"])
            got = sorted((round(o[1] * T.TICKS), o[2], o[3], o[4], o[5]) for o in T.decode(toks[sep + 1:], start) if o[0] == "note")
            want = sorted((e[0], *[int(c) for c in T.NAMES[e[2][0]][5:8]], int(T.NAMES[e[2][1]][4:]))
                          for e in ev if start + T.CONTEXT * T.TICKS <= e[0] < start + T.N_TIME and len(e[2]) == 2)
            checked += 1
            bad += got != want
    return checked, bad


def main():
    folder = sys.argv[1]
    print(f"vocabulary: {T.VOCAB} tokens")
    checked, bad = round_trip(folder)
    print(f"token round trip: {checked} charts, {bad} differ")
    t0 = time.time()
    cache = D.compile_dir(folder, workers=6)
    print(f"chart cache: {os.path.getsize(cache) // 1024} KB in {time.time() - t0:.1f} s")
    tr, te = D.Windows([folder], "train", 0.1), D.Windows([folder], "test", 0.1)
    overlap = {m["title"] for _, m, *_ in tr.items} & {m["title"] for _, m, *_ in te.items}
    print(f"charts: {len(tr)} train, {len(te)} held out ({len({m['key'] for _, m, *_ in te.items})} songs); songs in both: {len(overlap)}")
    mels, inp, tgt, mask = D.collate([tr.sample() for _ in range(8)])
    print(f"batch: mel {tuple(mels.shape)}, tokens {tuple(inp.shape)}, learned {int(mask.sum())}")
    sys.exit(1 if bad or overlap else 0)


if __name__ == "__main__":
    main()
