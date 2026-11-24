"""The cached decoder writes the same maps as the full re-read, faster.

    python3 -I train/tests/test_decode_cache.py CHECKPOINT SONG_AUDIO [--device mps|cuda|cpu]

1. Logits: for a real window's tokens, every position's logits from the
   cached step-by-step path match decode() over the whole sequence.
2. Maps: greedy (temperature 0) and sampled (same seed) maps of the song
   are identical with and without the cache.
3. Speed: wall time of both paths for one difficulty.
"""
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import generate as G  # noqa: E402
import prepare as P  # noqa: E402
import tokens as T  # noqa: E402
from model import MapModel  # noqa: E402


def maps(model, mel, dev, temperature, cache):
    original = model.generate

    def gen(*a, **k):
        k["cache"] = cache
        return original(*a, **k)

    model.generate = gen
    try:
        t = time.time()
        njs, objs = G.generate_difficulty(model, mel, 4, "Standard", dev, temperature, True, 1)
        return njs, objs, time.time() - t
    finally:
        model.generate = original


def main():
    ckpt, audio = sys.argv[1], sys.argv[2]
    name = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    dev = torch.device(name)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = MapModel(ck["config"]["size"])
    model.load_state_dict(ck["model"])
    model.to(dev).eval()
    mel, _ = P.mel(audio)
    mel = mel.astype(np.float32)

    # 1. logits along a sequence the model wrote itself
    _, objs, _ = maps(model, mel[:1600], dev, 0.0, True)
    events = [(o["tick"], o["order"], o["toks"], o["rot"]) for o in objs if o["tick"] < T.N_TIME]
    toks, _ = T.encode_window(events, 0, 4, "Standard", 16) if events else (T.header(4, "Standard", 16) + [T.SEP, T.EOS], 5)
    x = torch.tensor(G.window_mel(mel, 0), device=dev).unsqueeze(0)
    with torch.no_grad():
        memory = model.encode(x)
        full = model.decode(memory, torch.tensor([toks], device=dev))[0].float()
        layers, audio_kv = [None] * len(model.decoder.layers), model.audio_keys(memory)
        half = len(toks) // 2
        steps = [model.decode_step(torch.tensor([toks[:half]], device=dev), 0, layers, audio_kv)[0].float()]
        for i in range(half, len(toks)):
            steps.append(model.decode_step(torch.tensor([[toks[i]]], device=dev), i, layers, audio_kv)[0].float())
        cached = torch.cat(steps)
    diff = float((full - cached).abs().max())
    same_top = bool((full.argmax(-1) == cached.argmax(-1)).all())
    print(f"logits over {len(toks)} tokens: largest difference {diff:.2e}, same top token everywhere: {same_top}")
    ok = diff < 1e-3 and same_top

    # 2 and 3. whole maps
    for temperature in (0.0, 0.9):
        a = maps(model, mel, dev, temperature, False)
        b = maps(model, mel, dev, temperature, True)
        same = a[0] == b[0] and a[1] == b[1]
        notes = sum(1 for o in a[1] if o["kind"] == "note")
        print(f"temperature {temperature}: {notes} notes, identical maps: {same}; "
              f"without cache {a[2]:.1f} s, with cache {b[2]:.1f} s ({a[2] / max(b[2], 1e-9):.1f}x)")
        ok = ok and same
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
