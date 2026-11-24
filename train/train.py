"""Train the audio-to-map model.

    python3 -I train/train.py --data library/train/beatsaver [--data more/] --out runs/NAME \
        [--size small] [--steps 100000] [--batch 32] [--lr 3e-4] [--init runs/OTHER/last.pt]

Works on an NVIDIA card (bf16), on Apple silicon (MPS) or the CPU. Every
--eval-every steps it reports the held-out loss (songs the model never sees)
and saves `last.pt`; the best held-out loss is kept as `best.pt`. Progress
goes to stdout and `log.jsonl` in --out. `--init` starts from another
checkpoint (the fine-tune on a smaller, better set). `--ranks 3,4` trains on
some difficulties only (0 Easy .. 4 Expert+), e.g. a specialist fine-tuned
from the shared model.
"""
import argparse
import json
import math
import os
import random
import sys
import threading
import time
import queue

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D  # noqa: E402
import tokens as T  # noqa: E402
from model import MapModel, parameters_m  # noqa: E402


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def batches(ds, batch, seed, workers=4, depth=8):
    """Background threads keep a queue of ready batches (window cutting is
    numpy and file reads, which release the GIL often enough)."""
    q = queue.Queue(maxsize=depth)

    def work(k):
        rng = random.Random(seed * 1000 + k)
        while True:
            q.put(D.collate([ds.sample(rng) for _ in range(batch)]))

    for k in range(workers):
        threading.Thread(target=work, args=(k,), daemon=True).start()
    while True:
        yield q.get()


def mel_stats(ds, n=200):
    rng = random.Random(1)
    xs = np.concatenate([ds.sample(rng)[0] for _ in range(n)])
    xs = xs[xs > D.SILENCE + 1]
    return float(xs.mean()), float(xs.std())


def evaluate(model, ds, dev, n=256, batch=32):
    model.eval()
    rng = random.Random(1234)
    total, count, njs_right, njs_n = 0.0, 0, 0, 0
    with torch.no_grad():
        for _ in range(max(n // batch, 1)):
            mels, inp, tgt, mask = (t.to(dev) for t in D.collate([ds.sample(rng) for _ in range(batch)]))
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                logits = model(mels, inp)
            loss = F.cross_entropy(logits.float()[mask], tgt[mask], reduction="sum")
            total += float(loss)
            count += int(mask.sum())
            njs_right += int((logits[:, 2].argmax(-1) == tgt[:, 2]).sum())
            njs_n += tgt.shape[0]
    model.train()
    return total / max(count, 1), njs_right / max(njs_n, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", default="small")
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--holdout", type=float, default=0.02)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--init")
    ap.add_argument("--ranks", help="only these difficulties, e.g. 3,4 (0 Easy .. 4 Expert+)")
    ap.add_argument("--chars", help="only these characteristics, e.g. Standard,OneSaber")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    dev = device()
    ranks = {int(r) for r in a.ranks.split(",")} if a.ranks else None
    chars = set(a.chars.split(",")) if a.chars else None
    train = D.Windows(a.data, "train", a.holdout, ranks=ranks, chars=chars, seed=a.seed)
    test = D.Windows(a.data, "test", a.holdout, ranks=ranks, chars=chars, seed=a.seed + 1)
    print(f"{len(train)} training charts, {len(test)} held-out charts; device {dev}", flush=True)
    model = MapModel(a.size)
    start_step = 0
    if a.init:
        ck = torch.load(a.init, map_location="cpu")
        model = MapModel(ck["config"]["size"])
        model.load_state_dict(ck["model"])
        print(f"starting from {a.init} (step {ck.get('step')})", flush=True)
    else:
        mean, std = mel_stats(train)
        model.mel_mean.fill_(mean)
        model.mel_std.fill_(std)
    model.to(dev)
    print(f"model {model.config['size']}: {parameters_m(model):.1f} M parameters", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.98), weight_decay=0.01)
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    best = math.inf
    stream = batches(train, a.batch, a.seed, a.workers)
    t0, seen = time.time(), 0
    model.train()
    for step in range(start_step + 1, a.steps + 1):
        lr = a.lr * min(1.0, step / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(step / a.steps, 1.0)))
        for g in opt.param_groups:
            g["lr"] = lr
        mels, inp, tgt, mask = (t.to(dev, non_blocking=True) for t in next(stream))
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            logits = model(mels, inp)
        loss = F.cross_entropy(logits.float()[mask], tgt[mask])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        seen += mels.shape[0]
        if step % 50 == 0:
            rate = seen / (time.time() - t0)
            print(f"step {step} loss {float(loss):.4f} lr {lr:.2e} {rate:.1f} windows/s", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            val, njs_acc = evaluate(model, test, dev) if len(test) else (float("nan"), float("nan"))
            row = {"step": step, "train": float(loss), "heldout": val, "njsAccuracy": njs_acc, "lr": lr, "time": time.time()}
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(f"== step {step}: held-out loss {val:.4f}, note speed right {njs_acc:.0%}", flush=True)
            ck = {"model": model.state_dict(), "config": model.config, "step": step, "heldout": val, "args": vars(a)}
            torch.save(ck, os.path.join(a.out, "last.pt"))
            if val < best:
                best = val
                torch.save(ck, os.path.join(a.out, "best.pt"))
    print(f"done; best held-out loss {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
