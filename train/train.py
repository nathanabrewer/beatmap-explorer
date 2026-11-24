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
        failures = 0
        while True:
            try:
                q.put(D.collate([ds.sample(rng) for _ in range(batch)]))
            except Exception as error:  # noqa: BLE001 - a bad window must not stall training
                failures += 1
                if failures <= 5 or failures % 100 == 0:
                    print(f"batch worker {k}: {type(error).__name__}: {error} ({failures} so far)", flush=True)

    for k in range(workers):
        threading.Thread(target=work, args=(k,), daemon=True).start()
    while True:
        yield q.get()


HEADROOM = 0.8


def fit_batch(model, batch, dev):
    """The largest batch, at most `batch`, whose worst case fits the card
    with room to spare. Batches pad to their longest window, so one batch of
    dense windows can be far bigger than the average; the worst case (every
    window at the token limit) is tried once here, with room for the
    optimizer, instead of being met hours into a run. Merely fitting is not
    enough: a card filled to the brim spends its time freeing and
    re-allocating memory (a 202M model at batch 64 on an 80 GB H100 kept
    the GPU idle most of the time), so the worst case must stay under
    HEADROOM of the card."""
    if dev.type != "cuda":
        return batch
    n_mels, frames, max_len = model.config["n_mels"], model.config["frames"], model.config["max_len"]
    total = torch.cuda.get_device_properties(dev).total_memory
    while True:
        fits = True
        torch.cuda.reset_peak_memory_stats(dev)
        try:
            reserve = [torch.empty_like(p) for p in model.parameters() for _ in range(2)]  # AdamW's two moments
            mels = torch.zeros(batch, frames, n_mels, device=dev)
            inp = torch.full((batch, max_len), T.SEP, dtype=torch.long, device=dev)
            with torch.autocast(dev.type, dtype=torch.bfloat16):
                logits = model(mels, inp)
            F.cross_entropy(logits.float().flatten(0, 1), inp.flatten()).backward()
        except torch.OutOfMemoryError:
            fits = False
        peak = torch.cuda.max_memory_allocated(dev)
        reserve = mels = inp = logits = None
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        roomy = fits and peak <= HEADROOM * total
        print(f"worst-case batch of {batch}: " + (f"{peak / 1e9:.1f} of {total / 1e9:.1f} GB" if fits else "does not fit"), flush=True)
        if roomy or batch <= 4:
            return batch
        print(f"trying {batch // 2} (the worst case must stay under {HEADROOM:.0%} of the card)", flush=True)
        batch //= 2


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
    ap.add_argument("--test-songs", help="a file of song titles held out from training in every source (see data.read_test_songs)")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--init")
    ap.add_argument("--ranks", help="only these difficulties, e.g. 3,4 (0 Easy .. 4 Expert+)")
    ap.add_argument("--chars", help="only these characteristics, e.g. Standard,OneSaber")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-minutes", type=float, default=0, help="stop after this long; the learning-rate schedule follows the clock")
    ap.add_argument("--save-minutes", type=float, default=0, help="also save last.pt this often")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    dev = device()
    ranks = {int(r) for r in a.ranks.split(",")} if a.ranks else None
    chars = set(a.chars.split(",")) if a.chars else None
    test_songs = D.read_test_songs(a.test_songs)
    train = D.Windows(a.data, "train", a.holdout, ranks=ranks, chars=chars, seed=a.seed, test_songs=test_songs)
    # The held-out loss (which picks best.pt) uses the hash-held-out songs
    # only: choosing a checkpoint by its score on the test songs would leak
    # them into the result.
    test = D.Windows(a.data, "test", a.holdout, ranks=ranks, chars=chars, seed=a.seed + 1, drop=test_songs)
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
    a.batch = fit_batch(model, a.batch, dev)
    print(f"batch {a.batch}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.98), weight_decay=0.01)
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    best = math.inf
    stream = batches(train, a.batch, a.seed, a.workers)
    t0, seen, skipped = time.time(), 0, 0
    last_save = t0
    model.train()

    def checkpoint(step, val=None):
        ck = {"model": model.state_dict(), "config": model.config, "step": step, "heldout": val, "args": vars(a)}
        torch.save(ck, os.path.join(a.out, "last.pt"))
        return ck

    for step in range(start_step + 1, a.steps + 1):
        progress = step / a.steps
        if a.max_minutes:
            progress = max(progress, (time.time() - t0) / (a.max_minutes * 60))
        lr = a.lr * min(1.0, step / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
        for g in opt.param_groups:
            g["lr"] = lr
        mels, inp, tgt, mask = (t.to(dev, non_blocking=True) for t in next(stream))
        try:
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                logits = model(mels, inp)
            loss = F.cross_entropy(logits.float()[mask], tgt[mask])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            out_of_memory = False
        except torch.OutOfMemoryError:
            out_of_memory = True
        if out_of_memory:
            # fit_batch() should prevent this; if the card's memory is
            # fragmented, skip the batch rather than end a long run.
            logits = loss = None
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            skipped += 1
            print(f"step {step}: batch did not fit in memory, skipped ({skipped} so far)", flush=True)
            if skipped > 50 and skipped > step // 20:
                raise SystemExit("too many batches do not fit in memory; use a smaller --batch")
            continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        seen += mels.shape[0]
        if step % 50 == 0:
            rate = seen / (time.time() - t0)
            print(f"step {step} loss {float(loss):.4f} lr {lr:.2e} {rate:.1f} windows/s", flush=True)
        out_of_time = a.max_minutes and progress >= 1.0
        if a.save_minutes and time.time() - last_save > a.save_minutes * 60:
            checkpoint(step)
            last_save = time.time()
        if step % a.eval_every == 0 or step == a.steps or out_of_time:
            val, njs_acc = evaluate(model, test, dev) if len(test) else (float("nan"), float("nan"))
            row = {"step": step, "train": float(loss), "heldout": val, "njsAccuracy": njs_acc, "lr": lr, "time": time.time()}
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(f"== step {step}: held-out loss {val:.4f}, note speed right {njs_acc:.0%}", flush=True)
            ck = checkpoint(step, val)
            last_save = time.time()
            if val < best:
                best = val
                torch.save(ck, os.path.join(a.out, "best.pt"))
        if out_of_time:
            print(f"stopped at the {a.max_minutes} minute limit, step {step}", flush=True)
            break
    if not os.path.exists(os.path.join(a.out, "best.pt")) and os.path.exists(os.path.join(a.out, "last.pt")):
        import shutil
        shutil.copyfile(os.path.join(a.out, "last.pt"), os.path.join(a.out, "best.pt"))
    print(f"done; best held-out loss {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
