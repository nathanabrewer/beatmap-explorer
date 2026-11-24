"""Make a map from a song with a trained model.

    python3 -I train/generate.py --ckpt runs/NAME/best.pt --audio song.mp3 --title "Name" --artist "Who" \
        [--diff Expert --diff ExpertPlus] [--char Standard] [--out library/community/generated] \
        [--temperature 0.9] [--seed 1] [--no-rules]
    python3 -I train/generate.py --ckpt ... --prepared library/train/official/of-SONG   (a prepared song)

Writes a map folder (Info.dat v2.1 + one v3.3 file per difficulty + the song
as Ogg) that Beatmap Explorer lists, the game loads and other map tools read.

Generation runs window by window (8 s, overlapping by the 2 s of context),
feeding each window the notes already placed so the flow continues. Token
grammar is always enforced; with rules on (default) a hand also may not swing
the same way twice in under 0.5 s, the same playability rule mapgen uses.
"""
import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tokens as T  # noqa: E402
from model import MapModel  # noqa: E402

NAMES = ["Easy", "Normal", "Hard", "Expert", "ExpertPlus"]
RANK_NUMBER = [1, 3, 5, 7, 9]
UP, DOWN = {0, 4, 5}, {1, 6, 7}
# Playability guard rails (with rules on): the shortest time, in 10 ms ticks,
# between two notes of one hand per difficulty (mapgen's values for Hard and
# up), and between the starts of two walls. Notes at the same moment (stacks)
# are allowed.
HAND_GAP = {0: 60, 1: 45, 2: 40, 3: 25, 4: 20}
WALL_GAP = 100
# Bombs only when both hands are free and at most one every half second, so
# they are not used as filler while a hand waits.
BOMB_GAP = 50


def classes(dev):
    def mask(pred):
        return torch.tensor([pred(n) for n in T.NAMES], device=dev)
    return {
        "time": mask(lambda n: n.startswith("TIME_")), "note": mask(lambda n: n.startswith("NOTE_")),
        "dir": mask(lambda n: n.startswith("DIR_")), "bomb": mask(lambda n: n.startswith("BOMB_")),
        "wallx": mask(lambda n: n.startswith("WALLX_")), "wally": mask(lambda n: n.startswith("WALLY_")),
        "wdur": mask(lambda n: n.startswith("WDUR_")), "rot": mask(lambda n: n.startswith("ROT_")),
        "njs": mask(lambda n: n.startswith("NJS_")), "eos": mask(lambda n: n == "EOS"),
        "note0": mask(lambda n: n.startswith("NOTE_0")), "note1": mask(lambda n: n.startswith("NOTE_1")),
    }


def make_allow(C, sep_len, start_tick, hands, rules, rank=4, lasts=None):
    """Grammar (and optionally playability) for the tokens after SEP.
    `hands` = {colour: (song tick, "up"|"down")} and `lasts` = {colour or
    "wall": song tick} from earlier windows; what this window wrote so far is
    read from the tokens themselves."""
    ctx = int(T.CONTEXT * T.TICKS)
    none = torch.zeros_like(C["eos"])
    gap = HAND_GAP.get(rank, 20)

    def allow(toks):
        body = toks[sep_len:]
        last = T.NAMES[body[-1]] if body else "SEP"
        tick, state, seen = None, dict(hands), dict(lasts or {})
        for i, t in enumerate(body):
            name = T.NAMES[t]
            if T.is_time(t):
                tick = t - T.TIME0
            elif name.startswith("DIR_") and i > 0 and tick is not None:
                c = int(T.NAMES[body[i - 1]][5])
                seen[c] = start_tick + tick
                swing = note_swing(int(name[4:]))
                if swing:
                    state[c] = (start_tick + tick, swing)
            elif name.startswith("WALLX_") and tick is not None:
                seen["wall"] = start_tick + tick
            elif name.startswith("BOMB_") and tick is not None:
                seen["bomb"] = start_tick + tick
        now = start_tick + tick if tick is not None else None

        def hand_free(c):
            return c not in seen or now is None or now == seen[c] or now - seen[c] >= gap

        objects = torch.zeros_like(C["eos"])
        if not rules or (hand_free(0) and hand_free(1) and ("bomb" not in seen or now is None or now == seen["bomb"] or now - seen["bomb"] >= BOMB_GAP)):
            objects |= C["bomb"]
        if not rules or hand_free(0):
            objects |= C["note0"]
        if not rules or hand_free(1):
            objects |= C["note1"]
        if not rules or "wall" not in seen or now is None or now - seen["wall"] >= WALL_GAP:
            objects |= C["wallx"]
        later_time = C["time"].clone()
        floor = max(ctx, (tick + 1) if tick is not None else ctx)
        if rules and 0 in seen and 1 in seen:
            # The next moment comes when at least one hand is free again.
            floor = max(floor, min(seen[0], seen[1]) + gap - start_tick)
        later_time[T.TIME0:T.TIME0 + min(floor, T.N_TIME)] = False
        if last == "SEP":
            return later_time | C["eos"] | C["rot"]
        if last.startswith("ROT_"):
            return later_time | C["eos"] | (objects if tick is not None else none)
        if last.startswith("TIME_"):
            return objects
        if last.startswith("NOTE_"):
            ok = C["dir"].clone()
            c = int(last[5])
            if rules and c in state and tick is not None and start_tick + tick - state[c][0] < 50:
                for d in (UP if state[c][1] == "up" else DOWN):
                    ok[T.ID[f"DIR_{d}"]] = False
            return ok
        if last.startswith("WALLX_"):
            return C["wally"]
        if last.startswith("WALLY_"):
            return C["wdur"]
        return objects | later_time | C["eos"] | C["rot"]
    return allow


def note_swing(d):
    return "up" if d in UP else "down" if d in DOWN else None


def window_mel(mel, start):
    x = np.full((T.N_TIME, mel.shape[1]), float(np.log(1e-6)), dtype=np.float32)
    a, b = max(start, 0), min(start + T.N_TIME, mel.shape[0])
    if b > a:
        x[a - start:b - start] = mel[a:b]
    return x


@torch.no_grad()
def predict_njs(model, mel, rank, char, dev, C):
    memory = model.encode(torch.tensor(window_mel(mel, 0), device=dev).unsqueeze(0))
    header = torch.tensor([T.header(rank, char, 16)[:3]], device=dev)
    logits = model.decode(memory, header)[0, -1].float().cpu()
    return int(T.NAMES[int(logits.masked_fill(~C["njs"], float("-inf")).argmax())][4:])


def generate_difficulty(model, mel, rank, char, dev, temperature, rules, seed):
    torch.manual_seed(seed)
    # The rule masks and the sampling run on the CPU: per token they are a few
    # dozen tiny operations, slower on a GPU than the one copy of the logits.
    C = classes(torch.device("cpu"))
    frames = mel.shape[0]
    njs = predict_njs(model, mel, rank, char, dev, C)
    step = int((T.WINDOW - T.CONTEXT) * T.TICKS)
    ctx = int(T.CONTEXT * T.TICKS)
    objects, hands, rot, lasts = [], {}, 0.0, {}
    start = -ctx
    while start < frames:
        x = window_mel(mel, start)
        context = [o for o in objects if start <= o["tick"] < start + ctx]
        events = [(o["tick"], o["order"], o["toks"], o["rot"]) for o in context]
        if events:
            prefix, sep = T.encode_window(events, start, rank, char, njs)
            prefix = prefix[:-1]   # drop EOS; it ends with SEP
        else:
            prefix, sep = T.header(rank, char, njs, rot) + [T.SEP], 5
        out = model.generate(torch.tensor(x, device=dev), prefix, temperature=temperature,
                             allow=make_allow(C, len(prefix), start, hands, rules, rank, lasts))
        body = out[len(prefix):]
        cur_tick, i = None, 0
        while i < len(body):
            name = T.NAMES[body[i]]
            if name == "EOS":
                break
            if name.startswith("TIME_"):
                cur_tick = start + int(name[5:])
            elif name.startswith("ROT_"):
                rot = int(name[4:]) * 15.0
            elif cur_tick is not None and name.startswith("NOTE_") and i + 1 < len(body):
                d = int(T.NAMES[body[i + 1]][4:])
                c = int(name[5])
                objects.append({"tick": cur_tick, "order": 3, "toks": (body[i], body[i + 1]), "rot": rot, "kind": "note",
                                "c": c, "x": int(name[6]), "y": int(name[7]), "d": d})
                if note_swing(d):
                    hands[c] = (cur_tick, note_swing(d))
                lasts[c] = cur_tick
                i += 1
            elif cur_tick is not None and name.startswith("BOMB_"):
                lasts["bomb"] = cur_tick
                objects.append({"tick": cur_tick, "order": 2, "toks": (body[i],), "rot": rot, "kind": "bomb", "x": int(name[5]), "y": int(name[6])})
            elif cur_tick is not None and name.startswith("WALLX_") and i + 2 < len(body):
                wy, wd = T.NAMES[body[i + 1]], T.NAMES[body[i + 2]]
                lasts["wall"] = cur_tick
                objects.append({"tick": cur_tick, "order": 1, "toks": tuple(body[i:i + 3]), "rot": rot, "kind": "wall", "x": int(name[6]),
                                "w": int(name[7]), "y": int(wy[6]), "h": int(wy[7]), "dur": T.WALL_DURATIONS[int(wd[5:])]})
                i += 2
            i += 1
        start += step
    # Drop repeats: anything twice in the same place at the same moment, and a
    # hand's note in the cell it just used less than 0.15 s before (unplayable).
    seen, unique, last_cell = set(), [], {}
    for o in sorted(objects, key=lambda o: (o["tick"], o["order"])):
        k = (o["tick"], o["kind"], o.get("c"), o.get("x"), o.get("y"))
        if k in seen or not 0 <= o["tick"] < frames:
            continue
        if o["kind"] == "note":
            cell = (o["c"], o["x"], o["y"])
            if cell in last_cell and o["tick"] - last_cell[cell] < 15:
                continue
            last_cell[cell] = o["tick"]
        seen.add(k)
        unique.append(o)
    return njs, unique


def write_map(folder, title, artist, mapper, bpm, audio, sets):
    """sets: {char: [(rank, njs, objects)]} -> Info.dat v2.1 + v3.3 files + song.ogg."""
    os.makedirs(folder, exist_ok=True)
    beat = lambda tick: round(tick / T.TICKS * bpm / 60.0, 4)
    info_sets = []
    for char, diffs in sets.items():
        entries = []
        for rank, njs, objs in diffs:
            name = f"{NAMES[rank]}{char}.dat"
            rot_events, rot = [], 0.0
            for o in objs:
                if o["rot"] != rot:
                    rot_events.append({"b": beat(o["tick"]), "e": 0, "r": ((o["rot"] - rot + 180) % 360) - 180})
                    rot = o["rot"]
            chart = {"version": "3.3.0", "bpmEvents": [], "rotationEvents": rot_events,
                     "colorNotes": [{"b": beat(o["tick"]), "x": o["x"], "y": o["y"], "c": o["c"], "d": o["d"], "a": 0} for o in objs if o["kind"] == "note"],
                     "bombNotes": [{"b": beat(o["tick"]), "x": o["x"], "y": o["y"]} for o in objs if o["kind"] == "bomb"],
                     "obstacles": [{"b": beat(o["tick"]), "x": o["x"], "y": o["y"], "d": round(o["dur"] * bpm / 60.0, 4), "w": o["w"], "h": o["h"]} for o in objs if o["kind"] == "wall"],
                     "sliders": [], "burstSliders": [], "waypoints": [], "basicBeatmapEvents": [], "colorBoostBeatmapEvents": [],
                     "lightColorEventBoxGroups": [], "lightRotationEventBoxGroups": [], "lightTranslationEventBoxGroups": [],
                     "basicEventTypesWithKeywords": {}, "useNormalEventsAsCompatibleEvents": True}
            json.dump(chart, open(os.path.join(folder, name), "w"))
            entries.append({"_difficulty": NAMES[rank], "_difficultyRank": RANK_NUMBER[rank], "_beatmapFilename": name,
                            "_noteJumpMovementSpeed": njs, "_noteJumpStartBeatOffset": 0})
        info_sets.append({"_beatmapCharacteristicName": char, "_difficultyBeatmaps": sorted(entries, key=lambda e: e["_difficultyRank"])})
    song_file = "song.ogg"
    target = os.path.join(folder, song_file)
    if audio.lower().endswith((".ogg", ".egg")):
        shutil.copyfile(audio, target)
    else:
        sys.path.insert(0, os.path.dirname(HERE))
        from mapkit import encode_ogg
        if not encode_ogg(audio, target):
            raise SystemExit("Could not encode the song as Ogg (install sox).")
    info = {"_version": "2.1.0", "_songName": title, "_songSubName": "", "_songAuthorName": artist, "_levelAuthorName": mapper,
            "_beatsPerMinute": bpm, "_songTimeOffset": 0, "_shuffle": 0, "_shufflePeriod": 0.5, "_previewStartTime": 12,
            "_previewDuration": 10, "_songFilename": song_file, "_coverImageFilename": "", "_environmentName": "DefaultEnvironment",
            "_allDirectionsEnvironmentName": "GlassDesertEnvironment", "_difficultyBeatmapSets": info_sets}
    json.dump(info, open(os.path.join(folder, "Info.dat"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--audio")
    ap.add_argument("--prepared", help="a prepared song path without extension (uses its spectrogram and song details)")
    ap.add_argument("--song-audio", help="with --prepared: the song file to put in the map")
    ap.add_argument("--title", default="")
    ap.add_argument("--artist", default="")
    ap.add_argument("--diff", action="append")
    ap.add_argument("--char", default="Standard")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "library", "community", "generated"))
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-rules", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location="cpu")
    model = MapModel(ck["config"]["size"])
    model.load_state_dict(ck["model"])
    model.to(dev).eval()
    if a.prepared:
        from prepare import mel_float, read_mel
        mel = mel_float(read_mel(a.prepared + ".mel.npy"))
        song = json.load(gzip.open(a.prepared + ".json.gz", "rt"))
        title, artist, bpm = a.title or song.get("title", ""), a.artist or song.get("artist", ""), float(song.get("bpm") or 120)
        audio = a.song_audio
        if not audio:
            raise SystemExit("--prepared needs --song-audio (the song file to put in the map)")
    else:
        from prepare import mel as make_mel
        mel, _ = make_mel(a.audio)
        mel = mel.astype(np.float32)
        import librosa
        y, sr = librosa.load(a.audio, sr=22050, mono=True)
        bpm = float(np.round(np.atleast_1d(librosa.beat.beat_track(y=y, sr=sr)[0])[0], 2)) or 120.0
        title, artist, audio = a.title or os.path.splitext(os.path.basename(a.audio))[0], a.artist, a.audio
    diffs = a.diff or ["Hard", "Expert", "ExpertPlus"]
    sets = {a.char: []}
    for name in diffs:
        rank = NAMES.index(name)
        njs, objs = generate_difficulty(model, mel, rank, a.char, dev, a.temperature, not a.no_rules, a.seed + rank)
        notes = sum(1 for o in objs if o["kind"] == "note")
        print(f"{name}: {notes} notes ({notes / max(mel.shape[0] / T.TICKS, 1):.2f}/s), "
              f"{sum(1 for o in objs if o['kind'] == 'bomb')} bombs, {sum(1 for o in objs if o['kind'] == 'wall')} walls, note speed {njs}", flush=True)
        sets[a.char].append((rank, njs, objs))
    run = os.path.basename(os.path.dirname(os.path.abspath(a.ckpt)))
    safe = "".join(c if c.isalnum() else "_" for c in f"{title} {artist}".strip()).strip("_")[:60] or "song"
    folder = os.path.join(a.out, f"{safe}-{run}")
    write_map(folder, f"{title} [{run}]", artist, f"beatmap-explorer model ({run})", round(bpm, 3), audio, sets)
    print(f"map written to {folder}")


if __name__ == "__main__":
    main()
