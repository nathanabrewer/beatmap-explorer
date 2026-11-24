"""Maps as token sequences, for an audio-to-map transformer.

A training example is one window of WINDOW seconds of a song. Its first
CONTEXT seconds are given (the notes already placed there, so the model keeps
the flow going); the model writes the rest:

    BOS DIFF_<rank> CHAR_<characteristic> NJS_<speed> ROT_<lane rotation>
    <events in the context part> SEP <events in the rest> EOS

An event is a TIME token (10 ms steps from the window start, shared by
everything at that moment) followed by its objects:
    note   NOTE_<colour><lane><row>  DIR_<cut direction>
    bomb   BOMB_<lane><row>
    wall   WALLX_<lane><width>  WALLY_<row><height>  WDUR_<bucket>
    turn   ROT_<rotation>        (when the lane rotation changes, 360/90 maps)
Note speed (NJS) is predicted as a token, so speed is learned per difficulty.
Lights are a separate model (the "lighter"), not in this vocabulary.
"""
import math

WINDOW = 8.0          # seconds of audio per example
CONTEXT = 2.0         # seconds at the start whose events are given
TICKS = 100           # time steps per second (10 ms, the spectrogram's frame rate)
CHARACTERISTICS = ["Standard", "OneSaber", "NoArrows", "360Degree", "90Degree", "Legacy"]
NJS_RANGE = range(5, 31)
WALL_DURATIONS = [round(0.05 * (1.2 ** i), 4) for i in range(32)]   # 0.05 s .. ~14 s, log-spaced

SPECIALS = ["PAD", "BOS", "EOS", "SEP"]


def _build():
    names = list(SPECIALS)
    names += [f"DIFF_{r}" for r in range(5)]
    names += [f"CHAR_{c}" for c in CHARACTERISTICS]
    names += [f"NJS_{n}" for n in NJS_RANGE]
    names += [f"ROT_{k}" for k in range(24)]
    names += [f"TIME_{t}" for t in range(int(WINDOW * TICKS))]
    names += [f"NOTE_{c}{x}{y}" for c in range(2) for x in range(4) for y in range(3)]
    names += [f"DIR_{d}" for d in range(9)]
    names += [f"BOMB_{x}{y}" for x in range(4) for y in range(3)]
    names += [f"WALLX_{x}{w}" for x in range(4) for w in range(1, 5) if x + w <= 4]
    names += [f"WALLY_{y}{h}" for y in range(3) for h in range(1, 6)]
    names += [f"WDUR_{i}" for i in range(len(WALL_DURATIONS))]
    return names


NAMES = _build()
ID = {n: i for i, n in enumerate(NAMES)}
VOCAB = len(NAMES)
PAD, BOS, EOS, SEP = (ID[s] for s in SPECIALS)
TIME0 = ID["TIME_0"]
N_TIME = int(WINDOW * TICKS)


def is_time(tok):
    return TIME0 <= tok < TIME0 + N_TIME


def rot_bucket(degrees):
    return int(round(degrees / 15.0)) % 24


def wall_bucket(seconds):
    return min(range(len(WALL_DURATIONS)), key=lambda i: abs(math.log(max(seconds, 0.01)) - math.log(WALL_DURATIONS[i])))


def chart_events(chart):
    """A chart's objects as (tick, order, tokens, rotation) in song time,
    10 ms ticks; objects off the standard 4 x 3 grid are left out."""
    ev = []
    for t, b, x, y, c, d, a, r in chart["notes"]:
        xi, yi = int(round(x)), int(round(y))
        if x != xi or y != yi or not (0 <= xi < 4 and 0 <= yi < 3 and c in (0, 1) and 0 <= d <= 8):
            continue
        ev.append((int(round(t * TICKS)), 3, (ID[f"NOTE_{c}{xi}{yi}"], ID[f"DIR_{d}"]), r))
    for t, b, x, y, r in chart["bombs"]:
        xi, yi = int(round(x)), int(round(y))
        if x == xi and y == yi and 0 <= xi < 4 and 0 <= yi < 3:
            ev.append((int(round(t * TICKS)), 2, (ID[f"BOMB_{xi}{yi}"],), r))
    for t, b, x, y, w, h, t_end, beats, r in chart["walls"]:
        xi, wi, yi, hi = int(round(x)), int(round(w)), int(round(y)), int(round(h))
        if f"WALLX_{xi}{wi}" in ID and f"WALLY_{yi}{hi}" in ID and t_end > t:
            ev.append((int(round(t * TICKS)), 1, (ID[f"WALLX_{xi}{wi}"], ID[f"WALLY_{yi}{hi}"], ID[f"WDUR_{wall_bucket(t_end - t)}"]), r))
    ev.sort(key=lambda e: (e[0], e[1], e[2]))
    return ev


def header(rank, characteristic, njs, rotation=0.0):
    char = characteristic if characteristic in CHARACTERISTICS else "Standard"
    n = min(max(int(round(njs)), NJS_RANGE.start), NJS_RANGE.stop - 1)
    return [BOS, ID[f"DIFF_{rank}"], ID[f"CHAR_{char}"], ID[f"NJS_{n}"], ID[f"ROT_{rot_bucket(rotation)}"]]


def encode_window(events, start_tick, rank, characteristic, njs):
    """Token ids for the window starting at `start_tick`, and the index of SEP
    (everything after it is what the model learns to write)."""
    end_tick = start_tick + N_TIME
    ctx_end = start_tick + int(CONTEXT * TICKS)
    inside = [e for e in events if start_tick <= e[0] < end_tick]
    before = [e for e in events if e[0] < start_tick]
    rot = inside[0][3] if inside else (before[-1][3] if before else 0.0)
    toks = header(rank, characteristic, njs, rot)
    sep_at = None
    last_tick = None
    for tick, _, obj, r in inside:
        if sep_at is None and tick >= ctx_end:
            toks.append(SEP)
            sep_at = len(toks) - 1
            last_tick = None
        if rot_bucket(r) != rot_bucket(rot):
            toks.append(ID[f"ROT_{rot_bucket(r)}"])
            rot = r
        if tick != last_tick:
            toks.append(TIME0 + tick - start_tick)
            last_tick = tick
        toks.extend(obj)
    if sep_at is None:
        toks.append(SEP)
        sep_at = len(toks) - 1
    toks.append(EOS)
    return toks, sep_at


def decode(tokens, start_tick):
    """Generated tokens (after SEP) back to objects in song time:
    [("note", t, c, x, y, d, rot) | ("bomb", t, x, y, rot) | ("wall", t, x, w, y, h, dur, rot)]."""
    out, tick, rot, i = [], None, None, 0
    while i < len(tokens):
        name = NAMES[tokens[i]]
        if name == "EOS":
            break
        if name.startswith("TIME_"):
            tick = start_tick + int(name[5:])
        elif name.startswith("ROT_"):
            rot = int(name[4:]) * 15
        elif tick is not None and name.startswith("NOTE_") and i + 1 < len(tokens) and NAMES[tokens[i + 1]].startswith("DIR_"):
            c, x, y = (int(ch) for ch in name[5:8])
            out.append(("note", tick / TICKS, c, x, y, int(NAMES[tokens[i + 1]][4:]), rot))
            i += 1
        elif tick is not None and name.startswith("BOMB_"):
            out.append(("bomb", tick / TICKS, int(name[5]), int(name[6]), rot))
        elif tick is not None and name.startswith("WALLX_") and i + 2 < len(tokens):
            ny, nd = NAMES[tokens[i + 1]], NAMES[tokens[i + 2]]
            if ny.startswith("WALLY_") and nd.startswith("WDUR_"):
                out.append(("wall", tick / TICKS, int(name[6]), int(name[7]), int(ny[6]), int(ny[7]), WALL_DURATIONS[int(nd[5:])], rot))
                i += 2
        i += 1
    return out
