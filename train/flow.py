"""Swing flow: can a player swing this map smoothly, even at speed?

Each hand alternates its swings; the natural next swing goes back the way the
last one came (a 180 degree turn). Official Expert and Expert+ maps (166,890
swings) turn 180 degrees 60 % of the time, 135 degrees 30 %, 90 degrees 9 %
and hardly ever 45 or 0 degrees (a "reset", the hand winding back to swing the
same way again). The sharper the turn, the more time the mappers allow: the
fastest real turns are the limits below (their 2nd percentile gaps). A map
can be fast and hard and still flow; a turn faster than real mappers ever ask
for is a flow break.

    breaks(notes)          the flow breaks of a map
    fix(notes)             the same map with each breaking cut changed to the
                           nearest direction that flows (time, place and hand
                           kept; a post-filter, not the model)
    rate(notes)            flow breaks per 100 swings

Notes are tuples (time s, hand 0/1, lane x, layer y, cut direction 0-8).
Cut directions: 0 up, 1 down, 2 left, 3 right, 4 up-left, 5 up-right,
6 down-left, 7 down-right, 8 any (a dot, which swings whichever way flows).
"""
import math

VEC = {0: (0, 1), 1: (0, -1), 2: (-1, 0), 3: (1, 0), 4: (-1, 1), 5: (1, 1), 6: (-1, -1), 7: (1, -1)}
OPPOSITE = {0: 1, 1: 0, 2: 3, 3: 2, 4: 7, 7: 4, 5: 6, 6: 5}
# The shortest gap (s) official maps leave before a hand turns by this angle.
MIN_GAP = {180: 0.11, 135: 0.12, 90: 0.18, 45: 0.24, 0: 0.43}
STACK = 0.03   # notes of one hand closer than this are one swing


def turn(a, b):
    (x1, y1), (x2, y2) = VEC[a], VEC[b]
    c = (x1 * x2 + y1 * y2) / (math.hypot(x1, y1) * math.hypot(x2, y2))
    return int(round(math.degrees(math.acos(max(-1.0, min(1.0, c)))) / 45.0)) * 45


def swings(notes, hand):
    """[(time, [note indices], direction or None for all dots)] for one hand."""
    out = []
    for i in sorted((i for i, n in enumerate(notes) if n[1] == hand), key=lambda i: notes[i][0]):
        t = notes[i][0]
        if out and t - out[-1][0] < STACK:
            out[-1][1].append(i)
        else:
            out.append([t, [i]])
    result = []
    for t, idx in out:
        dirs = [notes[i][4] for i in idx if notes[i][4] != 8]
        result.append((t, idx, dirs[0] if dirs else None, len(set(dirs)) > 1))
    return result


def breaks(notes):
    """[(time, hand, kind, detail)]: kind is "turn" (too sharp for the time),
    "split" (one swing, two directions) or "crossed" (right hand left of the
    left hand at the same moment; reported, real maps do it on purpose)."""
    out = []
    for hand in (0, 1):
        prev = None  # (time, direction the hand last swung)
        for t, idx, d, split in swings(notes, hand):
            if split:
                out.append((t, hand, "split", ""))
            if d is None:
                if prev is not None:
                    prev = (t, OPPOSITE[prev[1]])
                continue
            if prev is not None:
                a, gap = turn(prev[1], d), t - prev[0]
                if gap < MIN_GAP[a]:
                    out.append((t, hand, "turn", f"{a} deg in {gap:.2f} s"))
            prev = (t, d)
    by_t = {}
    for n in notes:
        by_t.setdefault(round(n[0], 2), {}).setdefault(n[1], []).append(n[2])
    for t, v in by_t.items():
        if 0 in v and 1 in v and max(v[1]) < min(v[0]):
            out.append((t, 1, "crossed", ""))
    return sorted(out)


def rate(notes, kinds=("turn", "split")):
    n = len(swings(notes, 0)) + len(swings(notes, 1))
    return 100.0 * sum(1 for b in breaks(notes) if b[2] in kinds) / max(n, 1)


def fix(notes):
    """Change each breaking cut to the direction closest to it that flows
    from the hand's last swing in the time available; one swing's notes all
    take the same direction. Returns a new list in the same order."""
    notes = [tuple(n) for n in notes]
    for hand in (0, 1):
        prev = None
        for t, idx, d, split in swings(notes, hand):
            if d is None:
                if prev is not None:
                    prev = (t, OPPOSITE[prev[1]])
                continue
            want = d
            if prev is not None:
                gap = t - prev[0]
                ok = [c for c in VEC if gap >= MIN_GAP[turn(prev[1], c)]]
                if d not in ok and ok:
                    want = min(ok, key=lambda c: (turn(d, c), -turn(prev[1], c)))
            for i in idx:
                if notes[i][4] != 8 or split:
                    notes[i] = notes[i][:4] + (want,)
            prev = (t, want)
    return notes
