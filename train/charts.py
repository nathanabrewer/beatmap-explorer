"""Read any Beat Saber map into one normalised form for training.

A map folder (Info.dat v2 or v4 with its difficulty files, beatmap v2, v3 or
v4) or an extracted official level (an entry of mapkit's index.json) becomes a
song with its difficulties. Every object keeps its authored beat and grid
place and gains its exact time in seconds, worked out the same way as the
viewer (viewer/map_data.gd): the audio data file's sample-exact tempo regions
when there is one, else the map's own tempo changes, else the song's BPM.

    from charts import read_folder, read_extracted
    song = read_folder("path/to/map")          # or read_extracted(index_entry)
    chart = song["difficulties"][0]["chart"]    # notes, bombs, walls, arcs, ...

Rows are plain lists so they serialise compactly:
    notes   [t, b, x, y, c, d, a, r]          colour c (0 left, 1 right), cut d (0-8), angle a, lane rotation r
    bombs   [t, b, x, y, r]
    walls   [t, b, x, y, w, h, t_end, beats, r]
    arcs    [t, b, tt, tb, c, hx, hy, hd, tx, ty, td, m, tm, anchor, r]
    chains  [t, b, tt, tb, c, hx, hy, d, a, tx, ty, slices, squish, r]
    lights  [t, b, type, value, f]            basic light events
    boosts  [t, on]
    boxes   [t, group, color, brightness, transition]   event-box colour cues
"""
import gzip
import json
import os

RANK = {"Easy": 0, "Normal": 1, "Hard": 2, "Expert": 3, "ExpertPlus": 4}
DEFAULT_NJS = {0: 10.0, 1: 10.0, 2: 10.0, 3: 12.0, 4: 16.0}
V2_ROTATIONS = [-60, -45, -30, -15, 15, 30, 45, 60]


def load_json(path):
    with open(path, "rb") as f:
        raw = f.read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw.decode("utf-8-sig"))


def find_file(folder, name):
    """A file in `folder` by name, ignoring letter case (maps are made on
    case-insensitive systems)."""
    if not name:
        return ""
    exact = os.path.join(folder, name)
    if os.path.isfile(exact):
        return exact
    want = os.path.basename(name).lower()
    base = os.path.dirname(exact)
    if os.path.isdir(base):
        for f in os.listdir(base):
            if f.lower() == want:
                return os.path.join(base, f)
    return ""


class Tempo:
    """Beat <-> seconds, with the viewer's precedence: audio data regions,
    then tempo changes written in the map, then the song BPM."""

    def __init__(self, bpm):
        self.bpm = bpm if bpm and bpm > 0 else 120.0
        self.regions = []  # (si, ei, sb, eb) in samples and beats
        self.freq = 44100.0
        self.changes = []  # (b, t, bpm)

    def load_audio_data(self, path):
        if not path or not os.path.isfile(path):
            return
        try:
            d = load_json(path)
        except (ValueError, OSError):
            return
        self.freq = float(d.get("songFrequency", d.get("_songFrequency", 44100)) or 44100)
        for r in d.get("bpmData", []):
            self.regions.append((float(r.get("si", 0)), float(r.get("ei", 0)), float(r.get("sb", 0)), float(r.get("eb", 0))))
        for r in d.get("_regions", []):
            self.regions.append((float(r.get("_startSampleIndex", 0)), float(r.get("_endSampleIndex", 0)),
                                 float(r.get("_startBeat", 0)), float(r.get("_endBeat", 0))))

    def set_changes(self, changes):
        """Tempo changes from the map itself, [(beat, bpm)]; ignored when the
        audio data file gives regions."""
        if self.regions:
            return
        pts = sorted((float(b), float(m)) for b, m in changes if m and m > 0)
        if not pts:
            return
        if pts[0][0] > 0:
            pts.insert(0, (0.0, self.bpm))
        t = 0.0
        for i, (b, m) in enumerate(pts):
            if i:
                t += (b - pts[i - 1][0]) * 60.0 / pts[i - 1][1]
            self.changes.append((b, t, m))

    def time_of(self, b):
        if not self.regions and self.changes:
            c = self.changes[0]
            for ch in self.changes:
                if ch[0] <= b:
                    c = ch
            return c[1] + (b - c[0]) * 60.0 / c[2]
        if not self.regions:
            return b * 60.0 / self.bpm
        r = self.regions[-1]
        for reg in self.regions:
            if b < reg[3]:
                r = reg
                break
        per_beat = (r[1] - r[0]) / max(r[3] - r[2], 1e-6)
        return (r[0] + (b - r[2]) * per_beat) / self.freq

    def beat_of(self, t):
        if not self.regions and self.changes:
            c = self.changes[0]
            for ch in self.changes:
                if ch[1] <= t:
                    c = ch
            return c[0] + (t - c[1]) * c[2] / 60.0
        if not self.regions:
            return t * self.bpm / 60.0
        s = t * self.freq
        r = self.regions[-1]
        for reg in self.regions:
            if s < reg[1]:
                r = reg
                break
        return r[2] + (s - r[0]) * (r[3] - r[2]) / max(r[1] - r[0], 1.0)


def _n(d, key, default=0.0):
    v = d.get(key, default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def _row(table, i):
    return table[i] if isinstance(table, list) and 0 <= i < len(table) and isinstance(table[i], dict) else {}


def _precise(v):
    """Older maps placed objects between lanes as thousandths offset by 1000."""
    if v >= 1000:
        return v / 1000.0 - 1.0
    if v <= -1000:
        return v / 1000.0 + 1.0
    return v


def read_chart(beatmap_path, lightshow_path="", audio_data_path="", bpm=120.0):
    """One difficulty file (plus its v4 lightshow) -> chart dict, or raises
    ValueError for an unreadable or unsupported file."""
    tempo = Tempo(bpm)
    tempo.load_audio_data(audio_data_path)
    data = load_json(beatmap_path)
    if not isinstance(data, dict):
        raise ValueError("not a beatmap")
    version = str(data.get("version", data.get("_version", "")))
    o = {"notes": [], "bombs": [], "walls": [], "arcs": [], "chains": [], "lights": [], "boosts": [], "boxes": []}
    kinds = {}
    steps = []  # lane rotation steps (beat, late, degrees)
    if version.startswith("4"):
        _v4(data, o, steps)
    elif version.startswith("3"):
        _v3(data, o, steps, tempo, kinds)
    elif version.startswith("2") or "_notes" in data:
        _v2(data, o, steps, tempo)
    else:
        raise ValueError("unsupported beatmap version " + version)
    if lightshow_path and os.path.isfile(lightshow_path):
        _lightshow_v4(load_json(lightshow_path), o, kinds)
    if steps:
        _apply_rotations(o, steps)
    t = tempo.time_of
    chart = {
        "version": version,
        "notes": sorted([[t(b), b, x, y, c, d, a, r] for b, x, y, c, d, a, r in o["notes"]]),
        "bombs": sorted([[t(b), b, x, y, r] for b, x, y, r in o["bombs"]]),
        "walls": sorted([[t(b), b, x, y, w, h, t(b + dur), dur, r] for b, x, y, w, h, dur, r in o["walls"]]),
        "arcs": sorted([[t(b), b, t(tb), tb] + rest for b, tb, *rest in o["arcs"]]),
        "chains": sorted([[t(b), b, t(tb), tb] + rest for b, tb, *rest in o["chains"]]),
        "lights": sorted([[t(b), b, ty, v, f] for b, ty, v, f in o["lights"]]),
        "boosts": sorted([[t(b), on] for b, on in o["boosts"]]),
        "boxes": sorted([[t(b), g, c, br, tr] for b, g, c, br, tr in o["boxes"]]),
        "boxKinds": kinds,
        "njsEvents": len(data.get("njsEvents", []) or []),
        "tempo": {"bpm": tempo.bpm, "regions": len(tempo.regions), "changes": len(tempo.changes)},
    }
    return chart


def _v4(data, o, steps):
    nd = data.get("colorNotesData", [])
    for raw in data.get("colorNotes", []):
        m = _row(nd, int(raw.get("i", 0)))
        o["notes"].append((_n(raw, "b"), _n(m, "x"), _n(m, "y"), int(m.get("c", 0)), int(m.get("d", 0)), _n(m, "a"), _n(raw, "r")))
    bd = data.get("bombNotesData", [])
    for raw in data.get("bombNotes", []):
        m = _row(bd, int(raw.get("i", 0)))
        o["bombs"].append((_n(raw, "b"), _n(m, "x"), _n(m, "y"), _n(raw, "r")))
    od = data.get("obstaclesData", [])
    for raw in data.get("obstacles", []):
        m = _row(od, int(raw.get("i", 0)))
        o["walls"].append((_n(raw, "b"), _n(m, "x"), _n(m, "y"), _n(m, "w"), _n(m, "h"), _n(m, "d"), _n(raw, "r")))
    ad = data.get("arcsData", [])
    for raw in data.get("arcs", []):
        h, tl, m = _row(nd, int(raw.get("hi", 0))), _row(nd, int(raw.get("ti", 0))), _row(ad, int(raw.get("ai", 0)))
        o["arcs"].append((_n(raw, "hb"), _n(raw, "tb"), int(h.get("c", 0)), _n(h, "x"), _n(h, "y"), int(h.get("d", 0)),
                          _n(tl, "x"), _n(tl, "y"), int(tl.get("d", 0)), _n(m, "m"), _n(m, "tm"), int(m.get("a", 0)), _n(raw, "hr")))
    cd = data.get("chainsData", [])
    for raw in data.get("chains", []):
        h, m = _row(nd, int(raw.get("i", 0))), _row(cd, int(raw.get("ci", 0)))
        o["chains"].append((_n(raw, "hb"), _n(raw, "tb"), int(h.get("c", 0)), _n(h, "x"), _n(h, "y"), int(h.get("d", 0)), _n(h, "a"),
                            _n(m, "tx"), _n(m, "ty"), int(m.get("c", 0)), _n(m, "s"), _n(raw, "hr")))
    rotated = any(row[-1] for key in ("notes", "bombs", "walls") for row in o[key])
    rotations = data.get("spawnRotations", []) or []
    if rotations and not rotated:
        rd = data.get("spawnRotationsData", [])
        for raw in rotations:
            m = _row(rd, int(raw.get("i", 0)))
            steps.append((_n(raw, "b"), int(m.get("t", 0)) == 1, _n(m, "r")))


def _v3(data, o, steps, tempo, kinds):
    tempo.set_changes([(_n(r, "b"), _n(r, "m")) for r in data.get("bpmEvents", [])])
    for r in data.get("colorNotes", []):
        o["notes"].append((_n(r, "b"), _n(r, "x"), _n(r, "y"), int(r.get("c", 0)), int(r.get("d", 0)), _n(r, "a"), 0.0))
    for r in data.get("bombNotes", []):
        o["bombs"].append((_n(r, "b"), _n(r, "x"), _n(r, "y"), 0.0))
    for r in data.get("obstacles", []):
        o["walls"].append((_n(r, "b"), _n(r, "x"), _n(r, "y"), _n(r, "w"), _n(r, "h"), _n(r, "d"), 0.0))
    for r in data.get("sliders", []):
        o["arcs"].append((_n(r, "b"), _n(r, "tb"), int(r.get("c", 0)), _n(r, "x"), _n(r, "y"), int(r.get("d", 0)),
                          _n(r, "tx"), _n(r, "ty"), int(r.get("tc", 0)), _n(r, "mu"), _n(r, "tmu"), int(r.get("m", 0)), 0.0))
    for r in data.get("burstSliders", []):
        o["chains"].append((_n(r, "b"), _n(r, "tb"), int(r.get("c", 0)), _n(r, "x"), _n(r, "y"), int(r.get("d", 0)), 0.0,
                            _n(r, "tx"), _n(r, "ty"), int(r.get("sc", 0)), _n(r, "s", 1.0), 0.0))
    for r in data.get("rotationEvents", []):
        steps.append((_n(r, "b"), int(r.get("e", 0)) == 1, _n(r, "r")))
    # v3 files write every field, so a missing brightness means full.
    for r in data.get("basicBeatmapEvents", []):
        o["lights"].append((_n(r, "b"), int(r.get("et", 0)), int(r.get("i", 0)), _n(r, "f", 1.0)))
    for r in data.get("colorBoostBeatmapEvents", []):
        o["boosts"].append((_n(r, "b"), bool(r.get("o", False))))
    for g in data.get("lightColorEventBoxGroups", []):
        kinds[1] = kinds.get(1, 0) + 1
        for box in g.get("e", []):
            for step in box.get("e", []):
                o["boxes"].append((_n(g, "b") + _n(step, "b"), int(g.get("g", 0)), int(step.get("c", 0)), _n(step, "s", 1.0), int(step.get("i", 0))))
    for key, kind in (("lightRotationEventBoxGroups", 2), ("lightTranslationEventBoxGroups", 3)):
        n = len(data.get(key, []) or [])
        if n:
            kinds[kind] = kinds.get(kind, 0) + n


def _v2(data, o, steps, tempo):
    custom = data.get("_customData") if isinstance(data.get("_customData"), dict) else {}
    changes = [(_n(r, "_time"), _n(r, "_BPM", _n(r, "_bpm"))) for r in custom.get("_BPMChanges", custom.get("_bpmChanges", [])) or []]
    changes += [(_n(r, "_time"), _n(r, "_floatValue")) for r in data.get("_events", []) if int(r.get("_type", 0)) == 100]
    tempo.set_changes(changes)
    for r in data.get("_notes", []):
        kind = int(r.get("_type", 0))
        b, x, y = _n(r, "_time"), _precise(_n(r, "_lineIndex")), _precise(_n(r, "_lineLayer"))
        if kind == 3:
            o["bombs"].append((b, x, y, 0.0))
        elif kind in (0, 1):
            o["notes"].append((b, x, y, kind, int(r.get("_cutDirection", 0)), 0.0, 0.0))
    for r in data.get("_obstacles", []):
        crouch = int(r.get("_type", 0)) == 1
        o["walls"].append((_n(r, "_time"), _precise(_n(r, "_lineIndex")), 2.0 if crouch else 0.0, _n(r, "_width", 1),
                           3.0 if crouch else 5.0, _n(r, "_duration"), 0.0))
    for r in data.get("_sliders", []):
        o["arcs"].append((_n(r, "_headTime"), _n(r, "_tailTime"), int(r.get("_colorType", 0)), _n(r, "_headLineIndex"), _n(r, "_headLineLayer"),
                          int(r.get("_headCutDirection", 0)), _n(r, "_tailLineIndex"), _n(r, "_tailLineLayer"), int(r.get("_tailCutDirection", 0)),
                          _n(r, "_headControlPointLengthMultiplier"), _n(r, "_tailControlPointLengthMultiplier"), int(r.get("_sliderMidAnchorMode", 0)), 0.0))
    for r in data.get("_events", []):
        kind = int(r.get("_type", 0))
        if kind in (14, 15):
            v = int(r.get("_value", 0))
            if 0 <= v < len(V2_ROTATIONS):
                steps.append((_n(r, "_time"), kind == 15, float(V2_ROTATIONS[v])))
        elif kind == 5:
            o["boosts"].append((_n(r, "_time"), int(r.get("_value", 0)) == 1))
        elif kind != 100:
            o["lights"].append((_n(r, "_time"), kind, int(r.get("_value", 0)), _n(r, "_floatValue", 1.0)))


def _lightshow_v4(show, o, kinds):
    bd = show.get("basicEventsData", [])
    for r in show.get("basicEvents", []):
        m = _row(bd, int(r.get("i", 0)))
        # v4 leaves out values that are 0, so a missing brightness means 0.
        o["lights"].append((_n(r, "b"), int(m.get("t", 0)), int(m.get("i", 0)), _n(m, "f")))
    cbd = show.get("colorBoostEventsData", [])
    for r in show.get("colorBoostEvents", []):
        o["boosts"].append((_n(r, "b"), int(_row(cbd, int(r.get("i", 0))).get("b", 0)) == 1))
    colors = show.get("lightColorEvents", [])
    for g in show.get("eventBoxGroups", []):
        kind = int(g.get("t", 0))
        kinds[kind] = kinds.get(kind, 0) + 1
        if kind != 1:
            continue
        for box in g.get("e", []):
            for step in box.get("l", []):
                c = _row(colors, int(step.get("i", 0)))
                o["boxes"].append((_n(g, "b") + _n(step, "b"), int(g.get("g", 0)), int(c.get("c", 0)), _n(c, "b"), int(c.get("p", 0))))


def _apply_rotations(o, steps):
    """Running lane rotation: an early step applies on its beat, a late one
    only after it. Rotation is the last field of each row."""
    steps.sort(key=lambda s: s[0])
    for key in ("notes", "bombs", "walls", "arcs", "chains"):
        rows = []
        for row in o[key]:
            b, total = row[0], 0.0
            for sb, late, r in steps:
                if sb < b or (sb == b and not late):
                    total += r
                else:
                    break
            rows.append(tuple(row[:-1]) + (total,))
        o[key] = rows


# --- songs -----------------------------------------------------------------------

def _diff_entry(characteristic, name, rank, njs, offset, beatmap, lightshow="", env_idx=0, scheme_idx=-1, colors=None):
    speed = float(njs or 0)
    return {"characteristic": characteristic, "name": name, "rank": rank, "njs": speed or DEFAULT_NJS.get(rank, 10.0),
            "njsAuthored": speed, "offset": float(offset or 0), "beatmap": beatmap, "lightshow": lightshow,
            "envIdx": int(env_idx or 0), "schemeIdx": int(scheme_idx if scheme_idx is not None else -1), "colors": colors}


def read_info(folder):
    """A map folder's Info.dat -> song dict with difficulty entries (files
    resolved to paths), without reading the charts yet."""
    info_path = find_file(folder, "Info.dat")
    if not info_path:
        raise ValueError("no Info.dat")
    info = load_json(info_path)
    version = str(info.get("version", info.get("_version", "")))
    diffs = []
    if version.startswith("4"):
        song, audio = info.get("song", {}), info.get("audio", {})
        s = {"title": song.get("title", ""), "sub": song.get("subTitle", ""), "artist": song.get("author", ""),
             "mapper": ", ".join(sorted({m for d in info.get("difficultyBeatmaps", []) for m in (d.get("beatmapAuthors") or {}).get("mappers", [])})),
             "bpm": float(audio.get("bpm", 0) or 0), "audio": find_file(folder, audio.get("songFilename", "")),
             "audioData": find_file(folder, audio.get("audioDataFilename", "")),
             "env": "", "envAll": "", "envs": info.get("environmentNames", [])}
        for d in info.get("difficultyBeatmaps", []):
            name = d.get("difficulty", "")
            diffs.append(_diff_entry(d.get("characteristic", "Standard"), name, RANK.get(name, 0), d.get("noteJumpMovementSpeed"),
                                     d.get("noteJumpStartBeatOffset"), find_file(folder, d.get("beatmapDataFilename", "")),
                                     find_file(folder, d.get("lightshowDataFilename", "")), d.get("environmentNameIdx"), d.get("beatmapColorSchemeIdx")))
    else:
        s = {"title": info.get("_songName", ""), "sub": info.get("_songSubName", ""), "artist": info.get("_songAuthorName", ""),
             "mapper": info.get("_levelAuthorName", ""), "bpm": float(info.get("_beatsPerMinute", 0) or 0),
             "audio": find_file(folder, info.get("_songFilename", "")), "audioData": find_file(folder, "BPMInfo.dat"),
             "env": info.get("_environmentName", ""), "envAll": info.get("_allDirectionsEnvironmentName", ""),
             "envs": info.get("_environmentNames", []) or []}
        for st in info.get("_difficultyBeatmapSets", []):
            for d in st.get("_difficultyBeatmaps", []):
                name = d.get("_difficulty", "")
                custom = d.get("_customData") if isinstance(d.get("_customData"), dict) else {}
                diffs.append(_diff_entry(st.get("_beatmapCharacteristicName", "Standard"), name, RANK.get(name, 0),
                                         d.get("_noteJumpMovementSpeed"), d.get("_noteJumpStartBeatOffset"),
                                         find_file(folder, d.get("_beatmapFilename", "")), "", d.get("_environmentNameIdx"),
                                         d.get("_beatmapColorSchemeIdx"),
                                         {k: custom[k] for k in ("_colorLeft", "_colorRight", "_obstacleColor") if isinstance(custom.get(k), dict)} or None))
    s["infoVersion"] = version
    s["difficulties"] = diffs
    return s


def read_folder(folder, charts=True):
    """A community map folder -> song dict; each difficulty gains `chart`
    (or `error` when its file can't be read)."""
    song = read_info(folder)
    song["source"] = "map"
    song["folder"] = folder
    if charts:
        _read_charts(song)
    return song


def read_extracted(entry, charts=True):
    """An extracted official level (a song entry of mapkit's index.json)."""
    song = {k: entry.get(k, "") for k in ("title", "sub", "artist", "mapper", "pack", "env", "envAll")}
    song.update({"source": "extracted", "id": entry.get("id", ""), "bpm": float(entry.get("bpm") or 0),
                 "audio": entry.get("audio", ""), "audioData": entry.get("audioData", ""), "envs": entry.get("envs", []),
                 "difficulties": []})
    for st in entry.get("sets", []):
        for d in st.get("difficulties", []):
            song["difficulties"].append(_diff_entry(st["characteristic"], d["name"], int(d.get("difficulty", 0)), d.get("njsAuthored"),
                                                    d.get("offset"), d.get("beatmap", ""), d.get("lightshow", ""), d.get("envIdx"), d.get("schemeIdx")))
    if charts:
        _read_charts(song)
    return song


def _read_charts(song):
    for d in song["difficulties"]:
        if not d["beatmap"]:
            d["error"] = "difficulty file missing"
            continue
        try:
            d["chart"] = read_chart(d["beatmap"], d["lightshow"], song["audioData"], song["bpm"])
        except (ValueError, OSError, KeyError, TypeError) as error:
            d["error"] = f"{type(error).__name__}: {error}"[:200]
