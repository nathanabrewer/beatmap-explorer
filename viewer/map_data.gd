extends RefCounted
## One difficulty of a map, read straight from its file and normalised: every
## object keeps its authored beat, grid place, colour, cut direction, angle
## offset and lane rotation, plus its exact time in seconds from the map's own
## tempo data. Reads beatmap versions 2, 3 and 4 (official levels are v4;
## community maps are mostly v2 and v3).

## Lane centres are (x - 1.5) * 0.6 m; notes in the game were measured at
## exactly -0.9, -0.3, 0.3 and 0.9 m (2026-11-16).
const LANE_SPACING := 0.6
## Heights of the three layers at a note's beat time, measured the same way
## (player head at 1.67 m).
const LAYER_Y: Array[float] = [0.835, 1.383, 1.885]
const HEAD_Y := 1.67
## v2 rotation events (types 14 and 15) store an index into these angles.
const V2_ROTATIONS: Array[int] = [-60, -45, -30, -15, 15, 30, 45, 60]

var error := ""
var version := ""
var notes: Array[Dictionary] = []
var bombs: Array[Dictionary] = []
var walls: Array[Dictionary] = []
var arcs: Array[Dictionary] = []
var chains: Array[Dictionary] = []
## Basic light events: {b, t, type, value, f}.
var lights: Array[Dictionary] = []
## Colour boost switches: {t, on}.
var boosts: Array[Dictionary] = []
## Event-box colour cues per light group id: {t, group, color, brightness, transition}.
var box_cues: Array[Dictionary] = []
## Counts of the event-box kinds that move lights rather than colour them.
var box_kinds := {}
var njs_events := 0
var bpm := 120.0
var _regions: Array[Dictionary] = []
var _freq := 44100.0
## Tempo changes written into the map itself (v3 bpmEvents, v2 BPM events):
## {b, t, bpm}. Used only when there is no audio data file.
var _changes: Array[Dictionary] = []


## Seconds into the song of beat `b`, through the audio data's tempo regions
## (sample-exact), or the level BPM when a level has none.
func time_of(b: float) -> float:
	if _regions.is_empty() and not _changes.is_empty():
		var c: Dictionary = _changes[0]
		for change in _changes:
			if change.b <= b:
				c = change
		return c.t + (b - c.b) * 60.0 / c.bpm
	if _regions.is_empty():
		return b * 60.0 / bpm
	var r: Dictionary = _regions[-1]
	for region in _regions:
		if b < region.eb:
			r = region
			break
	var per_beat: float = (r.ei - r.si) / maxf(r.eb - r.sb, 0.000001)
	return (r.si + (b - r.sb) * per_beat) / _freq


func beat_of(t: float) -> float:
	if _regions.is_empty() and not _changes.is_empty():
		var c: Dictionary = _changes[0]
		for change in _changes:
			if change.t <= t:
				c = change
		return c.b + (t - c.t) * c.bpm / 60.0
	if _regions.is_empty():
		return t * bpm / 60.0
	var s := t * _freq
	var r: Dictionary = _regions[-1]
	for region in _regions:
		if s < region.ei:
			r = region
			break
	return r.sb + (s - r.si) * (r.eb - r.sb) / maxf(r.ei - r.si, 1.0)


## Beats from the moment a note lands on the lane to its beat time, as the
## game computes it from note jump speed and start beat offset.
static func half_jump_beats(njs: float, level_bpm: float, offset: float) -> float:
	var beat_s := 60.0 / level_bpm
	var hjd := 4.0
	while njs * beat_s * hjd > 17.999:
		hjd /= 2.0
	return maxf(hjd + offset, 0.25)


static func lane_x(x: float) -> float:
	return (x - 1.5) * LANE_SPACING


static func layer_y(y: float) -> float:
	var i := clampi(int(floor(y)), 0, 1)
	return lerpf(LAYER_Y[i], LAYER_Y[i + 1], y - i)


## Floor-to-top edge of grid row `r` (walls are measured in rows).
static func row_edge(r: float) -> float:
	var spacing := (LAYER_Y[2] - LAYER_Y[0]) / 2.0
	return LAYER_Y[0] - spacing / 2.0 + r * spacing


static func read_json(path: String) -> Variant:
	var bytes := FileAccess.get_file_as_bytes(path)
	if bytes.size() >= 2 and bytes[0] == 0x1f and bytes[1] == 0x8b:
		bytes = bytes.decompress_dynamic(-1, FileAccess.COMPRESSION_GZIP)
	return JSON.parse_string(bytes.get_string_from_utf8())


func load_files(beatmap_path: String, lightshow_path: String, audio_data_path: String, level_bpm: float) -> bool:
	bpm = level_bpm if level_bpm > 0.0 else 120.0
	if audio_data_path != "":
		var audio: Variant = read_json(audio_data_path)
		if audio is Dictionary:
			_freq = float(audio.get("songFrequency", audio.get("_songFrequency", 44100)))
			for r in audio.get("bpmData", []):
				_regions.append({"si": float(r.get("si", 0)), "ei": float(r.get("ei", 0)), "sb": float(r.get("sb", 0)), "eb": float(r.get("eb", 0))})
			# The v2 timing file (BPMInfo.dat) spells the same regions out.
			for r in audio.get("_regions", []):
				_regions.append({"si": float(r.get("_startSampleIndex", 0)), "ei": float(r.get("_endSampleIndex", 0)),
					"sb": float(r.get("_startBeat", 0)), "eb": float(r.get("_endBeat", 0))})
	var data: Variant = read_json(beatmap_path)
	if not data is Dictionary:
		error = "Could not read " + beatmap_path.get_file()
		return false
	version = str(data.get("version", data.get("_version", "")))
	if version.begins_with("4"):
		_read_v4(data)
	elif version.begins_with("3"):
		_read_v3(data)
	elif version.begins_with("2") or data.has("_notes"):
		_read_v2(data)
	else:
		error = "Unsupported beatmap version " + version
		return false
	if lightshow_path != "":
		var show: Variant = read_json(lightshow_path)
		if show is Dictionary:
			_read_lightshow_v4(show)
	for list in [notes, bombs, walls, arcs, chains, lights]:
		for o in list:
			o.t = time_of(o.b)
	for o in walls:
		o.t_end = time_of(o.b + o.d)
	for o in arcs:
		o.tt = time_of(o.tb)
	for o in chains:
		o.tt = time_of(o.tb)
	lights.sort_custom(func(a, b): return a.t < b.t)
	box_cues.sort_custom(func(a, b): return a.t < b.t)
	return true


static func _row(table: Array, i: int) -> Dictionary:
	return table[i] if i >= 0 and i < table.size() and table[i] is Dictionary else {}


static func _n(d: Dictionary, key: String, fallback := 0.0) -> float:
	return float(d.get(key, fallback))


func _read_v4(data: Dictionary) -> void:
	var nd: Array = data.get("colorNotesData", [])
	for raw in data.get("colorNotes", []):
		var m := _row(nd, int(raw.get("i", 0)))
		notes.append({"b": _n(raw, "b"), "r": _n(raw, "r"), "x": _n(m, "x"), "y": _n(m, "y"), "c": int(m.get("c", 0)), "d": int(m.get("d", 0)), "a": _n(m, "a")})
	var bd: Array = data.get("bombNotesData", [])
	for raw in data.get("bombNotes", []):
		var m := _row(bd, int(raw.get("i", 0)))
		bombs.append({"b": _n(raw, "b"), "r": _n(raw, "r"), "x": _n(m, "x"), "y": _n(m, "y")})
	var od: Array = data.get("obstaclesData", [])
	for raw in data.get("obstacles", []):
		var m := _row(od, int(raw.get("i", 0)))
		walls.append({"b": _n(raw, "b"), "r": _n(raw, "r"), "x": _n(m, "x"), "y": _n(m, "y"), "w": _n(m, "w"), "h": _n(m, "h"), "d": _n(m, "d")})
	var ad: Array = data.get("arcsData", [])
	for raw in data.get("arcs", []):
		var h := _row(nd, int(raw.get("hi", 0)))
		var t := _row(nd, int(raw.get("ti", 0)))
		var m := _row(ad, int(raw.get("ai", 0)))
		arcs.append({"b": _n(raw, "hb"), "tb": _n(raw, "tb"), "r": _n(raw, "hr"), "tr": _n(raw, "tr"), "c": int(h.get("c", 0)),
			"hx": _n(h, "x"), "hy": _n(h, "y"), "hd": int(h.get("d", 0)), "tx": _n(t, "x"), "ty": _n(t, "y"), "td": int(t.get("d", 0)),
			"m": _n(m, "m"), "tm": _n(m, "tm"), "a": int(m.get("a", 0))})
	var cd: Array = data.get("chainsData", [])
	for raw in data.get("chains", []):
		var h := _row(nd, int(raw.get("i", 0)))
		var m := _row(cd, int(raw.get("ci", 0)))
		chains.append({"b": _n(raw, "hb"), "tb": _n(raw, "tb"), "r": _n(raw, "hr"), "c": int(h.get("c", 0)), "hx": _n(h, "x"), "hy": _n(h, "y"),
			"d": int(h.get("d", 0)), "a": _n(h, "a"), "tx": _n(m, "tx"), "ty": _n(m, "ty"), "slices": int(m.get("c", 0)), "squish": _n(m, "s")})
	# Objects carry their own lane rotation in v4. A file that only has
	# spawn rotation events gets the running total applied the same way.
	var rotations: Array = data.get("spawnRotations", [])
	if not rotations.is_empty() and not _any_rotated():
		var rd: Array = data.get("spawnRotationsData", [])
		var steps: Array[Dictionary] = []
		for raw in rotations:
			var m := _row(rd, int(raw.get("i", 0)))
			steps.append({"b": _n(raw, "b"), "late": int(m.get("t", 0)) == 1, "r": _n(m, "r")})
		_apply_rotation_steps(steps)
	njs_events = (data.get("njsEvents", []) as Array).size()


func _read_v2(data: Dictionary) -> void:
	var tempo: Array = []
	var custom: Dictionary = data.get("_customData", {}) if data.get("_customData") is Dictionary else {}
	for raw in custom.get("_BPMChanges", custom.get("_bpmChanges", [])):
		tempo.append({"b": _n(raw, "_time"), "bpm": _n(raw, "_BPM", _n(raw, "_bpm"))})
	for raw in data.get("_events", []):
		if int(raw.get("_type", 0)) == 100:
			tempo.append({"b": _n(raw, "_time"), "bpm": _n(raw, "_floatValue")})
	_set_tempo(tempo)
	for raw in data.get("_notes", []):
		var kind := int(raw.get("_type", 0))
		var o := {"b": _n(raw, "_time"), "r": 0.0, "x": _precise(_n(raw, "_lineIndex")), "y": _precise(_n(raw, "_lineLayer"))}
		if kind == 3:
			bombs.append(o)
		elif kind in [0, 1]:
			o.merge({"c": kind, "d": int(raw.get("_cutDirection", 0)), "a": 0.0})
			notes.append(o)
	for raw in data.get("_obstacles", []):
		var crouch := int(raw.get("_type", 0)) == 1
		walls.append({"b": _n(raw, "_time"), "r": 0.0, "x": _precise(_n(raw, "_lineIndex")), "y": 2.0 if crouch else 0.0, "w": _n(raw, "_width", 1), "h": 3.0 if crouch else 5.0, "d": _n(raw, "_duration")})
	for raw in data.get("_sliders", []):
		arcs.append({"b": _n(raw, "_headTime"), "tb": _n(raw, "_tailTime"), "r": 0.0, "tr": 0.0, "c": int(raw.get("_colorType", 0)),
			"hx": _n(raw, "_headLineIndex"), "hy": _n(raw, "_headLineLayer"), "hd": int(raw.get("_headCutDirection", 0)),
			"tx": _n(raw, "_tailLineIndex"), "ty": _n(raw, "_tailLineLayer"), "td": int(raw.get("_tailCutDirection", 0)),
			"m": _n(raw, "_headControlPointLengthMultiplier"), "tm": _n(raw, "_tailControlPointLengthMultiplier"), "a": int(raw.get("_sliderMidAnchorMode", 0))})
	var steps: Array[Dictionary] = []
	for raw in data.get("_events", []):
		var type := int(raw.get("_type", 0))
		if type in [14, 15]:
			var v := int(raw.get("_value", 0))
			if v >= 0 and v < V2_ROTATIONS.size():
				steps.append({"b": _n(raw, "_time"), "late": type == 15, "r": float(V2_ROTATIONS[v])})
		elif type == 5:
			boosts.append({"b": _n(raw, "_time"), "on": int(raw.get("_value", 0)) == 1})
		elif type != 100:
			lights.append({"b": _n(raw, "_time"), "type": type, "value": int(raw.get("_value", 0)), "f": _n(raw, "_floatValue", 1.0)})
	if not steps.is_empty():
		_apply_rotation_steps(steps)
	for o in boosts:
		o.t = time_of(o.b)


func _read_lightshow_v4(show: Dictionary) -> void:
	var bd: Array = show.get("basicEventsData", [])
	for raw in show.get("basicEvents", []):
		var m := _row(bd, int(raw.get("i", 0)))
		# v4 leaves out values that are 0, so a missing brightness means 0.
		lights.append({"b": _n(raw, "b"), "type": int(m.get("t", 0)), "value": int(m.get("i", 0)), "f": _n(m, "f")})
	var cbd: Array = show.get("colorBoostEventsData", [])
	for raw in show.get("colorBoostEvents", []):
		var m := _row(cbd, int(raw.get("i", 0)))
		boosts.append({"t": time_of(_n(raw, "b")), "on": int(m.get("b", 0)) == 1})
	boosts.sort_custom(func(a, b): return a.t < b.t)
	var colors: Array = show.get("lightColorEvents", [])
	for group in show.get("eventBoxGroups", []):
		var kind := int(group.get("t", 0))
		box_kinds[kind] = int(box_kinds.get(kind, 0)) + 1
		if kind != 1:
			continue
		var at := _n(group, "b")
		var id := int(group.get("g", 0))
		for box in group.get("e", []):
			for step in box.get("l", []):
				var c := _row(colors, int(step.get("i", 0)))
				box_cues.append({"t": time_of(at + _n(step, "b")), "group": id, "color": int(c.get("c", 0)),
					"brightness": _n(c, "b"), "transition": int(c.get("p", 0))})


func _read_v3(data: Dictionary) -> void:
	var tempo: Array = []
	for raw in data.get("bpmEvents", []):
		tempo.append({"b": _n(raw, "b"), "bpm": _n(raw, "m")})
	_set_tempo(tempo)
	for raw in data.get("colorNotes", []):
		notes.append({"b": _n(raw, "b"), "r": 0.0, "x": _n(raw, "x"), "y": _n(raw, "y"), "c": int(raw.get("c", 0)), "d": int(raw.get("d", 0)), "a": _n(raw, "a")})
	for raw in data.get("bombNotes", []):
		bombs.append({"b": _n(raw, "b"), "r": 0.0, "x": _n(raw, "x"), "y": _n(raw, "y")})
	for raw in data.get("obstacles", []):
		walls.append({"b": _n(raw, "b"), "r": 0.0, "x": _n(raw, "x"), "y": _n(raw, "y"), "w": _n(raw, "w"), "h": _n(raw, "h"), "d": _n(raw, "d")})
	for raw in data.get("sliders", []):
		arcs.append({"b": _n(raw, "b"), "tb": _n(raw, "tb"), "r": 0.0, "tr": 0.0, "c": int(raw.get("c", 0)),
			"hx": _n(raw, "x"), "hy": _n(raw, "y"), "hd": int(raw.get("d", 0)), "tx": _n(raw, "tx"), "ty": _n(raw, "ty"), "td": int(raw.get("tc", 0)),
			"m": _n(raw, "mu"), "tm": _n(raw, "tmu"), "a": int(raw.get("m", 0))})
	for raw in data.get("burstSliders", []):
		chains.append({"b": _n(raw, "b"), "tb": _n(raw, "tb"), "r": 0.0, "c": int(raw.get("c", 0)), "hx": _n(raw, "x"), "hy": _n(raw, "y"),
			"d": int(raw.get("d", 0)), "a": 0.0, "tx": _n(raw, "tx"), "ty": _n(raw, "ty"), "slices": int(raw.get("sc", 0)), "squish": _n(raw, "s", 1.0)})
	var steps: Array[Dictionary] = []
	for raw in data.get("rotationEvents", []):
		steps.append({"b": _n(raw, "b"), "late": int(raw.get("e", 0)) == 1, "r": _n(raw, "r")})
	if not steps.is_empty():
		_apply_rotation_steps(steps)
	# v3 files write every field, so a missing brightness defaults to full.
	for raw in data.get("basicBeatmapEvents", []):
		lights.append({"b": _n(raw, "b"), "type": int(raw.get("et", 0)), "value": int(raw.get("i", 0)), "f": _n(raw, "f", 1.0)})
	for raw in data.get("colorBoostBeatmapEvents", []):
		boosts.append({"t": time_of(_n(raw, "b")), "on": bool(raw.get("o", false))})
	boosts.sort_custom(func(a, b): return a.t < b.t)
	for group in data.get("lightColorEventBoxGroups", []):
		box_kinds[1] = int(box_kinds.get(1, 0)) + 1
		var at := _n(group, "b")
		var id := int(group.get("g", 0))
		for box in group.get("e", []):
			for step in box.get("e", []):
				box_cues.append({"t": time_of(at + _n(step, "b")), "group": id, "color": int(step.get("c", 0)),
					"brightness": _n(step, "s", 1.0), "transition": int(step.get("i", 0))})
	for pair in [["lightRotationEventBoxGroups", 2], ["lightTranslationEventBoxGroups", 3]]:
		var n: int = (data.get(pair[0], []) as Array).size()
		if n > 0:
			box_kinds[pair[1]] = int(box_kinds.get(pair[1], 0)) + n


## Tempo changes from the map itself, as running {b, t, bpm} points. The
## audio data file, when there is one, is the more exact source and wins.
func _set_tempo(changes: Array) -> void:
	if not _regions.is_empty():
		return
	var list := changes.filter(func(c): return c.bpm > 0.0)
	if list.is_empty():
		return
	list.sort_custom(func(a, b): return a.b < b.b)
	if list[0].b > 0.0:
		list.insert(0, {"b": 0.0, "bpm": bpm})
	var t := 0.0
	for i in list.size():
		if i > 0:
			t += (list[i].b - list[i - 1].b) * 60.0 / list[i - 1].bpm
		_changes.append({"b": float(list[i].b), "t": t, "bpm": float(list[i].bpm)})


## Older maps placed notes between lanes by writing positions as thousandths
## offset by 1000 (1000 = lane 0, 2500 = lane 1.5; negatives mirror).
static func _precise(v: float) -> float:
	if v >= 1000.0:
		return v / 1000.0 - 1.0
	if v <= -1000.0:
		return v / 1000.0 + 1.0
	return v


func _any_rotated() -> bool:
	for list in [notes, bombs, walls]:
		for o in list:
			if o.r != 0.0:
				return true
	return false


## Running lane rotation: an early step applies to objects on its beat, a late
## one only to objects after it.
func _apply_rotation_steps(steps: Array[Dictionary]) -> void:
	steps.sort_custom(func(a, b): return a.b < b.b)
	for list in [notes, bombs, walls, arcs, chains]:
		for o in list:
			var total := 0.0
			for s in steps:
				if s.b < o.b or (s.b == o.b and not s.late):
					total += s.r
				else:
					break
			o.r = total
