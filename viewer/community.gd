extends RefCounted
## Community maps: a folder or zip holding an Info.dat (v2 or v4), its
## difficulty files and the song. Each is read into the same song entry that
## mapkit's index writes for extracted levels, so the viewer treats both alike.
## File names stay relative to the entry's `dir`; a zip gets its `dir` when
## it is first opened and unpacked into the cache.

const MapData := preload("res://map_data.gd")
const RANK := {"Easy": 0, "Normal": 1, "Hard": 2, "Expert": 3, "ExpertPlus": 4}
const DEFAULT_NJS := {0: 10.0, 1: 10.0, 2: 10.0, 3: 12.0, 4: 16.0}
const SCHEME_KEYS := {"saberAColor": "saberA", "saberBColor": "saberB", "obstaclesColor": "obstacles",
	"environmentColor0": "env0", "environmentColor1": "env1", "environmentColorW": "envW",
	"environmentColor0Boost": "env0Boost", "environmentColor1Boost": "env1Boost", "environmentColorWBoost": "envWBoost"}
## Per-difficulty colours set by the community song loader's custom data.
const CUSTOM_KEYS := {"_colorLeft": "saberA", "_colorRight": "saberB", "_obstacleColor": "obstacles",
	"_envColorLeft": "env0", "_envColorRight": "env1", "_envColorWhite": "envW",
	"_envColorLeftBoost": "env0Boost", "_envColorRightBoost": "env1Boost"}


## Every map in `root`: the folder itself, zips, and folders holding an
## Info.dat up to two levels down (packs of maps nest one level).
static func scan(root: String) -> Array:
	var out := []
	if not DirAccess.dir_exists_absolute(root):
		return out
	if _info_name(root) != "":
		out.append(read_folder(root))
	else:
		_scan_into(root, out, 2)
	return out.filter(func(e): return not e.is_empty() and not e.has("error"))


static func _scan_into(dir: String, out: Array, depth: int) -> void:
	for f in DirAccess.get_files_at(dir):
		if f.to_lower().ends_with(".zip") and not f.begins_with("."):
			out.append(peek_zip(dir.path_join(f)))
	if depth <= 0:
		return
	for d in DirAccess.get_directories_at(dir):
		if d.begins_with("."):
			continue
		var p := dir.path_join(d)
		if _info_name(p) != "":
			out.append(read_folder(p))
		else:
			_scan_into(p, out, depth - 1)


## A dropped or opened path: a map folder, its Info.dat, or a zip.
static func open_path(path: String) -> Dictionary:
	if path.to_lower().ends_with(".zip"):
		return peek_zip(path)
	if path.get_file().to_lower() == "info.dat":
		path = path.get_base_dir()
	if DirAccess.dir_exists_absolute(path) and _info_name(path) != "":
		return read_folder(path)
	return {"error": "No Info.dat in %s" % path.get_file()}


static func read_folder(dir: String) -> Dictionary:
	var name := _info_name(dir)
	var info: Variant = MapData.read_json(dir.path_join(name))
	if not info is Dictionary:
		return {}
	var entry := parse_info(info)
	if entry.is_empty():
		return {}
	entry.dir = dir
	entry.id = "map:" + dir
	return entry


## Song details from a zip's Info.dat without unpacking the song.
static func peek_zip(path: String) -> Dictionary:
	var zip := ZIPReader.new()
	if zip.open(path) != OK:
		return {"error": "Could not open %s" % path.get_file()}
	for f in zip.get_files():
		if not _safe_name(f):
			zip.close()
			return {"error": "Refused %s: it holds a file that would land outside its folder (%s)" % [path.get_file(), f]}
	var info_path := ""
	for f in zip.get_files():
		if f.get_file().to_lower() == "info.dat" and (info_path == "" or f.length() < info_path.length()):
			info_path = f
	var entry := {}
	if info_path != "":
		var info: Variant = JSON.parse_string(zip.read_file(info_path).get_string_from_utf8())
		if info is Dictionary:
			entry = parse_info(info)
	zip.close()
	if entry.is_empty():
		return {"error": "No readable Info.dat in %s" % path.get_file()}
	entry.zip = path
	entry.zip_root = info_path.get_base_dir()
	entry.dir = ""
	entry.id = "map:" + path
	return entry


## Unpack a zip entry into `cache` (once per zip and modification time) and
## give it its folder. Names that would escape the folder are refused.
static func materialize(entry: Dictionary, cache: String) -> String:
	if entry.get("dir", "") != "" or not entry.has("zip"):
		return ""
	var path: String = entry.zip
	var key := ("%s|%d" % [path, FileAccess.get_modified_time(path)]).md5_text().substr(0, 16)
	var dest := cache.path_join(key)
	var done := dest.path_join(".unpacked")
	if not FileAccess.file_exists(done):
		var zip := ZIPReader.new()
		if zip.open(path) != OK:
			return "Could not open %s" % path.get_file()
		var root: String = entry.zip_root
		for f in zip.get_files():
			if f.ends_with("/") or (root != "" and not f.begins_with(root + "/")):
				continue
			var rel := f.substr(root.length() + 1) if root != "" else f
			if not _safe_name(f) or not _safe_name(rel):
				zip.close()
				return "Refused %s: it holds a file that would land outside its folder" % path.get_file()
			var target := dest.path_join(rel)
			DirAccess.make_dir_recursive_absolute(target.get_base_dir())
			var out := FileAccess.open(target, FileAccess.WRITE)
			if out == null:
				zip.close()
				return "Could not write into the cache folder %s" % cache
			out.store_buffer(zip.read_file(f))
			out.close()
		zip.close()
		FileAccess.open(done, FileAccess.WRITE).store_string(path)
	entry.dir = dest
	return ""


## A zip entry name that stays inside the folder it unpacks into.
static func _safe_name(name: String) -> bool:
	return name != "" and not name.begins_with("/") and not name.is_absolute_path() and not "\\" in name \
		and not ".." in name.split("/") and not ":" in name


static func _info_name(dir: String) -> String:
	for f in DirAccess.get_files_at(dir):
		if f.to_lower() == "info.dat":
			return f
	return ""


static func parse_info(info: Dictionary) -> Dictionary:
	var version := str(info.get("version", info.get("_version", "")))
	if version.begins_with("4"):
		return _info_v4(info)
	if version.begins_with("2") or info.has("_difficultyBeatmapSets"):
		return _info_v2(info)
	return {}


static func _base(title: String, sub: String, artist: String, mapper: String, bpm: float) -> Dictionary:
	return {"source": "map", "levelId": "", "title": title if title != "" else "Untitled", "sub": sub, "artist": artist,
		"mapper": mapper, "pack": "", "bpm": bpm if bpm > 0.0 else 120.0, "env": "", "envAll": "", "envs": [],
		"schemes": [], "audio": "", "audioData": "", "cover": "", "sets": []}


static func _info_v4(info: Dictionary) -> Dictionary:
	var song: Dictionary = info.get("song", {})
	var audio: Dictionary = info.get("audio", {})
	var mappers := {}
	for d in info.get("difficultyBeatmaps", []):
		for m in (d.get("beatmapAuthors", {}) as Dictionary).get("mappers", []):
			mappers[str(m)] = true
	var e := _base(str(song.get("title", "")), str(song.get("subTitle", "")), str(song.get("author", "")),
		", ".join(mappers.keys()), float(audio.get("bpm", 0)))
	e.audio = str(audio.get("songFilename", ""))
	e.audioData = str(audio.get("audioDataFilename", ""))
	e.cover = str(info.get("coverImageFilename", ""))
	e.envs = info.get("environmentNames", [])
	for s in info.get("colorSchemes", []):
		if s is Dictionary:
			var override := bool(s.get("useOverride", s.get("overrideNotes", s.get("overrideLights", true))))
			e.schemes.append({"override": override, "colors": _scheme(s)})
	var sets := {}
	for d in info.get("difficultyBeatmaps", []):
		var rank: int = RANK.get(str(d.get("difficulty", "")), 0)
		var speed := float(d.get("noteJumpMovementSpeed", 0))
		_add_diff(sets, str(d.get("characteristic", "Standard")), {"difficulty": rank, "name": str(d.get("difficulty", "")),
			"beatmap": str(d.get("beatmapDataFilename", "")), "lightshow": str(d.get("lightshowDataFilename", "")),
			"njs": speed if speed > 0.0 else DEFAULT_NJS[rank], "njsAuthored": speed,
			"offset": float(d.get("noteJumpStartBeatOffset", 0)), "envIdx": int(d.get("environmentNameIdx", 0)),
			"schemeIdx": int(d.get("beatmapColorSchemeIdx", -1))})
	e.sets = _ordered(sets)
	return e


static func _info_v2(info: Dictionary) -> Dictionary:
	var e := _base(str(info.get("_songName", "")), str(info.get("_songSubName", "")), str(info.get("_songAuthorName", "")),
		str(info.get("_levelAuthorName", "")), float(info.get("_beatsPerMinute", 0)))
	e.audio = str(info.get("_songFilename", ""))
	e.cover = str(info.get("_coverImageFilename", ""))
	e.env = str(info.get("_environmentName", ""))
	e.envAll = str(info.get("_allDirectionsEnvironmentName", ""))
	e.envs = info.get("_environmentNames", [])
	for s in info.get("_colorSchemes", []):
		if s is Dictionary:
			e.schemes.append({"override": bool(s.get("useOverride", true)), "colors": _scheme(s.get("colorScheme", s))})
	var sets := {}
	for s in info.get("_difficultyBeatmapSets", []):
		var characteristic := str(s.get("_beatmapCharacteristicName", "Standard"))
		for d in s.get("_difficultyBeatmaps", []):
			var rank: int = RANK.get(str(d.get("_difficulty", "")), 0)
			var speed := float(d.get("_noteJumpMovementSpeed", 0))
			var custom: Dictionary = d.get("_customData", {}) if d.get("_customData") is Dictionary else {}
			var label := str(custom.get("_difficultyLabel", ""))
			var entry := {"difficulty": rank, "name": str(d.get("_difficulty", "")) + (" (%s)" % label if label != "" else ""),
				"beatmap": str(d.get("_beatmapFilename", "")), "lightshow": "",
				"njs": speed if speed > 0.0 else DEFAULT_NJS[rank], "njsAuthored": speed,
				"offset": float(d.get("_noteJumpStartBeatOffset", 0)), "envIdx": int(d.get("_environmentNameIdx", 0)),
				"schemeIdx": int(d.get("_beatmapColorSchemeIdx", -1))}
			var colors := {}
			for key in CUSTOM_KEYS:
				if custom.get(key) is Dictionary:
					colors[CUSTOM_KEYS[key]] = _color(custom[key])
			if not colors.is_empty():
				entry.colors = colors
			_add_diff(sets, characteristic, entry)
	e.sets = _ordered(sets)
	return e


static func _add_diff(sets: Dictionary, characteristic: String, d: Dictionary) -> void:
	if not sets.has(characteristic):
		sets[characteristic] = []
	sets[characteristic].append(d)


static func _ordered(sets: Dictionary) -> Array:
	var out := []
	for characteristic in sets:
		var diffs: Array = sets[characteristic]
		diffs.sort_custom(func(a, b): return a.difficulty < b.difficulty)
		out.append({"characteristic": characteristic, "difficulties": diffs})
	return out


static func _scheme(s: Dictionary) -> Dictionary:
	var out := {}
	for key in SCHEME_KEYS:
		if s.has(key):
			out[SCHEME_KEYS[key]] = _color(s[key])
	return out


## A colour given as {r, g, b, a} or as hex "RRGGBBAA", as [r, g, b, a].
static func _color(v: Variant) -> Array:
	if v is Dictionary:
		return [float(v.get("r", 0)), float(v.get("g", 0)), float(v.get("b", 0)), float(v.get("a", 1))]
	if v is String and Color.html_is_valid(v):
		var c := Color.html(v)
		return [c.r, c.g, c.b, c.a]
	return [1.0, 1.0, 1.0, 1.0]
