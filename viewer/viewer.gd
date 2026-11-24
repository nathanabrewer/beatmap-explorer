extends Node3D
## Beatmap Explorer: lays one difficulty of a map out in 3D exactly as
## authored. Every object sits on the measured note grid at distance =
## (its time - now) x note jump speed, so the spacing is what the player meets
## on the lane. Play runs the map toward the player with the song; "As the
## player sees it" hides what is beyond the distance where notes land.
##
## Songs come from mapkit's index.json (levels extracted from your own copy of
## the game) and from community maps (folders or zips) in the folders listed
## in config.json, plus any map dropped on the window or opened.
##
## Flags after `--`: --song=<id> --open=<map folder or zip> --bare (ignore the
## index and configured folders; only what is opened)
## --char=<characteristic> --diff=<name>
## --at=<seconds> or --beat=<beat> --cam=player|orbit|side|top
## --view=whole|player --play
## --snap=<png> (saves the view a few frames in, or --snap-frames=<n>, then
## quits).

const MapData := preload("res://map_data.gd")
const Community := preload("res://community.gd")
const EMPTY_HELP := """No songs yet.

Drop a map folder or .zip onto this window, or click Open.
Keep community maps in library/community to have them listed.
Songs from your own copy of the game on a Quest:
double-click "Get Songs From Quest.command"."""
const NOTE_SIZE := 0.45
## Where the arrow points for each cut direction, as a rotation about the
## player's view axis (0 = down, counter-clockwise positive, as in the game).
const DIR_ANGLE: Array[float] = [180.0, 0.0, -90.0, 90.0, -135.0, 135.0, -45.0, 45.0, 0.0]
const DIR_VEC: Array[Vector2] = [Vector2(0, 1), Vector2(0, -1), Vector2(-1, 0), Vector2(1, 0),
	Vector2(-0.7071, 0.7071), Vector2(0.7071, 0.7071), Vector2(-0.7071, -0.7071), Vector2(0.7071, -0.7071), Vector2.ZERO]
## Arc control-point reach per unit of the authored multiplier. The game's
## own spline is not public; this follows the authored directions and
## multipliers, so the shape is close rather than exact.
const ARC_REACH := 1.5
const LIGHT_NAMES := ["Back lasers", "Ring lights", "Left lasers", "Right lasers", "Center lights"]
const SPEEDS := [0.25, 0.5, 0.75, 1.0]
const DEFAULT_COLORS := {"saberA": [0.7843, 0.0784, 0.0784, 1], "saberB": [0.1569, 0.5569, 0.8235, 1], "obstacles": [1, 0.1882, 0.1882, 1],
	"env0": [0.85, 0.085, 0.085, 1], "env1": [0.1882, 0.6753, 1, 1], "envW": [0.7255, 0.7255, 0.7255, 1]}

var index := {}
var songs: Array = []
var song := {}
var set_entry := {}
var diff := {}
var map: MapData
var palette := {}
var env_name := ""
var njs := 10.0
var lead_s := 1.0
var now := 0.0
var duration := 1.0
var playing := false
var speed := 1.0
var view_mode := "whole"
var cam_mode := "orbit"

var _map_root: Node3D
var _lanes := {}
var _frame: Array[Node3D] = []
## Lane edges of rotated lanes: shown in the overview cameras only.
var _lane_lines: Array[Node3D] = []
var _objects: Array = []
var _light_events := {}
var _light_mats := {}
var _box_groups := {}
var _box_cells := {}
var _mats := {}
var _meshes := {}
var _audio: AudioStreamPlayer
var _cam: Camera3D
var _orbit := {"target": Vector3(0, 1.2, -6), "yaw": -0.55, "pitch": -0.32, "dist": 13.0}
var _look := Vector2.ZERO
## Notes in time order, for turning the player camera toward the lane of
## the next note on rotating maps.
var _turns: Array[Dictionary] = []
var _facing := 0.0
## Floor lane lines that turn with the player camera on rotating maps.
var _facing_lane: Node3D
var _rotating := false
var _drag := 0
var _snap := ""
var _snap_frames := -1

var _list: ItemList
var _search: LineEdit
var _char_opt: OptionButton
var _diff_opt: OptionButton
var _view_opt: OptionButton
var _cam_opt: OptionButton
var _speed_opt: OptionButton
var _play_btn: Button
var _slider: HSlider
var _time_label: Label
var _info: Label
var _board: GridContainer
var _left: PanelContainer
var _shown: Array = []
var _heading: Label
var _cover: TextureRect
var _file_dialog: FileDialog
var _root_dir := ""
var _community_roots: Array = []
var _cache := ""


func _ready() -> void:
	var args := _args()
	_read_settings()
	if args.has("bare"):
		_community_roots.clear()
	var index_path := _root_dir.path_join("index.json")
	var parsed: Variant = MapData.read_json(index_path) if FileAccess.file_exists(index_path) and not args.has("bare") else null
	index = parsed if parsed is Dictionary else {}
	if not index.get("envColors") is Dictionary:
		index.envColors = {}
	songs = index.get("songs", [])
	var found := []
	for root in _community_roots:
		found.append_array(Community.scan(root))
	found.sort_custom(func(a, b): return a.title.to_lower() < b.title.to_lower())
	songs.append_array(found)
	_build_world()
	_build_ui()
	get_window().files_dropped.connect(func(files): _open_paths(Array(files)))
	view_mode = args.get("view", "whole")
	_set_cam(args.get("cam", "orbit"))
	_snap = args.get("snap", "")
	if args.has("open"):
		_open_paths([args.open], args.get("char", "Standard"), args.get("diff", ""))
	elif not songs.is_empty():
		var pick := 0
		for i in songs.size():
			if songs[i].id == args.get("song", ""):
				pick = i
		_select_song(pick, args.get("char", "Standard"), args.get("diff", ""))
	else:
		_info.text = EMPTY_HELP
	if map:
		_seek(map.time_of(float(args.beat)) if args.has("beat") else float(args.get("at", "0")))
	if args.has("play"):
		_toggle_play()
	if _snap != "":
		_snap_frames = int(args.get("snap-frames", "8"))


## Where community maps are listed from and zips unpack to: config.json
## beside the project (paths may start with ~ or be relative to it), with
## ./library as the default home.
func _read_settings() -> void:
	_root_dir = ProjectSettings.globalize_path("res://").path_join("..").simplify_path()
	var cfg_path := _root_dir.path_join("config.json")
	var cfg: Variant = MapData.read_json(cfg_path) if FileAccess.file_exists(cfg_path) else {}
	if not cfg is Dictionary:
		cfg = {}
	var library := _resolve(str(cfg.get("library", "library")))
	_community_roots.clear()
	for p in cfg.get("community", []):
		_community_roots.append(_resolve(str(p)))
	if not library.path_join("community") in _community_roots:
		_community_roots.append(library.path_join("community"))
	_cache = _resolve(str(cfg.get("cache", library.path_join("cache"))))


func _resolve(p: String) -> String:
	if p.begins_with("~"):
		p = OS.get_environment("HOME") + p.substr(1)
	return (p if p.is_absolute_path() else _root_dir.path_join(p)).simplify_path()


## A file of the current song: index entries hold absolute paths, community
## maps names relative to their folder.
func _file(name: String) -> String:
	if name == "":
		return ""
	if name.is_absolute_path():
		return name
	var path := str(song.get("dir", "")).path_join(name)
	# Maps made on Windows or macOS may differ in letter case from Info.dat,
	# so match against the folder's real listing (case-insensitive disks
	# would otherwise accept the wrong spelling).
	var files := DirAccess.get_files_at(path.get_base_dir())
	if not name.get_file() in files:
		for f in files:
			if f.to_lower() == name.get_file().to_lower():
				return path.get_base_dir().path_join(f)
	return path


func _load_audio(path: String) -> AudioStream:
	if path == "" or not FileAccess.file_exists(path):
		return null
	match path.get_extension().to_lower():
		"wav":
			return AudioStreamWAV.load_from_file(path)
		"ogg", "egg":
			return AudioStreamOggVorbis.load_from_file(path)
		"mp3":
			var mp3 := AudioStreamMP3.new()
			mp3.data = FileAccess.get_file_as_bytes(path)
			return mp3
	return null


## Add dropped or opened maps to the list (once each) and show the first.
func _open_paths(paths: Array, wanted_char := "Standard", wanted_diff := "") -> void:
	var first := -1
	var problems := []
	for p in paths:
		var entry := Community.open_path(str(p))
		if entry.is_empty() or entry.has("error"):
			problems.append(entry.get("error", "Not a map: %s" % str(p).get_file()))
			continue
		var at := -1
		for i in songs.size():
			if songs[i].id == entry.id:
				at = i
		if at < 0:
			songs.append(entry)
			at = songs.size() - 1
		if first < 0:
			first = at
	_update_heading()
	_fill_list()
	if first >= 0:
		_select_song(first, wanted_char, wanted_diff)
	if not problems.is_empty():
		_info.text = "\n".join(problems) + ("\n\n" + _info.text if first >= 0 else "")


func _args() -> Dictionary:
	var out := {}
	for a in OS.get_cmdline_user_args():
		var kv: PackedStringArray = a.trim_prefix("--").split("=", true, 1)
		out[kv[0]] = kv[1] if kv.size() > 1 else ""
	return out


# --- world -------------------------------------------------------------------

func _build_world() -> void:
	var we := WorldEnvironment.new()
	var e := Environment.new()
	e.background_mode = Environment.BG_COLOR
	e.background_color = Color(0.008, 0.01, 0.02)
	e.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	e.ambient_light_color = Color(0.35, 0.38, 0.45)
	e.ambient_light_energy = 0.5
	e.tonemap_mode = Environment.TONE_MAPPER_FILMIC
	e.glow_enabled = true
	e.glow_intensity = 0.9
	e.glow_bloom = 0.08
	e.fog_enabled = true
	e.fog_light_color = Color(0.02, 0.025, 0.05)
	e.fog_density = 0.006
	we.environment = e
	add_child(we)
	var sun := DirectionalLight3D.new()
	sun.rotation = Vector3(deg_to_rad(-50), deg_to_rad(20), 0)
	sun.light_energy = 0.9
	add_child(sun)
	_cam = Camera3D.new()
	_cam.fov = 70
	_cam.far = 600
	add_child(_cam)
	_audio = AudioStreamPlayer.new()
	add_child(_audio)
	_meshes.note = _box(Vector3.ONE * NOTE_SIZE)
	var arrow := PrismMesh.new()
	arrow.size = Vector3(0.3, 0.13, 0.02)
	_meshes.arrow = arrow
	var dot := CylinderMesh.new()
	dot.top_radius = 0.075
	dot.bottom_radius = 0.075
	dot.height = 0.02
	_meshes.dot = dot
	var bomb := SphereMesh.new()
	bomb.radius = 0.21
	bomb.height = 0.42
	_meshes.bomb = bomb
	_meshes.link = _box(Vector3(0.42, 0.06, 0.42))
	_meshes.unit = _box(Vector3.ONE)
	_mats.white = _mat(Color.WHITE, 2.0)
	_mats.bomb = _mat(Color(0.12, 0.12, 0.14), 0.0, false, 0.6)
	_mats.grid = _mat(Color(0.3, 0.4, 0.6), 0.25)
	_mats.beat = _mat(Color(0.25, 0.3, 0.45), 0.15)
	_mats.bar = _mat(Color(0.45, 0.55, 0.8), 0.5)
	# The fixed scene: the strike frame (the 4 x 3 grid at the beat plane, at
	# the player; hidden from the player camera, which it would cut across)
	# and the lane edges running out ahead.
	for i in 5:
		_frame.append(_static_bar(Vector3((i - 2) * MapData.LANE_SPACING, MapData.row_edge(1.5), 0), Vector3(0.012, MapData.row_edge(3) - MapData.row_edge(0), 0.012), _mats.grid))
		_static_bar(Vector3((i - 2) * MapData.LANE_SPACING, 0.005, -150), Vector3(0.015, 0.01, 300), _mats.grid)
	for r in 4:
		_frame.append(_static_bar(Vector3(0, MapData.row_edge(r), 0), Vector3(4 * MapData.LANE_SPACING, 0.012, 0.012), _mats.grid))
	var floor_mesh := PlaneMesh.new()
	floor_mesh.size = Vector2(3.0, 2.0)
	var floor_node := MeshInstance3D.new()
	floor_node.mesh = floor_mesh
	floor_node.material_override = _mat(Color(0.05, 0.06, 0.09), 0.0)
	floor_node.position = Vector3(0, -0.001, 0.4)
	add_child(floor_node)
	_build_light_rig()
	_facing_lane = Node3D.new()
	add_child(_facing_lane)
	for i in 5:
		_static_bar(Vector3((i - 2) * MapData.LANE_SPACING, 0.006, -20), Vector3(0.015, 0.01, 40), _mats.grid, _facing_lane)
	_facing_lane.visible = false
	_map_root = Node3D.new()
	add_child(_map_root)


## An abstract rig with one part per basic light type, so the light show's
## cues can be seen in time with the map. It is not any game's stage.
func _build_light_rig() -> void:
	for type in 5:
		_light_mats[type] = _mat(Color.BLACK, 3.0)
	for i in 9:
		_static_bar(Vector3(-16 + i * 4, 12, -90), Vector3(0.25, 24, 0.25), _light_mats[0])
	var ring := TorusMesh.new()
	ring.inner_radius = 7.6
	ring.outer_radius = 8.0
	for z in [-24.0, -36.0, -48.0]:
		var n := MeshInstance3D.new()
		n.mesh = ring
		n.material_override = _light_mats[1]
		n.position = Vector3(0, 3, z)
		n.rotation.x = PI / 2
		add_child(n)
	for side in [-1, 1]:
		for i in 5:
			var n := MeshInstance3D.new()
			n.mesh = _meshes.unit
			n.material_override = _light_mats[2 if side < 0 else 3]
			n.scale = Vector3(0.12, 30, 0.12)
			n.position = Vector3(side * (7 + i * 1.5), 10, -20 - i * 7)
			n.rotation.z = side * deg_to_rad(-18)
			add_child(n)
		_static_bar(Vector3(side * 1.6, 0.02, -40), Vector3(0.08, 0.03, 80), _light_mats[4])


func _box(size: Vector3) -> BoxMesh:
	var m := BoxMesh.new()
	m.size = size
	return m


func _mat(color: Color, glow: float, transparent := false, metal := 0.15) -> StandardMaterial3D:
	var m := StandardMaterial3D.new()
	m.albedo_color = color
	m.metallic = metal
	m.roughness = 0.35
	if glow > 0.0:
		m.emission_enabled = true
		m.emission = Color(color.r, color.g, color.b)
		m.emission_energy_multiplier = glow
	if transparent:
		m.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
		m.cull_mode = BaseMaterial3D.CULL_DISABLED
	return m


func _static_bar(at: Vector3, size: Vector3, mat: Material, parent: Node = self) -> MeshInstance3D:
	var n := MeshInstance3D.new()
	n.mesh = _meshes.unit
	n.material_override = mat
	n.position = at
	n.scale = size
	parent.add_child(n)
	return n


# --- loading a difficulty -------------------------------------------------------

func _select_song(i: int, wanted_char := "Standard", wanted_diff := "") -> void:
	song = songs[i]
	if song.has("zip") and song.get("dir", "") == "":
		var problem := Community.materialize(song, _cache)
		if problem != "":
			_info.text = problem
			return
	if song.sets.is_empty():
		_info.text = "%s has no difficulties listed in its Info.dat." % song.title
		return
	_char_opt.clear()
	var pick := 0
	for k in song.sets.size():
		_char_opt.add_item(song.sets[k].characteristic)
		if song.sets[k].characteristic == wanted_char:
			pick = k
	_char_opt.select(pick)
	_select_set(pick, wanted_diff)
	if not _list.is_anything_selected() or _shown[_list.get_selected_items()[0]] != i:
		var row := _shown.find(i)
		if row >= 0:
			_list.select(row)
			_list.ensure_current_is_visible()


func _select_set(k: int, wanted_diff := "") -> void:
	set_entry = song.sets[k]
	_diff_opt.clear()
	var diffs: Array = set_entry.difficulties
	var pick := diffs.size() - 1
	for j in diffs.size():
		_diff_opt.add_item(diffs[j].name)
		if diffs[j].name == wanted_diff:
			pick = j
	_diff_opt.select(pick)
	_load_difficulty(diffs[pick])


func _load_difficulty(d: Dictionary) -> void:
	var was_playing := playing
	_pause()
	diff = d
	for c in _map_root.get_children():
		c.queue_free()
	_lanes.clear()
	_lane_lines.clear()
	_objects.clear()
	map = MapData.new()
	if not map.load_files(_file(d.beatmap), _file(d.lightshow), _file(song.audioData), float(song.bpm)):
		_info.text = map.error
		return
	_resolve_palette()
	njs = float(d.njs)
	lead_s = MapData.half_jump_beats(njs, float(song.bpm), float(d.offset)) * 60.0 / float(song.bpm)
	var stream := _load_audio(_file(song.audio))
	_audio.stream = stream
	duration = stream.get_length() if stream else 0.0
	for list in [map.notes, map.walls, map.arcs, map.chains, map.bombs]:
		for o in list:
			duration = maxf(duration, float(o.get("t_end", o.get("tt", o.t))) + 1.0)
	_slider.max_value = duration
	for o in map.notes:
		_add_note(o)
	for o in map.bombs:
		var n := _mesh(_meshes.bomb, _mats.bomb)
		_place(n, o.r, Vector3(MapData.lane_x(o.x), MapData.layer_y(o.y), -o.t * njs), o.t, o.t)
	for o in map.walls:
		_add_wall(o)
	for o in map.arcs:
		_add_arc(o)
	for o in map.chains:
		_add_chain(o)
	_add_beat_lines()
	_turns.assign(map.notes.duplicate())
	_turns.sort_custom(func(a, b): return a.t < b.t)
	_rotating = _lanes.size() > 1
	_facing = 0.0
	if cam_mode == "top":
		_set_cam("top")
	_index_lights()
	_update_info()
	_apply_view()
	_seek(now)
	if was_playing:
		_toggle_play()


## The difficulty's environment, and the colours the game would use: the
## level's own override scheme, else the environment's default.
func _resolve_palette() -> void:
	var envs: Array = song.envs
	var rotating: bool = set_entry.characteristic in ["360Degree", "90Degree"]
	var env_idx := int(diff.envIdx)
	env_name = envs[env_idx] if env_idx >= 0 and env_idx < envs.size() else (song.envAll if rotating and song.envAll != "" else song.env)
	var source: Dictionary = DEFAULT_COLORS
	var schemes: Array = song.schemes
	# A scheme index of -1 means the level has no override.
	var idx := int(diff.schemeIdx)
	var env_entry: Dictionary = index.envColors.get(env_name, {})
	if idx >= 0 and idx < schemes.size() and schemes[idx].override and schemes[idx].colors is Dictionary:
		source = schemes[idx].colors
	elif env_entry.get("colors") is Dictionary:
		source = env_entry.colors
	palette.clear()
	for key in source:
		if source[key] is Array:
			var c: Array = source[key]
			palette[key] = Color(c[0], c[1], c[2], c[3])
	# Community maps can set colours per difficulty; those win.
	var own: Variant = diff.get("colors")
	if own is Dictionary:
		for key in own:
			var c: Array = own[key]
			palette[key] = Color(c[0], c[1], c[2], c[3])
	for key in DEFAULT_COLORS:
		if not palette.has(key) or palette[key].a == 0.0 and not key.ends_with("Boost"):
			var c: Array = DEFAULT_COLORS[key]
			palette[key] = Color(c[0], c[1], c[2], c[3])
	_mats.note0 = _mat(palette.saberA, 0.7)
	_mats.note1 = _mat(palette.saberB, 0.7)
	var wall: Color = palette.obstacles
	_mats.wall = _mat(Color(wall.r, wall.g, wall.b, 0.22), 0.6, true)


func _lane(rotation_deg: float) -> Node3D:
	var key := snappedf(rotation_deg, 0.01)
	if not _lanes.has(key):
		var pivot := Node3D.new()
		# Positive lane rotation turns the lane to the player's right.
		pivot.rotation.y = -deg_to_rad(rotation_deg)
		var track := Node3D.new()
		pivot.add_child(track)
		_map_root.add_child(pivot)
		if key != 0.0:
			for i in 5:
				var line := _static_bar(Vector3((i - 2) * MapData.LANE_SPACING, 0.005, -40), Vector3(0.015, 0.01, 80), _mats.beat, pivot)
				line.visible = cam_mode != "player"
				_lane_lines.append(line)
		_lanes[key] = track
	return _lanes[key]


func _mesh(mesh: Mesh, mat: Material) -> MeshInstance3D:
	var n := MeshInstance3D.new()
	n.mesh = mesh
	n.material_override = mat
	return n


func _place(n: Node3D, rotation_deg: float, at: Vector3, t0: float, t1: float) -> void:
	n.position = at
	_lane(rotation_deg).add_child(n)
	_objects.append([n, t0, t1])


func _add_note(o: Dictionary) -> void:
	var n := _mesh(_meshes.note, _mats.note0 if o.c == 0 else _mats.note1)
	var d := clampi(int(o.d), 0, 8)
	var mark := _mesh(_meshes.dot if d == 8 else _meshes.arrow, _mats.white)
	mark.position = Vector3(0, 0 if d == 8 else -0.06, NOTE_SIZE / 2.0 + 0.012)
	if d == 8:
		mark.rotation.x = PI / 2
	else:
		mark.rotation.z = PI
	n.add_child(mark)
	n.rotation.z = deg_to_rad(DIR_ANGLE[d] + float(o.a))
	_place(n, o.r, Vector3(MapData.lane_x(o.x), MapData.layer_y(o.y), -o.t * njs), o.t, o.t)


func _add_wall(o: Dictionary) -> void:
	var x0 := (float(o.x) - 2.0) * MapData.LANE_SPACING
	var x1 := (float(o.x) + float(o.w) - 2.0) * MapData.LANE_SPACING
	var y0 := MapData.row_edge(o.y)
	var y1 := MapData.row_edge(float(o.y) + float(o.h))
	var z0: float = -o.t * njs
	var z1: float = -o.t_end * njs
	var n := _mesh(_meshes.unit, _mats.wall)
	n.scale = Vector3(maxf(absf(x1 - x0), 0.02), maxf(absf(y1 - y0), 0.02), maxf(absf(z1 - z0), 0.02))
	_place(n, o.r, Vector3((x0 + x1) / 2.0, (y0 + y1) / 2.0, (z0 + z1) / 2.0), o.t, o.t_end)


func _add_arc(o: Dictionary) -> void:
	var p0 := Vector3(MapData.lane_x(o.hx), MapData.layer_y(o.hy), -o.t * njs)
	var p3 := Vector3(MapData.lane_x(o.tx), MapData.layer_y(o.ty), -o.tt * njs)
	var hd: Vector2 = DIR_VEC[clampi(int(o.hd), 0, 8)] * float(o.m) * ARC_REACH
	var td: Vector2 = DIR_VEC[clampi(int(o.td), 0, 8)] * float(o.tm) * ARC_REACH
	var dz := p3.z - p0.z
	var p1 := p0 + Vector3(hd.x, hd.y, dz / 3.0)
	var p2 := p3 - Vector3(td.x, td.y, dz / 3.0)
	var pts := PackedVector3Array()
	for i in 33:
		var s := i / 32.0
		var u := 1.0 - s
		pts.append(p0 * u * u * u + p1 * 3.0 * u * u * s + p2 * 3.0 * u * s * s + p3 * s * s * s)
	var n := _mesh(_tube(pts, 0.035), _mats.note0 if o.c == 0 else _mats.note1)
	_place(n, o.r, Vector3.ZERO, o.t, o.tt)


func _add_chain(o: Dictionary) -> void:
	var slices := int(o.slices)
	if slices < 2:
		return
	var head := Vector2(MapData.lane_x(o.hx), MapData.layer_y(o.hy))
	var tail := Vector2(MapData.lane_x(o.tx), MapData.layer_y(o.ty))
	var dir: Vector2 = DIR_VEC[clampi(int(o.d), 0, 8)].rotated(deg_to_rad(float(o.a)))
	var mid := head + dir * head.distance_to(tail) / 2.0
	for k in range(1, slices):
		var f := float(k) / (slices - 1)
		var s := f * float(o.squish)
		var at := head.lerp(mid, s).lerp(mid.lerp(tail, s), s)
		var tangent := (mid - head) * 2.0 * (1.0 - s) + (tail - mid) * 2.0 * s
		var t: float = lerpf(o.t, o.tt, f)
		var n := _mesh(_meshes.link, _mats.note0 if o.c == 0 else _mats.note1)
		var dot := _mesh(_meshes.dot, _mats.white)
		dot.position = Vector3(0, 0, 0.222)
		dot.rotation.x = PI / 2
		n.add_child(dot)
		if tangent.length() > 0.0001:
			n.rotation.z = atan2(tangent.x, -tangent.y)
		_place(n, o.r, Vector3(at.x, at.y, -t * njs), t, t)


## A beat line on the floor for every beat (brighter each 4), with the bar
## and time, so spacing can be read against the music.
func _add_beat_lines() -> void:
	var last := ceili(map.beat_of(duration))
	var track := _lane(0.0)
	for b in range(0, last + 1):
		var t := map.time_of(b)
		var n := _mesh(_meshes.unit, _mats.bar if b % 4 == 0 else _mats.beat)
		n.scale = Vector3(4 * MapData.LANE_SPACING, 0.008, 0.02 if b % 4 == 0 else 0.01)
		n.position = Vector3(0, 0.006, -t * njs)
		track.add_child(n)
		if b % 4 == 0:
			var label := Label3D.new()
			label.text = "bar %d  %s" % [b / 4 + 1, _clock(t)]
			label.font_size = 48
			label.pixel_size = 0.004
			label.modulate = Color(0.6, 0.7, 0.95, 0.8)
			label.billboard = BaseMaterial3D.BILLBOARD_ENABLED
			label.position = Vector3(-1.75, 0.15, -t * njs)
			track.add_child(label)


func _tube(points: PackedVector3Array, radius: float) -> ArrayMesh:
	var st := SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	var sides := 6
	var rings: Array = []
	for i in points.size():
		var fwd := (points[mini(i + 1, points.size() - 1)] - points[maxi(i - 1, 0)]).normalized()
		var side := fwd.cross(Vector3.UP)
		if side.length() < 0.01:
			side = fwd.cross(Vector3.RIGHT)
		side = side.normalized()
		var up := side.cross(fwd).normalized()
		var ring := []
		for k in sides:
			var a := TAU * k / sides
			ring.append(points[i] + (side * cos(a) + up * sin(a)) * radius)
		rings.append(ring)
	for i in rings.size() - 1:
		for k in sides:
			var a: Vector3 = rings[i][k]
			var b: Vector3 = rings[i][(k + 1) % sides]
			var c: Vector3 = rings[i + 1][k]
			var e: Vector3 = rings[i + 1][(k + 1) % sides]
			for v in [a, b, c, b, e, c]:
				st.add_vertex(v)
	st.generate_normals()
	return st.commit()


# --- lights ------------------------------------------------------------------

func _index_lights() -> void:
	_light_events.clear()
	for e in map.lights:
		if e.type >= 0 and e.type <= 4:
			if not _light_events.has(e.type):
				_light_events[e.type] = []
			_light_events[e.type].append(e)
	_box_groups.clear()
	for cue in map.box_cues:
		if not _box_groups.has(cue.group):
			_box_groups[cue.group] = []
		_box_groups[cue.group].append(cue)
	for c in _board.get_children():
		c.queue_free()
	_box_cells.clear()
	var ids := _box_groups.keys()
	ids.sort()
	for id in ids:
		var cell := ColorRect.new()
		cell.custom_minimum_size = Vector2(34, 22)
		cell.tooltip_text = "light group %d" % id
		var l := Label.new()
		l.text = str(id)
		l.add_theme_font_size_override("font_size", 11)
		l.add_theme_color_override("font_color", Color(1, 1, 1, 0.7))
		cell.add_child(l)
		_board.add_child(cell)
		_box_cells[id] = cell


static func _last_at(list: Array, t: float) -> int:
	var lo := 0
	var hi := list.size() - 1
	var found := -1
	while lo <= hi:
		var mid := (lo + hi) >> 1
		if list[mid].t <= t:
			found = mid
			lo = mid + 1
		else:
			hi = mid - 1
	return found


func _boosted() -> bool:
	var i := _last_at(map.boosts, now)
	return i >= 0 and map.boosts[i].on


func _env_color(which: int) -> Color:
	var key: String = ["env0", "env1", "envW"][clampi(which, 0, 2)]
	var boost: Color = palette.get(key + "Boost", Color(0, 0, 0, 0))
	return boost if _boosted() and boost.a > 0.0 else palette.get(key, Color.WHITE)


## Colour and brightness of a basic light event `dt` seconds after it fired.
## Values 1-4 use the right colour, 5-8 the left, 9-12 white; within each:
## on, flash, fade, transition. Flash and fade timings are approximate.
func _basic_state(e: Dictionary, dt: float) -> Color:
	var v: int = e.value
	if v <= 0 or v > 12:
		return Color.BLACK
	var col := _env_color(1 if v <= 4 else (0 if v <= 8 else 2))
	var kind := (v - 1) % 4
	var level: float = e.f
	if kind == 1:
		level *= 1.0 + 0.8 * exp(-dt * 6.0)
	elif kind == 2:
		level *= (1.0 + 0.8 * exp(-dt * 6.0)) * clampf(1.0 - dt / 1.2, 0.0, 1.0)
	return col * level


func _update_lights() -> void:
	for type in 5:
		var evs: Array = _light_events.get(type, [])
		var i := _last_at(evs, now)
		var c := Color.BLACK
		if i >= 0:
			c = _basic_state(evs[i], now - evs[i].t)
			if i + 1 < evs.size() and (int(evs[i + 1].value) - 1) % 4 == 3 and int(evs[i + 1].value) > 0:
				var nxt: Dictionary = evs[i + 1]
				var f := clampf((now - evs[i].t) / maxf(nxt.t - evs[i].t, 0.001), 0.0, 1.0)
				c = c.lerp(_basic_state(nxt, 0.0), f)
		var m: StandardMaterial3D = _light_mats[type]
		m.albedo_color = Color(c.r, c.g, c.b).clamp() * 0.6
		m.emission = Color(c.r, c.g, c.b).clamp()
		m.emission_energy_multiplier = 3.0 * maxf(maxf(c.r, c.g), c.b)
	for id in _box_cells:
		var cues: Array = _box_groups[id]
		var i := _last_at(cues, now)
		var c := Color(0.06, 0.06, 0.08)
		if i >= 0:
			c = _env_color(cues[i].color) * float(cues[i].brightness)
			if i + 1 < cues.size() and int(cues[i + 1].transition) == 1:
				var nxt: Dictionary = cues[i + 1]
				var f := clampf((now - cues[i].t) / maxf(nxt.t - cues[i].t, 0.001), 0.0, 1.0)
				c = c.lerp(_env_color(nxt.color) * float(nxt.brightness), f)
		(_box_cells[id] as ColorRect).color = Color(c.r, c.g, c.b).clamp()


# --- time, view and camera ---------------------------------------------------------

func _process(delta: float) -> void:
	if playing:
		if _audio.stream and _audio.playing:
			now = _audio.get_playback_position() + AudioServer.get_time_since_last_mix() - AudioServer.get_output_latency()
		else:
			now += delta * speed
		if now >= duration:
			now = duration
			_pause()
		_slider.set_value_no_signal(now)
	for key in _lanes:
		(_lanes[key] as Node3D).position.z = now * njs
	if view_mode == "player":
		for o in _objects:
			(o[0] as Node3D).visible = o[2] >= now - 0.05 and o[1] <= now + lead_s
	if map:
		_update_lights()
		_time_label.text = "%s / %s   beat %.2f" % [_clock(now), _clock(duration), map.beat_of(now)]
	_update_camera()
	if _snap_frames > 0:
		_snap_frames -= 1
		if _snap_frames == 0:
			await RenderingServer.frame_post_draw
			get_viewport().get_texture().get_image().save_png(_snap)
			print("snapshot %s at %.2f s (audio %s)" % [_snap, now, "playing" if _audio.playing else "stopped"])
			get_tree().quit()


func _seek(t: float) -> void:
	now = clampf(t, 0.0, duration)
	_slider.set_value_no_signal(now)
	if _rotating and not _turns.is_empty():
		_facing = -deg_to_rad(float(_turns[clampi(_last_at(_turns, now - 0.05) + 1, 0, _turns.size() - 1)].r))
	if playing and _audio.stream:
		_audio.play(now)


func _toggle_play() -> void:
	if playing:
		_pause()
		return
	playing = true
	_play_btn.text = "Pause"
	if _audio.stream:
		_audio.pitch_scale = speed
		_audio.play(now)


func _pause() -> void:
	playing = false
	if _play_btn:
		_play_btn.text = "Play"
	if _audio:
		_audio.stop()


func _apply_view() -> void:
	if view_mode == "whole":
		for o in _objects:
			(o[0] as Node3D).visible = true
	if _view_opt:
		_view_opt.select(0 if view_mode == "whole" else 1)


func _set_cam(mode: String) -> void:
	cam_mode = mode
	_look = Vector2.ZERO
	match mode:
		"side":
			_orbit = {"target": Vector3(0, 1.2, -14), "yaw": -PI / 2, "pitch": -0.12, "dist": 22.0}
		"top":
			# Rotating maps surround the player, so look down on the player.
			_orbit = {"target": Vector3(0, 0, 0 if _rotating else -8), "yaw": 0.0, "pitch": -1.45, "dist": 30.0 if _rotating else 20.0}
		"orbit":
			_orbit = {"target": Vector3(0, 1.2, -6), "yaw": -0.55, "pitch": -0.32, "dist": 13.0}
	for bar in _frame + _lane_lines:
		bar.visible = mode != "player"
	if _cam_opt:
		_cam_opt.select(["player", "orbit", "side", "top"].find(mode))


func _update_camera() -> void:
	if cam_mode == "player":
		if _rotating:
			var i := clampi(_last_at(_turns, now - 0.05) + 1, 0, _turns.size() - 1)
			_facing = lerp_angle(_facing, -deg_to_rad(float(_turns[i].r)), clampf(get_process_delta_time() * 4.0, 0.0, 1.0))
		_facing_lane.visible = _rotating
		_facing_lane.rotation.y = _facing
		var turn := Basis(Vector3.UP, _facing)
		_cam.position = turn * Vector3(0, MapData.HEAD_Y, 0.6)
		_cam.rotation = Vector3(_look.y, _facing + _look.x, 0)
		return
	_facing_lane.visible = false
	var dir := Vector3(sin(_orbit.yaw) * cos(_orbit.pitch), -sin(_orbit.pitch), cos(_orbit.yaw) * cos(_orbit.pitch))
	_cam.position = _orbit.target + dir * _orbit.dist
	_cam.look_at(_orbit.target, Vector3.UP)


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventMouseButton:
		if event.button_index in [MOUSE_BUTTON_LEFT, MOUSE_BUTTON_RIGHT, MOUSE_BUTTON_MIDDLE]:
			_drag = event.button_index if event.pressed else 0
		elif event.pressed and event.button_index in [MOUSE_BUTTON_WHEEL_UP, MOUSE_BUTTON_WHEEL_DOWN] and cam_mode != "player":
			_orbit.dist = clampf(_orbit.dist * (0.9 if event.button_index == MOUSE_BUTTON_WHEEL_UP else 1.1), 1.5, 250.0)
	elif event is InputEventMouseMotion and _drag != 0:
		var m: Vector2 = event.relative * 0.005
		if cam_mode == "player":
			_look = Vector2(_look.x - m.x, clampf(_look.y - m.y, -1.4, 1.4))
		elif _drag == MOUSE_BUTTON_LEFT:
			_orbit.yaw -= m.x
			_orbit.pitch = clampf(_orbit.pitch - m.y, -1.55, 1.2)
		else:
			var right := _cam.global_basis.x
			var up := _cam.global_basis.y
			_orbit.target += (-right * m.x + up * m.y) * _orbit.dist * 0.4
	elif event is InputEventKey and event.pressed and not _search.has_focus():
		var step := 10.0 if event.shift_pressed else 2.0
		match event.keycode:
			KEY_SPACE: _toggle_play()
			KEY_LEFT: _seek(now - step)
			KEY_RIGHT: _seek(now + step)
			KEY_BRACKETLEFT: if map: _seek(map.time_of(floor(map.beat_of(now) - 0.001)))
			KEY_BRACKETRIGHT: if map: _seek(map.time_of(floor(map.beat_of(now)) + 1.0))
			KEY_HOME: _seek(0.0)
			KEY_1: _set_cam("player")
			KEY_2: _set_cam("orbit")
			KEY_3: _set_cam("side")
			KEY_4: _set_cam("top")
			KEY_V:
				view_mode = "player" if view_mode == "whole" else "whole"
				_apply_view()
			KEY_TAB: _left.visible = not _left.visible
			_: return
		get_viewport().set_input_as_handled()


static func _clock(t: float) -> String:
	return "%d:%04.1f" % [int(t) / 60, fmod(t, 60.0)]


# --- interface ---------------------------------------------------------------

func _build_ui() -> void:
	var layer := CanvasLayer.new()
	add_child(layer)
	var root := Control.new()
	root.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	root.mouse_filter = Control.MOUSE_FILTER_IGNORE
	var theme := Theme.new()
	theme.default_font_size = 15
	root.theme = theme
	layer.add_child(root)

	_left = _panel(root)
	_left.set_anchors_and_offsets_preset(Control.PRESET_LEFT_WIDE)
	_left.offset_right = 330
	_left.offset_bottom = -64
	var lv := VBoxContainer.new()
	_left.add_child(lv)
	_heading = Label.new()
	lv.add_child(_heading)
	var open := Button.new()
	open.text = "Open map folder or zip…"
	open.tooltip_text = "Or drop map folders and zips onto the window"
	open.pressed.connect(func(): _file_dialog.popup_centered_ratio(0.6))
	lv.add_child(open)
	_file_dialog = FileDialog.new()
	_file_dialog.title = "Open a map folder or zip"
	_file_dialog.file_mode = FileDialog.FILE_MODE_OPEN_ANY
	_file_dialog.access = FileDialog.ACCESS_FILESYSTEM
	_file_dialog.use_native_dialog = true
	_file_dialog.filters = PackedStringArray(["*.zip ; Map zips", "Info.dat ; Map info files"])
	_file_dialog.file_selected.connect(func(p): _open_paths([p]))
	_file_dialog.dir_selected.connect(func(p): _open_paths([p]))
	add_child(_file_dialog)
	_search = LineEdit.new()
	_search.placeholder_text = "Search title, artist, mapper, pack"
	_search.text_changed.connect(func(_t): _fill_list())
	lv.add_child(_search)
	_list = ItemList.new()
	_list.size_flags_vertical = Control.SIZE_EXPAND_FILL
	_list.item_selected.connect(func(row): _select_song(_shown[row]))
	lv.add_child(_list)
	_update_heading()
	_fill_list()

	var top := HBoxContainer.new()
	top.set_anchors_and_offsets_preset(Control.PRESET_TOP_WIDE)
	top.offset_left = 345
	top.offset_top = 8
	root.add_child(top)
	_char_opt = OptionButton.new()
	_char_opt.item_selected.connect(func(k): _select_set(k))
	_diff_opt = OptionButton.new()
	_diff_opt.item_selected.connect(func(j): _load_difficulty(set_entry.difficulties[j]))
	_view_opt = _options(["Whole map", "As the player sees it"], func(j):
		view_mode = "whole" if j == 0 else "player"
		_apply_view())
	_cam_opt = _options(["Player view", "Orbit", "Side", "Top"], func(j): _set_cam(["player", "orbit", "side", "top"][j]))
	for c in [_char_opt, _diff_opt, _view_opt, _cam_opt]:
		top.add_child(c)

	var right := _panel(root)
	right.set_anchors_and_offsets_preset(Control.PRESET_TOP_RIGHT)
	right.offset_left = -340
	right.offset_top = 8
	right.offset_right = -8
	var rv := VBoxContainer.new()
	right.add_child(rv)
	_cover = TextureRect.new()
	_cover.custom_minimum_size = Vector2(96, 96)
	_cover.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	_cover.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT
	_cover.visible = false
	rv.add_child(_cover)
	_info = Label.new()
	_info.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	_info.custom_minimum_size = Vector2(320, 0)
	rv.add_child(_info)
	_board = GridContainer.new()
	_board.columns = 8
	rv.add_child(_board)

	var bottom := _panel(root)
	bottom.set_anchors_and_offsets_preset(Control.PRESET_BOTTOM_WIDE)
	bottom.offset_top = -56
	var bh := HBoxContainer.new()
	bottom.add_child(bh)
	_play_btn = Button.new()
	_play_btn.text = "Play"
	_play_btn.custom_minimum_size = Vector2(80, 0)
	_play_btn.pressed.connect(_toggle_play)
	bh.add_child(_play_btn)
	_slider = HSlider.new()
	_slider.step = 0.01
	_slider.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_slider.size_flags_vertical = Control.SIZE_SHRINK_CENTER
	_slider.value_changed.connect(_seek)
	bh.add_child(_slider)
	_time_label = Label.new()
	_time_label.custom_minimum_size = Vector2(250, 0)
	bh.add_child(_time_label)
	_speed_opt = _options(["0.25x", "0.5x", "0.75x", "1x"], func(j):
		speed = SPEEDS[j]
		_audio.pitch_scale = speed)
	_speed_opt.select(3)
	bh.add_child(_speed_opt)
	var keys := Label.new()
	keys.text = "Space play · ←/→ 2 s (Shift 10 s) · [ ] beat · 1-4 camera · V view · Tab list · drag to look"
	keys.add_theme_font_size_override("font_size", 12)
	keys.modulate = Color(1, 1, 1, 0.6)
	keys.set_anchors_and_offsets_preset(Control.PRESET_BOTTOM_WIDE)
	keys.offset_top = -80
	keys.offset_bottom = -60
	keys.offset_left = 345
	root.add_child(keys)


func _panel(parent: Control) -> PanelContainer:
	var p := PanelContainer.new()
	var style := StyleBoxFlat.new()
	style.bg_color = Color(0.03, 0.035, 0.06, 0.82)
	style.set_content_margin_all(8)
	style.set_corner_radius_all(6)
	p.add_theme_stylebox_override("panel", style)
	parent.add_child(p)
	return p


func _options(labels: Array, on_pick: Callable) -> OptionButton:
	var o := OptionButton.new()
	for l in labels:
		o.add_item(l)
	o.item_selected.connect(on_pick)
	return o


func _update_heading() -> void:
	var maps := songs.filter(func(s): return s.source == "map").size()
	var extracted := songs.size() - maps
	_heading.text = "%d extracted song%s · %d map%s" % [extracted, "" if extracted == 1 else "s", maps, "" if maps == 1 else "s"]


func _fill_list() -> void:
	_list.clear()
	_shown.clear()
	var q := _search.text.strip_edges().to_lower()
	for i in songs.size():
		var s: Dictionary = songs[i]
		var hay := ("%s %s %s %s %s" % [s.title, s.sub, s.artist, s.mapper, s.pack]).to_lower()
		if q == "" or hay.contains(q):
			_list.add_item("%s — %s%s" % [s.title, s.artist, "  · map" if s.source == "map" else ""])
			_shown.append(i)
	if not song.is_empty():
		var row := _shown.find(songs.find(song))
		if row >= 0:
			_list.select(row)


func _update_info() -> void:
	var counts := "%d notes · %d bombs · %d walls · %d arcs · %d chains" % [map.notes.size(), map.bombs.size(), map.walls.size(), map.arcs.size(), map.chains.size()]
	var rotations := {}
	for o in map.notes:
		rotations[o.r] = true
	var lines := [
		"%s%s" % [song.title, (" " + song.sub) if song.sub != "" else ""],
		"%s%s%s" % [song.artist, " · mapped by " + song.mapper if song.mapper != "" else "",
			" · %s pack" % song.pack if song.pack != "" else (" · community map" if song.source == "map" else "")],
		"%s %s · beatmap v%s" % [set_entry.characteristic, diff.name, map.version],
		counts,
		"BPM %s · note jump speed %.1f%s · start offset %+.2f beats" % [song.bpm, njs, "" if float(diff.njsAuthored) > 0 else " (game default)", float(diff.offset)],
		"Notes land %.1f m ahead, %.2f s before their beat" % [lead_s * njs, lead_s],
		"Environment: %s" % (index.envColors.get(env_name, {}).get("name", env_name) if env_name != "" else "not set"),
	]
	if rotations.size() > 1:
		lines.append("Lane rotations: %d directions" % rotations.size())
	if map.njs_events > 0:
		lines.append("Note speed changes: %d (not applied to the layout)" % map.njs_events)
	if map.lights.is_empty() and map.box_cues.is_empty():
		lines.append("No light show for this difficulty")
	else:
		lines.append("Lights: %d basic cues%s" % [map.lights.size(), ", %d light groups below (event boxes)" % _box_groups.size() if not _box_groups.is_empty() else ""])
		var moves: int = int(map.box_kinds.get(2, 0)) + int(map.box_kinds.get(3, 0))
		if moves > 0:
			lines.append("Plus %d light rotation/translation groups (not shown)" % moves)
	_info.text = "\n".join(lines)
	var cover := _file(str(song.get("cover", "")))
	var image := Image.load_from_file(cover) if cover != "" and FileAccess.file_exists(cover) else null
	_cover.texture = ImageTexture.create_from_image(image) if image else null
	_cover.visible = image != null
	get_window().title = "Beatmap Explorer · %s · %s %s" % [song.title, set_entry.characteristic, diff.name]
