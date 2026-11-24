extends SceneTree
## Dump what the viewer's reader (viewer/map_data.gd) makes of one difficulty,
## for train/tests/compare_with_viewer.py. Arguments after `--`:
## BEATMAP LIGHTSHOW AUDIO_DATA BPM OUT_JSON (use "" for none).

const MapData := preload("res://map_data.gd")


func _init() -> void:
	var a := OS.get_cmdline_user_args()
	var m := MapData.new()
	var ok := m.load_files(a[0], a[1], a[2], float(a[3]))
	var out := {"ok": ok, "error": m.error, "version": m.version,
		"notes": m.notes.map(func(o): return [o.t, o.b, o.x, o.y, o.c, o.d, o.a, o.r]),
		"bombs": m.bombs.map(func(o): return [o.t, o.b, o.x, o.y, o.r]),
		"walls": m.walls.map(func(o): return [o.t, o.b, o.x, o.y, o.w, o.h, o.t_end, o.d, o.r]),
		"arcs": m.arcs.map(func(o): return [o.t, o.b, o.tt, o.tb]),
		"chains": m.chains.map(func(o): return [o.t, o.b, o.tt, o.tb, o.slices]),
		"lights": m.lights.map(func(o): return [o.t, o.b, o.type, o.value, o.f]),
		"boxes": m.box_cues.map(func(o): return [o.t, o.group, o.color, o.brightness, o.transition])}
	var f := FileAccess.open(a[4], FileAccess.WRITE)
	f.store_string(JSON.stringify(out))
	f.close()
	quit()
