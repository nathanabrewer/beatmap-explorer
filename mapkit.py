#!/usr/bin/env python3
"""beatmap-explorer toolkit: get your own songs off a Quest, unpack them,
index them, and open them in the 3D viewer.

    python3 mapkit.py all             pull from the Quest, extract, index (one step)
    python3 mapkit.py pull            copy the game's song files and custom maps off a Quest (adb)
    python3 mapkit.py extract [SRC]   unpack every level from a pulled folder or a main .obb
    python3 mapkit.py index           rebuild index.json (and covers) from the extracted levels
    python3 mapkit.py open [FLAGS]    open the viewer; flags pass through (see viewer/viewer.gd)
    python3 mapkit.py zip LEVEL       export an extracted level as a standard map zip
    python3 mapkit.py inspect BUNDLE [NAME]   list a Unity bundle's objects, or dump matching ones

Settings come from config.json beside this file (see config.example.json);
without one, everything lives in ./library. Use `--config FILE` for another.

Use this only with a copy of the game you own, for your own viewing. This
project ships no game content, and what it extracts should stay with you.
"""
import argparse
import datetime
import gzip
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
GAME = "com.beatgames.beatsaber"
QUEST_GAME_DIR = f"/sdcard/Android/obb/{GAME}"
QUEST_CUSTOM_DIRS = [f"/sdcard/ModData/{GAME}/Mods/SongCore/CustomLevels",
                     f"/sdcard/ModData/{GAME}/Mods/SongCore/CustomWIPLevels"]
# Test and calibration levels shipped with the game, not songs.
SKIP_LEVELS = {"artteamtest", "performancetest", "metronome"}
# Bundles holding level metadata, characteristics, colour schemes and covers.
METADATA_BUNDLES = ("_pack_assets_", "environment_assets", "sharedassets", "default_assets")
DIFFICULTY = {0: "Easy", 1: "Normal", 2: "Hard", 3: "Expert", 4: "ExpertPlus"}
# The game's speed when a difficulty leaves note jump speed at 0.
DEFAULT_NJS = {0: 10.0, 1: 10.0, 2: 10.0, 3: 12.0, 4: 16.0}
COLOR_KEYS = {"_saberAColor": "saberA", "_saberBColor": "saberB", "_obstaclesColor": "obstacles",
              "_environmentColor0": "env0", "_environmentColor1": "env1", "_environmentColorW": "envW",
              "_environmentColor0Boost": "env0Boost", "_environmentColor1Boost": "env1Boost",
              "_environmentColorWBoost": "envWBoost"}


def fail(message):
    print(message, file=sys.stderr)
    sys.exit(1)


# --- settings ------------------------------------------------------------------

def resolve(value):
    p = Path(os.path.expanduser(str(value)))
    return p if p.is_absolute() else (HERE / p).resolve()


def unique(paths):
    seen, out = set(), []
    for p in paths:
        if str(p) not in seen:
            seen.add(str(p))
            out.append(p)
    return out


def load_config(path):
    raw = json.load(open(path)) if path.exists() else {}
    lib = resolve(raw.get("library", "library"))
    return {
        "raw": raw,
        "library": lib,
        "pulled": resolve(raw.get("pulled", lib / "pulled")),
        "levels": unique([resolve(p) for p in raw.get("levels", [])] + [lib / "levels"]),
        "bundles": unique([resolve(p) for p in raw.get("bundles", [])] + [lib / "bundles"]),
        "community": unique([resolve(p) for p in raw.get("community", [])] + [lib / "community"]),
        "covers": lib / "covers",
        "adb": raw.get("adb", ""),
    }


def need_unitypy(argv):
    """Import UnityPy, setting up a private Python environment for it the
    first time (config "python", else ./.venv)."""
    try:
        import UnityPy  # noqa: F401
        return
    except ImportError:
        pass
    cfg_python = CONFIG["raw"].get("python")
    candidates = [resolve(cfg_python)] if cfg_python else []
    venv = HERE / ".venv"
    candidates.append(venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))
    # A venv's python is a symlink to the base interpreter, so compare the
    # paths as invoked, and never hop more than once.
    if os.environ.get("MAPKIT_HOPPED"):
        fail(f"{sys.executable} can't import UnityPy: install requirements.txt there, or remove \"python\" from config.json.")
    os.environ["MAPKIT_HOPPED"] = "1"
    for py in candidates:
        if py.exists() and os.path.abspath(sys.executable) != os.path.abspath(py):
            os.execv(str(py), [str(py), str(HERE / "mapkit.py")] + argv)
    print("Setting up the Python package this needs (UnityPy) in .venv, once...")
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    subprocess.run([str(candidates[-1]), "-m", "pip", "install", "-q", "-r", str(HERE / "requirements.txt")], check=True)
    os.execv(str(candidates[-1]), [str(candidates[-1]), str(HERE / "mapkit.py")] + argv)


def read_json(path):
    raw = Path(path).read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw)


# --- pull ------------------------------------------------------------------------

def find_adb():
    if CONFIG["adb"]:
        return str(resolve(CONFIG["adb"]))
    found = shutil.which("adb")
    if found:
        return found
    for p in ["~/Library/Android/sdk/platform-tools/adb", "~/Android/Sdk/platform-tools/adb",
              "~/AppData/Local/Android/Sdk/platform-tools/adb.exe"]:
        if Path(os.path.expanduser(p)).exists():
            return os.path.expanduser(p)
    return ""


def cmd_pull(args):
    """Copy the game's song files off the headset, plus any custom maps."""
    dest = CONFIG["pulled"] / datetime.date.today().isoformat()
    custom = CONFIG["library"] / "community" / "quest"
    adb = find_adb() or ("adb" if args.dry_run else "")
    if not adb:
        fail("adb was not found. Install Android platform tools (macOS: brew install android-platform-tools)\n"
             "or set \"adb\" in config.json.")
    base = [adb] + (["-s", args.serial] if args.serial else [])
    plan = [base + ["pull", QUEST_GAME_DIR, str(dest)]]
    if not args.no_custom:
        plan += [base + ["pull", d, str(custom)] for d in QUEST_CUSTOM_DIRS]
    if args.dry_run:
        print("Would run (nothing touched):")
        for step in plan:
            print("  " + " ".join(step))
        return dest / GAME
    lines = subprocess.run([adb, "devices"], capture_output=True, text=True).stdout.splitlines()[1:]
    devices = [l.split()[0] for l in lines if l.strip().endswith("\tdevice")]
    waiting = [l.split()[0] for l in lines if "unauthorized" in l]
    if args.serial:
        devices = [d for d in devices if d == args.serial]
    if not devices:
        if waiting:
            fail("The headset is connected but hasn't allowed this computer yet: put it on and accept "
                 "the USB debugging prompt, then run this again.")
        fail("No headset found. Connect it with a USB cable (developer mode on), accept the USB debugging\n"
             "prompt inside the headset, and run this again.")
    if len(devices) > 1:
        fail("Several devices are connected: " + ", ".join(devices) + ". Pick one with --serial.")
    base = [adb, "-s", devices[0]]
    if subprocess.run(base + ["shell", "ls", QUEST_GAME_DIR], capture_output=True).returncode != 0:
        fail(f"The game's files aren't on this headset ({QUEST_GAME_DIR}). Is the game installed?")
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Copying the game's song files to {dest} (a few GB; this takes a while)...")
    subprocess.run(base + ["pull", QUEST_GAME_DIR, str(dest)], check=True)
    if not args.no_custom:
        for d in QUEST_CUSTOM_DIRS:
            if subprocess.run(base + ["shell", "ls", d], capture_output=True).returncode == 0:
                custom.mkdir(parents=True, exist_ok=True)
                print(f"Copying custom maps from {d}...")
                subprocess.run(base + ["pull", d, str(custom)], check=True)
    return dest / GAME


# --- extract -------------------------------------------------------------------------

def latest_pull():
    pulls = sorted(CONFIG["pulled"].glob(f"*/{GAME}"))
    return pulls[-1] if pulls else None


def extract_level(bundle, out):
    """Unpack one level bundle: beatmaps and lightshows (JSON), timing data,
    the song (WAV) and the level's data. Returns what was written."""
    import UnityPy
    out.mkdir(parents=True, exist_ok=True)
    env = UnityPy.load(str(bundle))
    written = {"text": 0, "audio": 0, "level": 0}
    for obj in env.objects:
        kind = obj.type.name
        if kind == "TextAsset":
            data = obj.read()
            raw = data.m_Script
            if isinstance(raw, str):
                raw = raw.encode("utf-8", "surrogateescape")
            name = data.m_Name
            if name.endswith(".gz"):
                raw, name = gzip.decompress(raw), name[:-3]
            (out / (name + ".json")).write_bytes(raw)
            written["text"] += 1
        elif kind == "AudioClip":
            data = obj.read()
            for sample_name, sample in data.samples.items():
                (out / (sample_name if "." in sample_name else sample_name + ".wav")).write_bytes(sample)
                written["audio"] += 1
        elif kind == "MonoBehaviour":
            try:
                tree = obj.read_typetree()
            except Exception:  # noqa: BLE001 - scripts without a type tree are not level data
                continue
            if "_difficultyBeatmapSets" in tree:
                json.dump(tree, open(out / "level-data.json", "w"), indent=1, default=str)
                written["level"] += 1
    return written


def cmd_extract(args):
    """Unpack every level from a pulled folder (main .obb plus one file per
    purchased level) or from a main .obb on its own."""
    src = Path(args.source).expanduser() if args.source else latest_pull()
    if src is None or not src.exists():
        fail("Nothing to extract: run `mapkit.py pull` first, or give the pulled folder or a main .obb.")
    if src.is_file():
        obbs, singles = [src], []
    else:
        obbs = sorted(src.glob("main.*.obb"))
        singles = sorted(f for f in src.iterdir() if f.is_file() and not f.name.startswith(".") and f.suffix != ".obb")
    if not obbs and not singles:
        fail(f"No game files in {src}.")
    only = set(args.only.split(",")) if args.only else None
    levels_out = CONFIG["library"] / "levels"
    bundles_out = CONFIG["library"] / "bundles"
    bundles_out.mkdir(parents=True, exist_ok=True)
    report = {"ok": [], "skipped": [], "failed": []}

    def do(level, write_bundle):
        if level in SKIP_LEVELS or (only and level not in only):
            return
        out = levels_out / level
        if (out / "level-data.json").exists() and not args.force:
            report["skipped"].append(level)
            return
        out.mkdir(parents=True, exist_ok=True)
        write_bundle(out / "bundle")
        try:
            got = extract_level(out / "bundle", out)
        except Exception as error:  # noqa: BLE001 - report and carry on with the next level
            report["failed"].append(f"{level}: {type(error).__name__}: {str(error)[:120]}")
            return
        if not got["level"] or not got["audio"]:
            report["failed"].append(f"{level}: no level data or song in the bundle (protected or not a level)")
            return
        report["ok"].append(level)
        print(f"  {level}: {got['text']} map files, {got['audio']} song")

    for obb in obbs:
        print(f"Reading {obb.name}...")
        with zipfile.ZipFile(obb) as z:
            for name in z.namelist():
                if name.startswith("assets/aa/Android/") and any(k in name for k in METADATA_BUNDLES):
                    target = bundles_out / name.rsplit("/", 1)[1]
                    if not target.exists() or target.stat().st_size != z.getinfo(name).file_size:
                        with z.open(name) as a, open(target, "wb") as b:
                            shutil.copyfileobj(a, b)
            for name in sorted(z.namelist()):
                if name.startswith("assets/BeatmapLevelsData/") and not name.endswith("/"):
                    def write(target, name=name):
                        with z.open(name) as a, open(target, "wb") as b:
                            shutil.copyfileobj(a, b)
                    do(name.rsplit("/", 1)[1], write)
    for f in singles:
        do(f.name, lambda target, f=f: shutil.copyfile(f, target))
    print(f"{len(report['ok'])} levels extracted, {len(report['skipped'])} already there, {len(report['failed'])} failed")
    for line in report["failed"]:
        print("  failed " + line)
    if not args.no_index:
        cmd_index(args)


# --- index ---------------------------------------------------------------------------

def trees(env, wanted):
    for obj in env.objects:
        if obj.type.name != "MonoBehaviour":
            continue
        try:
            tree = obj.read_typetree()
        except Exception:  # noqa: BLE001
            continue
        if wanted(tree):
            yield obj, tree


def cab(assets_file, file_id):
    """Name of the serialized file a reference's m_FileID points into."""
    if file_id == 0:
        return assets_file.name.lower()
    return assets_file.externals[file_id - 1].path.rsplit("/", 1)[-1].lower()


def colors(scheme):
    if not isinstance(scheme, dict) or "_saberAColor" not in scheme:
        return None
    out = {}
    for key, name in COLOR_KEYS.items():
        c = scheme.get(key)
        if isinstance(c, dict):
            out[name] = [round(float(c.get(k, 1.0)), 4) for k in ("r", "g", "b", "a")]
    return out


def bundle_files(predicate=lambda n: True):
    files = []
    for root in CONFIG["bundles"]:
        if root.is_dir():
            files += [str(p) for p in sorted(root.iterdir()) if p.is_file() and predicate(p.name)]
    return files


def make_covers(level_ids):
    """Cover art (256 px PNG) for each level, from the pack bundles. All
    metadata bundles load together so a cover kept in another one resolves."""
    import UnityPy
    out = CONFIG["covers"]
    out.mkdir(parents=True, exist_ok=True)
    wanted = {i.lower() for i in level_ids if not (out / f"{i.lower()}.png").exists()}
    if not wanted:
        return
    env = UnityPy.load(*bundle_files(lambda n: any(k in n for k in METADATA_BUNDLES)))
    by_name = {}
    for f in env.files.values():
        for sub in getattr(f, "files", {}).values():
            if hasattr(sub, "objects"):
                by_name[os.path.basename(getattr(sub, "name", "")).lower()] = sub
    made = 0
    for obj, tree in trees(env, lambda t: "_songName" in t and "_coverImage" in t):
        level = str(tree.get("_levelID", "")).lower()
        if level not in wanted:
            continue
        ref = tree["_coverImage"]
        try:
            target_file = obj.assets_file if ref["m_FileID"] == 0 else by_name.get(cab(obj.assets_file, ref["m_FileID"]))
            target = target_file.objects.get(ref["m_PathID"]) if target_file else None
            if target is not None:
                target.read().image.convert("RGB").resize((256, 256)).save(out / f"{level}.png")
                made += 1
        except Exception as error:  # noqa: BLE001 - a missing cover is not fatal
            print(f"  no cover for {level}: {type(error).__name__}")
    print(f"{made} covers written")


def cmd_index(args):
    """Index every extracted level: the exact beatmap and lightshow file of
    each characteristic and difficulty (matched through the asset ids in the
    level bundle), note jump speed and offset, environment, colour scheme."""
    import UnityPy
    shared = bundle_files(lambda n: n.startswith("sharedassets"))
    characteristic = {}
    if shared:
        characteristic = {o.path_id: t["_serializedName"] for o, t in trees(UnityPy.load(*shared), lambda t: "_serializedName" in t)}

    env_files = bundle_files(lambda n: "environment" in n or n.startswith(("sharedassets", "default_assets")))
    env_colors = {}
    if env_files:
        env = UnityPy.load(*env_files)
        schemes, environments = {}, []
        for obj, t in trees(env, lambda t: "_colorScheme" in t):
            c = colors(t["_colorScheme"])
            if c:
                schemes[(obj.assets_file.name.lower(), obj.path_id)] = c
            elif isinstance(t["_colorScheme"], dict) and "m_PathID" in t["_colorScheme"] and "_serializedName" in t:
                environments.append((obj.assets_file, t))
        for f, t in environments:
            ref = t["_colorScheme"]
            try:
                c = schemes.get((cab(f, ref["m_FileID"]), ref["m_PathID"]))
            except IndexError:
                c = None
            env_colors[t["_serializedName"]] = {"name": t.get("_environmentName", ""), "colors": c}

    meta = {}
    for path in bundle_files(lambda n: "_pack_assets_" in n):
        pack = os.path.basename(path).split("_pack_assets_")[0]
        for _, t in trees(UnityPy.load(path), lambda t: "_songName" in t):
            sets = {}
            for s in t.get("_previewDifficultyBeatmapSets", []):
                name = characteristic.get(s["_beatmapCharacteristic"]["m_PathID"], "Unknown")
                sets[name] = {int(d["_difficulty"]): d for d in s.get("_previewDifficultyBeatmaps", [])}
            level_schemes = []
            for entry in t.get("_colorSchemes", []) or []:
                c = colors(entry.get("_colorScheme", entry) if isinstance(entry, dict) else None)
                level_schemes.append({"override": bool(entry.get("_useOverride", True)) if isinstance(entry, dict) else False, "colors": c})
            meta[str(t["_levelID"]).lower()] = {
                "pack": pack, "title": t.get("_songName", ""), "sub": t.get("_songSubName", ""),
                "artist": t.get("_songAuthorName", ""), "mapper": t.get("_levelAuthorName", ""),
                "bpm": float(t.get("_beatsPerMinute", 0)),
                "env": (t.get("_environmentName") or {}).get("_environmentName", ""),
                "envAll": (t.get("_allDirectionsEnvironmentName") or {}).get("_environmentName", ""),
                "envs": [e.get("_environmentName", "") for e in t.get("_environmentNames", []) or []],
                "schemes": level_schemes, "sets": sets}

    songs, seen = [], set()
    for root in CONFIG["levels"]:
        if not root.is_dir():
            continue
        for d in sorted(p for p in root.iterdir() if (p / "bundle").is_file()):
            if d.name in seen:
                continue
            lv = UnityPy.load(str(d / "bundle"))
            text, level = {}, None
            for obj in lv.objects:
                if obj.type.name == "TextAsset":
                    name = obj.read().m_Name
                    text[obj.path_id] = (name[:-3] if name.endswith(".gz") else name) + ".json"
                elif obj.type.name == "MonoBehaviour" and level is None:
                    try:
                        t = obj.read_typetree()
                    except Exception:  # noqa: BLE001
                        continue
                    if "_difficultyBeatmapSets" in t:
                        level = t
            if level is None:
                continue
            seen.add(d.name)
            level_id = level["m_Name"][:-len("BeatmapLevelData")] if level["m_Name"].endswith("BeatmapLevelData") else level["m_Name"]
            m = meta.get(level_id.lower()) or meta.get(d.name) or {}
            wavs = sorted(f.name for f in d.iterdir() if f.suffix == ".wav" and not f.is_symlink())
            audio_data = sorted(f.name for f in d.iterdir() if f.name.endswith(".audio.json"))
            sets = []
            for s in level["_difficultyBeatmapSets"]:
                name = s["_beatmapCharacteristicSerializedName"]
                diffs = []
                for b in s["_difficultyBeatmaps"]:
                    n = int(b["_difficulty"])
                    info = m.get("sets", {}).get(name, {}).get(n, {})
                    njs = float(info.get("_noteJumpMovementSpeed", 0) or 0)
                    beatmap = text.get(b["_beatmapAsset"]["m_PathID"], "")
                    light = text.get((b.get("_lightshowAsset") or {}).get("m_PathID"), "")
                    diffs.append({
                        "difficulty": n, "name": DIFFICULTY.get(n, str(n)),
                        "beatmap": str(d / beatmap) if beatmap else "", "lightshow": str(d / light) if light else "",
                        "njs": njs or DEFAULT_NJS.get(n, 10.0), "njsAuthored": njs,
                        "offset": float(info.get("_noteJumpStartBeatOffset", 0) or 0),
                        "envIdx": int(info.get("_environmentNameIdx", 0) or 0),
                        "schemeIdx": int(info.get("_beatmapColorSchemeIdx", 0) or 0)})
                sets.append({"characteristic": name, "difficulties": diffs})
            songs.append({
                "id": d.name, "source": "extracted", "levelId": level_id,
                "title": m.get("title") or level_id, "sub": m.get("sub", ""), "artist": m.get("artist", ""),
                "mapper": m.get("mapper", ""), "pack": m.get("pack", ""), "bpm": m.get("bpm", 0.0),
                "env": m.get("env", ""), "envAll": m.get("envAll", ""), "envs": m.get("envs", []),
                "schemes": m.get("schemes", []), "dir": "",
                "audio": str(d / wavs[0]) if wavs else "",
                "audioData": str(d / audio_data[0]) if audio_data else "",
                "sets": sets})
    if songs and env_files:
        make_covers([s["levelId"] for s in songs])
    for s in songs:
        cover = CONFIG["covers"] / f"{s['levelId'].lower()}.png"
        s["cover"] = str(cover) if cover.exists() else ""
    songs.sort(key=lambda s: (s["title"].lower(), s["id"]))
    json.dump({"envColors": env_colors, "songs": songs}, open(HERE / "index.json", "w"), indent=1)
    print(f"{len(songs)} songs indexed ({len(env_colors)} environments)")


# --- zip export -----------------------------------------------------------------------

def encode_ogg(src, dst):
    """Song as Ogg Vorbis with the first encoder that works. Some ffmpeg
    builds lack libvorbis, so its built-in encoder is the last resort."""
    attempts = [["sox", src, "-C", "6", dst], ["oggenc", "-Q", "-q", "6", "-o", dst, src],
                ["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-c:a", "libvorbis", "-q:a", "6", dst],
                ["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-c:a", "vorbis", "-strict", "-2", "-q:a", "6", dst]]
    for cmd in attempts:
        if shutil.which(cmd[0]) and subprocess.run(cmd, capture_output=True).returncode == 0:
            return True
    return False


def cmd_zip(args):
    """Export an extracted level as a standard map zip (v4 Info.dat, every
    characteristic and difficulty, lightshows, timing, song as Ogg, cover)."""
    index = json.load(open(HERE / "index.json")) if (HERE / "index.json").exists() else {"songs": []}
    want = args.level.lower()
    matches = [s for s in index["songs"] if s["id"].lower() == want or s["title"].lower() == want]
    if not matches:
        fail(f"No extracted level called {args.level} (use its folder name or title).")
    song = matches[0]
    safe = "".join(c if c.isalnum() else "_" for c in f"{song['title']} {song['sub']}".strip()).strip("_")
    out = Path(args.out).expanduser() if args.out else CONFIG["library"] / "zips" / f"{safe}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)
    envs = song["envs"] or [song["env"] or "DefaultEnvironment"]
    if not song["envs"] and song["envAll"]:
        envs = envs + [song["envAll"]]
    files, lights, beatmaps = {}, {}, []
    for s in song["sets"]:
        rotating = s["characteristic"] in ("360Degree", "90Degree")
        for d in s["difficulties"]:
            name = f"{d['name']}{s['characteristic']}.dat"
            files[name] = json.dumps(read_json(d["beatmap"]))
            light = ""
            if d["lightshow"]:
                light = lights.get(d["lightshow"]) or Path(d["lightshow"]).name.replace(".lightshow.json", ".lightshow.dat")
                lights[d["lightshow"]] = light
                files[light] = json.dumps(read_json(d["lightshow"]))
            env_idx = d["envIdx"] if song["envs"] else (len(envs) - 1 if rotating and song["envAll"] else 0)
            beatmaps.append({"characteristic": s["characteristic"], "difficulty": d["name"],
                             "beatmapAuthors": {"mappers": [song["mapper"]] if song["mapper"] else [], "lighters": []},
                             "environmentNameIdx": env_idx, "beatmapColorSchemeIdx": max(d["schemeIdx"], 0),
                             "noteJumpMovementSpeed": d["njsAuthored"] or d["njs"], "noteJumpStartBeatOffset": d["offset"],
                             "lightshowDataFilename": light, "beatmapDataFilename": name})
    audio = read_json(song["audioData"]) if song["audioData"] else {}
    files["BPMInfo.dat"] = json.dumps(audio)
    duration = audio.get("songSampleCount", 0) / max(audio.get("songFrequency", 44100), 1)
    info = {"version": "4.0.1", "song": {"title": song["title"], "subTitle": song["sub"], "author": song["artist"]},
            "audio": {"songFilename": "song.ogg", "songDuration": round(duration, 3), "audioDataFilename": "BPMInfo.dat",
                      "bpm": song["bpm"], "lufs": 0, "previewStartTime": 0, "previewDuration": 10},
            "songPreviewFilename": "song.ogg", "coverImageFilename": "cover.png" if song.get("cover") else "",
            "environmentNames": envs, "colorSchemes": [], "difficultyBeatmaps": beatmaps}
    ogg = out.parent / (out.stem + ".song.ogg")
    if not encode_ogg(song["audio"], str(ogg)):
        fail("Exporting needs an Ogg Vorbis encoder: install sox (macOS: brew install sox).")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("Info.dat", json.dumps(info, indent=1))
        for name, data in files.items():
            z.writestr(name, data)
        z.write(ogg, "song.ogg")
        if song.get("cover"):
            z.write(song["cover"], "cover.png")
    ogg.unlink()
    print(f"{out} ({out.stat().st_size // 1024} KB, {len(beatmaps)} difficulties)")


# --- inspect and open ------------------------------------------------------------------

def cmd_inspect(args):
    """List a Unity bundle's objects, or dump the data of those whose name
    contains NAME."""
    import UnityPy
    env = UnityPy.load(args.bundle)
    needle = (args.name or "").lower()
    for obj in env.objects:
        kind = obj.type.name
        try:
            data = obj.read()
            name = getattr(data, "m_Name", "") or ""
        except Exception as error:  # noqa: BLE001
            data, name = None, f"<unreadable: {error}>"
        if needle:
            if needle in str(name).lower() and kind == "MonoBehaviour":
                print(name)
                print(json.dumps(obj.read_typetree(), indent=1, default=str)[:8000])
                print("----")
            continue
        extra = ""
        if kind == "AudioClip" and data is not None:
            extra = f" length={getattr(data, 'm_Length', '?')} freq={getattr(data, 'm_Frequency', '?')}"
        elif kind == "TextAsset" and data is not None:
            extra = f" bytes={len(getattr(data, 'm_Script', b''))}"
        print(f"{obj.path_id:>22} {kind:<16} {name}{extra}")


def cmd_open(args):
    run = HERE / "run.sh"
    os.execv("/bin/sh", ["/bin/sh", str(run)] + args.flags)


def cmd_all(args):
    args.source = str(cmd_pull(args))
    if args.dry_run:
        print(f"Would then extract {args.source} and index it.")
        return
    cmd_extract(args)
    print("Done. Open the viewer with \"Open Beatmap Explorer.command\" (or: python3 mapkit.py open).")


def main():
    global CONFIG
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default=str(HERE / "config.json"), help="settings file (default: config.json here)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("pull", "all"):
        p = sub.add_parser(name)
        p.add_argument("--serial", help="which headset, when several are connected (see `adb devices`)")
        p.add_argument("--no-custom", action="store_true", help="skip custom maps installed on the headset")
        p.add_argument("--dry-run", action="store_true", help="show the adb commands without running them")
        if name == "all":
            p.add_argument("--only", help="comma-separated level names to extract")
            p.add_argument("--force", action="store_true")
            p.add_argument("--no-index", action="store_true")
    p = sub.add_parser("extract")
    p.add_argument("source", nargs="?", help="pulled folder or main .obb (default: the latest pull)")
    p.add_argument("--only", help="comma-separated level names, e.g. 100bills,takemybreath")
    p.add_argument("--force", action="store_true", help="extract again even if already there")
    p.add_argument("--no-index", action="store_true", help="don't rebuild the index afterwards")
    sub.add_parser("index")
    p = sub.add_parser("zip")
    p.add_argument("level", help="folder name or title of an extracted level")
    p.add_argument("--out", help="zip to write (default: library/zips/<title>.zip)")
    p = sub.add_parser("inspect")
    p.add_argument("bundle")
    p.add_argument("name", nargs="?")
    p = sub.add_parser("open")
    p.add_argument("flags", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    CONFIG = load_config(Path(args.config).expanduser())
    if args.cmd in ("extract", "index", "inspect") or (args.cmd == "all" and not args.dry_run):
        need_unitypy(sys.argv[1:])
    {"pull": cmd_pull, "extract": cmd_extract, "index": cmd_index, "zip": cmd_zip,
     "inspect": cmd_inspect, "open": cmd_open, "all": cmd_all}[args.cmd](args)


CONFIG = {}
if __name__ == "__main__":
    main()
