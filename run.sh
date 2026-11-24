#!/bin/sh
# Open Beatmap Explorer in Godot 4.4 or newer. Extra flags go to the viewer,
# e.g.  ./run.sh --open="$HOME/Downloads/some-map.zip"
# Godot is found from $GODOT, "godot" in config.json, the PATH, or /Applications.
here=$(cd "$(dirname "$0")" && pwd)
godot="${GODOT:-}"
if [ -z "$godot" ] && [ -f "$here/config.json" ]; then
  godot=$(sed -n 's/.*"godot"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$here/config.json" | head -1)
  case "$godot" in "~"*) godot="$HOME${godot#\~}" ;; esac
fi
if [ -z "$godot" ]; then
  for g in "$(command -v godot 2>/dev/null)" "$(command -v godot4 2>/dev/null)" \
           /Applications/Godot.app/Contents/MacOS/Godot "$HOME/Applications/Godot.app/Contents/MacOS/Godot" \
           /Applications/Godot_mono.app/Contents/MacOS/Godot; do
    if [ -n "$g" ] && [ -x "$g" ]; then godot="$g"; break; fi
  done
fi
if [ ! -x "$godot" ]; then
  echo "Beatmap Explorer needs Godot 4.4 or newer: https://godotengine.org/download" >&2
  echo "(or set \"godot\" in config.json to where it is)" >&2
  exit 1
fi
exec "$godot" --path "$here/viewer" -- "$@"
