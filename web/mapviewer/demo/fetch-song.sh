#!/bin/sh
# The demo's music: "Raving Energy (faster)" by Kevin MacLeod (incompetech.com),
# licensed under Creative Commons: By Attribution 4.0
# (https://creativecommons.org/licenses/by/4.0/). Not kept in this repository.
set -eu
cd "$(dirname "$0")"
curl -fsSL -A "beatmap-explorer/0.1" -o raving-energy-faster.mp3 \
  "https://incompetech.com/music/royalty-free/mp3-royaltyfree/Raving%20Energy%20(faster).mp3"
echo "saved raving-energy-faster.mp3"
