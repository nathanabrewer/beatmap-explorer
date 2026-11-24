#!/bin/sh
# Double-click in Finder: copies the song files of your own copy of the game
# off your Quest (USB cable, developer mode on), unpacks and indexes them.
cd "$(dirname "$0")" || exit 1
python3 mapkit.py all
status=$?
echo
if [ "$status" -eq 0 ]; then
  echo "All done. Double-click \"Open Beatmap Explorer.command\" to look at them."
else
  echo "That didn't finish (see the message above)."
fi
printf "Press Return to close this window. "
read -r _
