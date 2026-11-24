# Training a map maker

An audio-to-map model: it listens to a song and writes a Beat Saber map for a
chosen difficulty, note by note, the way a speech model writes a transcript.
Beatmap Explorer is how its maps are judged: open them beside the real ones.

**Status:** pipeline built and checked end to end on a small local run; the
first full training run (on community maps, on a rented GPU) is next. Results
will be added here as they come in, good or bad.

## The idea

- **Ears:** an encoder hears an 8-second window of the song as a log-mel
  spectrogram (96 bands, 100 frames per second).
- **Hands:** a decoder writes the map for that window as tokens: the moment
  (10 ms steps), then each object at it: a note (hand, lane, row, cut
  direction), a bomb, a wall (lane, width, row, height, length) or a lane turn
  (360°/90° maps). It also predicts the note jump speed for the difficulty.
- **One model, every difficulty:** the difficulty and the characteristic
  (Standard, One Saber, No Arrows, 360°, 90°, Legacy) are the first tokens, so
  what it learns about music is shared by all of them.
- **Flow:** windows overlap by 2 seconds and each one is given the notes already
  placed there, so a long map stays continuous.
- **Guard rails:** while it writes, the token grammar is enforced and a hand may
  not swing the same way twice within half a second.
- **Lights:** a second model (a "lighter") that hears the song and reads the
  finished notes; not built yet.

## Data

Maps come from [BeatSaver](https://beatsaver.com)'s public API, in three tiers:
curated, ranked, and maps by verified mappers rated 80 % or better. Left out:
maps declared AI-made (by the uploader, BeatSaver's AI check or an admin) and
difficulties that need Noodle Extensions or Mapping Extensions. Songs are
split into training and held-out sets by title and artist, so the scores are on
songs the model never heard.

The maps themselves are not in this repository (they belong to their mappers,
and the songs are copyrighted). `beatsaver.py` writes the list of map ids and
hashes, and `prepare.py` rebuilds the dataset from it for anyone who runs it.

## Steps

```sh
python3 -I train/beatsaver.py                         # the list of maps (metadata only)
python3 -I train/prepare.py beatsaver --workers 8     # download, read, spectrogram; keeps only the result
python3 -I train/train.py --data library/train/beatsaver --out library/train/runs/base --size base
python3 -I train/evaluate.py --ckpt library/train/runs/base/best.pt --data library/train/beatsaver
python3 -I train/generate.py --ckpt library/train/runs/base/best.pt --audio song.mp3 --title "..." --artist "..."
```

`generate.py` writes a map folder into `library/community/generated/`, which
the viewer lists. `prepare.py official` prepares levels extracted with mapkit
from your own copy of the game; that output and any model tuned on it are for
your own machine only.

Needs Python 3.10+, PyTorch 2.7+, librosa, numpy (`train/requirements.txt`),
and sox for writing songs as Ogg.

## Checks

- `tests/compare_with_viewer.py`: the training reader and the viewer read every
  object of a map identically (time to the microsecond, place, colour, cut,
  rotation, lights). Last run: 611 difficulties (v2.0 to v4.1), all identical.
- `tests/test_pipeline.py`: token round trip, chart cache, split by song,
  batch shapes.

## What the scores mean

`evaluate.py` reports, per held-out song and difficulty: timing F1 (notes at
the right moments within 30 ms), colour F1, placement and cut agreement on the
matched notes, density (generated / real notes per second) and swing errors per
100 notes (with the real map's own count beside it). A model can be good and
still differ from the real map, since songs can be mapped many ways, so the
numbers go together with looking at the maps.
