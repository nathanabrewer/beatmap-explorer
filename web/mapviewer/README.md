# Map viewer

`mapviewer.js` plays a beatmap in 3D: blocks fly down a lane at the map's note
jump speed and reach the player on their beat, and a figure swings what the map
asks for. One lane, or two side by side to compare maps of the same song. It
needs only three.js (r160 or later, with `OrbitControls`), passed in by the page.

```js
import * as THREE from './vendor/three.min.js';            // an ES module build of three.js
import { createMapViewer, notesFromChart } from './mapviewer.js';

const viewer = createMapViewer(canvas, { three: THREE, look: { hands: ['#ff3d9a', '#14c2dd'] } });
viewer.load([{ notes: notesFromChart(difficultyJson, bpm), njs: 16, name: 'Expert+' }]);
viewer.follow(audioElement);      // the picture follows the audio's clock
viewer.soundDelay = 0.12;         // seconds the sound reaches the listener late (Bluetooth)
viewer.camera('eyes0');           // glide to 'behind', 'eyes0', 'eyes1', 'side', 'top', 'wide'
```

- **Notes:** `[seconds, hand (0 left, 1 right), lane x 0-3, layer y 0-2, cut direction 0-8]`.
  `notesFromChart(json, bpm)` reads a v2 (`_notes`) or v3 (`colorNotes`)
  difficulty file with one BPM.
- **Motion:** a model, not a recording. Each cut is a stroke through its note in
  its arrow's direction. Between cuts the saber tip curves from one cut into the
  next. The arms reach the grips by two-bone inverse kinematics. A smooth
  pendulum means the map flows; a snap means it asks the hand to jump.
- **Camera:** always free: drag to orbit, right-drag to pan, scroll to zoom.
  `camera(name)` glides to a preset from wherever it is; a head the camera is
  inside of is hidden.
- **Look:** `look` overrides the colours: hands, suit, floor, lines, frame, background.
- **Not yet:** walls, bombs, arcs and chains, and cut effects.

The demo (`index.html`) plays a generated map of a CC BY track (fetch the song once
with `demo/fetch-song.sh`: "Raving Energy (faster)" Kevin MacLeod (incompetech.com),
licensed under Creative Commons: By Attribution 4.0), or a map and song you open
(read in the browser, never uploaded). Serve the folder over http
(ES modules don't load from `file://`): `python3 -m http.server` here, then open
http://127.0.0.1:8000/.

MIT licence. three.js: MIT, three.js authors (vendor/LICENSE-three.txt).
