// mapviewer.js: play a beatmap in 3D. Blocks fly down a lane at the map's
// note jump speed and reach the strike plane (z = 0) on their beat; a figure
// stands behind it and swings what the map asks for. One lane, or two side by
// side to compare maps of the same song. No dependencies beyond three.js,
// which the caller passes in (so the page decides how to load it).
//
//   import * as THREE from './vendor/three.min.js';   // needs OrbitControls too
//   import { createMapViewer, notesFromChart } from './mapviewer.js';
//   const viewer = createMapViewer(canvas, { three: THREE });
//   viewer.load([{ notes, njs: 16, name: 'My map' }]);  // one or two lanes
//   viewer.follow(audioElement);                          // the clock follows the audio
//
// Notes are [time in seconds, hand (0 left, 1 right), lane x (0-3), layer y
// (0-2), cut direction (0 up, 1 down, 2 left, 3 right, 4 up-left, 5 up-right,
// 6 down-left, 7 down-right, 8 any)]. notesFromChart() reads a v2 or v3 map
// difficulty file with one BPM.
//
// The motion is a model, not a recording: each cut is a stroke through its
// note in its arrow's direction; between cuts the saber tip curves from one
// cut into the next (cubic Hermite); the arms reach the grips by two-bone
// inverse kinematics from the shoulders. A smooth pendulum means the map
// flows; a sudden snap means it asks the hand to jump.
//
// MIT licence. three.js: MIT, three.js authors.

const LANE = 0.6;
const LAYER = [0.85, 1.38, 1.89];
const REACH = 0.45;
const DIRS = { 0: [0, 1], 1: [0, -1], 2: [-1, 0], 3: [1, 0], 4: [-0.7071, 0.7071], 5: [0.7071, 0.7071], 6: [-0.7071, -0.7071], 7: [0.7071, -0.7071] };
const OPPOSITE = { 0: 1, 1: 0, 2: 3, 3: 2, 4: 7, 7: 4, 5: 6, 6: 5 };
const REST = [[-0.35, 1.05], [0.35, 1.05]];
const SHOULDER_Y = 1.45, BODY_Z = 0.95, BLADE = 1.0;

export const DEFAULT_LOOK = { hands: ['#d64d8a', '#2fa8d8'], suit: '#2a3550', joint: '#3a4868', head: '#4a5a80', visor: '#9fb6ff', floor: '#0b1322', lines: '#22345e', frame: '#2b4170', background: '#060a12', label: '#e8ecf4' };

const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
const layerY = (y) => { const i = Math.max(0, Math.min(2, y)); const f = Math.floor(Math.min(i, 1.999)); return LAYER[f] + (LAYER[f + 1] - LAYER[f]) * (i - f); };

// ---- reading maps ----------------------------------------------------------

/** Notes from a map difficulty file (v2 `_notes` or v3 `colorNotes`), times from one BPM. */
export function notesFromChart(chart, bpm) {
  const sec = (beat) => beat * 60 / bpm;
  if (Array.isArray(chart.colorNotes)) return chart.colorNotes.map((n) => [sec(n.b), n.c ?? 0, n.x ?? 0, n.y ?? 0, n.d ?? 8]).sort((a, b) => a[0] - b[0]);
  if (Array.isArray(chart._notes)) return chart._notes.filter((n) => n._type === 0 || n._type === 1).map((n) => [sec(n._time), n._type, n._lineIndex, n._lineLayer, n._cutDirection]).sort((a, b) => a[0] - b[0]);
  return [];
}

// ---- the motion model ------------------------------------------------------

/** One hand's swings: notes of that hand within 30 ms are one swing; dots take the direction that flows. */
export function swings(notes, hand) {
  const sw = [];
  for (const n of notes.filter((n) => n[1] === hand)) {
    const last = sw[sw.length - 1];
    if (last && n[0] - last.t < 0.03) { last.xs.push(n[2]); last.ys.push(n[3]); if (last.d === null && n[4] !== 8) last.d = n[4]; }
    else sw.push({ t: n[0], xs: [n[2]], ys: [n[3]], d: n[4] === 8 ? null : n[4] });
  }
  let prev = null;
  for (const s of sw) {
    s.x = (mean(s.xs) - 1.5) * LANE; s.y = layerY(mean(s.ys));
    if (s.d === null) s.d = prev === null ? 1 : OPPOSITE[prev];
    prev = s.d; s.v = DIRS[s.d];
  }
  sw.forEach((s, i) => { const a = i ? s.t - sw[i - 1].t : 1, b = i < sw.length - 1 ? sw[i + 1].t - s.t : 1; s.c = Math.max(0.02, Math.min(0.07, a / 3, b / 3)); });
  return sw;
}

function hermite(p0, m0, p1, m1, u) {
  const u2 = u * u, u3 = u2 * u;
  return [0, 1].map((k) => (2 * u3 - 3 * u2 + 1) * p0[k] + (u3 - 2 * u2 + u) * m0[k] + (-2 * u3 + 3 * u2) * p1[k] + (u3 - u2) * m1[k]);
}

/** Where a hand's saber tip is at time t, in the strike plane (x across, y up, metres). */
export function tipAt(sw, hand, t) {
  const rest = REST[hand];
  if (!sw.length) return rest;
  if (t < sw[0].t - sw[0].c) {
    const s = sw[0], entry = [s.x - s.v[0] * REACH, s.y - s.v[1] * REACH], u = Math.max(0, 1 - (s.t - s.c - t) / 0.5);
    return hermite(rest, [0, 0], entry, [s.v[0] * 0.6, s.v[1] * 0.6], u);
  }
  let lo = 0, hi = sw.length - 1;
  while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (sw[mid].t - sw[mid].c <= t) lo = mid; else hi = mid - 1; }
  const s = sw[lo];
  if (t <= s.t + s.c) { const k = (t - s.t) / s.c; return [s.x + s.v[0] * REACH * k, s.y + s.v[1] * REACH * k]; }
  const exit = [s.x + s.v[0] * REACH, s.y + s.v[1] * REACH], n = sw[lo + 1];
  if (!n) { const u = Math.min(1, (t - s.t - s.c) / 0.6); return hermite(exit, [s.v[0] * 0.6, s.v[1] * 0.6], rest, [0, 0], u); }
  const t0 = s.t + s.c, t1 = n.t - n.c, T = Math.max(t1 - t0, 1e-3), u = Math.min(1, (t - t0) / T);
  const entry = [n.x - n.v[0] * REACH, n.y - n.v[1] * REACH];
  const m0 = Math.min(REACH / s.c * T, 1.1), m1 = Math.min(REACH / n.c * T, 1.1);
  return hermite(exit, [s.v[0] * m0, s.v[1] * m0], entry, [n.v[0] * m1, n.v[1] * m1], u);
}

// ---- the viewer ------------------------------------------------------------

export function createMapViewer(canvas, { three: T, look = {}, laneGap = 3.4 } = {}) {
  if (!T || !T.WebGLRenderer) throw new Error('createMapViewer needs { three }: a three.js module namespace');
  const L = { ...DEFAULT_LOOK, ...look };
  const renderer = new T.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  const scene = new T.Scene();
  scene.background = new T.Color(L.background);
  scene.fog = new T.Fog(L.background, 10, 75);
  const camera = new T.PerspectiveCamera(62, 2, 0.05, 200);
  const orbit = T.OrbitControls ? new T.OrbitControls(camera, canvas) : null;
  if (orbit) orbit.enabled = false;
  scene.add(new T.HemisphereLight('#8899ff', '#080810', 0.9));
  const handColors = L.hands.map((c) => new T.Color(c));
  let lanes = [], cameraName = 'behind', time = 0, followed = null, delay = 0, raf = 0, last = 0, playing = false;

  const v = (x, y, z) => new T.Vector3(x, y, z);
  function limb(mesh, a, b) {
    const d = new T.Vector3().subVectors(b, a), len = d.length() || 1e-4;
    mesh.position.copy(a).addScaledVector(d, 0.5); mesh.scale.set(1, len, 1);
    mesh.quaternion.setFromUnitVectors(v(0, 1, 0), d.multiplyScalar(1 / len));
  }
  function label(text) {
    const c = document.createElement('canvas'); c.width = 1024; c.height = 128;
    const g = c.getContext('2d'); g.font = '600 52px system-ui, sans-serif'; g.fillStyle = L.label; g.textAlign = 'center'; g.fillText(text, 512, 80);
    const tex = new T.CanvasTexture(c); if (T.SRGBColorSpace) tex.colorSpace = T.SRGBColorSpace;
    const sp = new T.Sprite(new T.SpriteMaterial({ map: tex, transparent: true, depthWrite: false })); sp.scale.set(3.2, 0.4, 1); return sp;
  }
  function figure(root) {
    const suit = new T.MeshBasicMaterial({ color: L.suit }), joint = new T.MeshBasicMaterial({ color: L.joint });
    const f = {};
    f.head = new T.Mesh(new T.SphereGeometry(0.11, 20, 14), new T.MeshBasicMaterial({ color: L.head })); root.add(f.head);
    const visor = new T.Mesh(new T.BoxGeometry(0.17, 0.045, 0.02), new T.MeshBasicMaterial({ color: L.visor })); visor.position.set(0, 0.01, -0.105); f.head.add(visor);
    const torso = new T.Mesh(new T.CapsuleGeometry(0.15, 0.42, 6, 14), suit); torso.scale.set(1, 1, 0.6); torso.position.set(0, 1.17, BODY_Z); root.add(torso);
    const shoulders = new T.Mesh(new T.CapsuleGeometry(0.06, 0.34, 4, 10), suit); shoulders.rotation.z = Math.PI / 2; shoulders.position.set(0, SHOULDER_Y, BODY_Z); root.add(shoulders);
    for (const sd of [-1, 1]) { const m = new T.Mesh(new T.CylinderGeometry(0.06, 0.05, 1, 10), suit); root.add(m); limb(m, v(0.1 * sd, 0.92, BODY_Z), v(0.13 * sd, 0, BODY_Z - 0.05)); }
    f.arms = [0, 1].map((h) => {
      const upper = new T.Mesh(new T.CylinderGeometry(0.04, 0.035, 1, 10), suit), fore = new T.Mesh(new T.CylinderGeometry(0.035, 0.03, 1, 10), suit);
      const elbow = new T.Mesh(new T.SphereGeometry(0.042, 10, 8), joint), hand = new T.Mesh(new T.SphereGeometry(0.045, 10, 8), joint);
      const color = handColors[h];
      const core = new T.Mesh(new T.CylinderGeometry(0.009, 0.009, 1, 8), new T.MeshBasicMaterial({ color: color.clone().lerp(new T.Color('#ffffff'), 0.65) }));
      const glow = new T.Mesh(new T.CylinderGeometry(0.03, 0.03, 1, 10, 1, true), new T.MeshBasicMaterial({ color, transparent: true, opacity: 0.35, blending: T.AdditiveBlending, depthWrite: false }));
      const geo = new T.BufferGeometry(); geo.setAttribute('position', new T.BufferAttribute(new Float32Array(26 * 3), 3));
      const trail = new T.Line(geo, new T.LineBasicMaterial({ color, transparent: true, opacity: 0.7, blending: T.AdditiveBlending }));
      for (const m of [upper, fore, elbow, hand, core, glow, trail]) root.add(m);
      return { upper, fore, elbow, hand, core, glow, trail };
    });
    return f;
  }
  function grip(S, tip) {
    const toTip = new T.Vector3().subVectors(tip, S), dist = toTip.length();
    const G = S.clone().addScaledVector(toTip.normalize(), Math.min(0.52, dist * 0.55)); G.z -= 0.12; return G;
  }
  function pose(f, sw, t) {
    const tips = [0, 1].map((h) => tipAt(sw[h], h, t));
    f.head.position.set(0, 1.66, BODY_Z + 0.02);
    f.head.lookAt((tips[0][0] + tips[1][0]) * 0.2, 1.62, BODY_Z - 1); f.head.rotateY(Math.PI);
    for (const h of [0, 1]) {
      const a = f.arms[h], side = h === 0 ? -1 : 1, S = v(0.2 * side, SHOULDER_Y, BODY_Z), tip = v(tips[h][0], tips[h][1], 0), G = grip(S, tip);
      const n = new T.Vector3().subVectors(G, S), d = Math.min(Math.max(n.length(), 0.05), 0.599); n.normalize();
      const alpha = Math.acos(Math.min(1, Math.max(-1, d / 0.6)));
      const pole = v(0.6 * side, -0.7, 0.25); pole.addScaledVector(n, -n.dot(pole)).normalize();
      const E = S.clone().addScaledVector(n, 0.3 * Math.cos(alpha)).addScaledVector(pole, 0.3 * Math.sin(alpha));
      limb(a.upper, S, E); limb(a.fore, E, G); a.elbow.position.copy(E); a.hand.position.copy(G);
      const end = G.clone().addScaledVector(new T.Vector3().subVectors(tip, G).normalize(), BLADE);
      limb(a.core, G, end); limb(a.glow, G, end);
      const pos = a.trail.geometry.attributes.position;
      for (let k = 0; k < 26; k++) {
        const tk = tipAt(sw[h], h, t - k * 0.01), tp = v(tk[0], tk[1], 0), gk = grip(S, tp);
        const ek = gk.clone().addScaledVector(tp.sub(gk).normalize(), BLADE); pos.setXYZ(k, ek.x, ek.y, ek.z);
      }
      pos.needsUpdate = true;
    }
  }
  function lane(x) {
    const root = new T.Group(); root.position.x = x; scene.add(root);
    const floor = new T.Mesh(new T.PlaneGeometry(2.6, 90), new T.MeshBasicMaterial({ color: L.floor })); floor.rotation.x = -Math.PI / 2; floor.position.set(0, 0, -43); root.add(floor);
    const lineMat = new T.MeshBasicMaterial({ color: L.lines });
    for (let k = 0; k <= 4; k++) { const l = new T.Mesh(new T.BoxGeometry(0.012, 0.004, 90), lineMat); l.position.set((k - 2) * LANE, 0.003, -43); root.add(l); }
    const frameMat = new T.MeshBasicMaterial({ color: L.frame, transparent: true, opacity: 0.6 });
    for (const yy of [0.58, 1.11, 1.63, 2.16]) { const b = new T.Mesh(new T.BoxGeometry(4 * LANE, 0.006, 0.006), frameMat); b.position.set(0, yy, 0); root.add(b); }
    for (let k = 0; k <= 4; k++) { const b = new T.Mesh(new T.BoxGeometry(0.006, 1.58, 0.006), frameMat); b.position.set((k - 2) * LANE, 1.37, 0); root.add(b); }
    const rings = [];
    for (let k = 0; k < 16; k++) {
      const m = new T.Mesh(new T.TorusGeometry(2.3, 0.025, 6, 48), new T.MeshBasicMaterial({ color: handColors[k % 2], transparent: true, opacity: 0.25, blending: T.AdditiveBlending, depthWrite: false }));
      m.position.set(0, 1.35, -4 - k * 5); root.add(m); rings.push(m);
    }
    const box = new T.BoxGeometry(0.42, 0.42, 0.42), arrow = new T.BufferGeometry();
    arrow.setAttribute('position', new T.BufferAttribute(new Float32Array([0, -0.12, 0, -0.14, 0.06, 0, 0.14, 0.06, 0]), 3));
    const dot = new T.SphereGeometry(0.06, 10, 8), mats = handColors.map((c) => new T.MeshBasicMaterial({ color: c }));
    const white = new T.MeshBasicMaterial({ color: '#ffffff', side: T.DoubleSide });
    const pool = [];
    for (let k = 0; k < 240; k++) {
      const g = new T.Group(), b = new T.Mesh(box, mats[0]), ar = new T.Mesh(arrow, white), dt = new T.Mesh(dot, white);
      ar.position.z = 0.215; dt.position.z = 0.2; dt.scale.z = 0.3; g.add(b, ar, dt); g.visible = false; root.add(g); pool.push({ g, b, ar, dt });
    }
    return { root, rings, pool, mats, figure: figure(root), label: null };
  }
  function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== Math.round(w * renderer.getPixelRatio()) || canvas.height !== Math.round(h * renderer.getPixelRatio())) {
      renderer.setSize(w, h, false); camera.aspect = w / Math.max(h, 1); camera.updateProjectionMatrix();
    }
  }
  function placeCamera() {
    const xs = lanes.map((l) => l.root.position.x), mid = xs.length ? (Math.min(...xs) + Math.max(...xs)) / 2 : 0;
    const set = (p, l) => { camera.position.set(...p); camera.lookAt(...l); if (orbit) orbit.target.set(...l); };
    if (orbit) orbit.enabled = cameraName === 'orbit';
    // Looking through a figure's eyes: hide that figure's own head.
    const eyes = cameraName.startsWith('eyes') ? Math.min(+(cameraName.slice(4) || 0), lanes.length - 1) : -1;
    lanes.forEach((l, i) => { l.figure.head.visible = i !== eyes; });
    if (cameraName === 'behind') set([mid, 2.5, lanes.length > 1 ? 4.8 : 3.6], [mid, 1.25, -6]);
    else if (cameraName.startsWith('eyes')) { const x = xs[Math.min(+(cameraName.slice(4) || 0), xs.length - 1)] || 0; set([x, 1.68, BODY_Z + 0.1], [x, 1.3, -10]); }
    else if (cameraName === 'side') set([mid + 7.5, 2.2, -1.5], [mid, 1.2, -3]);
    else if (cameraName === 'orbit' && orbit) orbit.update();
  }

  const api = {
    /** maps: [{ notes, njs, name }], one or two lanes. */
    load(maps) {
      for (const l of lanes) scene.remove(l.root);
      lanes = maps.map((m, i) => {
        const l = lane((i - (maps.length - 1) / 2) * laneGap);
        l.notes = [...m.notes].sort((a, b) => a[0] - b[0]); l.times = l.notes.map((n) => n[0]); l.njs = m.njs || 16;
        l.sw = [swings(l.notes, 0), swings(l.notes, 1)];
        if (m.name) { l.label = label(`${m.name} · note speed ${l.njs}`); l.label.position.set(0, 2.75, -0.6); l.root.add(l.label); }
        return l;
      });
      placeCamera(); api.render();
    },
    /** 'behind', 'eyes0', 'eyes1', 'side' or 'orbit' (drag to look around). */
    camera(name) { cameraName = name; placeCamera(); api.render(); },
    seek(t) { time = Math.max(0, t); api.render(); },
    get time() { return time; },
    /** Seconds the sound reaches the listener after the audio element reports it (Bluetooth: more). */
    set soundDelay(s) { delay = s; },
    /** Follow an audio element's clock while it plays (or null to stop following). */
    follow(audio) {
      followed = audio; cancelAnimationFrame(raf);
      const tick = () => { if (followed) { time = Math.max(0, followed.currentTime - delay); api.render(); raf = requestAnimationFrame(tick); } };
      if (audio) raf = requestAnimationFrame(tick);
    },
    /** Play without audio, at a speed (1 = real time). */
    play(speed = 1) {
      playing = true; last = performance.now(); cancelAnimationFrame(raf);
      const tick = (now) => { if (!playing) return; time += (now - last) / 1000 * speed; last = now; api.render(); raf = requestAnimationFrame(tick); };
      raf = requestAnimationFrame(tick);
    },
    pause() { playing = false; cancelAnimationFrame(raf); },
    render() {
      resize();
      for (const l of lanes) {
        let lo = 0, hi = l.times.length; while (lo < hi) { const mid = (lo + hi) >> 1; if (l.times[mid] < time - 0.15) lo = mid + 1; else hi = mid; }
        let k = 0;
        for (let j = lo; j < l.times.length && l.times[j] <= time + 2.2 && k < l.pool.length; j++, k++) {
          const n = l.notes[j], dt = l.times[j] - time, p = l.pool[k];
          p.g.visible = true; p.b.material = l.mats[n[1]];
          p.g.position.set((n[2] - 1.5) * LANE, layerY(n[3]), -dt * l.njs);
          p.g.scale.setScalar(Math.max(0.05, dt < 0 ? 1 + dt / 0.15 : 1));
          p.g.rotation.set(0, 0, n[4] === 8 ? 0 : Math.atan2(DIRS[n[4]][0], -DIRS[n[4]][1]));
          p.ar.visible = n[4] !== 8; p.dt.visible = n[4] === 8;
        }
        for (; k < l.pool.length; k++) l.pool[k].g.visible = false;
        let lastCut = lo; while (lastCut + 1 < l.times.length && l.times[lastCut + 1] <= time) lastCut++;
        const since = l.times[lastCut] <= time ? time - l.times[lastCut] : 9;
        l.rings.forEach((r, q) => { r.material.opacity = 0.18 + 0.5 * Math.exp(-since * 5) * Math.exp(-q * 0.12); r.rotation.z = time * 0.15 * (q % 2 ? 1 : -1); });
        pose(l.figure, l.sw, time);
      }
      if (cameraName === 'orbit' && orbit) orbit.update();
      renderer.render(scene, camera);
    },
    destroy() { api.pause(); api.follow(null); renderer.dispose(); },
  };
  return api;
}
