/* 3D views (Three.js r128, UMD global THREE), registered through the HM extension API.
 *  - 'hero3d':    the 3 best held-out cities (top 3 by R² inside the city, per_city.json) as pillars
 *                 at their real lon/lat; height = the selected score, drawn to scale from 0.
 *  - 'terrain3d': HM.grid() as a rotatable surface in #viz-3d (placeholder-labelled until a real grid lands).
 * Both render only while on screen; with prefers-reduced-motion they do not auto-spin. */
(function () {
'use strict';
const HM = window.HM; if (!HM) return;
const D = HM.data || {};
const reduce = !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
const hasGL = (function () { try { const c = document.createElement('canvas'); return !!(window.WebGLRenderingContext && (c.getContext('webgl') || c.getContext('experimental-webgl'))); } catch (e) { return false; } })();
const ok = !!window.THREE && hasGL;
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
const f2 = v => (typeof v === 'number' && isFinite(v)) ? v.toFixed(2) : 'n/a';

function fallback(el, msg) {
  if (el.querySelector('.h3-fallback')) return;
  const d = document.createElement('div'); d.className = 'h3-fallback'; d.textContent = msg; el.appendChild(d);
}

/* Renderer + camera orbiting a target. Drag to rotate, idle auto-spin, renders only while visible. */
function Orbit(el, o) {
  const T = THREE;
  const renderer = new T.WebGLRenderer({antialias: true, alpha: true});
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  el.insertBefore(renderer.domElement, el.firstChild);
  const scene = new T.Scene();
  const camera = new T.PerspectiveCamera(o.fov || 36, 1, 0.1, 200);
  const st = {az: o.az, el: o.el, r: o.r, drag: false, moved: false, lx: 0, ly: 0, idle: 3, visible: true, raf: 0};
  const cv = renderer.domElement;
  function place() {
    camera.position.set(o.target.x + st.r * Math.cos(st.el) * Math.sin(st.az), o.target.y + st.r * Math.sin(st.el), o.target.z + st.r * Math.cos(st.el) * Math.cos(st.az));
    camera.lookAt(o.target);
  }
  function size() {
    const w = el.clientWidth || 300, h = el.clientHeight || 300;
    renderer.setSize(w, h, false); camera.aspect = w / h;
    // keep the scene's width in frame on tall/narrow stages by backing the camera off
    st.r = o.r * (camera.aspect < 1.6 ? Math.pow(1.6 / camera.aspect, 1.1) : 1);
    camera.updateProjectionMatrix();
  }
  let last = 0;
  function frame(t) {
    st.raf = 0;
    const dt = last ? Math.min(0.05, (t - last) / 1000) : 0.016; last = t;
    if (!st.drag && !reduce) { st.idle += dt; if (st.idle > 2.5) st.az += dt * (o.spin || 0.12); }
    place();
    const busy = o.tick ? o.tick(dt, t, camera) : false;
    renderer.render(scene, camera);
    if (o.after) o.after(camera);
    if (st.visible && !document.hidden && (!reduce || st.drag || busy)) st.raf = requestAnimationFrame(frame);
  }
  function start() { if (!st.raf) { last = 0; st.raf = requestAnimationFrame(frame); } }
  cv.addEventListener('pointerdown', e => { st.drag = true; st.moved = false; st.lx = e.clientX; st.ly = e.clientY; st.idle = 0; cv.style.cursor = 'grabbing'; start(); });
  cv.addEventListener('pointermove', e => {
    if (!st.drag) return;
    const dx = e.clientX - st.lx, dy = e.clientY - st.ly;
    if (Math.abs(dx) + Math.abs(dy) > 2) st.moved = true;
    st.az -= dx * 0.008; st.el = Math.max(o.minEl || 0.2, Math.min(1.35, st.el + dy * 0.006));
    st.lx = e.clientX; st.ly = e.clientY; start();
  });
  const end = () => { st.drag = false; st.idle = 0; cv.style.cursor = ''; };
  cv.addEventListener('pointerup', end); cv.addEventListener('pointercancel', end); cv.addEventListener('pointerleave', end);
  if ('IntersectionObserver' in window) new IntersectionObserver(es => { st.visible = es[0].isIntersecting; if (st.visible) start(); }, {threshold: 0.01}).observe(el);
  document.addEventListener('visibilitychange', () => { if (!document.hidden) start(); });
  if ('ResizeObserver' in window) new ResizeObserver(() => { size(); start(); }).observe(el);
  size(); place();
  return {T: T, renderer: renderer, scene: scene, camera: camera, st: st, start: start, canvas: cv, el: el};
}

function picker(O, meshes, onHit) {
  const ray = new O.T.Raycaster(), v = new O.T.Vector2();
  return function (e) {
    const r = O.canvas.getBoundingClientRect();
    v.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    ray.setFromCamera(v, O.camera);
    const hit = ray.intersectObjects(meshes, false)[0];
    onHit(hit || null, e.clientX - r.left, e.clientY - r.top);
  };
}

/* ======================= hero: 3 best held-out cities ======================= */
const METRICS = {
  R2_within: {label: 'R² inside city', short: 'R²'},
  spearman: {label: 'Spearman rank correlation', short: 'Spearman'},
  hot_AUC: {label: 'Hot-pixel AUC (hottest 10 %)', short: 'AUC'},
  hot_F1: {label: 'Hot-pixel F1 (hottest 10 %)', short: 'F1'}
};
const HEIGHT_PER_UNIT = 7;   // world units per 1.0 of score; every pillar starts at 0, so heights are to scale
let hero = null, heroMetric = 'R2_within';

function topCities() {
  const PC = D.perCity, CI = D.cities;
  if (!PC || !Array.isArray(PC.cities) || !CI || !Array.isArray(CI.cities)) return [];
  const where = {}; CI.cities.forEach(c => { where[c.city] = c; });
  return PC.cities.filter(r => where[r.city] && typeof r.R2_within === 'number')
    .sort((a, b) => (b.R2_within - a.R2_within) || (b.R2_anomaly - a.R2_anomaly) || (b.spearman - a.spearman))
    .slice(0, 3).map(r => ({row: r, geo: where[r.city]}));
}

function buildHero() {
  if (hero !== null) return;
  const el = document.getElementById('hero-3d'); if (!el) { hero = false; return; }
  const top = topCities();
  if (!ok || top.length === 0) { hero = false; fallback(el, top.length ? 'The 3D view needs WebGL. The Results section shows the same scores as charts.' : 'City scores unavailable.'); return; }
  const lon0 = top.reduce((s, c) => s + c.geo.lon, 0) / top.length, lat0 = top.reduce((s, c) => s + c.geo.lat, 0) / top.length, k = 0.62;
  const O = Orbit(el, {az: -0.55, el: 0.42, r: 15.5, target: new THREE.Vector3(0, 2.4, 0), spin: 0.14, minEl: 0.12, tick: tick, after: after});
  const T = O.T, S = O.scene;
  S.add(new T.AmbientLight(0xffffff, 0.5));
  const key = new T.DirectionalLight(0xffffff, 0.95); key.position.set(6, 14, 9); S.add(key);
  const rim = new T.DirectionalLight(0xff8a4c, 0.6); rim.position.set(-8, 4, -10); S.add(rim);

  // ground: thermal disc + degree grid (1 cell = 1° of lon/lat at this scale)
  const disc = new T.Mesh(new T.CircleGeometry(9.5, 64), new T.MeshBasicMaterial({color: 0x17131d, transparent: true, opacity: 0.85}));
  disc.rotation.x = -Math.PI / 2; S.add(disc);
  const grid = new T.GridHelper(19, Math.round(19 / k), 0x3b3447, 0x26212e); grid.position.y = 0.01; S.add(grid);

  const blue = new T.Color('#5aa2ff');
  const labels = document.createElement('div'); labels.className = 'h3-labels'; el.appendChild(labels);
  const pillars = top.map((c, i) => {
    const x = (c.geo.lon - lon0) * k, z = -(c.geo.lat - lat0) * k;
    const geo = new T.BoxGeometry(1.6, 1, 1.6); geo.translate(0, 0.5, 0);
    { const pos = geo.attributes.position, col = new Float32Array(pos.count * 3), c0 = new T.Color('#1c5cab'), c1 = new T.Color('#ff7a3d'), cc = new T.Color(); for (let q = 0; q < pos.count; q++) { cc.copy(c0).lerp(c1, Math.pow(Math.max(0, Math.min(1, pos.getY(q))), 1.4)); col[3 * q] = cc.r; col[3 * q + 1] = cc.g; col[3 * q + 2] = cc.b; } geo.setAttribute('color', new T.BufferAttribute(col, 3)); }
    const mat = new T.MeshStandardMaterial({vertexColors: true, emissive: 0x331a10, emissiveIntensity: 0.6, roughness: 0.4, metalness: 0.2});
    const mesh = new T.Mesh(geo, mat); mesh.position.set(x, 0, z); mesh.scale.y = reduce ? 1 : 0.001; S.add(mesh);
    const cap = new T.Mesh(new T.PlaneGeometry(1.9, 1.9), new T.MeshBasicMaterial({color: 0x8fa9c9}));
    cap.rotation.x = -Math.PI / 2; cap.position.set(x, 0, z); S.add(cap);
    const ring = new T.Mesh(new T.RingGeometry(1.25, 1.35, 4), new T.MeshBasicMaterial({color: blue, transparent: true, opacity: 0.5, side: T.DoubleSide}));
    ring.rotation.x = -Math.PI / 2; ring.position.set(x, 0.03, z); S.add(ring);
    const lab = document.createElement('span'); labels.appendChild(lab);
    mesh.userData.idx = i;
    return {c: c, mesh: mesh, cap: cap, ring: ring, lab: lab, h: reduce ? 0 : 0.001, target: 0, phase: i * 0.66};
  });

  function setTargets() {
    pillars.forEach(p => {
      const v = p.c.row[heroMetric];
      p.target = (typeof v === 'number' && isFinite(v)) ? Math.max(0.001, v) * HEIGHT_PER_UNIT : 0.001;
      if (reduce) p.h = p.target;
      p.lab.innerHTML = esc(p.c.row.city) + '<b>' + esc(METRICS[heroMetric].short) + ' ' + f2(v) + '</b>';
    });
  }
  function tick(dt, t) {
    let busy = false;
    pillars.forEach(p => {
      const d = p.target - p.h;
      if (Math.abs(d) > 0.002) { p.h += d * Math.min(1, dt * 3.2); busy = true; } else p.h = p.target;
      p.mesh.scale.y = Math.max(0.001, p.h); p.cap.position.y = p.h + 0.02;
      if (!reduce) { const ph = ((t / 1000) * 0.6 + p.phase) % 1; p.ring.scale.setScalar(1 + ph * 1.6); p.ring.material.opacity = 0.55 * (1 - ph); }
    });
    return busy || !reduce;
  }
  const tmp = new T.Vector3();
  function after(cam) {
    const w = el.clientWidth, h = el.clientHeight;
    pillars.forEach(p => {
      tmp.set(p.mesh.position.x, p.h + 0.45, p.mesh.position.z).project(cam);
      const sx = (tmp.x * 0.5 + 0.5) * w, sy = (-tmp.y * 0.5 + 0.5) * h;
      p.lab.style.transform = 'translate(' + sx.toFixed(1) + 'px,' + sy.toFixed(1) + 'px) translate(-50%,-100%)';
      p.lab.hidden = tmp.z > 1;
    });
  }

  // hover / tap tooltip with all scores for that city
  const tip = document.getElementById('hero-tip'), stage = el.parentNode;
  const pick = picker(O, pillars.map(p => p.mesh), (hit, x, y) => {
    if (!tip) return;
    if (!hit) { tip.hidden = true; return; }
    const r = pillars[hit.object.userData.idx].c.row;
    tip.innerHTML = '<b>' + esc(r.city) + '</b> &middot; held out<br>' +
      '<span class="k">R² inside</span> ' + f2(r.R2_within) + ' &middot; <span class="k">offset removed</span> ' + f2(r.R2_anomaly) + '<br>' +
      '<span class="k">Spearman</span> ' + f2(r.spearman) + ' &middot; <span class="k">MAE</span> ' + f2(r.MAE) + ' °C<br>' +
      '<span class="k">Hot AUC</span> ' + f2(r.hot_AUC) + ' &middot; <span class="k">Hot F1</span> ' + f2(r.hot_F1);
    tip.hidden = false;
    const sw = stage.clientWidth, tw = tip.offsetWidth;
    tip.style.left = Math.max(8, Math.min(sw - tw - 8, x + 14)) + 'px';
    tip.style.top = Math.max(8, y - tip.offsetHeight - 12) + 'px';
  });
  O.canvas.addEventListener('pointermove', e => { if (!O.st.drag) pick(e); });
  O.canvas.addEventListener('pointerup', e => { if (!O.st.moved) pick(e); });
  O.canvas.addEventListener('pointerleave', () => { if (tip) tip.hidden = true; });

  document.querySelectorAll('[data-h3m]').forEach(b => b.addEventListener('click', () => {
    heroMetric = b.dataset.h3m;
    document.querySelectorAll('[data-h3m]').forEach(x => x.setAttribute('aria-pressed', String(x === b)));
    const lg = document.getElementById('h3-metric'); if (lg) lg.textContent = METRICS[heroMetric].label;
    setTargets(); O.start();
  }));
  el.setAttribute('aria-label', 'Rotatable 3D chart of the three best held-out cities by R² inside the city: ' +
    top.map(c => c.row.city + ' ' + f2(c.row.R2_within)).join(', ') + '. Pillars stand at each city\'s real position; height is the selected score.');
  // satellite tops: the real Sentinel-2 image of each city, once the imagery block has loaded
  let tries = 0;
  const texTimer = setInterval(() => {
    tries++;
    const IMG = window.HM_IMAGERY; if (!IMG && tries < 40) return;
    pillars.forEach(p => {
      if (p.texDone || !IMG || !IMG[p.c.row.city]) return; p.texDone = true;
      new T.TextureLoader().load(IMG[p.c.row.city].img, tex => { p.cap.material.map = tex; p.cap.material.color.set(0xffffff); p.cap.material.needsUpdate = true; O.start(); });
    });
    if (tries >= 40 || pillars.every(p => p.texDone)) clearInterval(texTimer);
  }, 300);
  setTargets();
  hero = O;
}

/* ======================= terrain: HM.grid() as a 3D surface ======================= */
let terr = null;
const EXAG = 0.3;   // vertical units per °C (exaggerated so the pattern reads; colour carries the value)

function buildTerrain() {
  if (terr !== null) return;
  const el = document.getElementById('viz-3d'), g = HM.grid && HM.grid();
  if (!el || !g || !ok) { terr = false; return; }
  el.classList.add('is-filled'); el.removeAttribute('aria-hidden');
  el.setAttribute('role', 'img');
  el.setAttribute('aria-label', g.placeholder
    ? 'Rotatable 3D surface of the same placeholder grid. Height and colour show the synthetic anomaly. Not model output.'
    : 'Rotatable 3D surface of the predicted anomaly grid. Height and colour show °C above or below the city mean.');
  el.insertAdjacentHTML('beforeend',
    '<div class="t3-top"><span class="t3-title">' + (g.placeholder ? '<span class="ph-tag">PLACEHOLDER</span> ' : '') + '3D view: drag to spin</span><span class="t3-read" id="t3-read">hover the surface</span></div>' +
    (g.placeholder ? '<div class="t3-wm">PLACEHOLDER: synthetic pattern,<br>not model output</div>' : '') +
    '<div class="t3-cap">Same ' + g.rows + ' &times; ' + g.cols + ' grid as the map above (' + g.cell_m + ' m cells). Height = anomaly, exaggerated; colour uses the same scale.</div>');
  const O = Orbit(el, {az: 0.75, el: 0.78, r: 10.5, target: new THREE.Vector3(0, -0.3, 0), spin: 0.16, tick: () => false});
  const T = O.T;
  O.scene.add(new T.AmbientLight(0xffffff, 0.72));
  const sun = new T.DirectionalLight(0xffffff, 0.7); sun.position.set(-4, 9, 5); O.scene.add(sun);
  const W = 6, H = W * g.rows / g.cols;
  const geo = new T.PlaneGeometry(W, H, g.cols - 1, g.rows - 1); geo.rotateX(-Math.PI / 2);
  const pos = geo.attributes.position, n = pos.count, cols = new Float32Array(n * 3);
  const val = i => { const v = g.values[Math.floor(i / g.cols)][i % g.cols]; return (typeof v === 'number' && isFinite(v)) ? v : 0; };
  for (let i = 0; i < n; i++) pos.setY(i, val(i) * EXAG);
  geo.setAttribute('color', new T.BufferAttribute(cols, 3));
  geo.computeVertexNormals();
  const mesh = new T.Mesh(geo, new T.MeshStandardMaterial({vertexColors: true, roughness: 0.85, metalness: 0, side: T.DoubleSide}));
  O.scene.add(mesh);
  const base = new T.LineSegments(new T.EdgesGeometry(new T.BoxGeometry(W, 1.6, H)), new T.LineBasicMaterial({transparent: true, opacity: 0.35}));
  base.position.y = -0.8 - 4 * EXAG * 0.25; O.scene.add(base);

  function recolor() {
    for (let i = 0; i < n; i++) { const c = HM.scales.anomalyRgb(val(i)); cols[i * 3] = c[0] / 255; cols[i * 3 + 1] = c[1] / 255; cols[i * 3 + 2] = c[2] / 255; }
    geo.attributes.color.needsUpdate = true;
    base.material.color.set(HM.colors('axis') || '#888888');
  }
  const read = document.getElementById('t3-read');
  const pick = picker(O, [mesh], hit => {
    if (!read) return;
    if (!hit || !hit.uv) { read.textContent = 'hover the surface'; return; }
    const c = Math.min(g.cols - 1, Math.max(0, Math.round(hit.uv.x * (g.cols - 1))));
    const r = Math.min(g.rows - 1, Math.max(0, Math.round((1 - hit.uv.y) * (g.rows - 1))));
    const v = g.values[r][c];
    read.textContent = 'row ' + (r + 1) + ', col ' + (c + 1) + ': ' + ((typeof v === 'number' && isFinite(v)) ? (v > 0 ? '+' : '') + v.toFixed(1) + ' °C' : 'no data') + (g.placeholder ? ' (synthetic)' : '');
  });
  O.canvas.addEventListener('pointermove', e => { if (!O.st.drag) pick(e); });
  O.canvas.addEventListener('pointerup', e => { if (!O.st.moved) pick(e); });
  recolor();
  terr = {O: O, recolor: recolor};
}

// hero3d removed: the hero is now a 2D India map (src/hero_map.js)
HM.register('terrain3d', () => { buildTerrain(); if (terr) { terr.recolor(); terr.O.start(); } });
})();
