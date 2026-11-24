(function () {
  try {
    var IM = window.HM_IMAGERY, host = document.getElementById('pipeline-diagram');
    if (!IM || !host || document.getElementById('hw')) return;
    var S = 640, DUR = 4600, reduce = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;
    var PCR = window.HM_DATA.perCity.cities.filter(function (r) { return IM[r.city]; }).sort(function (a, b) { return b.R2_within - a.R2_within; });
    var ROW = {}; PCR.forEach(function (r) { ROW[r.city] = r; });
    var CITIES = PCR.map(function (r) { return r.city; });
    if (!CITIES.length) return;
    var STEPS = [
      ['Fetch scenes', '40 Landsat scenes become one clean picture'],
      ['Read features', '51 layers per pixel: green, built-up, surroundings'],
      ['Cut patches', '64 × 64 pixel tiles, 1.9 km each'],
      ['Predict heat', 'The U-Net scores every 30 m pixel'],
      ['Test cooling', 'Plant trees, re-predict, watch the gap close'],
      ['Score it', 'R² on the 10 held-out cities']
    ];
    var box = document.createElement('div'); box.id = 'hw'; box.className = 'hw';
    box.innerHTML =
      '<div class="hw-left"><div class="hw-stage"><canvas id="hw-c" role="img" aria-label="Animated walkthrough of the pipeline on a satellite image of a city"></canvas>' +
      '<span class="xp-tag tl" id="hw-tag"></span><span class="xp-tag bl" id="hw-note"></span><span class="xp-tag br" id="hw-val" hidden></span></div>' +
      '<div class="hw-name" id="hw-name"></div><div class="xp-tiles" id="hw-score"></div></div>' +
      '<div class="hw-side"><div class="controls" id="hw-city" role="group" aria-label="City">' + CITIES.map(function (c, i) { return '<button type="button" class="chip" data-c="' + c + '" aria-pressed="' + (i === 0) + '">' + c + '</button>'; }).join('') +
      '<button type="button" class="chip" id="hw-play" aria-pressed="true">Auto-play</button></div>' +
      '<ol class="hw-steps" id="hw-steps">' + STEPS.map(function (s, i) { return '<li><button type="button" data-s="' + i + '"><span class="n">' + (i + 1) + '</span><span><b>' + s[0] + '</b><em>' + s[1] + '</em></span><i class="bar"></i></button></li>'; }).join('') + '</ol>' +
      '<svg class="hw-unet" id="hw-unet" viewBox="0 0 340 74" aria-hidden="true"></svg>' +
      '<p class="xp-credit">Imagery: Sentinel-2 cloudless 2020 by EOX IT Services GmbH (contains modified Copernicus Sentinel data), CC BY-NC 4.0. Layers are colour-derived from the image, not model output.</p></div>';
    host.parentNode.insertBefore(box, host); host.style.display = 'none';

    var cv = document.getElementById('hw-c'), ctx = cv.getContext('2d'); cv.width = S; cv.height = S;
    var $ = function (s) { return document.querySelector(s); };

    /* mini U-Net drawing */
    (function () {
      var enc = [[8, 22, 30], [60, 31, 22], [112, 40, 14]], g = '';
      var blocks = [[10, 6, 22, 50], [44, 14, 22, 36], [78, 22, 22, 22], [112, 30, 28, 14], [150, 22, 22, 22], [184, 14, 22, 36], [218, 6, 22, 50]];
      blocks.forEach(function (b, i) { g += '<rect class="u" style="animation-delay:' + (i * 0.18) + 's" x="' + b[0] + '" y="' + b[1] + '" width="' + b[2] + '" height="' + b[3] + '" rx="4"/>'; });
      g += '<path class="sk" d="M32 12 C 120 -4 130 -4 218 12M66 20 C 120 6 140 6 184 20M100 28 C 120 20 130 20 150 28"/>';
      g += '<text x="256" y="34" class="ut">→ heat</text><text x="256" y="48" class="ut">per pixel</text>';
      $('#hw-unet').innerHTML = g;
    })();

    function mk() { var c = document.createElement('canvas'); c.width = c.height = S; return c; }
    function ramp(t, st) { t = Math.max(0, Math.min(1, t)); var k = t * (st.length - 1), i = Math.floor(k), f = k - i, a = st[i], b = st[Math.min(i + 1, st.length - 1)]; return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f]; }
    var GREEN = [[40, 36, 32], [96, 88, 70], [150, 160, 90], [60, 170, 70], [10, 120, 50]];
    var HOT = [[30, 40, 90], [70, 110, 190], [240, 220, 160], [240, 120, 70], [180, 30, 40]];
    function blur(src, rad) {
      var n = S, w = n + 1, I = new Float32Array(w * w), out = new Float32Array(n * n), y, x;
      for (y = 0; y < n; y++) { var row = 0; for (x = 0; x < n; x++) { row += src[y * n + x]; I[(y + 1) * w + x + 1] = I[y * w + x + 1] + row; } }
      for (y = 0; y < n; y++) for (x = 0; x < n; x++) {
        var x0 = Math.max(0, x - rad), x1 = Math.min(n, x + rad + 1), y0 = Math.max(0, y - rad), y1 = Math.min(n, y + rad + 1);
        out[y * n + x] = (I[y1 * w + x1] - I[y0 * w + x1] - I[y1 * w + x0] + I[y0 * w + x0]) / ((x1 - x0) * (y1 - y0));
      }
      return out;
    }
    function fill(c, fn) { var x = c.getContext('2d'), o = x.createImageData(S, S), p = o.data; for (var i = 0; i < S * S; i++) { var v = fn(i); p[4 * i] = v[0]; p[4 * i + 1] = v[1]; p[4 * i + 2] = v[2]; p[4 * i + 3] = v.length > 3 ? v[3] : 255; } x.putImageData(o, 0, 0); }

    var cache = {};
    function prep(city, cb) {
      if (cache[city]) return cb(cache[city]);
      var im = new Image();
      im.onload = function () {
        var base = mk(), bx = base.getContext('2d'); bx.drawImage(im, 0, 0, S, S);
        var d = bx.getImageData(0, 0, S, S).data, n = S * S, veg = new Float32Array(n), built = new Float32Array(n), i;
        for (i = 0; i < n; i++) {
          var r = d[4 * i], g = d[4 * i + 1], b = d[4 * i + 2], mx = Math.max(r, g, b), mn = Math.min(r, g, b);
          veg[i] = (2 * g - r - b) / 255 > 0.14 ? 1 : 0;
          built[i] = (!veg[i] && mx > 0 && (mx - mn) / mx < 0.3) ? 1 : 0;
        }
        var bb = blur(built, 8), bv = blur(veg, 8), risk = new Float32Array(n);
        for (i = 0; i < n; i++) risk[i] = bb[i] * (1 - bv[i]);
        var srt = Float32Array.from(risk).sort(), thr = srt[Math.floor(n * 0.9)], rmax = srt[n - 1] || 1;
        var o = {img: im, base: base, thr: thr};
        o.cVeg = mk(); fill(o.cVeg, function (i) { return ramp(((2 * d[4 * i + 1] - d[4 * i] - d[4 * i + 2]) / 255 + 0.05) / 0.3, GREEN); });
        o.cBuilt = mk(); fill(o.cBuilt, function (i) { return built[i] ? [235, 235, 240] : (veg[i] ? [24, 60, 34] : [60, 52, 46]); });
        o.cRisk = mk(); fill(o.cRisk, function (i) { return ramp(risk[i] / rmax * 1.15, HOT); });
        o.cDim = mk(); fill(o.cDim, function (i) { var g = (d[4 * i] * 0.3 + d[4 * i + 1] * 0.59 + d[4 * i + 2] * 0.11) * 0.55 + 20; return [g, g, g + 6]; });
        var hot = [], seed = 12345;
        for (i = 0; i < n; i++) if (risk[i] >= thr && risk[i] > 0.05 && !veg[i]) { seed = (seed * 1664525 + 1013904223) >>> 0; hot.push([i, seed / 4294967296]); }
        hot.sort(function (a, b) { return a[1] - b[1]; });
        var cut = Math.round(hot.length * 0.4), keep = {}, k;
        o.cKeep = mk(); o.cDoneR = mk(); o.cDoneG = mk();
        var doneSet = {}; for (k = 0; k < cut; k++) doneSet[hot[k][0]] = 1;
        var hs = {}; hot.forEach(function (h) { hs[h[0]] = 1; });
        function red(i) { return ramp((risk[i] - thr) / (1 - thr + 1e-6) * 1.6, [[250, 190, 60], [240, 110, 50], [200, 30, 40]]); }
        fill(o.cKeep, function (i) { if (hs[i] && !doneSet[i]) { var c = red(i); return [c[0], c[1], c[2], 255]; } return [0, 0, 0, 0]; });
        fill(o.cDoneR, function (i) { if (doneSet[i]) { var c = red(i); return [c[0], c[1], c[2], 255]; } return [0, 0, 0, 0]; });
        fill(o.cDoneG, function (i) { return doneSet[i] ? [40, 190, 80, 255] : [0, 0, 0, 0]; });
        cache[city] = o; cb(o);
      };
      im.src = IM[city].img;
    }

    var city = CITIES[0], step = 0, t0 = performance.now(), auto = !reduce, visible = true, raf = 0, cur = null;
    function ease(x) { return x < 0 ? 0 : x > 1 ? 1 : 1 - Math.pow(1 - x, 3); }
    function gapFor(c) { var rs = window.HM_DATA.cooling.rows; if (c === 'Jaipur') return rs[0].pred_gap_C; return rs.reduce(function (a, r) { return a + r.pred_gap_C; }, 0) / rs.length; }
    function label(text, x, y, align) { ctx.font = '700 15px "Bricolage Grotesque", sans-serif'; ctx.textAlign = align || 'left'; var w = ctx.measureText(text).width + 14, bx = align === 'right' ? x - w : x; ctx.fillStyle = 'rgba(10,10,14,.7)'; ctx.fillRect(bx, y - 18, w, 26); ctx.fillStyle = '#fff'; ctx.fillText(text, align === 'right' ? x - 7 : x + 7, y); }

    function draw(p) {
      var o = cur; if (!o) return;
      ctx.clearRect(0, 0, S, S); ctx.imageSmoothingEnabled = true;
      var tag = $('#hw-tag'), note = $('#hw-note'), val = $('#hw-val'); val.hidden = true; note.textContent = '';
      if (step === 0) {
        var e = ease(p / 0.8);
        ctx.fillStyle = '#0c0b10'; ctx.fillRect(0, 0, S, S);
        for (var k = 3; k >= 1; k--) { ctx.globalAlpha = 0.5 * (1 - e) + 0.0; var off = (1 - e) * k * 34; ctx.drawImage(o.base, off, -off * 0.7, S, S); }
        ctx.globalAlpha = 1; ctx.drawImage(o.base, 0, 0, S, S);
        var sy = p * 1.15 * S; ctx.fillStyle = 'rgba(255,255,255,.55)'; if (p < 0.87) ctx.fillRect(0, sy, S, 2);
        tag.textContent = 'Landsat 8/9 · 30 m · Mar–May'; note.textContent = '40 scenes → 1 composite';
      } else if (step === 1) {
        var cells = [[o.base, 'Satellite'], [o.cVeg, 'Vegetation'], [o.cBuilt, 'Built-up'], [o.cRisk, 'Surroundings']];
        ctx.fillStyle = '#0c0b10'; ctx.fillRect(0, 0, S, S);
        cells.forEach(function (c, i) {
          var a = ease(p * 4.2 - i * 0.9), x = (i % 2) * S / 2, y = Math.floor(i / 2) * S / 2;
          ctx.globalAlpha = a; ctx.drawImage(c[0], x + 2, y + 2 + (1 - a) * 18, S / 2 - 4, S / 2 - 4); label(c[1], x + 10, y + 28);
        });
        ctx.globalAlpha = 1; tag.textContent = '51 channels per pixel'; note.textContent = '4 of 51 shown';
      } else if (step === 2) {
        ctx.drawImage(o.cDim, 0, 0, S, S); var cs = S * 1.9 / 40, nx = Math.floor(S / cs), cells2 = nx * nx, shown = Math.floor(ease(p / 0.85) * cells2), q;
        ctx.fillStyle = 'rgba(90,162,255,.2)'; ctx.strokeStyle = 'rgba(255,255,255,.35)'; ctx.lineWidth = 1;
        for (q = 0; q < shown; q++) { var cx = (q % nx) * cs, cy = Math.floor(q / nx) * cs; ctx.fillRect(cx, cy, cs, cs); ctx.strokeRect(cx + .5, cy + .5, cs, cs); }
        var pi = Math.min(cells2 - 1, Math.max(0, shown - 1)), px = (Math.floor(S / 2 / cs) * cs), py = (Math.floor(S / 2 / cs) * cs);
        ctx.strokeStyle = '#ffb36b'; ctx.lineWidth = 3; ctx.strokeRect(px, py, cs, cs);
        var iz = 150; ctx.imageSmoothingEnabled = false; ctx.drawImage(o.base, px, py, cs, cs, S - iz - 14, S - iz - 50, iz, iz); ctx.imageSmoothingEnabled = true;
        ctx.strokeStyle = '#ffb36b'; ctx.strokeRect(S - iz - 14, S - iz - 50, iz, iz);
        tag.textContent = '64 × 64 px patches'; note.textContent = 'held-out city: patches used for scoring only';
      } else if (step === 3) {
        var x = ease(p / 0.85) * S; ctx.drawImage(o.base, 0, 0, S, S);
        ctx.save(); ctx.beginPath(); ctx.rect(0, 0, x, S); ctx.clip(); ctx.drawImage(o.cRisk, 0, 0, S, S); ctx.restore();
        ctx.fillStyle = '#fff'; ctx.fillRect(x - 1, 0, 2, S);
        tag.textContent = 'U-Net → heat for every pixel'; note.textContent = 'illustration: colour-based proxy, not model output';
        document.getElementById('hw-unet').classList.add('on');
      } else if (step === 5) {
        ctx.fillStyle = '#0c0b10'; ctx.fillRect(0, 0, S, S);
        var rowH = (S - 120) / PCR.length, e5 = ease(p / 0.8), left = 130, maxW = S - left - 70, acc = getComputedStyle(document.documentElement).getPropertyValue('--held-out').trim() || '#2a78d6';
        PCR.forEach(function (r, i) {
          var y = 80 + i * rowH, sel = r.city === city, w = Math.max(0, r.R2_within) * maxW * e5;
          ctx.fillStyle = 'rgba(255,255,255,.07)'; ctx.fillRect(left, y + 4, maxW, rowH - 10);
          ctx.fillStyle = sel ? '#ffb36b' : '#5aa2ff'; ctx.globalAlpha = sel ? 1 : 0.7; ctx.fillRect(left, y + 4, w, rowH - 10); ctx.globalAlpha = 1;
          ctx.font = (sel ? '700 ' : '500 ') + '15px "IBM Plex Sans", sans-serif'; ctx.textAlign = 'right'; ctx.fillStyle = sel ? '#fff' : '#cfc9bf'; ctx.fillText(r.city, left - 10, y + rowH / 2 + 5);
          ctx.textAlign = 'left'; ctx.fillText(r.R2_within.toFixed(2), left + w + 8, y + rowH / 2 + 5);
        });
        var mean = window.HM_DATA.perCity.mean.R2_within; var mx = left + mean * maxW * e5;
        ctx.strokeStyle = '#fff'; ctx.setLineDash([5, 4]); ctx.beginPath(); ctx.moveTo(mx, 66); ctx.lineTo(mx, S - 40); ctx.stroke(); ctx.setLineDash([]);
        ctx.font = '600 13px "IBM Plex Mono", monospace'; ctx.textAlign = 'center'; ctx.fillStyle = '#fff'; ctx.fillText('mean ' + mean.toFixed(2), mx, 58);
        tag.textContent = 'R² inside each held-out city'; note.textContent = 'scored only, never trained on';
      } else {
        ctx.drawImage(o.cDim, 0, 0, S, S); ctx.drawImage(o.cKeep, 0, 0, S, S);
        var q2 = ease(p / 0.8); ctx.globalAlpha = 1 - q2; ctx.drawImage(o.cDoneR, 0, 0, S, S); ctx.globalAlpha = q2; ctx.drawImage(o.cDoneG, 0, 0, S, S); ctx.globalAlpha = 1;
        var gap = gapFor(city); val.hidden = false; val.textContent = 'gap ' + (gap * (1 - q2)).toFixed(1) + '°C';
        tag.textContent = 'Plant trees on 40 % of open hotspot ground'; note.textContent = 'cooling from README results; hotspots from image colour';
      }
      if (step !== 3) document.getElementById('hw-unet').classList.remove('on');
    }

    function scoreTile(v, l, frac) { return '<div class="xp-tile"><div class="v">' + v + '</div><div class="l">' + l + '</div>' + (frac != null ? '<div class="xp-bar"><i style="width:' + Math.round(frac * 100) + '%"></i></div>' : '') + '</div>'; }
    function scores() {
      var r = ROW[city]; if (!r) return;
      $('#hw-name').innerHTML = '<b>' + city + '</b> <span>held-out city · rank ' + (CITIES.indexOf(city) + 1) + ' of ' + CITIES.length + ' by R²</span>';
      $('#hw-score').innerHTML = scoreTile(r.R2_within.toFixed(2), 'R² inside city', r.R2_within) + scoreTile(r.hot_AUC.toFixed(2), 'Hotspot AUC', r.hot_AUC) +
        scoreTile(r.spearman.toFixed(2), 'Rank match (Spearman)', r.spearman) + scoreTile(r.MAE.toFixed(2) + '°C', 'Typical error', null);
    }
    function sync() {
      document.querySelectorAll('#hw-steps button').forEach(function (b) { var on = +b.dataset.s === step; b.classList.toggle('on', on); b.setAttribute('aria-current', on ? 'step' : 'false'); if (!on) b.querySelector('.bar').style.width = '0'; });
    }
    function loop(now) {
      raf = 0; var p = Math.min(1, (now - t0) / DUR);
      if (reduce) p = 1;
      draw(p);
      var bar = document.querySelector('#hw-steps button.on .bar'); if (bar) bar.style.width = (auto ? Math.min(100, (now - t0) / (DUR + 1400) * 100) : 100) + '%';
      if (auto && now - t0 > DUR + 1400) { step = (step + 1) % STEPS.length; t0 = now; sync(); }
      if (!reduce && visible && !document.hidden) raf = requestAnimationFrame(loop);
    }
    function kick() { if (!raf) raf = requestAnimationFrame(loop); }
    function go(s) { step = s; t0 = performance.now(); sync(); kick(); }
    document.getElementById('hw-steps').addEventListener('click', function (e) { var b = e.target.closest('button'); if (!b) return; auto = false; $('#hw-play').setAttribute('aria-pressed', 'false'); go(+b.dataset.s); });
    $('#hw-play').addEventListener('click', function () { auto = !auto; this.setAttribute('aria-pressed', String(auto)); t0 = performance.now(); kick(); });
    $('#hw-city').addEventListener('click', function (e) {
      var b = e.target.closest('[data-c]'); if (!b) return; city = b.dataset.c;
      document.querySelectorAll('#hw-city [data-c]').forEach(function (x) { x.setAttribute('aria-pressed', String(x === b)); });
      scores(); prep(city, function (o) { cur = o; t0 = performance.now(); kick(); });
    });
    if (window.IntersectionObserver) new IntersectionObserver(function (en) { visible = en[0].isIntersecting; if (visible) { t0 = performance.now(); kick(); } }).observe(box);
    document.addEventListener('visibilitychange', kick);
    sync(); scores(); prep(city, function (o) { cur = o; t0 = performance.now(); kick(); });
  } catch (e) { console.error('how', e); }
})();
