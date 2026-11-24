(function () {
  try {
    var IM = window.HM_IMAGERY, PC = window.HM_DATA.perCity.cities;
    if (!IM || !document.getElementById('xp')) return;
    var $ = function (s) { return document.querySelector(s); };
    var rows = PC.filter(function (r) { return IM[r.city]; }).sort(function (a, b) { return b.R2_within - a.R2_within; });
    if (!rows.length) return;
    var cv = $('#xp-canvas'), ctx = cv.getContext('2d'), S = 640; cv.width = S; cv.height = S;
    var cache = {}, cur = rows[0].city, layer = 'sat';
    function load(city, cb) {
      if (cache[city]) return cb(cache[city]);
      var im = new Image();
      im.onload = function () {
        var t = document.createElement('canvas'); t.width = t.height = S; var c = t.getContext('2d'); c.drawImage(im, 0, 0, S, S);
        var d = c.getImageData(0, 0, S, S).data, n = S * S, veg = new Float32Array(n), g = 0;
        for (var i = 0; i < n; i++) {
          var exg = (2 * d[4 * i + 1] - d[4 * i] - d[4 * i + 2]) / 255;
          veg[i] = exg; if (exg > 0.14) g++;
        }
        cache[city] = { img: im, veg: veg, green: g / n, base: d };
        cb(cache[city]);
      };
      im.src = IM[city].img;
    }
    function ramp(t, st) { t = Math.max(0, Math.min(1, t)); var k = t * (st.length - 1), i = Math.floor(k), f = k - i, a = st[i], b = st[Math.min(i + 1, st.length - 1)]; return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f]; }
    var GREEN = [[40, 36, 32], [96, 88, 70], [150, 160, 90], [60, 170, 70], [10, 120, 50]];
    var HOT = [[30, 40, 70], [90, 100, 120], [200, 170, 120], [240, 120, 70], [200, 40, 40]];

    var PLAN = {trees: 0, roofs: 0}, PX_HA = Math.pow(40000 / S, 2) / 10000;
    function blur(src, rad) {
      var n = S, w = n + 1, I = new Float32Array(w * w), out = new Float32Array(n * n), y, x;
      for (y = 0; y < n; y++) { var row = 0; for (x = 0; x < n; x++) { row += src[y * n + x]; I[(y + 1) * w + x + 1] = I[y * w + x + 1] + row; } }
      for (y = 0; y < n; y++) for (x = 0; x < n; x++) {
        var x0 = Math.max(0, x - rad), x1 = Math.min(n, x + rad + 1), y0 = Math.max(0, y - rad), y1 = Math.min(n, y + rad + 1);
        out[y * n + x] = (I[y1 * w + x1] - I[y0 * w + x1] - I[y1 * w + x0] + I[y0 * w + x0]) / ((x1 - x0) * (y1 - y0));
      }
      return out;
    }
    function buildPlan(o) {
      if (o.plan) return o.plan;
      var n = S * S, d = o.base, built = new Float32Array(n), veg = new Float32Array(n), i;
      for (i = 0; i < n; i++) {
        var r = d[4 * i], g = d[4 * i + 1], b = d[4 * i + 2], mx = Math.max(r, g, b), mn = Math.min(r, g, b);
        veg[i] = o.veg[i] > 0.14 ? 1 : 0;
        built[i] = (!veg[i] && mx > 0 && (mx - mn) / mx < 0.3) ? 1 : 0;
      }
      var bb = blur(built, 8), bv = blur(veg, 8), risk = new Float32Array(n);
      for (i = 0; i < n; i++) risk[i] = bb[i] * (1 - bv[i]);
      var sorted = Float32Array.from(risk).sort(), thr = sorted[Math.floor(n * 0.9)];
      var hot = [], seed = 12345;
      for (i = 0; i < n; i++) if (risk[i] >= thr && risk[i] > 0.05 && !veg[i]) { seed = (seed * 1664525 + 1013904223) >>> 0; hot.push({i: i, k: seed / 4294967296, b: built[i]}); }
      hot.sort(function (a, b) { return a.k - b.k; });
      var roof = hot.filter(function (h) { return h.b; });
      o.plan = {risk: risk, thr: thr, hot: hot, roof: roof, veg: veg};
      return o.plan;
    }
    /* cooling per README (models/simulate_cooling.py): Jaipur hotspot closes its 2.2 C gap at 40 % trees; Delhi block closes with 99 % cool roofs */
    function gapFor(city) {
      var rs = window.HM_DATA.cooling.rows, jp = rs[0];
      if (city === 'Jaipur') return {gap: jp.pred_gap_C, vc: jp.veg_dose_pct, rc: 99};
      var m = rs.reduce(function (a, r) { return a + r.pred_gap_C; }, 0) / rs.length, v = rs.reduce(function (a, r) { return a + r.veg_dose_pct; }, 0) / rs.length;
      return {gap: m, vc: v, rc: 99};
    }
    function drawPlan(o) {
      var pl = buildPlan(o), n = S * S, d = o.base, out = ctx.createImageData(S, S), p = out.data, i;
      for (i = 0; i < n; i++) { var g = (d[4 * i] * 0.3 + d[4 * i + 1] * 0.59 + d[4 * i + 2] * 0.11) * 0.55 + 20; p[4 * i] = g; p[4 * i + 1] = g; p[4 * i + 2] = g + 6; p[4 * i + 3] = 255; }
      var nt = Math.round(pl.hot.length * PLAN.trees / 100), nr = Math.round(pl.roof.length * PLAN.roofs / 100), done = {}, k, h, rgb;
      for (k = 0; k < nt; k++) done[pl.hot[k].i] = 1;
      for (k = 0; k < nr; k++) if (!done[pl.roof[k].i]) done[pl.roof[k].i] = 2;
      for (k = 0; k < pl.hot.length; k++) {
        h = pl.hot[k]; i = h.i;
        if (done[i] === 1) rgb = [40, 190, 80]; else if (done[i] === 2) rgb = [150, 225, 245];
        else rgb = ramp((pl.risk[i] - pl.thr) / (1 - pl.thr + 1e-6) * 1.6, [[250, 190, 60], [240, 110, 50], [200, 30, 40]]);
        p[4 * i] = rgb[0]; p[4 * i + 1] = rgb[1]; p[4 * i + 2] = rgb[2];
      }
      ctx.putImageData(out, 0, 0);
      var G = gapFor(cur), f = Math.min(1, PLAN.trees / G.vc + PLAN.roofs / G.rc), left = G.gap * (1 - f);
      var el = $('#xp-plan-out');
      if (el) el.innerHTML = tile(Math.round(pl.hot.length * PX_HA).toLocaleString() + ' ha', 'Hotspot area found', null) +
        tile('−' + (G.gap - left).toFixed(1) + '°C', 'Estimated cooling in hotspots', f) +
        tile(Math.round(nt * PX_HA).toLocaleString() + ' ha', 'Trees planted', null) + tile(Math.round(nr * PX_HA).toLocaleString() + ' ha', 'Roofs made reflective', null) +
        '<div class="xp-tile" style="grid-column:1/-1"><div class="l" style="margin:0 0 6px">Hotspot is ' + G.gap.toFixed(1) + '°C warmer than its surroundings. After the plan: <b>' + left.toFixed(1) + '°C</b>' + (f >= 1 ? ' (gap closed)' : '') + '</div><div class="xp-bar"><i style="width:' + Math.round(f * 100) + '%;background:var(--success-text)"></i></div></div>';
    }
    function draw() {
      var city = cur;
      load(city, function (o) {
        if (city !== cur) return;
        ctx.clearRect(0, 0, S, S);
        if (layer === 'sat') { ctx.drawImage(o.img, 0, 0, S, S); }
        else if (layer === 'plan') { drawPlan(o); }
        else {
          var out = ctx.createImageData(S, S), p = out.data, d = o.base;
          for (var i = 0; i < S * S; i++) {
            var rgb, v = o.veg[i];
            if (layer === 'veg') rgb = ramp((v + 0.05) / 0.3, GREEN);
            else rgb = ramp(v > 0.14 ? 0 : ((d[4 * i] + d[4 * i + 1] + d[4 * i + 2]) / 765 - 0.15) / 0.5, HOT);
            p[4 * i] = rgb[0]; p[4 * i + 1] = rgb[1]; p[4 * i + 2] = rgb[2]; p[4 * i + 3] = 255;
          }
          ctx.putImageData(out, 0, 0);
        }
        ctx.strokeStyle = 'rgba(255,255,255,.9)'; ctx.fillStyle = 'rgba(255,255,255,.9)'; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(S / 2 - 14, S / 2); ctx.lineTo(S / 2 + 14, S / 2); ctx.moveTo(S / 2, S / 2 - 14); ctx.lineTo(S / 2, S / 2 + 14); ctx.stroke();
        ctx.beginPath(); ctx.arc(S / 2, S / 2, 7, 0, 6.3); ctx.stroke();
        ctx.fillRect(S - 20 - S / 4, S - 56, S / 4, 3); /* 10 km of the 40 km box */
      });
    }
    function tile(v, label, frac, id) { return '<div class="xp-tile"' + (id ? ' id="' + id + '"' : '') + '><div class="v">' + v + '</div><div class="l">' + label + '</div>' + (frac != null ? '<div class="xp-bar"><i style="width:' + Math.round(frac * 100) + '%"></i></div>' : '') + '</div>'; }
    function render() {
      var city = cur, r = rows.filter(function (x) { return x.city === city; })[0], im = IM[city];
      $('#xp-name').textContent = city;
      $('#xp-sub').textContent = im.lat.toFixed(3) + '°N, ' + im.lon.toFixed(3) + '°E · rank ' + (rows.indexOf(r) + 1) + ' of ' + rows.length + ' by R²';
      $('#xp-ll').textContent = im.bbox[3].toFixed(2) + '°N ' + im.bbox[0].toFixed(2) + '°E';
      $('#xp-tiles').innerHTML = tile(r.R2_within.toFixed(2), 'R² inside city', r.R2_within) + tile(r.hot_AUC.toFixed(2), 'Hotspot AUC', r.hot_AUC) +
        tile(r.spearman.toFixed(2), 'Rank match (Spearman)', r.spearman) + tile(r.MAE.toFixed(2) + '°C', 'Typical error', null) +
        tile('…', 'Vegetation in image (incl. crops)', null, 'xp-green');
      document.querySelectorAll('#xp-cities .chip').forEach(function (b) { b.setAttribute('aria-pressed', b.dataset.c === city ? 'true' : 'false'); });
      load(city, function (o) { if (city !== cur) return; var g = $('#xp-green'); if (g) g.outerHTML = tile(Math.round(o.green * 100) + '%', 'Vegetation in image (incl. crops)', o.green, 'xp-green'); });
      draw();
    }
    $('#xp-cities').innerHTML = rows.map(function (r) { return '<button type="button" class="chip" data-c="' + r.city + '" aria-pressed="false">' + r.city + '</button>'; }).join('');
    $('#xp-cities').addEventListener('click', function (e) { var b = e.target.closest('.chip'); if (b) { cur = b.dataset.c; render(); } });
    document.querySelectorAll('[data-xl]').forEach(function (b) { b.addEventListener('click', function () { layer = b.dataset.xl; $('#xp-plan').hidden = layer !== 'plan'; $('#xp-tiles').hidden = layer === 'plan'; document.querySelectorAll('[data-xl]').forEach(function (x) { x.setAttribute('aria-pressed', x === b ? 'true' : 'false'); }); draw(); }); });
    $('#xp-stage').addEventListener('pointermove', function (e) {
      var rc = this.getBoundingClientRect(), fx = (e.clientX - rc.left) / rc.width, fy = (e.clientY - rc.top) / rc.height, bb = IM[cur].bbox;
      $('#xp-read').textContent = (bb[3] - fy * (bb[3] - bb[1])).toFixed(3) + '°N ' + (bb[0] + fx * (bb[2] - bb[0])).toFixed(3) + '°E';
    });
    ['trees', 'roofs'].forEach(function (k) { $('#xp-' + k).addEventListener('input', function () { PLAN[k] = +this.value; $('#xp-' + k + '-v').textContent = this.value + ' %'; if (layer === 'plan') draw(); }); });
    $('#xp-best').addEventListener('click', function () { PLAN.trees = 40; PLAN.roofs = 0; $('#xp-trees').value = 40; $('#xp-roofs').value = 0; $('#xp-trees-v').textContent = '40 %'; $('#xp-roofs-v').textContent = '0 %'; draw(); });
    render();
  } catch (e) { console.error('explorer', e); }
})();
(function () {
  try {
    function fold(sec, startEl, label) {
      var wrap = sec && sec.querySelector('.wrap'); if (!wrap) return;
      var kids = Array.prototype.slice.call(wrap.children), i = startEl ? kids.indexOf(startEl) : kids.findIndex(function (k) { return !k.classList.contains('sec-head'); });
      if (i < 0) return;
      var d = document.createElement('details'); d.className = 'fold';
      var s = document.createElement('summary'); s.textContent = label; d.appendChild(s);
      kids.slice(i).forEach(function (k) { d.appendChild(k); });
      wrap.appendChild(d);
    }
    var m = document.getElementById('method');
    var subs = m ? Array.prototype.slice.call(m.querySelectorAll('.wrap > .sub-h')) : [];
    fold(m, subs.filter(function (h) { return /Data sources/.test(h.textContent); })[0], 'Data sources, model and evaluation details');
    fold(document.getElementById('limits'), null, 'Show limitations and notes');
  } catch (e) { console.error('fold', e); }
})();

(function () {
  try {  /* the Delhi map figure gets its own card beside the hotspot-quality card */
    var fig = document.querySelector('.figs figure[data-name="fig_delhi_map.png"]'), v7 = document.getElementById('v7');
    if (fig && v7 && !document.getElementById('v7fig')) {
      var card = document.createElement('div'); card.className = 'card s6'; card.id = 'v7fig';
      card.appendChild(fig);
      var facts = document.getElementById('hq-facts'), tw = facts && facts.parentNode;
      if (tw && tw.classList.contains('tbl-wrap')) { tw.style.marginTop = '16px'; card.appendChild(tw); }
      v7.parentNode.insertBefore(card, v7.nextSibling);
    }
  } catch (e) { console.error('move fig', e); }
})();
