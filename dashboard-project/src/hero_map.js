(function () {
  try {
    var D = window.HM_DATA, el = document.getElementById('hero-3d');
    if (!D || !el || el.dataset.ready) return; el.dataset.ready = '1';
    var OUT = /*INDIA_OUTLINE*/[]/*END*/;
    var cities = D.cities.cities, rows = {}; D.perCity.cities.forEach(function (r) { rows[r.city] = r; });
    var METRICS = {R2_within: 'R² inside city', spearman: 'Spearman', hot_AUC: 'Hotspot AUC', hot_F1: 'Hotspot F1'};
    var metric = 'R2_within', reduce = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;
    var LEFT = {Coimbatore: 1, Bengaluru: 1, Jaipur: 1, Amritsar: 1};
    var cv = document.createElement('canvas'); cv.style.cssText = 'width:100%;height:100%;display:block;cursor:crosshair;touch-action:pan-y'; el.appendChild(cv);
    var ctx = cv.getContext('2d'), W = 0, H = 0, dpr = 1, dots = [], hover = null, raf = 0, visible = true;
    var LON0 = 67, LON1 = 98.5, LAT0 = 5.5, LAT1 = 37.5, KX = Math.cos(21 * Math.PI / 180);
    function size() {
      var r = el.getBoundingClientRect(); dpr = Math.min(2, window.devicePixelRatio || 1);
      W = Math.max(200, r.width); H = Math.max(200, r.height); cv.width = W * dpr; cv.height = H * dpr; layout(); draw(0);
    }
    var sc = 1, ox = 0, oy = 0;
    function proj(lon, lat) { return [ox + (lon - LON0) * KX * sc, oy + (LAT1 - lat) * sc]; }
    function layout() {
      var padT = 34, padB = 64, w = (LON1 - LON0) * KX, h = LAT1 - LAT0;
      sc = Math.min((W - 24) / w, (H - padT - padB) / h); ox = (W - w * sc) / 2; oy = padT + (H - padT - padB - h * sc) / 2;
      var vals = cities.filter(function (c) { return c.split === 'val' && rows[c.city]; }).map(function (c) { return rows[c.city][metric]; }), mn = Math.min.apply(null, vals), mx = Math.max.apply(null, vals);
      dots = cities.map(function (c) {
        var p = proj(c.lon, c.lat), val = c.split === 'val' && rows[c.city] ? rows[c.city][metric] : null;
        var t = val == null ? 0 : (mx > mn ? (val - mn) / (mx - mn) : 0.5);
        return {c: c, x: p[0], y: p[1], val: val, t: t, r: val == null ? 3.5 : 6 + 11 * t};
      });
    }
    function css(n, fb) { return getComputedStyle(document.documentElement).getPropertyValue(n).trim() || fb; }
    function draw(now) {
      var C = {land: css('--surface-raised', '#fff'), edge: css('--axis', '#bbb'), text: css('--text-primary', '#111'), chip: css('--surface-raised', '#fff'), border: css('--border', 'rgba(0,0,0,.15)'), train: css('--train', '#eb6834'), held: css('--held-out', '#2a78d6')};
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
      // land
      ctx.beginPath();
      OUT.forEach(function (ring) { ring.forEach(function (pt, i) { var p = proj(pt[0], pt[1]); if (i) ctx.lineTo(p[0], p[1]); else ctx.moveTo(p[0], p[1]); }); ctx.closePath(); });
      ctx.fillStyle = C.land; ctx.fill(); ctx.strokeStyle = C.edge; ctx.lineWidth = 1.4; ctx.stroke();
      var tt = now / 1000;
      // trained-on cities: warm glow + dot
      dots.forEach(function (d) {
        if (d.val != null) return;
        ctx.fillStyle = C.train; ctx.globalAlpha = .85; ctx.beginPath(); ctx.arc(d.x, d.y, d.r, 0, 6.3); ctx.fill(); ctx.globalAlpha = 1;
      });
      // held-out cities: sized and shaded by the chosen score
      dots.forEach(function (d, i) {
        if (d.val == null) return;
        var ph = reduce ? 0.4 : ((tt * 0.5 + i * 0.13) % 1);
        ctx.strokeStyle = C.held; ctx.globalAlpha = 0.5 * (1 - ph); ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(d.x, d.y, d.r + 2 + ph * 14, 0, 6.3); ctx.stroke(); ctx.globalAlpha = 1;
        ctx.globalAlpha = 0.55 + 0.45 * d.t; ctx.fillStyle = C.held; ctx.strokeStyle = C.chip; ctx.lineWidth = hover === d ? 3 : 1.5; ctx.beginPath(); ctx.arc(d.x, d.y, d.r, 0, 6.3); ctx.fill(); ctx.stroke(); ctx.globalAlpha = 1;
        ctx.font = '600 11px "IBM Plex Mono", monospace'; var lab = d.c.city, left = LEFT[d.c.city];
        ctx.textAlign = left ? 'right' : 'left'; ctx.fillStyle = C.chip; ctx.globalAlpha = .92; var w = ctx.measureText(lab).width + 8, lx = left ? d.x - d.r - 5 : d.x + d.r + 5;
        ctx.fillRect(left ? lx - w + 4 : lx - 4, d.y - 9, w, 17); ctx.globalAlpha = 1; ctx.fillStyle = C.text; ctx.fillText(lab, lx, d.y + 4);
      });
    }
    function loop(now) { raf = 0; draw(now); if (!reduce && visible && !document.hidden) raf = requestAnimationFrame(loop); }
    function kick() { if (!raf) raf = requestAnimationFrame(loop); }
    // tooltip
    var tip = document.getElementById('hero-tip'), stage = el.parentNode;
    function f2(v) { return Number(v).toFixed(2); }
    function pick(e) {
      var rc = cv.getBoundingClientRect(), x = e.clientX - rc.left, y = e.clientY - rc.top, best = null, bd = 18;
      dots.forEach(function (d) { var dd = Math.hypot(d.x - x, d.y - y); if (dd < Math.max(bd, 0) && dd < d.r + 10) { bd = dd; best = d; } });
      hover = best && best.val != null ? best : (best || null); if (raf === 0) draw(performance.now());
      if (!tip) return;
      if (!best) { tip.hidden = true; return; }
      var c = best.c, r = rows[c.city];
      tip.innerHTML = '<b>' + c.city + '</b> &middot; ' + (r ? 'held out, scored' : 'trained on') + '<br><span class="k">' + c.climate + '</span>' +
        (r ? '<br><span class="k">R² inside</span> ' + f2(r.R2_within) + ' &middot; <span class="k">Spearman</span> ' + f2(r.spearman) + '<br><span class="k">Hot AUC</span> ' + f2(r.hot_AUC) + ' &middot; <span class="k">Hot F1</span> ' + f2(r.hot_F1) + '<br><span class="k">Typical error</span> ' + f2(r.MAE) + ' °C' : '');
      tip.hidden = false; var sw = stage.clientWidth, tw = tip.offsetWidth, sr = stage.getBoundingClientRect();
      tip.style.left = Math.max(8, Math.min(sw - tw - 8, rc.left - sr.left + best.x + 14)) + 'px'; tip.style.top = Math.max(8, rc.top - sr.top + best.y - tip.offsetHeight - 12) + 'px';
    }
    cv.addEventListener('pointermove', pick); cv.addEventListener('pointerdown', pick);
    cv.addEventListener('pointerleave', function () { hover = null; if (tip) tip.hidden = true; });
    document.querySelectorAll('[data-h3m]').forEach(function (b) { b.addEventListener('click', function () {
      metric = b.dataset.h3m; document.querySelectorAll('[data-h3m]').forEach(function (x) { x.setAttribute('aria-pressed', String(x === b)); });
      var lg = document.getElementById('h3-metric'); if (lg) lg.textContent = METRICS[metric]; layout(); kick(); if (reduce) draw(0);
    }); });
    if (window.ResizeObserver) new ResizeObserver(size).observe(el); else window.addEventListener('resize', size);
    if (window.IntersectionObserver) new IntersectionObserver(function (en) { visible = en[0].isIntersecting; if (visible) kick(); }).observe(el);
    document.addEventListener('visibilitychange', kick);
    el.setAttribute('aria-label', 'Map of India with 50 city dots. 40 orange cities were used for training. 10 blue cities were held out and scored; dot size shows the selected score.');
    size(); kick();
  } catch (e) { console.error('hero map', e); }
})();
