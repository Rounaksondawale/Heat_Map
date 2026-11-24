(function () {
  try {
    var D = window.HM_DATA, M = D.metrics, PC = D.perCity, CO = D.cooling;
    var $ = function (s) { return document.querySelector(s); };
    var f1 = function (v) { return Number(v).toFixed(1); }, f2 = function (v) { return Number(v).toFixed(2); };
    function ico(path) { return '<svg viewBox="0 0 32 32" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' + path + '</svg>'; }

    /* ---- hero lede and headings: short ---- */
    var lede = $('.hero .lede'); if (lede) lede.innerHTML = 'A U-Net spots the blocks that run hotter than their city. Trained on <b>40</b> cities, scored on <b>10</b> it never saw in training.';
    var H = {overview: 'Find the heat. Explain it. Cool it.', method: 'Satellite scenes in, heat map out', results: 'How well it works'};
    Object.keys(H).forEach(function (k) { var h = document.querySelector('#' + k + ' .sec-head h2'); if (h) h.textContent = H[k]; });

    /* ---- stat strip ---- */
    var stats = [
      ['50', '', 'Indian cities', ico('<rect x="5" y="14" width="6" height="13"/><rect x="13" y="6" width="7" height="21"/><rect x="22" y="11" width="6" height="16"/>')],
      ['89', 'M', 'pixels of satellite data', ico('<rect x="4" y="4" width="10" height="10"/><rect x="18" y="4" width="10" height="10"/><rect x="4" y="18" width="10" height="10"/><rect x="18" y="18" width="10" height="10"/>')],
      ['30', 'm', 'per pixel', ico('<path d="M4 22h24M4 18v8M28 18v8M10 22v-3M16 22v-3M22 22v-3"/>')],
      ['40', 'km', 'box per city', ico('<rect x="5" y="5" width="22" height="22" rx="2"/><path d="M5 12h22M12 5v22" opacity=".5"/>')],
      ['51', '', 'input channels', ico('<path d="M4 9l12-5 12 5-12 5zM4 16l12 5 12-5M4 22l12 5 12-5"/>')],
      ['1.9', 'M', 'model parameters', ico('<circle cx="7" cy="9" r="3"/><circle cx="7" cy="23" r="3"/><circle cx="16" cy="16" r="3"/><circle cx="25" cy="16" r="3"/><path d="M10 9l3 6M10 23l3-6M19 16h3"/>')]
    ];
    $('#vz-stats').innerHTML = stats.map(function (s) { return '<div class="vz-stat">' + s[3] + '<div class="n">' + s[0] + '<small>' + s[1] + '</small></div><div class="l">' + s[2] + '</div></div>'; }).join('');

    /* ---- three tools as pictures ---- */
    var gi = M.group_importance.rows, ja = CO.rows[0];
    var maxd = Math.max.apply(null, gi.map(function (r) { return r.R2_drop; }));
    var artHot = '<svg class="vz-art" viewBox="0 0 220 96" preserveAspectRatio="xMidYMid slice" aria-hidden="true"><defs><linearGradient id="vzg" x1="0" x2="1"><stop offset="0" stop-color="#2a78d6"/><stop offset=".5" stop-color="#9ec5f4"/><stop offset="1" stop-color="#f1aea8"/></linearGradient><radialGradient id="vzh1"><stop offset="0" stop-color="#b13f3c"/><stop offset="1" stop-color="#b13f3c" stop-opacity="0"/></radialGradient></defs><rect width="220" height="96" fill="url(#vzg)" opacity=".5"/>' +
      '<circle cx="150" cy="42" r="34" fill="url(#vzh1)"/><circle cx="70" cy="62" r="24" fill="url(#vzh1)" opacity=".7"/>' +
      '<path d="M128 28l18-8 22 10 6 20-14 16-22-2-12-18z" fill="none" stroke="#fff" stroke-width="2" stroke-dasharray="5 3"/><path d="M52 50l14-6 16 8 2 14-12 10-16-4z" fill="none" stroke="#fff" stroke-width="2" stroke-dasharray="5 3"/></svg>';
    var artDrv = '<div class="vz-art" style="height:auto;padding:10px 12px">' + gi.map(function (r, i) {
      return '<div class="vz-row" style="margin-top:' + (i ? 7 : 0) + 'px"><span>' + r.group + '</span><span class="t"><i class="c-hot" style="width:' + Math.round(r.R2_drop / maxd * 100) + '%;opacity:' + (1 - i * 0.22) + '"></i></span><b>' + f2(r.R2_drop) + '</b></div>';
    }).join('') + '<div class="vz-row" style="margin-top:7px"><span>the rest</span><span class="t"><i class="c-first" style="width:' + Math.round(M.group_importance.rest_below / maxd * 100) + '%"></i></span><b>&lt;' + f1(M.group_importance.rest_below) + '</b></div></div>';
    var artCool = '<div class="vz-art" style="height:auto;padding:12px"><div class="vz-row" style="grid-template-columns:70px minmax(0,1fr) 44px;margin-top:0"><span>Gap now</span><span class="t"><i class="c-hot" style="width:100%"></i></span><b>' + f1(ja.pred_gap_C) + '&deg;C</b></div>' +
      '<div class="vz-row" style="grid-template-columns:70px minmax(0,1fr) 44px"><span>' + ja.veg_dose_pct + '% trees</span><span class="t"><i class="c-good" style="width:3%"></i></span><b>0&deg;C</b></div>' +
      '<div style="font-size:11px;color:var(--text-muted);margin-top:8px">Jaipur hotspot, README result</div></div>';
    $('#vz-tools').innerHTML = [['1 &middot; Where', 'Find hotspots', artHot], ['2 &middot; Why', 'Explain the heat', artDrv], ['3 &middot; What if', 'Test the cooling', artCool]].map(function (t) {
      return '<div class="vz-card"><span class="step">' + t[0] + '</span><h3>' + t[1] + '</h3>' + t[2] + '</div>'; }).join('');

    /* ---- findings as mini charts ---- */
    var mr = M.models.rows, first = mr[0], fin = mr[2], de = CO.rows[1];
    var unexpl = Math.round(((1 - first.R2_anomaly) - (1 - fin.R2_anomaly)) / (1 - first.R2_anomaly) * 100);
    function row(lab, val, max, cls, txt) { return '<div class="vz-row"><span>' + lab + '</span><span class="t"><i class="' + cls + '" style="width:' + Math.round(val / max * 100) + '%"></i></span><b>' + txt + '</b></div>'; }
    var vc = M.vegetation_check.per_0_1_NDVI_cooling_C, vmax = Math.max.apply(null, vc.map(function (r) { return Math.max(r.model, r.data); }));
    var cards = [
      ['−' + unexpl + '%', 'Unexplained heat variance vs first model',
        row('First', 1 - first.R2_anomaly, 1 - first.R2_anomaly, 'c-first', f2(1 - first.R2_anomaly)) + row('Final', 1 - fin.R2_anomaly, 1 - first.R2_anomaly, 'c-final', f2(1 - fin.R2_anomaly))],
      ['½', 'Hottest blocks: gap the model sees vs observed',
        row('Jaipur seen', ja.pred_gap_C, ja.obs_gap_C, 'c-final', f1(ja.pred_gap_C) + '°') + row('Jaipur real', ja.obs_gap_C, ja.obs_gap_C, 'c-hot', f1(ja.obs_gap_C) + '°') +
        row('Delhi seen', de.pred_gap_C, de.obs_gap_C, 'c-final', f1(de.pred_gap_C) + '°') + row('Delhi real', de.obs_gap_C, de.obs_gap_C, 'c-hot', f1(de.obs_gap_C) + '°')],
      [f2(gi[0].R2_drop), 'R² lost when vegetation is shuffled',
        gi.map(function (r, i) { return row(r.group, r.R2_drop, gi[0].R2_drop, i ? 'c-first' : 'c-good', f2(r.R2_drop)); }).join('')],
      ['±0.2', '°C cooling per +0.1 NDVI: model vs data',
        vc.map(function (r) { return row(r.city + ' model', r.model, vmax, 'c-final', f1(r.model)) + row(r.city + ' data', r.data, vmax, 'c-good', f1(r.data)); }).join('')]
    ];
    cards[3][0] = '+' + Math.round(Math.min.apply(null, vc.map(function (r) { return (r.model / r.data - 1) * 100; }))) + '–' + Math.round(Math.max.apply(null, vc.map(function (r) { return (r.model / r.data - 1) * 100; }))) + '%';
    cards[3][1] = 'Cooling per +0.1 NDVI (°C): model vs satellite data';
    cards[1][0] = '~½';
    var host = $('#findings');
    if (host) {
      var box = document.createElement('div'); box.className = 'vz-find'; box.id = 'vz-find';
      box.innerHTML = cards.map(function (c) { return '<div class="vz-card"><div class="vz-big">' + c[0] + '</div><div class="vz-cap">' + c[1] + '</div>' + c[2] + '</div>'; }).join('');
      host.parentNode.insertBefore(box, host);
    }
  } catch (e) { console.error('visuals', e); }
})();
