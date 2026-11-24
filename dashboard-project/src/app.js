(function () {
  try {
    var sec = document.getElementById('azure'); if (!sec) return;
    var wrap = sec.querySelector('.wrap'), head = wrap && wrap.querySelector('.sec-head');
    if (!wrap || !head || wrap.dataset.app) return; wrap.dataset.app = '1';

    var css = document.createElement('style');
    css.textContent =
      '.app-flow{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:28px;margin-top:var(--s4)}' +
      '.app-step{position:relative;background:var(--surface);border:1px solid var(--border);border-radius:var(--r-card);padding:14px;min-width:0;transition:transform .2s ease,box-shadow .2s ease}' +
      '.app-step:hover{transform:translateY(-3px);box-shadow:0 14px 30px -18px rgba(0,0,0,.4)}' +
      '.app-step:not(:last-child)::after{content:"";position:absolute;right:-22px;top:50%;width:16px;height:16px;margin-top:-8px;border-top:3px solid var(--accent);border-right:3px solid var(--accent);transform:rotate(45deg) scale(.7);opacity:.8}' +
      '.app-n{position:absolute;top:10px;left:10px;width:26px;height:26px;border-radius:50%;background:var(--accent);color:#fff;font:800 14px/26px var(--display);text-align:center;z-index:1}' +
      '.app-step svg{display:block;width:100%;height:auto;margin-bottom:10px}' +
      '.app-step h3{font:750 18px/1.15 var(--display);letter-spacing:-.02em;margin:0 0 4px}' +
      '.app-step p{font-size:13px;color:var(--text-secondary);margin:0}' +
      '.app-chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:var(--s5);align-items:center}' +
      '.app-chips b{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--text-muted);margin-right:4px}' +
      '.app-chips span{border:1px solid var(--border);background:var(--surface);border-radius:999px;padding:5px 12px;font-size:13px;color:var(--text-secondary)}' +
      '@media (max-width:1100px){.app-flow{grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.app-step::after{display:none!important}}' +
      '@media (max-width:520px){.app-flow{grid-template-columns:minmax(0,1fr)}}' +
      '@media (prefers-reduced-motion:reduce){.app-step{transition:none}}';
    document.head.appendChild(css);

    var bg = '<rect width="200" height="120" rx="8" fill="var(--surface-raised)"/>';
    var art = [
      /* 1 draw a box on a map */
      '<svg viewBox="0 0 200 120" aria-hidden="true">' + bg +
      '<path d="M70 12L120 16L134 40L152 52L128 72L116 106L100 114L86 96L64 66L48 44L60 28Z" fill="var(--grid)" stroke="var(--axis)" stroke-width="1.5"/>' +
      '<rect x="84" y="46" width="42" height="30" fill="rgba(37,106,191,.18)" stroke="var(--accent)" stroke-width="2" stroke-dasharray="5 3"/>' +
      '<g fill="var(--accent)"><circle cx="84" cy="46" r="3.5"/><circle cx="126" cy="46" r="3.5"/><circle cx="84" cy="76" r="3.5"/><circle cx="126" cy="76" r="3.5"/></g>' +
      '<path d="M132 78l0 16 4-4 3 7 3-1-3-7 6 0z" fill="var(--text-primary)"/></svg>',
      /* 2 fetch layers */
      '<svg viewBox="0 0 200 120" aria-hidden="true">' + bg +
      '<g stroke="rgba(0,0,0,.25)" stroke-width="1"><polygon points="30,86 100,66 170,86 100,106" fill="#c9a96a"/><polygon points="30,64 100,44 170,64 100,84" fill="#9aa3ad"/><polygon points="30,42 100,22 170,42 100,62" fill="#4f8a4c"/></g>' +
      '<g font-family="IBM Plex Mono,monospace" font-size="9" font-weight="600" fill="#fff" text-anchor="middle"><text x="100" y="46">Landsat</text><text x="100" y="68">Elevation</text><text x="100" y="90">Land cover</text></g></svg>',
      /* 3 predict */
      '<svg viewBox="0 0 200 120" aria-hidden="true"><defs><linearGradient id="appg3" x1="0" x2="1" y1="1" y2="0"><stop offset="0" stop-color="#2a78d6"/><stop offset=".5" stop-color="#f0c27a"/><stop offset="1" stop-color="#b13f3c"/></linearGradient></defs>' + bg +
      '<g fill="#8f8a7c"><rect x="12" y="38" width="16" height="16"/><rect x="30" y="38" width="16" height="16" fill="#4f8a4c"/><rect x="12" y="56" width="16" height="16" fill="#b9b2a2"/><rect x="30" y="56" width="16" height="16"/></g>' +
      '<g fill="var(--accent)" opacity=".85"><rect x="62" y="30" width="9" height="52" rx="2"/><rect x="76" y="38" width="9" height="36" rx="2"/><rect x="90" y="46" width="9" height="20" rx="2"/><rect x="104" y="38" width="9" height="36" rx="2"/><rect x="118" y="30" width="9" height="52" rx="2"/></g>' +
      '<path d="M50 56h8M130 56h8" stroke="var(--text-muted)" stroke-width="2"/><rect x="146" y="36" width="42" height="42" rx="3" fill="url(#appg3)"/>' +
      '<text x="100" y="104" text-anchor="middle" font-family="IBM Plex Mono,monospace" font-size="9" fill="var(--text-muted)">U-Net</text></svg>',
      /* 4 hotspots */
      '<svg viewBox="0 0 200 120" aria-hidden="true"><defs><linearGradient id="appg4" x1="0" x2="1"><stop offset="0" stop-color="#2a78d6" stop-opacity=".55"/><stop offset=".5" stop-color="#9ec5f4" stop-opacity=".55"/><stop offset="1" stop-color="#f1aea8" stop-opacity=".7"/></linearGradient><radialGradient id="appr4"><stop offset="0" stop-color="#b13f3c"/><stop offset="1" stop-color="#b13f3c" stop-opacity="0"/></radialGradient></defs>' + bg +
      '<rect x="40" y="14" width="120" height="92" rx="4" fill="url(#appg4)"/><circle cx="122" cy="52" r="26" fill="url(#appr4)"/><circle cx="76" cy="78" r="18" fill="url(#appr4)" opacity=".75"/>' +
      '<path d="M104 36l16-6 18 10 4 18-12 14-18-2-10-14z" fill="none" stroke="#fff" stroke-width="2" stroke-dasharray="5 3"/><path d="M62 70l12-4 12 6 2 12-10 8-12-4z" fill="none" stroke="#fff" stroke-width="2" stroke-dasharray="5 3"/>' +
      '<g font-family="IBM Plex Mono,monospace" font-size="10" font-weight="700" fill="#fff"><text x="116" y="56">#1</text><text x="70" y="82">#2</text></g></svg>',
      /* 5 cooling */
      '<svg viewBox="0 0 200 120" aria-hidden="true">' + bg +
      '<g><rect x="34" y="62" width="6" height="22" fill="#7a5a3a"/><circle cx="37" cy="52" r="17" fill="#3da05a"/><circle cx="28" cy="60" r="11" fill="#2f8a4c"/><circle cx="47" cy="60" r="11" fill="#2f8a4c"/></g>' +
      '<g><polygon points="120,58 146,40 172,58" fill="#9fdcf0" stroke="#5aa8c4" stroke-width="1.5"/><rect x="126" y="58" width="40" height="26" fill="var(--grid)" stroke="var(--axis)"/></g>' +
      '<rect x="22" y="96" width="156" height="8" rx="4" fill="var(--grid)"/><rect x="22" y="96" width="156" height="8" rx="4" fill="#b13f3c"/><rect x="22" y="96" width="18" height="8" rx="4" fill="#2b9a4b"/>' +
      '<text x="100" y="92" text-anchor="middle" font-family="IBM Plex Mono,monospace" font-size="9" font-weight="700" fill="var(--text-primary)">gap 2.2°C → 0</text></svg>'
    ];
    var steps = [
      ['Draw a box', 'Anywhere in India, 1–10 km a side'],
      ['Fetch the data', 'Landsat, elevation, land cover, buildings'],
      ['Predict heat', 'The U-Net scores every 30 m pixel'],
      ['Find hotspots', 'Ranked clusters, with what drives each'],
      ['Test cooling', 'Add trees or cool roofs, watch the gap close']
    ];

    /* hide the old Azure diagram and text (kept in the page so other code still finds its elements) */
    var old = document.createElement('div'); old.hidden = true; old.style.display = 'none';
    Array.prototype.slice.call(wrap.children).forEach(function (k) { if (k !== head) old.appendChild(k); });
    wrap.appendChild(old);

    head.querySelector('.eyebrow').innerHTML = '04 &middot; The app';
    head.querySelector('h2').textContent = 'Draw a box. Get the hotspots.';
    var nav = document.querySelector('.nav a[href="#azure"]'); if (nav) nav.textContent = 'The app';

    var box = document.createElement('div');
    box.innerHTML = '<div class="app-flow">' + steps.map(function (s, i) {
      return '<div class="app-step"><span class="app-n">' + (i + 1) + '</span>' + art[i] + '<h3>' + s[0] + '</h3><p>' + s[1] + '</p></div>';
    }).join('') + '</div>' +
      '<div class="app-chips"><b>Runs as</b><span>Streamlit app</span><span>Docker image, CPU only</span><span>CI tests on every push</span><span>Azure Container Apps or Hugging Face</span></div>';
    wrap.insertBefore(box, old);

    /* ---- what it cost: Azure cost analysis (accumulated cost, UAE North), INR ---- */
    var items = [
      ['Compute', 'Virtual machines', 3742, '--held-out', 'Feature extraction and U-Net training on the 4-core ML instance'],
      ['Storage', 'Disks and files', 2172, '--train', 'Satellite grids, patches and checkpoints'],
      ['Network', 'Load balancer, bandwidth', 64.81, '--bias-warm', 'Traffic and routing'],
      ['Secrets', 'Key Vault', 0.01, '--success-text', 'Keys and tokens']
    ];
    var total = items.reduce(function (a, r) { return a + r[2]; }, 0), R = 54, C = 2 * Math.PI * R, off = 0;
    function inr(v) { return '₹' + (v < 1 ? v.toFixed(2) : Math.round(v).toLocaleString('en-IN')); }
    var arcs = items.map(function (r) {
      var f = r[2] / total, len = Math.max(f * C, f > 0 ? 1.5 : 0), s = '<circle cx="70" cy="70" r="' + R + '" fill="none" stroke="var(' + r[3] + ')" stroke-width="22" stroke-dasharray="' + len.toFixed(2) + ' ' + (C - len).toFixed(2) + '" stroke-dashoffset="' + (-off).toFixed(2) + '" transform="rotate(-90 70 70)"/>';
      off += f * C; return s;
    }).join('');
    var cs = document.createElement('style');
    cs.textContent =
      '.cost{margin-top:var(--s6)}.cost h3.sub-h{margin-top:0}' +
      '.cost-grid{display:grid;grid-template-columns:minmax(0,300px) minmax(0,1fr);gap:var(--s6);align-items:center;background:var(--surface);border:1px solid var(--border);border-radius:var(--r-card);padding:var(--s5)}' +
      '.cost-donut{position:relative;width:220px;max-width:100%;margin:0 auto}.cost-donut svg{display:block;width:100%;height:auto}' +
      '.cost-mid{position:absolute;inset:0;display:grid;place-content:center;text-align:center}' +
      '.cost-mid b{font:800 30px/1 var(--display);letter-spacing:-.03em;font-variant-numeric:tabular-nums}.cost-mid span{font-size:12px;color:var(--text-muted);margin-top:4px}' +
      '.cost-row{display:grid;grid-template-columns:12px minmax(0,1fr) auto;gap:6px 12px;align-items:center;padding:10px 0;border-bottom:1px solid var(--grid)}' +
      '.cost-row:last-child{border-bottom:0}.cost-row .d{width:12px;height:12px;border-radius:3px}' +
      '.cost-row b{font:700 16px var(--display)}.cost-row small{display:block;font-size:12px;color:var(--text-muted);margin-top:1px}' +
      '.cost-row .amt{font:800 20px var(--display);letter-spacing:-.02em;font-variant-numeric:tabular-nums;text-align:right}.cost-row .amt small{font:500 12px var(--font)}' +
      '.cost-bar{grid-column:2 / -1;height:6px;border-radius:3px;background:var(--grid);overflow:hidden}.cost-bar i{display:block;height:100%;border-radius:3px}' +
      '.cost-row.off{opacity:.8}.cost-row.off .d{background:transparent;border:2px dashed var(--axis)}' +
      '.cost-stats{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:var(--s3);margin-top:var(--s3)}' +
      '.cost-stats div{background:var(--surface);border:1px solid var(--border);border-radius:var(--r-control);padding:12px}.cost-stats b{display:block;font:800 24px/1 var(--display);letter-spacing:-.02em}.cost-stats span{font-size:12px;color:var(--text-muted)}' +
      '@media (max-width:760px){.cost-grid{grid-template-columns:minmax(0,1fr);padding:var(--s4)}.cost-stats{grid-template-columns:minmax(0,1fr)}}';
    document.head.appendChild(cs);
    var cost = document.createElement('div'); cost.className = 'cost';
    cost.innerHTML = '<h3 class="sub-h">What it cost on Azure</h3><div class="cost-grid">' +
      '<div class="cost-donut"><svg viewBox="0 0 140 140" role="img" aria-label="Donut chart of Azure spend by component"><circle cx="70" cy="70" r="' + R + '" fill="none" stroke="var(--grid)" stroke-width="22"/>' + arcs + '</svg><div class="cost-mid"><b>' + inr(total) + '</b><span>total spend</span></div></div>' +
      '<div>' + items.map(function (r) {
        return '<div class="cost-row"><span class="d" style="background:var(' + r[3] + ')"></span><div><b>' + r[0] + '</b><small>' + r[1] + ' · ' + r[4] + '</small></div><div class="amt">' + inr(r[2]) + '<small> ' + Math.round(r[2] / total * 100) + '%</small></div><div class="cost-bar"><i style="width:' + Math.max(1, r[2] / total * 100).toFixed(1) + '%;background:var(' + r[3] + ')"></i></div></div>';
      }).join('') +
      '<div class="cost-row off"><span class="d"></span><div><b>App hosting</b><small>Container Apps or Hugging Face · no line in this bill</small></div><div class="amt">–</div></div></div></div>' +
      '<div class="cost-stats"><div><b>' + inr(total / 50) + '</b><span>per city (total ÷ 50 cities)</span></div><div><b>' + Math.round(items[0][2] / total * 100) + '%</b><span>of spend is compute</span></div><div><b>UAE North</b><span>Azure region · accumulated cost</span></div></div>';
    wrap.insertBefore(cost, old);
  } catch (e) { console.error('app flow', e); }
})();
