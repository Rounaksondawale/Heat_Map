(function () {
  try {
    var view = document.getElementById('storyView'), track = document.getElementById('storyTrack'), stage = document.getElementById('storyStage');
    var IMGS = window.STORY_IMGS || [];
    if (!view || !track || view.dataset.ready || !IMGS.length) return; view.dataset.ready = '1';
    var D = window.HM_DATA || {}, reduce = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;
    var PC = (D.perCity && D.perCity.cities) || [], CO = (D.cooling && D.cooling.rows) || [];
    var f2 = function (v) { return Number(v).toFixed(2); }, f1 = function (v) { return Number(v).toFixed(1); };
    var meanR2 = D.perCity && D.perCity.mean ? D.perCity.mean.R2_within : 0.72, meanAUC = D.perCity && D.perCity.mean ? D.perCity.mean.hot_AUC : 0.93;
    var jaip = CO[0] || {pred_gap_C: 2.2, veg_dose_pct: 40};

    /* One step per picture. Numbers come from the dashboard data (Delhi drivers: README figure). */
    var STEPS = [
      {t: 'Satellites first', p: 'Landsat 8 and 9 fly over each city. We keep the clearest March to May scenes, up to 40 per city.', f: '50 cities · 40 km tiles · 30 m pixels'},
      {t: 'Pixels become clues', p: 'Every 30 m pixel gets 51 numbers: how green, how built-up, how wet, how high, and what surrounds it. A U-Net learns from them on 40 cities.', f: '51 channels · about 2 M parameters'},
      {t: 'Spot the hotspots', p: 'The trained U-Net predicts how much hotter or cooler each pixel is than its own city’s average. Blocks far hotter than their neighbours join into clusters, ranked from worst.', f: '600 m blocks · Gi* z of 2.58 or more'},
      {t: 'Why is it hot?', p: 'Six feature groups share the extra heat. In Delhi’s biggest hotspot, vegetation adds +2.11 °C and built-up +1.10 °C, while water cools by 0.63.', f: 'Shapley split in °C'},
      {t: 'Try a cooler city', p: 'Plant trees or paint roofs white, re-predict, and watch the gap close. Jaipur’s modelled gap of ' + f1(jaip.pred_gap_C) + ' °C closes at ' + jaip.veg_dose_pct + ' % trees.', f: f1(jaip.pred_gap_C) + ' °C gap · ' + jaip.veg_dose_pct + ' % trees'},
      {t: 'Check the score', p: 'On 10 cities it never trained on: R² ' + f2(meanR2) + ' inside the city and hotspot AUC ' + f2(meanAUC) + '. Held-out means held out.', f: 'R² ' + f2(meanR2) + ' · hot AUC ' + f2(meanAUC)}
    ];
    var N = Math.min(STEPS.length, IMGS.length);

    /* pictures: stacked, cross-faded. A blurred copy fills the sides of the two narrower pictures. */
    var html = '';
    for (var i = 0; i < N; i++) html += '<div class="shot" data-i="' + i + '"><div class="back" style="background-image:url(' + IMGS[i] + ')"></div><img src="' + IMGS[i] + '" alt="" decoding="async"></div>';
    view.insertAdjacentHTML('afterbegin', html);
    view.querySelectorAll('.shot img').forEach(function (im) { function mark() { if (im.naturalWidth / im.naturalHeight < 1.4) im.parentNode.classList.add('narrow'); } if (im.complete) mark(); else im.addEventListener('load', mark); });

    var $ = function (id) { return document.getElementById(id); };
    $('sDots').innerHTML = STEPS.slice(0, N).map(function (s, k) { return '<button type="button" data-i="' + k + '" aria-label="Step ' + (k + 1) + ': ' + s.t + '" aria-current="false"></button>'; }).join('') + '<span>scroll</span>';
    var cur = -1;
    function show(i) {
      if (i === cur) return; cur = i; var S = STEPS[i];
      view.querySelectorAll('.shot').forEach(function (g, k) { g.classList.toggle('on', k === i); });
      $('sN').textContent = 'Step ' + (i + 1) + ' of ' + N; $('sT').textContent = S.t; $('sP').textContent = S.p; $('sF').textContent = S.f;
      $('sDots').querySelectorAll('button').forEach(function (b, k) { b.setAttribute('aria-current', String(k === i)); });
    }

    /* scroll-driven: the sticky stage stays while the page scrolls through N steps */
    var PER = 0.6;                           /* viewport heights of scrolling per step */
    function size() { track.style.height = (stage.offsetHeight + N * PER * window.innerHeight) + 'px'; }
    function stickyTop() { return parseFloat(getComputedStyle(stage).top) || 0; }
    function onScroll() {
      var r = track.getBoundingClientRect(), span = track.offsetHeight - stage.offsetHeight, p = span > 0 ? (stickyTop() - r.top) / span : 0;
      p = Math.max(0, Math.min(0.9999, p)); show(Math.floor(p * N));
    }
    $('sDots').addEventListener('click', function (e) {
      var b = e.target.closest('button'); if (!b) return;
      var span = track.offsetHeight - stage.offsetHeight, y = window.scrollY + track.getBoundingClientRect().top - stickyTop() + span * ((+b.dataset.i + .5) / N);
      window.scrollTo({top: y, behavior: reduce ? 'auto' : 'smooth'});
    });
    window.addEventListener('scroll', onScroll, {passive: true}); window.addEventListener('resize', function () { size(); onScroll(); });
    size(); show(0); onScroll();
    setTimeout(function () { size(); onScroll(); }, 600);
  } catch (e) { console.error('story', e); }
})();
