/* Assessment wizard: step navigation, validation, live BMI, loading screen, result rendering. */
(function () {
  var form = document.getElementById('assessForm');
  var wizardCard = document.getElementById('wizardCard');
  var loader = document.getElementById('loader');
  var resultEl = document.getElementById('result');
  var stepper = document.getElementById('stepper');
  var panels = Array.prototype.slice.call(document.querySelectorAll('.wizard-step'));
  var dots = Array.prototype.slice.call(stepper.querySelectorAll('.step'));
  var prevBtn = document.getElementById('prevBtn');
  var nextBtn = document.getElementById('nextBtn');
  var formError = document.getElementById('formError');
  var current = 0;

  var ICONS = {
    check: '<svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>',
    alert: '<svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>'
  };

  /* ---------- helpers ---------- */
  function val(name) {
    var el = form.elements[name];
    if (!el) return '';
    if (el.length !== undefined && !el.tagName) { // RadioNodeList
      return el.value;
    }
    return el.value.trim();
  }
  function setError(name, msg) {
    var p = form.querySelector('.error[data-for="' + name + '"]');
    if (p) p.textContent = msg || '';
  }
  function esc(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }
  function sleep(ms) { return new Promise(function (r) { setTimeout(r, ms); }); }

  /* ---------- BMI ---------- */
  var bmiOut = document.getElementById('bmiReadout');
  function updateBmi() {
    var h = parseFloat(val('height_cm')), w = parseFloat(val('weight_kg'));
    if (!(h >= 100 && h <= 250 && w >= 30 && w <= 300)) { bmiOut.textContent = '--'; return; }
    var bmi = w / Math.pow(h / 100, 2);
    var cat = bmi < 18.5 ? 'Underweight' : bmi < 25 ? 'Normal' : bmi < 30 ? 'Overweight' : 'Obese';
    bmiOut.textContent = bmi.toFixed(1) + '  ·  ' + cat;
  }
  ['height_cm', 'weight_kg'].forEach(function (n) { form.elements[n].addEventListener('input', updateBmi); });

  /* ---------- validation ---------- */
  function checkNumber(name, lo, hi, required) {
    var v = val(name);
    setError(name, '');
    if (v === '') { if (required) { setError(name, 'Required'); return false; } return true; }
    var n = Number(v);
    if (isNaN(n) || n < lo || n > hi) { setError(name, 'Enter a value between ' + lo + ' and ' + hi); return false; }
    return true;
  }
  function checkChoice(name) {
    setError(name, '');
    if (!val(name)) { setError(name, 'Please choose an option'); return false; }
    return true;
  }
  var validators = [
    function () { var a = checkNumber('age', 18, 100, true), b = checkChoice('sex'),
                  c = checkNumber('height_cm', 100, 250), d = checkNumber('weight_kg', 30, 300); return a && b && c && d; },
    function () { var a = checkChoice('cp'), b = checkChoice('exang'); return a && b; },
    function () { var a = checkNumber('trestbps', 70, 250), b = checkNumber('chol', 100, 600); return a && b; },
    function () { var a = checkNumber('thalch', 60, 220), b = checkNumber('oldpeak', -3, 7); return a && b; }
  ];

  /* ---------- navigation ---------- */
  function show(i) {
    current = i;
    panels.forEach(function (p, k) { p.classList.toggle('active', k === i); });
    dots.forEach(function (d, k) {
      d.classList.toggle('active', k === i);
      d.classList.toggle('done', k < i);
    });
    prevBtn.hidden = i === 0;
    nextBtn.innerHTML = i === panels.length - 1
      ? 'See my results'
      : 'Next step <svg class="icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>';
    formError.textContent = '';
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }
  prevBtn.addEventListener('click', function () { if (current > 0) show(current - 1); });
  nextBtn.addEventListener('click', function () {
    if (!validators[current]()) return;
    if (current < panels.length - 1) show(current + 1); else submit();
  });
  // clear a field's error as soon as the user edits it
  form.addEventListener('input', function (e) { if (e.target.name) setError(e.target.name, ''); });
  form.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && e.target.tagName !== 'BUTTON') { e.preventDefault(); nextBtn.click(); }
  });

  /* ---------- submit ---------- */
  function payload() {
    var out = {};
    ['age', 'sex', 'cp', 'exang', 'trestbps', 'chol', 'fbs', 'restecg', 'thalch', 'oldpeak',
     'height_cm', 'weight_kg', 'model'].forEach(function (k) { out[k] = val(k); });
    return out;
  }

  var messages = ['Preparing your data...', 'Running the model...', 'Calculating your probability...'];
  async function submit() {
    wizardCard.hidden = true; resultEl.hidden = true; loader.hidden = false;
    stepper.hidden = true;
    var msgEl = document.getElementById('loaderMsg'), k = 0;
    msgEl.textContent = messages[0];
    var timer = setInterval(function () { k = (k + 1) % messages.length; msgEl.textContent = messages[k]; }, 700);
    try {
      var results = await Promise.all([
        fetch('/api/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify(payload()) }),
        sleep(2100)  // keep the loading screen up long enough to read
      ]);
      var res = results[0], data = await res.json();
      if (!res.ok) throw { fields: data.errors || {} };
      render(data);
    } catch (err) {
      loader.hidden = true; wizardCard.hidden = false; stepper.hidden = false;
      var fields = err && err.fields ? Object.keys(err.fields) : [];
      fields.forEach(function (f) { setError(f, err.fields[f]); });
      formError.textContent = fields.length
        ? 'Some answers need fixing: ' + fields.join(', ') + '.'
        : 'Something went wrong. Please try again.';
    } finally {
      clearInterval(timer);
    }
  }

  /* ---------- result ---------- */
  function list(items, kind) {
    return items.map(function (i) {
      var sign = i.effect > 0 ? '+' : '';
      return '<li class="' + kind + '"><span class="dot"></span><span><b>' + esc(i.feature) + '</b>: ' + esc(i.value) +
             ' <em>(' + sign + i.effect + ' points)</em></span></li>';
    }).join('');
  }

  function render(d) {
    var high = d.high_risk;
    var ins = d.insights;
    resultEl.innerHTML =
      '<span class="badge ' + (high ? 'bad' : 'good') + ' big">' + (high ? 'ELEVATED RISK' : 'LOWER RISK') + '</span>' +
      '<div class="result-icon ' + (high ? 'bad' : 'good') + '">' + (high ? ICONS.alert : ICONS.check) + '</div>' +
      '<div class="percent" id="pct">0%</div>' +
      '<div class="percent-label">ESTIMATED PROBABILITY OF HEART DISEASE</div>' +
      '<div class="model-chip">' + esc(d.model) + ' · test recall ' + Math.round(d.model_recall * 100) + '%</div>' +
      '<h2 class="verdict">' + (high ? 'Flagged for follow-up' : 'No flag raised') + '</h2>' +
      '<p class="muted center-text">Flagging cutoff for this model: ' + Math.round(d.threshold * 100) + '%. It is set low on purpose so that few ' +
      'sick patients are missed, which means some healthy people get flagged too.</p>' +
      '<div class="grid-2 result-cards">' +
        '<div class="panel"><h3>What influenced this estimate</h3><ul class="insights">' +
          (ins.raising.length || ins.lowering.length
            ? list(ins.raising, 'up') + list(ins.lowering, 'down')
            : '<li class="none">No single answer stood out.</li>') + '</ul>' +
          '<p class="tiny">Points are percentage points of risk compared with typical patients in the training data. ' +
          'This describes the model, not what causes disease.</p></div>' +
        '<div class="panel"><h3>General guidance</h3><ul class="plain">' +
          d.guidance.map(function (g) { return '<li>' + esc(g) + '</li>'; }).join('') + '</ul></div>' +
      '</div>' +
      '<div class="result-actions"><a class="btn btn-ghost" href="/history">Full history</a>' +
      '<button class="btn btn-ghost" id="againBtn" type="button">New assessment</button></div>' +
      '<p class="tiny center-text">Not a medical diagnosis. If you have severe chest pain, trouble breathing or fainting, call your local emergency number.</p>';

    loader.hidden = true; resultEl.hidden = false;
    window.scrollTo({ top: 0, behavior: 'smooth' });

    // count the percentage up
    var pct = document.getElementById('pct'), start = null, target = d.percent;
    function tick(ts) {
      if (start === null) start = ts;
      var p = Math.min((ts - start) / 900, 1);
      pct.textContent = Math.round(target * (1 - Math.pow(1 - p, 3))) + '%';
      if (p < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
    setTimeout(function () { pct.textContent = target + '%'; }, 1100);  // fallback if frames are throttled

    document.getElementById('againBtn').addEventListener('click', function () {
      form.reset(); updateBmi();
      resultEl.hidden = true; wizardCard.hidden = false; stepper.hidden = false;
      show(0);
    });
  }

  show(0);
})();
