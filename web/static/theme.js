/* Vertex — 테마 토글 + Plotly 차트 테마 헬퍼 */

(function () {
  var saved;
  try { saved = localStorage.getItem('vertex-theme'); } catch (e) { saved = null; }
  document.documentElement.setAttribute('data-theme', saved === 'light' ? 'light' : 'dark');
})();

function vxTheme() {
  return document.documentElement.getAttribute('data-theme') || 'dark';
}

function vxToggleTheme() {
  var next = vxTheme() === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  try { localStorage.setItem('vertex-theme', next); } catch (e) {}
  var btn = document.getElementById('theme-toggle');
  if (btn) btn.textContent = next === 'dark' ? '☀' : '☾';
  window.dispatchEvent(new CustomEvent('vx-themechange', { detail: { theme: next } }));
}

function vxInitToggle() {
  var btn = document.getElementById('theme-toggle');
  if (!btn) return;
  btn.textContent = vxTheme() === 'dark' ? '☀' : '☾';
  btn.onclick = vxToggleTheme;
}

/* CSS 변수 → Plotly 색상 */
function vxChartColors() {
  var s = getComputedStyle(document.documentElement);
  function v(name) { return s.getPropertyValue(name).trim(); }
  return {
    paper: v('--surface'),
    plot:  v('--surface-2'),
    grid:  v('--chart-grid'),
    line:  v('--chart-line'),
    text:  v('--text-dim'),
    accent: v('--accent'),
    ok: v('--ok'), warn: v('--warn'), bad: v('--bad'), danger: v('--danger'),
  };
}

/* 시리즈 팔레트 — 양쪽 테마에서 모두 가독되는 중채도 색 */
var VX_SERIES = ['#3a7bf4', '#e05252', '#17b287', '#e09c2e', '#9a55e0',
                 '#3fa8dc', '#e052a8', '#71b83a', '#e0763a', '#4a9c9c'];

/* 공통 Plotly 레이아웃 */
function vxBaseLayout(extra) {
  var c = vxChartColors();
  var base = {
    paper_bgcolor: c.paper,
    plot_bgcolor:  c.plot,
    font: { color: c.text, size: 12, family: "'Pretendard Variable', Pretendard, 'Segoe UI', sans-serif" },
    margin: { l: 65, r: 20, t: 24, b: 56 },
    legend: { bgcolor: 'rgba(0,0,0,0)', bordercolor: c.line, borderwidth: 1,
              orientation: 'v', yanchor: 'top', y: 1, xanchor: 'left', x: 1.01 },
    xaxis: { gridcolor: c.grid, linecolor: c.line, zerolinecolor: c.line },
    yaxis: { gridcolor: c.grid, linecolor: c.line, zerolinecolor: c.line },
    hovermode: 'x unified',
    transition: { duration: 280, easing: 'cubic-in-out' },
  };
  if (!extra) return base;
  var out = Object.assign({}, base, extra);
  if (extra.xaxis) out.xaxis = Object.assign({}, base.xaxis, extra.xaxis);
  if (extra.yaxis) out.yaxis = Object.assign({}, base.yaxis, extra.yaxis);
  if (extra.legend) out.legend = Object.assign({}, base.legend, extra.legend);
  return out;
}

/* 수명 히트맵 컬러스케일 (테마별) */
function vxHeatmapScale() {
  if (vxTheme() === 'dark') {
    return [[0, '#0f1117'], [0.1, '#1a1f30'], [0.25, '#2a1f50'], [0.4, '#6b2070'],
            [0.55, '#c03060'], [0.7, '#e87840'], [0.85, '#f7c66b'], [1, '#ffffff']];
  }
  /* 라이트: 짧은 수명 = 진한 적색 → 긴 수명 = 청록 */
  return [[0, '#8c1d18'], [0.2, '#d94801'], [0.4, '#f59f3c'], [0.6, '#f7dd72'],
          [0.8, '#7fd4a8'], [1, '#1b8a6b']];
}

/* 공용 로딩 오버레이 — #lov 마크업이 있는 페이지라면 어디서든 사용 가능 */
var _vxLovRaf = null, _vxLovStepTimer = null, _vxLovElapsedTimer = null;
function vxShowLoading(opts) {
  opts = opts || {};
  var title = opts.title || '처리 중...';
  var steps = opts.steps && opts.steps.length ? opts.steps : ['처리 중…'];
  var titleEl = document.querySelector('#lov .lv-title');
  var stepEl = document.getElementById('lv-step');
  var pipelineEl = document.getElementById('lv-pipeline');
  var elapsedEl = document.getElementById('lv-elapsed');
  if (titleEl) titleEl.textContent = title;

  var canvas = document.getElementById('lv-canvas');
  if (canvas) {
    var ctx = canvas.getContext('2d');
    canvas.width = window.innerWidth; canvas.height = window.innerHeight;
    var pts = Array.from({ length: 70 }, function () {
      return { x: Math.random() * canvas.width, y: Math.random() * canvas.height,
        r: Math.random() * 2 + .8, vx: (Math.random() - .5) * .5, vy: (Math.random() - .5) * .5,
        a: Math.random() * .6 + .15 };
    });
    (function frame() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      for (var i = 0; i < pts.length; i++) {
        var p = pts[i]; p.x += p.vx; p.y += p.vy;
        if (p.x < 0) p.x = canvas.width; if (p.x > canvas.width) p.x = 0;
        if (p.y < 0) p.y = canvas.height; if (p.y > canvas.height) p.y = 0;
        ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(74,124,247,' + p.a + ')'; ctx.fill();
        for (var j = i + 1; j < pts.length; j++) {
          var dx = p.x - pts[j].x, dy = p.y - pts[j].y, d = Math.sqrt(dx * dx + dy * dy);
          if (d < 110) {
            ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(pts[j].x, pts[j].y);
            ctx.strokeStyle = 'rgba(74,124,247,' + ((1 - d / 110) * .15) + ')';
            ctx.lineWidth = .7; ctx.stroke();
          }
        }
      }
      _vxLovRaf = requestAnimationFrame(frame);
    })();
  }

  if (stepEl) {
    var si = 0; stepEl.textContent = steps[0];
    function renderPipeline() {
      if (!pipelineEl) return;
      pipelineEl.innerHTML = steps.map(function (step, i) {
        var cls = i < si ? 'done' : i === si ? 'active' : '';
        return '<span class="lv-stage ' + cls + '">' + step.replace(/…/g, '') + '</span>';
      }).join('');
    }
    renderPipeline();
    clearInterval(_vxLovStepTimer);
    _vxLovStepTimer = setInterval(function () {
      if (si < steps.length - 1) si += 1;
      stepEl.textContent = steps[si];
      renderPipeline();
    }, opts.interval || 1200);
  }
  var startedAt = Date.now();
  clearInterval(_vxLovElapsedTimer);
  if (elapsedEl) {
    elapsedEl.textContent = '경과 0초';
    _vxLovElapsedTimer = setInterval(function () {
      elapsedEl.textContent = '경과 ' + Math.floor((Date.now() - startedAt) / 1000) + '초';
    }, 250);
  }
  var lov = document.getElementById('lov');
  if (lov) lov.classList.add('show');
}
function vxHideLoading() {
  if (_vxLovRaf) { cancelAnimationFrame(_vxLovRaf); _vxLovRaf = null; }
  clearInterval(_vxLovStepTimer);
  clearInterval(_vxLovElapsedTimer);
  var lov = document.getElementById('lov');
  if (lov) lov.classList.remove('show');
}

/* 전체 화면의 은은한 결정 격자 배경 */
var _vxAmbientRaf = null;
function vxInitAmbient() {
  if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  var canvas = document.createElement('canvas');
  canvas.id = 'vx-ambient';
  canvas.setAttribute('aria-hidden', 'true');
  document.body.insertBefore(canvas, document.body.firstChild);
  var ctx = canvas.getContext('2d'), points = [], mouse = { x: -9999, y: -9999 };
  function resize() {
    var ratio = Math.min(window.devicePixelRatio || 1, 1.5);
    canvas.width = Math.floor(window.innerWidth * ratio); canvas.height = Math.floor(window.innerHeight * ratio);
    canvas.style.width = window.innerWidth + 'px'; canvas.style.height = window.innerHeight + 'px';
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    points = Array.from({ length: Math.max(18, Math.min(34, Math.floor(window.innerWidth / 45))) }, function () {
      return { x: Math.random() * window.innerWidth, y: Math.random() * window.innerHeight,
        vx: (Math.random() - .5) * .12, vy: (Math.random() - .5) * .12, r: Math.random() * 1.2 + .6 };
    });
  }
  function frame() {
    ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
    var dark = vxTheme() === 'dark';
    for (var i = 0; i < points.length; i++) {
      var p = points[i], dxm = p.x - mouse.x, dym = p.y - mouse.y, dm = Math.sqrt(dxm * dxm + dym * dym);
      if (dm < 130 && dm > 1) { p.x += dxm / dm * .08; p.y += dym / dm * .08; }
      p.x += p.vx; p.y += p.vy;
      if (p.x < -10) p.x = window.innerWidth + 10; if (p.x > window.innerWidth + 10) p.x = -10;
      if (p.y < -10) p.y = window.innerHeight + 10; if (p.y > window.innerHeight + 10) p.y = -10;
      ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = dark ? 'rgba(126,184,247,.24)' : 'rgba(45,95,208,.16)'; ctx.fill();
      for (var j = i + 1; j < points.length; j++) {
        var dx = p.x - points[j].x, dy = p.y - points[j].y, d = Math.sqrt(dx * dx + dy * dy);
        if (d < 150) {
          ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(points[j].x, points[j].y);
          ctx.strokeStyle = dark ? 'rgba(126,184,247,' + ((1 - d / 150) * .08) + ')' : 'rgba(45,95,208,' + ((1 - d / 150) * .06) + ')';
          ctx.lineWidth = .7; ctx.stroke();
        }
      }
    }
    _vxAmbientRaf = requestAnimationFrame(frame);
  }
  window.addEventListener('resize', resize, { passive: true });
  window.addEventListener('pointermove', function (e) { mouse.x = e.clientX; mouse.y = e.clientY; }, { passive: true });
  window.addEventListener('pointerleave', function () { mouse.x = -9999; mouse.y = -9999; }, { passive: true });
  resize(); frame();
}

document.addEventListener('DOMContentLoaded', function () { vxInitToggle(); vxInitAmbient(); });
