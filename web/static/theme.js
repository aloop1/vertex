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
var _vxLovRaf = null, _vxLovStepTimer = null;
function vxShowLoading(opts) {
  opts = opts || {};
  var title = opts.title || '처리 중...';
  var steps = opts.steps && opts.steps.length ? opts.steps : ['처리 중…'];
  var titleEl = document.querySelector('#lov .lv-title');
  var stepEl = document.getElementById('lv-step');
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
    clearInterval(_vxLovStepTimer);
    _vxLovStepTimer = setInterval(function () {
      si = (si + 1) % steps.length;
      stepEl.textContent = steps[si];
    }, opts.interval || 1200);
  }
  var lov = document.getElementById('lov');
  if (lov) lov.classList.add('show');
}
function vxHideLoading() {
  if (_vxLovRaf) { cancelAnimationFrame(_vxLovRaf); _vxLovRaf = null; }
  clearInterval(_vxLovStepTimer);
  var lov = document.getElementById('lov');
  if (lov) lov.classList.remove('show');
}

document.addEventListener('DOMContentLoaded', vxInitToggle);
