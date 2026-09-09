/* ROSA blog widgets — vanilla JS, no dependencies.
 *
 * Widgets, sharing small numeric utilities:
 *   1. #calib-widget  — Fig-7-style ridge world: reward-function uncertainty over a
 *                       1-D task; entropy-reg PG vs ROSA+Max, BOTH optimized live
 *                       per column (same SGD+momentum/steps; PG fixed point = softmax(rbar/lam)).
 *   2. #adv-widget-bags — distill-style advantage-computation trace (DOM, not
 *                       canvas): standard PG w/ LOO mean vs ROSA+Max w/ LOO max,
 *                       hover a row for the worked line; bag / preset buttons.
 *   3. #flow-sgdm-tr-widget — exact-gradient trajectories on the 3-action simplex,
 *                       GD + heavy-ball momentum from 24 starts sampled near the
 *                       broccoli vertex (fixed seed; resample draws a new set), plus a
 *                       reward / entropy trace column. Vertices: A=🍎, B=🍓, C=🥦.
 *   4. #ctrl-widget   — controllability: exact landscapes + optima as rho shifts
 *                       (ROSA optimum via the paper's closed form) + 10 fixed
 *                       starts, re-run when rho / n change.
 *   5. #setfn-tr-widget — build-your-own success-count function f~; induced simplex
 *                       landscape + optimal set, exact binomial expectations;
 *                       overlay: random starts, SGD+momentum on exact ∇J, resample;
 *                       reward / entropy trace column.
 *
 * Every canvas widget animates only while on screen and idles once its runs
 * converge (see animateWhileVisible); landscapes are tabulated per axis (lut1).
 */
(() => {
'use strict';

/* ---------- shared utilities ---------- */

const INK = '#16181d', MUTED = '#5b6069', FAINT = '#9aa0a8', HAIR = '#e7e8ea';
const C1 = '#d6336c', C2 = '#f76707', C3 = '#7048e8', C4 = '#0ca678';   // reward-function colors
const MONO = '12px "JetBrains Mono", monospace';

function softmax(theta) {
  let mx = -Infinity;
  for (const t of theta) mx = Math.max(mx, t);
  const e = theta.map(t => Math.exp(t - mx));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map(v => v / s);
}

function makeSgdm(dim, lr, mu = 0.9) {
  const v = new Float64Array(dim);
  return {
    step(theta, grad, scale = 1) {   // gradient ASCENT with heavy-ball momentum
      for (let i = 0; i < dim; i++) { v[i] = mu * v[i] + grad[i]; theta[i] += lr * scale * v[i]; }
    },
    reset() { v.fill(0); },
  };
}

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function hexStops(hexes) {
  return hexes.map(h => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)]);
}
function lerpMap(stops, t) {
  t = Math.min(1, Math.max(0, t));
  const x = t * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(x)), f = x - i;
  const a = stops[i], b = stops[i + 1];
  return [a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2])];
}
/* matplotlib plasma — policy / landscape mass */
const PLASMA = hexStops(['#0d0887', '#41049d', '#6a00a8', '#8f0da4', '#b12a90', '#cc4778',
                         '#e16462', '#f2844b', '#fca636', '#fcce25', '#f0f921']);
function plasma(t) { return lerpMap(PLASMA, t); }
/* cool sequential — reward landscapes; never enters plasma yellow */
const COOL = hexStops(['#16181d', '#2c4a62', '#3d7a88', '#d7ece6']);
function cool(t) { return lerpMap(COOL, t); }

/* Hi-DPI canvas setup; returns 2d context sized in CSS pixels */
function setupCanvas(canvas, cssW, cssH) {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  canvas.style.width = cssW + 'px';
  canvas.style.height = cssH + 'px';
  canvas.style.minWidth = '0';   // flex items otherwise use the bitmap width as min-content
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  return ctx;
}

/* run cb via requestAnimationFrame only while el is on screen.
 * cb returns false when it has nothing left to animate (runs converged, no
 * pending input); the loop then idles instead of repainting an identical frame
 * at 60 fps. Returns wake(): call it after any input that changes what should
 * be drawn (slider, drag, button, resize). Also wakes once when web fonts
 * finish loading so canvas labels re-render in the right typeface. */
function animateWhileVisible(el, cb) {
  let visible = false, rafId = null, lastTime = null;
  const loop = now => {
    rafId = null;
    if (!visible) { lastTime = null; return; }
    const dt = lastTime === null ? 0 : Math.max(0, Math.min(50, now - lastTime));
    lastTime = now;
    if (cb(dt) !== false) rafId = requestAnimationFrame(loop);
    else lastTime = null;
  };
  const wake = () => {
    if (visible && rafId === null) {
      lastTime = null;
      rafId = requestAnimationFrame(loop);
    }
  };
  new IntersectionObserver(entries => {
    for (const e of entries) {
      visible = e.isIntersecting;
      if (visible) wake();
      else {
        lastTime = null;
        if (rafId !== null) { cancelAnimationFrame(rafId); rafId = null; }
      }
    }
  }, { threshold: 0.05 }).observe(el);
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(wake);
  return wake;
}

/* call fn when el's layout width (or the device pixel ratio) actually changes.
 * Plain `resize` events also fire on mobile when the address bar collapses
 * during scroll, which would otherwise rebuild landscapes and restart runs
 * mid-read; events are coalesced to one check per frame. */
function watchWidth(el, fn) {
  const key = () => `${el.clientWidth}:${window.devicePixelRatio || 1}`;
  let last = key(), pending = false;
  window.addEventListener('resize', () => {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => {
      pending = false;
      const k = key();
      if (k !== last) { last = k; fn(); }
    });
  });
}

/* tabulate a smooth fn on [0,1] once; returns a linearly interpolating lookup.
 * Every landscape on this page is separable, J(p) = f(pA) + g(pB), so the
 * per-pixel work in renderSimplexBg drops from repeated Math.pow to two array
 * reads. With 2048 knots the interpolation error is ~1e-7, far below one
 * colormap step. */
function lut1(fn, n = 2048) {
  const t = new Float64Array(n + 1);
  for (let i = 0; i <= n; i++) t[i] = fn(i / n);
  return x => {
    if (x <= 0) return t[0];
    if (x >= 1) return t[n];
    const u = x * n, i = u | 0, f = u - i;
    return t[i] + f * (t[i + 1] - t[i]);
  };
}

function star(ctx, x, y, r, fill, stroke) {
  ctx.beginPath();
  for (let i = 0; i < 10; i++) {
    const rad = (i % 2 === 0) ? r : r * 0.45;
    const a = -Math.PI / 2 + i * Math.PI / 5;
    const px = x + rad * Math.cos(a), py = y + rad * Math.sin(a);
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  }
  ctx.closePath();
  ctx.fillStyle = fill; ctx.fill();
  if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 1; ctx.stroke(); }
}

/* Stacked traces vs optimization step (expected reward on top, entropy below),
 * one thin polyline per run per series. series: [{ name, color }]. Values are
 * pushed per step from the owning widget; draw() replays up to the current step. */
function makeTraces(canvas, { series, total, reward = p => 0.5 * (p[0] + p[1]), logX = true }) {
  const metrics = [
    { label: 'reward', lo: 0, hi: 0.5, ref: 0.5, refLabel: '.5', fn: reward },
    { label: 'entropy', lo: 0, hi: Math.log(3), ref: Math.log(2), refLabel: 'ln2',
      fn: p => -p.reduce((s, v) => s + (v > 1e-12 ? v * Math.log(v) : 0), 0) },
  ];
  const P = { l: 28, r: 8, t: 15, b: 14 };
  /* SGD+momentum settles within ~100 steps of 1200: a log step axis keeps the transient readable */
  const xFrac = st => logX ? Math.log10(Math.max(st, 1)) / Math.log10(total) : st / total;
  const xTicks = logX ? [1, 10, 100, 1000].filter(t => t <= total) : [0, total / 2, total];
  let W = 0, H = 0, ctx = null, data = [];
  return {
    resize(w, h) { W = w; H = h; ctx = setupCanvas(canvas, W, H); },
    reset(nRuns) {
      data = series.map(() => metrics.map(() =>
        Array.from({ length: nRuns }, () => new Float32Array(total + 1))));
    },
    push(s, run, step, p) {
      if (step > total) return;
      for (let m = 0; m < metrics.length; m++) data[s][m][run][step] = metrics[m].fn(p);
    },
    draw(steps) {
      if (!ctx) return;
      const n = Math.min(steps, total);
      ctx.clearRect(0, 0, W, H);
      const ch = H / metrics.length;
      const x0 = P.l, x1 = W - P.r;
      const xOf = st => x0 + xFrac(st) * (x1 - x0);
      ctx.font = '10px "JetBrains Mono", monospace';
      metrics.forEach((mt, m) => {
        const top = m * ch + P.t, bot = (m + 1) * ch - P.b;
        const yOf = v => bot - ((v - mt.lo) / (mt.hi - mt.lo)) * (bot - top);
        // frame + reference line
        ctx.strokeStyle = HAIR; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(x0 + 0.5, top); ctx.lineTo(x0 + 0.5, bot + 0.5); ctx.lineTo(x1, bot + 0.5); ctx.stroke();
        for (const t of xTicks) {
          const tx = Math.round(xOf(t)) + 0.5;
          ctx.beginPath(); ctx.moveTo(tx, bot + 0.5); ctx.lineTo(tx, bot + 3.5); ctx.stroke();
        }
        ctx.setLineDash([2, 4]); ctx.strokeStyle = '#22d3ee'; ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.moveTo(x0, yOf(mt.ref) + 0.5); ctx.lineTo(x1, yOf(mt.ref) + 0.5); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = FAINT; ctx.textAlign = 'right';
        ctx.fillText(mt.refLabel, x0 - 4, yOf(mt.ref) + 3);
        ctx.fillText('0', x0 - 4, yOf(mt.lo) + 3);
        ctx.fillStyle = MUTED; ctx.textAlign = 'left';
        ctx.fillText(mt.label, x0, top - 5);
        // one polyline per run, per series
        ctx.lineWidth = 1.1; ctx.lineJoin = 'round';
        data.forEach((sd, s) => {
          ctx.strokeStyle = series[s].color; ctx.globalAlpha = series[s].alpha ?? 0.45;
          for (const arr of sd[m]) {
            ctx.beginPath();
            for (let st = 0; st <= n; st++) {
              const x = xOf(st), y = yOf(arr[st]);
              if (st === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            }
            ctx.stroke();
          }
        });
        ctx.globalAlpha = 1;
      });
      // step ticks under the last chart; series legend top-right of the first
      ctx.fillStyle = FAINT; ctx.textAlign = 'center';
      for (const t of xTicks) ctx.fillText(String(t), xOf(t), H - 3);
      ctx.textAlign = 'right';
      let lx = x1;
      for (let s = series.length - 1; s >= 0; s--) {
        ctx.fillStyle = series[s].color; ctx.fillText(series[s].name, lx, P.t - 5);
        lx -= ctx.measureText(series[s].name).width + 10;
      }
    },
  };
}

/* Per-pixel objective landscape over the 3-simplex. J takes [pA,pB,pC];
 * norm = 'auto' rescales colors to the landscape's own [min,max], or pass [lo,hi]. */
function renderSimplexBg(J, W, H, V, norm) {
  const dpr = window.devicePixelRatio || 1;
  const off = document.createElement('canvas');
  const pw = Math.round(W * dpr), ph = Math.round(H * dpr);
  off.width = pw; off.height = ph;
  const octx = off.getContext('2d');
  const img = octx.createImageData(pw, ph);
  const [ax, ay] = V.A, [bx, by] = V.B, [cx, cy] = V.C;
  const det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy);
  const vals = new Float64Array(pw * ph).fill(NaN);
  let lo = Infinity, hi = -Infinity;
  for (let py = 0; py < ph; py++) {
    for (let px = 0; px < pw; px++) {
      const x = px / dpr, y = py / dpr;
      const l1 = ((by - cy) * (x - cx) + (cx - bx) * (y - cy)) / det;
      const l2 = ((cy - ay) * (x - cx) + (ax - cx) * (y - cy)) / det;
      const l3 = 1 - l1 - l2;
      if (l1 < 0 || l2 < 0 || l3 < 0) continue;
      const v = J([l1, l2, l3]);
      vals[py * pw + px] = v;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  if (norm !== 'auto') { lo = norm[0]; hi = norm[1]; }
  for (let i = 0; i < vals.length; i++) {
    if (Number.isNaN(vals[i])) continue;
    const [r, g, b] = plasma((vals[i] - lo) / (hi - lo + 1e-12));
    img.data[4 * i] = r; img.data[4 * i + 1] = g; img.data[4 * i + 2] = b; img.data[4 * i + 3] = 255;
  }
  octx.putImageData(img, 0, 0);
  return off;
}

/* triangle frame + vertex labels, shared by the simplex widgets.
 * A/B/C = apple / strawberry / broccoli (same world as the advantage-trace bags). */
function drawSimplexFrame(ctx, V) {
  ctx.strokeStyle = INK; ctx.lineWidth = 1.4;
  ctx.beginPath(); ctx.moveTo(...V.A); ctx.lineTo(...V.B); ctx.lineTo(...V.C); ctx.closePath(); ctx.stroke();
  ctx.font = '18px "Apple Color Emoji", "Segoe UI Emoji", "Noto Color Emoji", sans-serif';
  ctx.fillStyle = INK;
  ctx.textAlign = 'right';  ctx.fillText('🍎', V.A[0] - 4, V.A[1] + 16);
  ctx.textAlign = 'left';   ctx.fillText('🍓', V.B[0] + 4, V.B[1] + 16);
  ctx.textAlign = 'center'; ctx.fillText('🥦', V.C[0], V.C[1] - 6);
}

/* =========================================================================
 * 1. CALIBRATION — reward-function uncertainty over a 1-D task (Fig-7 world)
 *    Left: reward ridges. Middle: optimal entropy-reg PG = softmax(rbar/lam),
 *    closed form. Right: ROSA+Max, per-column exact E[max] gradient ascent.
 * ========================================================================= */
function initCalibWidget() {
  const root = document.getElementById('calib-widget');
  if (!root) return;
  const canvases = root.querySelectorAll('canvas');
  if (canvases.length < 3) return;
  const lamSlider = root.querySelector('.lam-slider');
  const lamLabel = root.querySelector('.lam-label');
  const nSlider = root.querySelector('.n-slider');
  const nLabel = root.querySelector('.n-label');
  const resetBtn = root.querySelector('.reset-btn');

  const XC = 84, YB = 56;                 // grid: x columns, y bins
  const XMAX = 10, YMAX = 4;
  const WGT = [0.4, 0.3, 0.2, 0.1];       // rho over the four reward functions
  const NF = WGT.length;
  const RIDGE = [C1, C2, C3, C4];          // identity colors for R1–R4
  const SIG = 0.22;

  /* ridges share a trunk on the left, split mid-domain, partially re-merge */
  const trunk = x => 2 + 0.45 * Math.sin(0.55 * x + 0.6);
  const env = x => Math.pow(Math.sin(Math.PI * x / XMAX), 1.4);
  const OFF0 = [0.95, 0.32, -0.38, -1.0];
  const OFF = OFF0.slice();                // ridge offsets — user-draggable
  const gk = (k, x) => trunk(x) + OFF[k] * 1.05 * env(x) + 0.12 * Math.sin(1.3 * x + k * 2.1) * env(x);

  const yOfBin = b => (b + 0.5) * YMAX / YB;
  const xOfCol = c => (c + 0.5) * XMAX / XC;

  /* per column: per-fn values, scalarized rbar, ascending sort order (recomputed on ridge drag) */
  const V = [], RBAR = [], ORD = [];
  let rbarMax = 0;
  function recomputeGrid() {
    V.length = 0; RBAR.length = 0; ORD.length = 0; rbarMax = 0;
    for (let c = 0; c < XC; c++) {
      const x = xOfCol(c);
      const vk = [], rb = new Float64Array(YB);
      for (let k = 0; k < NF; k++) {
        const v = new Float64Array(YB);
        for (let b = 0; b < YB; b++) {
          const d = yOfBin(b) - gk(k, x);
          v[b] = Math.exp(-d * d / (2 * SIG * SIG));
        }
        vk.push(v);
        for (let b = 0; b < YB; b++) rb[b] += WGT[k] * v[b];
      }
      for (let b = 0; b < YB; b++) rbarMax = Math.max(rbarMax, rb[b]);
      V.push(vk); RBAR.push(rb);
      ORD.push(vk.map(v => Array.from({ length: YB }, (_, b) => b).sort((a, b2) => v[a] - v[b2])));
    }
  }
  recomputeGrid();

  const MAX_STEPS = 2400, STEPS_PER_SECOND = 480, MAX_PER_FRAME = 8;
  const FRAME_BUDGET_MS = 7;
  const MIN_CONV_STEPS = 800, CONV_INTERVAL = 100, CONV_TV = 0.0025;
  let stepCarry = 0;
  const makeConvergence = () => ({
    snapshot: new Float64Array(XC * YB),
    hasSnapshot: false,
    nextCheck: CONV_INTERVAL,
    stableChecks: 0,
    converged: false,
  });
  const convP = makeConvergence(), convR = makeConvergence();
  const resetConvergence = tracker => {
    tracker.hasSnapshot = false;
    tracker.nextCheck = CONV_INTERVAL;
    tracker.stableChecks = 0;
    tracker.converged = false;
  };

  const state = {
    lam: 0.05, n: 8,
    theta: new Float64Array(XC * YB), opt: null, steps: 0,       // ROSA logits
    thetaP: new Float64Array(XC * YB), optP: null, stepsP: 0,    // PG+entropy logits
    runR: true, runP: true,
  };
  const resetOpt = () => {
    state.theta.fill(0); state.opt = makeSgdm(XC * YB, 0.25); state.steps = 0;
    state.thetaP.fill(0); state.optP = makeSgdm(XC * YB, 0.25); state.stepsP = 0;
    resetConvergence(convR); resetConvergence(convP);
    state.runR = true; state.runP = true; stepCarry = 0;
  };
  resetOpt();

  /* exact dJ/dp for one column: J = sum_k w_k E[max of n iid draws of v_k].
   * E[max] = sum_s v_(s) (C_s^n - C_{s-1}^n) over ascending sorted bins, so
   * dE/dp_y = n * [ sum_{s>=rank(y)}^{S-1} C_s^{n-1}(v_s - v_{s+1}) + v_(S) ]. */
  const cumC = new Float64Array(YB), suf = new Float64Array(YB);
  function rosaColU(c, p, n, out) {
    out.fill(0);
    for (let k = 0; k < NF; k++) {
      const v = V[c][k], ord = ORD[c][k];
      let C = 0;
      for (let s = 0; s < YB; s++) { C += p[ord[s]]; cumC[s] = Math.min(1, C); }
      suf[YB - 1] = v[ord[YB - 1]];
      for (let s = YB - 2; s >= 0; s--) {
        suf[s] = suf[s + 1] + Math.pow(cumC[s], n - 1) * (v[ord[s]] - v[ord[s + 1]]);
      }
      const wn = WGT[k] * n;
      for (let s = 0; s < YB; s++) out[ord[s]] += wn * suf[s];
    }
  }

  const colP = new Float64Array(YB), colU = new Float64Array(YB), colG = new Float64Array(XC * YB);
  function updateConvergence(theta, steps, tracker) {
    if (steps < tracker.nextCheck) return;
    const hadSnapshot = tracker.hasSnapshot;
    let totalL1 = 0;
    for (let c = 0; c < XC; c++) {
      const off = c * YB;
      let mx = -Infinity;
      for (let b = 0; b < YB; b++) mx = Math.max(mx, theta[off + b]);
      let Z = 0;
      for (let b = 0; b < YB; b++) { colP[b] = Math.exp(theta[off + b] - mx); Z += colP[b]; }
      for (let b = 0; b < YB; b++) {
        colP[b] /= Z;
        if (hadSnapshot) totalL1 += Math.abs(colP[b] - tracker.snapshot[off + b]);
        tracker.snapshot[off + b] = colP[b];
      }
    }
    if (hadSnapshot && steps >= MIN_CONV_STEPS) {
      const meanTv = totalL1 / (2 * XC);
      tracker.stableChecks = meanTv < CONV_TV ? tracker.stableChecks + 1 : 0;
      tracker.converged = tracker.stableChecks >= 2;
    }
    tracker.hasSnapshot = true;
    while (tracker.nextCheck <= steps) tracker.nextCheck += CONV_INTERVAL;
  }

  const ENT = 0.015;                       // small entropy bonus, display smoothness only
  function stepRosa(iters) {
    for (let it = 0; it < iters; it++) {
      for (let c = 0; c < XC; c++) {
        const off = c * YB;
        let mx = -Infinity;
        for (let b = 0; b < YB; b++) mx = Math.max(mx, state.theta[off + b]);
        let Z = 0;
        for (let b = 0; b < YB; b++) { colP[b] = Math.exp(state.theta[off + b] - mx); Z += colP[b]; }
        for (let b = 0; b < YB; b++) colP[b] /= Z;
        rosaColU(c, colP, state.n, colU);
        let mean = 0, Hc = 0;
        for (let b = 0; b < YB; b++) {
          mean += colP[b] * colU[b];
          Hc -= colP[b] * Math.log(colP[b] + 1e-12);
        }
        for (let b = 0; b < YB; b++) {
          colG[off + b] = colP[b] * (colU[b] - mean)
            - ENT * colP[b] * (Math.log(colP[b] + 1e-12) + Hc);   // dH/dtheta
        }
      }
      state.opt.step(state.theta, colG);
      for (let i = 0; i < state.theta.length; i++) state.theta[i] = Math.max(-8, Math.min(8, state.theta[i]));
      state.steps += 1;
    }
  }

  /* PG+entropy panel: same live setup as ROSA (per-column softmax logits,
   * SGD+momentum, warm restarts), ascending E_p[rbar] + lam*H(p);
   * fixed point = softmax(rbar/lam). */
  function stepPg(iters) {
    for (let it = 0; it < iters; it++) {
      for (let c = 0; c < XC; c++) {
        const off = c * YB, rb = RBAR[c];
        let mx = -Infinity;
        for (let b = 0; b < YB; b++) mx = Math.max(mx, state.thetaP[off + b]);
        let Z = 0;
        for (let b = 0; b < YB; b++) { colP[b] = Math.exp(state.thetaP[off + b] - mx); Z += colP[b]; }
        for (let b = 0; b < YB; b++) colP[b] /= Z;
        let mean = 0, Hc = 0;
        for (let b = 0; b < YB; b++) {
          mean += colP[b] * rb[b];
          Hc -= colP[b] * Math.log(colP[b] + 1e-12);
        }
        for (let b = 0; b < YB; b++) {
          colG[off + b] = colP[b] * (rb[b] - mean)
            - state.lam * colP[b] * (Math.log(colP[b] + 1e-12) + Hc);   // lam * dH/dtheta
        }
      }
      state.optP.step(state.thetaP, colG);
      for (let i = 0; i < state.thetaP.length; i++) state.thetaP[i] = Math.max(-8, Math.min(8, state.thetaP[i]));
      state.stepsP += 1;
    }
  }

  /* ---- rendering: draw grids to offscreen canvases, scale up smoothly ---- */
  /* Layout knobs — heatmap width is (row − CSS gaps − sum of l+r) / 3;
   * height tracks width at ASPECT, so tightening pads/gaps scales the plot up
   * in both directions. CSS gaps: #calib-widget .panels (left↔mid) and .cgroup (mid↔right). */
  const RLAB = ['R₁', 'R₂', 'R₃', 'R₄'];
  const ASPECT = 0.72;          // heatH / heatW
  const HEAT_MAX = 300;         // px cap on heatmap width
  const PAD_T = 4, PAD_R = 2, PAD_B = 18;  // PAD_B: x-ticks + "X" (no tick numbers)
  const PADS = [
    { l: 32, r: PAD_R, t: PAD_T, b: PAD_B, yTitle: 'Y',      yTicks: true },
    { l: 32, r: PAD_R, t: PAD_T, b: PAD_B, yTitle: 'π(Y|X)', yTicks: true },
    { l: 2,  r: PAD_R, t: PAD_T, b: PAD_B, yTitle: null,     yTicks: false },
  ];
  let geom = [], ctxs;
  let rewardPanelDirty = true;
  const offReward = document.createElement('canvas');
  const offPg = document.createElement('canvas');
  const offRosa = document.createElement('canvas');
  for (const o of [offReward, offPg, offRosa]) { o.width = XC; o.height = YB; }

  function paintGrid(off, valueAt, cmap) {   // valueAt(c, b) in [0,1], b=0 is bottom
    const octx = off.getContext('2d');
    const img = octx.createImageData(XC, YB);
    for (let b = 0; b < YB; b++) {
      for (let c = 0; c < XC; c++) {
        const [r, g, bl] = cmap(valueAt(c, b));
        const i = 4 * ((YB - 1 - b) * XC + c);
        img.data[i] = r; img.data[i + 1] = g; img.data[i + 2] = bl; img.data[i + 3] = 255;
      }
    }
    octx.putImageData(img, 0, 0);
  }

  function paintReward() {
    paintGrid(offReward, (c, b) => 0.85 * RBAR[c][b] / rbarMax, cool);
    rewardPanelDirty = true;
  }

  function paintPg() {
    for (let c = 0; c < XC; c++) {
      const off = c * YB;
      let mx = -Infinity;
      for (let b = 0; b < YB; b++) mx = Math.max(mx, state.thetaP[off + b]);
      let Z = 0;
      for (let b = 0; b < YB; b++) { colP[b] = Math.exp(state.thetaP[off + b] - mx); Z += colP[b]; }
      let cm = 0;
      for (let b = 0; b < YB; b++) { colP[b] /= Z; cm = Math.max(cm, colP[b]); }
      for (let b = 0; b < YB; b++) PGIMG[off + b] = Math.pow(colP[b] / cm, 0.85);
    }
    paintGrid(offPg, (c, b) => PGIMG[c * YB + b], plasma);
  }
  const PGIMG = new Float64Array(XC * YB);

  function paintRosa() {
    for (let c = 0; c < XC; c++) {
      const off = c * YB;
      let mx = -Infinity;
      for (let b = 0; b < YB; b++) mx = Math.max(mx, state.theta[off + b]);
      let Z = 0;
      for (let b = 0; b < YB; b++) { colP[b] = Math.exp(state.theta[off + b] - mx); Z += colP[b]; }
      let cm = 0;
      for (let b = 0; b < YB; b++) { colP[b] /= Z; cm = Math.max(cm, colP[b]); }
      for (let b = 0; b < YB; b++) ROIMG[off + b] = Math.pow(colP[b] / cm, 0.85);
    }
    paintGrid(offRosa, (c, b) => ROIMG[c * YB + b], plasma);
  }
  const ROIMG = new Float64Array(XC * YB);

  function plotBox(g) {
    return { x0: g.pad.l, y0: g.pad.t, pW: g.cW - g.pad.l - g.pad.r, pH: g.cH - g.pad.t - g.pad.b };
  }

  function drawAxes(ctx, g) {
    const { x0, y0, pW, pH } = plotBox(g);
    const x1 = x0 + pW, y1 = y0 + pH;
    const xOf = x => x0 + (x / XMAX) * pW;
    const yOf = y => y0 + (1 - y / YMAX) * pH;

    ctx.strokeStyle = 'rgba(22,24,29,0.55)';
    ctx.lineWidth = 1;
    ctx.strokeRect(x0 + 0.5, y0 + 0.5, pW - 1, pH - 1);

    ctx.strokeStyle = 'rgba(22,24,29,0.55)';
    for (const x of [0, 5, 10]) {
      const px = Math.round(xOf(x)) + 0.5;
      ctx.beginPath(); ctx.moveTo(px, y1); ctx.lineTo(px, y1 + 3.5); ctx.stroke();
    }
    if (g.pad.yTicks) {
      ctx.font = '9px "JetBrains Mono", monospace';
      ctx.fillStyle = FAINT;
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      for (const y of [1, 3]) {
        const py = Math.round(yOf(y)) + 0.5;
        ctx.beginPath(); ctx.moveTo(x0, py); ctx.lineTo(x0 - 3.5, py); ctx.stroke();
        ctx.fillText(String(y), x0 - 5, py);
      }
    }

    ctx.fillStyle = MUTED;
    ctx.font = '11px Inter, system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'alphabetic';
    ctx.fillText('X', (x0 + x1) / 2, g.cH - 3);

    if (g.pad.yTitle) {
      ctx.save();
      ctx.translate(12, (y0 + y1) / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textBaseline = 'middle';
      ctx.font = g.pad.yTitle === 'Y'
        ? '11px Inter, system-ui, sans-serif'
        : 'italic 11px Inter, system-ui, sans-serif';
      ctx.fillText(g.pad.yTitle, 0, 0);
      ctx.restore();
    }
  }

  function drawPanel(i, off, ridges) {
    const ctx = ctxs[i], g = geom[i];
    if (!ctx || !g) return;
    const { x0, y0, pW, pH } = plotBox(g);
    const xToPx = x => x0 + (x / XMAX) * pW;
    const yToPx = y => y0 + (1 - y / YMAX) * pH;
    ctx.clearRect(0, 0, g.cW, g.cH);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(off, x0, y0, pW, pH);
    if (ridges) {
      const wMax = Math.max(...WGT);
      ctx.save();
      ctx.beginPath(); ctx.rect(x0, y0, pW, pH); ctx.clip();
      ctx.lineJoin = 'round'; ctx.lineCap = 'round';
      for (let k = 0; k < NF; k++) {
        const lw = 0.7 + 1.9 * WGT[k] / wMax;
        ctx.beginPath();
        for (let c = 0; c <= 100; c++) {
          const x = (c / 100) * XMAX;
          const px = xToPx(x), py = yToPx(gk(k, x));
          if (c === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        }
        ctx.strokeStyle = 'rgba(22,24,29,0.55)'; ctx.lineWidth = lw + 1.8; ctx.stroke();
        ctx.strokeStyle = RIDGE[k]; ctx.lineWidth = lw; ctx.stroke();
        const lx = 5.35, ly = gk(k, lx);
        ctx.font = '10px "JetBrains Mono", monospace';
        ctx.fillStyle = RIDGE[k];
        ctx.textAlign = 'left';
        ctx.textBaseline = 'bottom';
        ctx.fillText(`${RLAB[k]} ${Math.round(100 * WGT[k])}%`, xToPx(lx) + 5, yToPx(ly) - 3);
      }
      ctx.restore();
    }
    drawAxes(ctx, g);
  }

  function resize() {
    const panels = root.querySelector('.panels');
    const group = root.querySelector('.cgroup');
    const g0 = parseFloat(getComputedStyle(panels).gap);
    const g1 = group ? parseFloat(getComputedStyle(group).gap) : 0;
    const padX = PADS.reduce((s, p) => s + p.l + p.r, 0);
    const heatW = Math.min(HEAT_MAX, Math.max(72, Math.floor((panels.clientWidth - (g0 || 0) - (g1 || 0) - padX) / 3)));
    const heatH = Math.round(heatW * ASPECT);
    geom = PADS.map(pad => ({ pad, cW: heatW + pad.l + pad.r, cH: heatH + pad.t + pad.b }));
    ctxs = Array.from(canvases).map((cv, i) => setupCanvas(cv, geom[i].cW, geom[i].cH));
    rewardPanelDirty = true;
    root.querySelectorAll('.pcap').forEach((el, i) => {
      el.style.paddingLeft = (PADS[i].l - PADS[i].r) + 'px';
    });
    const hint = root.querySelector('.drag-hint');
    if (hint) {
      hint.style.left = (PADS[0].l + 6) + 'px';
      hint.style.bottom = (PADS[0].b + 6) + 'px';
    }
  }

  let wake = () => {};                     // assigned by animateWhileVisible below
  lamSlider.addEventListener('input', () => {
    state.lam = 0.02 * Math.pow(40, lamSlider.value / 100);
    lamLabel.textContent = `λ = ${state.lam.toFixed(2)}`;
    state.optP.reset();                   // drop leftover velocity; keep current policy
    state.stepsP = 0;
    resetConvergence(convP); state.runP = true; stepCarry = 0;
    wake();
  });
  nSlider.addEventListener('input', () => {
    state.n = parseInt(nSlider.value, 10);
    nLabel.textContent = `n = ${state.n}`;
    state.opt.reset();
    state.steps = 0;
    resetConvergence(convR); state.runR = true; stepCarry = 0;
    wake();
  });
  resetBtn.addEventListener('click', () => {
    for (let k = 0; k < NF; k++) OFF[k] = OFF0[k];   // restore default ridges too
    recomputeGrid(); paintReward();
    resetOpt();                            // both panels restart from uniform
    wake();
  });

  /* drag a ridge vertically on the reward panel to reshape that reward function */
  const rewardCv = canvases[0];
  let dragK = -1;
  const evData = e => {
    const g = geom[0];
    if (!g) return [NaN, NaN];
    const r = rewardCv.getBoundingClientRect();
    const { x0, y0, pW, pH } = plotBox(g);
    const scaleX = r.width / g.cW, scaleY = r.height / g.cH;
    const x = ((e.clientX - r.left) / scaleX - x0) / pW * XMAX;
    const y = (1 - ((e.clientY - r.top) / scaleY - y0) / pH) * YMAX;
    return [x, y];
  };
  function inPlot(x, y) {
    return x >= -0.15 && x <= XMAX + 0.15 && y >= -0.15 && y <= YMAX + 0.15;
  }
  function nearestRidge(x, y) {
    let best = -1, bd = 0.34;
    for (let k = 0; k < NF; k++) {
      const d = Math.abs(gk(k, x) - y);
      if (d < bd) { bd = d; best = k; }
    }
    return best;
  }
  let pendingRidge = null, ridgeMoveRaf = null;
  function applyRidgeMove() {
    ridgeMoveRaf = null;
    if (!pendingRidge) return;
    const { x, y, k } = pendingRidge;
    pendingRidge = null;
    if (!Number.isFinite(x) || !Number.isFinite(y)) return;
    const xc = Math.max(0, Math.min(XMAX, x));
    const denom = 1.05 * Math.max(env(xc), 0.25);      // ridges pinch together at the domain edges
    OFF[k] = Math.max(-1.4, Math.min(1.4, OFF[k] + (y - gk(k, xc)) / denom));
    recomputeGrid(); paintReward();
    state.opt.reset(); state.optP.reset();
    state.steps = 0; state.stepsP = 0;    // both re-optimize live from current policy
    resetConvergence(convR); resetConvergence(convP);
    state.runR = true; state.runP = true; stepCarry = 0;
    wake();
  }
  function moveRidge(e) {
    const [x, y] = evData(e);
    pendingRidge = { x, y, k: dragK };
    if (ridgeMoveRaf === null) ridgeMoveRaf = requestAnimationFrame(applyRidgeMove);
  }
  rewardCv.addEventListener('pointerdown', e => {
    const [x, y] = evData(e);
    if (!inPlot(x, y)) return;
    const k = nearestRidge(x, y);
    if (k >= 0) { dragK = k; rewardCv.setPointerCapture(e.pointerId); moveRidge(e); }
  });
  rewardCv.addEventListener('pointermove', e => {
    if (dragK >= 0) { moveRidge(e); return; }
    const [x, y] = evData(e);
    rewardCv.style.cursor = inPlot(x, y) && nearestRidge(x, y) >= 0 ? 'ns-resize' : 'default';
  });
  const endRidgeDrag = () => { dragK = -1; };
  rewardCv.addEventListener('pointerup', endRidgeDrag);
  rewardCv.addEventListener('pointercancel', endRidgeDrag);

  resize();
  watchWidth(root.querySelector('.panels'), () => { resize(); wake(); });
  paintReward(); paintPg(); paintRosa();
  const pgNeedsWork = () => {
    if (!state.runP || state.stepsP >= MAX_STEPS) return false;
    if (!state.runR) return !convP.converged;
    return !(convP.converged && (convR.converged || state.steps >= MAX_STEPS));
  };
  const rosaNeedsWork = () => {
    if (!state.runR || state.steps >= MAX_STEPS) return false;
    if (!state.runP) return !convR.converged;
    return !(convR.converged && (convP.converged || state.stepsP >= MAX_STEPS));
  };
  wake = animateWhileVisible(root, dt => {
    stepCarry = Math.min(MAX_PER_FRAME, stepCarry + dt * STEPS_PER_SECOND / 1000);
    const requested = Math.min(MAX_PER_FRAME, Math.floor(stepCarry));
    const frameStart = performance.now();
    let performed = 0, updatedP = false, updatedR = false;
    while (performed < requested && performance.now() - frameStart < FRAME_BUDGET_MS) {
      const doP = pgNeedsWork(), doR = rosaNeedsWork();
      if (!doP && !doR) break;
      if (doP) { stepPg(1); updateConvergence(state.thetaP, state.stepsP, convP); updatedP = true; }
      if (doR) { stepRosa(1); updateConvergence(state.theta, state.steps, convR); updatedR = true; }
      performed += 1;
    }
    stepCarry -= performed;
    if (updatedP) paintPg();
    if (updatedR) paintRosa();

    const runningP = pgNeedsWork(), runningR = rosaNeedsWork();
    if (!runningP) state.runP = false;
    if (!runningR) state.runR = false;
    if (rewardPanelDirty) {
      drawPanel(0, offReward, true);
      rewardPanelDirty = false;
    }
    drawPanel(1, offPg, false);
    drawPanel(2, offRosa, false);
    return runningP || runningR;           // idle once both panels have visually converged
  });
}

/* =========================================================================
 * 2. ADVANTAGE TRACE — distill-style computation diagram (DOM, not canvas)
 * ========================================================================= */
const advShuffle = a => a.map(v => [Math.random(), v]).sort((x, y) => x[0] - y[0]).map(x => x[1]);
const advFlatten = counts => {
  const a = [];
  counts.forEach((n, g) => { for (let i = 0; i < n; i++) a.push(g); });
  return a;
};

function initAdvWidget(rootId) {
  const root = document.getElementById(rootId);
  if (!root) return;
  const host = root.querySelector('.adv-panels');
  const scenarioEl = root.querySelector('.adv-scenario');
  const explainEl = root.querySelector('.adv-explain');

  const NR = 4;                                   // samples Y1..Y4
  const RHO = [0.5, 0.5];
  const COLC = [C2, C3];                          // reward-fn colors; magenta stays the credit color
  const SUB = ['₁', '₂', '₃', '₄'];
  const GLYPHS = ['🍎', '🍓', '🥦'];
  const parseBag = btn => btn.dataset.bag.split(',').map(Number);
  const rhoS = k => String(RHO[k]).replace(/^0(?=\.)/, '');

  /* One world (apple / strawberry / broccoli) and one draw of answers per bag click;
   * presets differ ONLY in the reward design, so switching compares the same samples
   * under the two designs directly. */
  const PRESETS = {
    multiple: {
      scenario: `<b style="color:${COLC[0]}">R₁</b>: +1 🍎, 0 other · ` +
        `<b style="color:${COLC[1]}">R₂</b>: +1 🍓, 0 other`,
      rewardOf: ans => [ans === 0 ? 1 : 0, ans === 1 ? 1 : 0],
    },
    competing: {
      scenario: `<b style="color:${COLC[0]}">R₁</b>: +1 🍎, −1 🍓, 0 other · ` +
        `<b style="color:${COLC[1]}">R₂</b>: +1 🍓, −1 🍎, 0 other`,
      rewardOf: ans => [
        ans === 0 ? 1 : ans === 1 ? -1 : 0,
        ans === 1 ? 1 : ans === 0 ? -1 : 0,
      ],
    },
  };

  let bagCounts = [2, 1, 1];
  let answers = [0, 0, 1, 2];                     // flatten([2,1,1]) — stable until a bag click
  let preset = 'multiple';
  let M = answers.map(PRESETS[preset].rewardOf);
  let sel = 2;
  let cycling = true;                             // auto-advance the hovered row until the reader interacts
  const glyphOf = i => GLYPHS[answers[i]];

  const fmt = v => {
    if (Math.abs(v) < 1e-9) return '0';
    const av = Math.abs(v);
    const s = Math.abs(av - Math.round(av)) < 1e-9
      ? String(Math.round(av))
      : '.' + String(Math.round(av * 100)).padStart(2, '0');
    return (v < 0 ? '−' : '') + s;
  };
  const fmtSigned = v => (Math.abs(v) < 1e-9 ? '0' : (v > 0 ? '+' : '−') + fmt(Math.abs(v)));

  function compute() {
    const rbar = M.map(row => row.reduce((a, v, k) => a + v * RHO[k], 0));
    const rlooA = rbar.map((r, i) => r - (rbar.reduce((a, b) => a + b, 0) - r) / (NR - 1));
    const fullMax = [0, 1].map(k => Math.max(...M.map(row => row[k])));
    const looMax = M.map((_, i) => [0, 1].map(k =>
      Math.max(...M.filter((_, j) => j !== i).map(row => row[k]))));
    const marg = M.map((_, i) => [0, 1].map(k => fullMax[k] - looMax[i][k]));
    const rosaA = marg.map(mi => mi.reduce((a, v, k) => a + v * RHO[k], 0));
    return { rbar, rlooA, fullMax, looMax, marg, rosaA };
  }

  const advClass = v => Math.abs(v) < 1e-9 ? 'zero' : (v > 0 ? 'pos' : 'neg');
  const rewardCls = v => (v > 0 ? '' : v < 0 ? 'neg' : 'zero');
  const rewardStyl = (v, k) => {
    const c = COLC[k];
    if (v === 0) return `background:#fff;border-color:${c}40;color:${c};--dot:${c}`;
    return `background:${c}${v > 0 ? '22' : '14'};border-color:${c}${v > 0 ? '55' : '40'};color:${c};--dot:${c}`;
  };
  const termStyl = (k, hot) => {
    const c = COLC[k];
    if (hot) return `background:${c}22;border-color:${c}55;color:${c};--term:${c}`;
    return `background:#fff;border-color:${c}40;color:${c};--term:${c}`;
  };
  const chip = (k, text, hot) => {
    const c = COLC[k];
    const bg = hot ? `${c}22` : '#fff';
    const bd = hot ? `${c}55` : `${c}40`;
    return `<span class="chip ${hot ? 'hot' : 'cold'}" style="color:${c};background:${bg};border-color:${bd};--term:${c}">${text}</span>`;
  };
  const chipRbar = text => `<span class="chip rbar-chip">${text}</span>`;
  const chipMean = text => `<span class="chip mean-chip">${text}</span>`;
  const chipAdv = (v, text = fmtSigned(v)) =>
    `<span class="chip adv-chip ${advClass(v)}">${text}</span>`;

  function cell(txt, cls, style) {
    return `<div class="ac ${cls || ''}" ${style ? `style="${style}"` : ''}>${txt}</div>`;
  }

  const sp = `<div class="sp"></div>`;

  function syncChrome() {
    const key = bagCounts.join(',');
    root.querySelectorAll('.bag-btn').forEach(b =>
      b.classList.toggle('active', b.dataset.bag === key));
    root.querySelectorAll('.preset-btn').forEach(b =>
      b.classList.toggle('active', b.dataset.preset === preset));
  }

  /* full rebuild — bag / preset; hover uses select() */
  function render() {
    const c = compute();
    const { rbar, rlooA, fullMax, rosaA } = c;
    scenarioEl.innerHTML = PRESETS[preset].scenario;
    const isColMax = (i, k) => M[i][k] === fullMax[k] && M.some(row => row[k] < fullMax[k]);
    const chead = (txt, k) => `<div class="chead"${k != null ? ` style="color:${COLC[k]}"` : ''}>${txt}</div>`;

    let h = `<div class="adv-grid">`;

    h += `<div class="trow">` +
      `<div class="ap-title" style="grid-column:span 4">samples</div>` + sp +
      `<div class="ap-title" style="grid-column:span 2">standard PG</div>` + sp +
      `<div class="ap-title rosa" style="grid-column:span 3">ROSA+Max</div>` +
      `</div>`;

    h += `<div class="trow">` +
      `<div></div><div></div>` +
      chead(`R${SUB[0]}<span class="rho">ρ=${fmt(RHO[0])}</span>`, 0) +
      chead(`R${SUB[1]}<span class="rho">ρ=${fmt(RHO[1])}</span>`, 1) + sp +
      chead('r̄ᵢ') + chead('Aᵢ') + sp +
      chead(`Δ${SUB[0]}`, 0) + chead(`Δ${SUB[1]}`, 1) + chead('Aᵢ') +
      `</div>`;

    for (let i = 0; i < NR; i++) {
      h += `<div class="trow body" data-row="${i}">` +
        `<div class="glab">Y${SUB[i]}</div><div class="gglyph">${glyphOf(i)}</div>`;
      for (let k = 0; k < 2; k++) {
        const cls = [`w${k}`, rewardCls(M[i][k]), isColMax(i, k) ? 'setmax' : ''].filter(Boolean).join(' ');
        h += cell(fmt(M[i][k]), cls, rewardStyl(M[i][k], k));
      }
      h += sp;
      h += cell(fmt(rbar[i]), 'rbar src') +
        cell(fmtSigned(rlooA[i]), 'adv-final pg-final ' + advClass(rlooA[i]));
      h += sp;
      for (let k = 0; k < 2; k++) {
        // result only, coloured when it credits the sample
        const pos = c.marg[i][k] > 0;
        h += cell(fmt(c.marg[i][k]), 'delta ' + advClass(c.marg[i][k]),
          `${termStyl(k, pos)};--pos:${pos ? 1 : 0}`);
      }
      h += cell(fmtSigned(rosaA[i]), 'adv-final rosa-final ' + advClass(rosaA[i]));
      h += `</div>`;
    }

    h += `</div>`;
    host.innerHTML = h;

    host.querySelectorAll('.trow.body').forEach(tr => {
      const i = parseInt(tr.dataset.row, 10);
      const pick = () => { cycling = false; select(i); };
      tr.addEventListener('pointerenter', pick);
      tr.addEventListener('click', pick);
    });
    select(sel);
  }

  function rosaExplain(c) {
    const { fullMax, looMax, marg, rosaA } = c;
    const loo = looMax[sel];
    const A = fmtSigned(rosaA[sel]);
    const symbolic =
      `Σ<sub>k</sub> ρ<sub>k</sub>(max<sub>1≤j≤n</sub> R<sub>k</sub>(Y<sub>j</sub>) − ` +
      `max<sub>j≠${sel + 1}</sub> R<sub>k</sub>(Y<sub>j</sub>))`;
    return `A${SUB[sel]} = ${symbolic} = ` +
      `${rhoS(0)}·${chip(0, `(${fmt(fullMax[0])} − ${fmt(loo[0])})`, marg[sel][0] > 0)} + ` +
      `${rhoS(1)}·${chip(1, `(${fmt(fullMax[1])} − ${fmt(loo[1])})`, marg[sel][1] > 0)} = ` +
      chipAdv(rosaA[sel], A);
  }

  /* selection-only update: toggle classes + refresh the worked-example line */
  function select(i) {
    sel = ((i % NR) + NR) % NR;
    const c = compute();
    const { rbar, rlooA } = c;
    if (!host.querySelector('.trow.body')) return;

    host.querySelectorAll('.trow.body').forEach(tr => {
      const on = parseInt(tr.dataset.row, 10) === sel;
      tr.classList.toggle('sel', on);
      const rb = tr.querySelector('.ac.rbar');
      if (rb) { rb.classList.toggle('selsrc', on); rb.classList.toggle('src', !on); }
      // ring only the result cells that credit the hovered sample
      tr.querySelectorAll('.ac.delta').forEach(d =>
        d.classList.toggle('credit', on && d.style.getPropertyValue('--pos').trim() === '1'));
    });

    const rest = rbar.filter((_, j) => j !== sel).map(fmt).join(', ');
    const b = (rbar.reduce((a, v) => a + v, 0) - rbar[sel]) / (NR - 1);
    const pgLine = `A${SUB[sel]} = r̄${SUB[sel]} − mean(${rest}) = ` +
      `${chipRbar(fmt(rbar[sel]))} − ${chipMean(fmt(b))} = ` +
      chipAdv(rlooA[sel]);
    explainEl.innerHTML =
      `<span class="ex"><b class="tag">PG</b>${pgLine}</span>` +
      `<span class="ex"><b class="tag rosa">ROSA</b>${rosaExplain(c)}</span>`;
  }

  root.querySelectorAll('.preset-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      preset = btn.dataset.preset;
      cycling = false;
      M = answers.map(PRESETS[preset].rewardOf);
      syncChrome();
      render();
    });
  });
  root.querySelectorAll('.bag-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      bagCounts = parseBag(btn);
      answers = advShuffle(advFlatten(bagCounts));
      cycling = false;
      M = answers.map(PRESETS[preset].rewardOf);
      syncChrome();
      render();
    });
  });
  setInterval(() => { if (cycling) select(sel + 1); }, 2400);
  render();
}

/* =========================================================================
 * 3. FLOW — exact-gradient trajectories on the simplex, GD + heavy-ball
 *    momentum, from starts sampled near the broccoli vertex; both objectives
 *    side by side; replay resamples the starts.
 * ========================================================================= */
function sampleNearC(rnd, n = 24) {
  const starts = [];
  for (let i = 0; i < n; i++) {
    const pc = 0.55 + 0.37 * rnd();
    /* A's share of non-C mass: two lobes, no midline (no ~50/50 A–B) */
    const t = rnd() < 0.5
      ? 0.05 + 0.40 * rnd()    // [0.05, 0.45]
      : 0.55 + 0.40 * rnd();   // [0.55, 0.95]
    starts.push([(1 - pc) * t, (1 - pc) * (1 - t), pc]);
  }
  return starts;
}

function initFlowGdmWidget(id = 'flow-sgdm-tr-widget') {
  const wrap = document.getElementById(id);
  if (!wrap) return;
  /* lr = 0.05, μ = 0.8 → asymptotic step lr/(1-μ) = 0.25 per unit gradient */
  let seed = 20260826;
  initFlowVariant(wrap, {
    sampleStarts: () => sampleNearC(mulberry32(seed++)),
    momentum: true,
    lr: 0.05, mu: 0.8,
  });
}

function initFlowVariant(root, opts) {
  if (!root) return;
  const {
    starts: fixedStarts,
    sampleStarts,
    momentum = false,
    lr = 1.5,
    mu = 0.9,
  } = opts;
  const N = 4;
  const TOTAL = 1200, PER_FRAME = 8;
  const rosa1 = lut1(q => 0.5 * (1 - Math.pow(1 - q, N)));   // per-axis ROSA+Max term, tabulated for the landscape
  const panels = [
    {
      name: 'standard PG', optima: 'edge', color: INK, fill: MUTED,
      J: p => 0.5 * p[0] + 0.5 * p[1],
      w: () => [0.5, 0.5, 0],
      canvas: root.querySelectorAll('canvas')[0],
    },
    {
      name: 'ROSA+Max', optima: 'mid', color: C1,
      J: p => rosa1(p[0]) + rosa1(p[1]),
      w: p => [0.5 * N * Math.pow(1 - p[0], N - 1), 0.5 * N * Math.pow(1 - p[1], N - 1), 0],
      canvas: root.querySelectorAll('canvas')[1],
    },
  ];
  /* optional third column: reward / entropy traces for every run of both panels */
  const traceCanvas = root.querySelector('.trace-canvas');
  const traces = traceCanvas ? makeTraces(traceCanvas, {
    series: [{ name: 'PG', color: INK, alpha: 0.35 }, { name: 'ROSA+Max', color: C1, alpha: 0.45 }],
    total: TOTAL,
  }) : null;

  let W, H, V;
  const toXY = p => [
    p[0] * V.A[0] + p[1] * V.B[0] + p[2] * V.C[0],
    p[0] * V.A[1] + p[1] * V.B[1] + p[2] * V.C[1],
  ];

  let starts = fixedStarts || [];
  let steps = 0;
  function resetRuns(resample = false) {
    steps = 0;
    if (sampleStarts && (resample || !starts.length)) starts = sampleStarts();
    if (traces) traces.reset(starts.length);
    panels.forEach((p, pi) => {
      p.runs = starts.map((p0, ri) => {
        if (traces) traces.push(pi, ri, 0, p0);
        return {
          theta: new Float64Array(p0.map(Math.log)),
          cur: p0.slice(),
          opt: momentum ? makeSgdm(3, lr, mu) : null,
        };
      });
      for (const layer of [p.halo, p.core]) {
        const c = layer.getContext('2d');
        c.setTransform(1, 0, 0, 1, 0, 0);
        c.clearRect(0, 0, layer.width, layer.height);
        const dpr = window.devicePixelRatio || 1;
        c.scale(dpr, dpr);
        c.lineJoin = 'round'; c.lineCap = 'round';
      }
    });
  }

  /* exact gradient ascent; trails drawn incrementally onto halo/core layers */
  function stepAll(iters) {
    const g = new Float64Array(3);
    for (let it = 0; it < iters; it++) {
      const a = 0.2 + 0.75 * (steps / TOTAL);
      panels.forEach((p, pi) => {
        const hc = p.halo.getContext('2d'), cc = p.core.getContext('2d');
        hc.strokeStyle = `rgba(22,24,29,${(0.5 * a).toFixed(3)})`; hc.lineWidth = 3.2;
        cc.strokeStyle = `rgba(255,255,255,${a.toFixed(3)})`; cc.lineWidth = 1.4;
        p.runs.forEach((run, ri) => {
          const pr = run.cur, w = p.w(pr);
          const mean = pr[0] * w[0] + pr[1] * w[1] + pr[2] * w[2];
          for (let k = 0; k < 3; k++) g[k] = pr[k] * (w[k] - mean);
          if (run.opt) run.opt.step(run.theta, g);
          else for (let k = 0; k < 3; k++) run.theta[k] += lr * g[k];
          const nxt = softmax(Array.from(run.theta));
          const [x0, y0] = toXY(pr), [x1, y1] = toXY(nxt);
          hc.beginPath(); hc.moveTo(x0, y0); hc.lineTo(x1, y1); hc.stroke();
          cc.beginPath(); cc.moveTo(x0, y0); cc.lineTo(x1, y1); cc.stroke();
          run.cur = nxt;
          if (traces) traces.push(pi, ri, steps + 1, nxt);
        });
      });
      steps += 1;
    }
  }

  function drawPanel(p) {
    const ctx = p.ctx;
    ctx.clearRect(0, 0, W, H);
    ctx.drawImage(p.bg, 0, 0, W, H);
    drawSimplexFrame(ctx, V);
    if (p.optima === 'edge') {
      ctx.setLineDash([2, 5]); ctx.strokeStyle = '#22d3ee'; ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.moveTo(...toXY([1, 0, 0])); ctx.lineTo(...toXY([0, 1, 0])); ctx.stroke();
      ctx.setLineDash([]);
    } else {
      const [sx, sy] = toXY([0.5, 0.5, 0]);
      star(ctx, sx, sy - 2, 8, '#22d3ee', INK);
    }
    ctx.drawImage(p.halo, 0, 0, W, H);
    ctx.drawImage(p.core, 0, 0, W, H);
    for (const p0 of starts) {                       // start markers
      const [x, y] = toXY(p0);
      ctx.beginPath(); ctx.arc(x, y, 2.6, 0, 2 * Math.PI);
      ctx.fillStyle = 'rgba(255,255,255,0.85)'; ctx.fill();
      ctx.strokeStyle = INK; ctx.lineWidth = 1; ctx.stroke();
    }
    for (const run of p.runs) {                      // current points
      const [x, y] = toXY(run.cur);
      ctx.beginPath(); ctx.arc(x, y, 3.4, 0, 2 * Math.PI);
      ctx.fillStyle = p.fill || p.color; ctx.fill();
      ctx.strokeStyle = INK; ctx.lineWidth = 1.3; ctx.stroke();
    }
    ctx.font = MONO; ctx.fillStyle = p.color; ctx.textAlign = 'center';
    ctx.fillText(p.name, W / 2, H - 8);
  }

  function resize() {
    const panelsEl = root.querySelector('.panels');
    const cw = panelsEl.clientWidth;
    const gap = parseFloat(getComputedStyle(panelsEl).gap) || 14.4;
    if (traces && cw >= 540) {
      /* three columns: two simplexes + trace column taking the remainder */
      W = Math.floor((cw - 2 * gap) * 0.35);
      H = Math.round(W * 0.92);
      traces.resize(cw - 2 * W - 2 * gap - 1, H);
    } else {
      W = Math.max(230, Math.floor(cw / 2) - 10);
      H = Math.round(W * 0.92);
      if (traces) traces.resize(cw, 220);        // narrow: traces wrap below the simplexes
    }
    const pad = 38;
    V = { A: [pad, H - pad], B: [W - pad, H - pad], C: [W / 2, pad * 1.15] };
    const dpr = window.devicePixelRatio || 1;
    for (const p of panels) {
      p.ctx = setupCanvas(p.canvas, W, H);
      p.bg = renderSimplexBg(p.J, W, H, V, 'auto');
      p.halo = document.createElement('canvas');
      p.core = document.createElement('canvas');
      for (const layer of [p.halo, p.core]) {
        layer.width = Math.round(W * dpr); layer.height = Math.round(H * dpr);
      }
    }
    resetRuns();
  }

  const stepsEl = root.querySelector('.step-count');
  let wake = () => {};
  root.querySelector('.replay-btn').addEventListener('click', () => { resetRuns(true); wake(); });
  resize();
  watchWidth(root.querySelector('.panels'), () => { resize(); wake(); });
  const method = momentum ? 'GD + momentum' : 'exact gradient flow';
  wake = animateWhileVisible(root, () => {
    if (steps < TOTAL) stepAll(PER_FRAME);
    for (const p of panels) drawPanel(p);
    if (traces) traces.draw(steps);
    stepsEl.textContent = `${method} · step ${Math.min(steps, TOTAL)}/${TOTAL}`;
    return steps < TOTAL;                  // idle once every run has converged
  });
}

/* =========================================================================
 * 4. CONTROLLABILITY — exact landscapes + optima as rho shifts
 *    ROSA optimum from the paper's closed form (prop:non-uniform, m=2):
 *    p_i* = 1 - alpha_i^{-1/(n-1)} / sum_j alpha_j^{-1/(n-1)},  p_C* = 0.
 * ========================================================================= */
function initCtrlWidget() {
  const root = document.getElementById('ctrl-widget');
  if (!root) return;
  const canvases = root.querySelectorAll('canvas');
  if (canvases.length < 2) return;
  const rhoSlider = root.querySelector('.rho-slider');
  const rhoLabelA = root.querySelector('.rho-label-a');
  const rhoLabelB = root.querySelector('.rho-label-b');
  const nSlider = root.querySelector('.n-slider');
  const nLabel = root.querySelector('.n-label');
  const optEl = root.querySelector('.opt-readout');

  const state = { aB: 0.4, n: 2, dirty: true };

  let W, H, V;
  const toXY = p => [
    p[0] * V.A[0] + p[1] * V.B[0] + p[2] * V.C[0],
    p[0] * V.A[1] + p[1] * V.B[1] + p[2] * V.C[1],
  ];

  function pStar() {
    const n = state.n, aA = 1 - state.aB, aB = state.aB;
    if (aB <= 0) return [1, 0, 0];                  // one reward fn left: k = 1, deterministic
    if (aA <= 0) return [0, 1, 0];
    const eA = Math.pow(aA, -1 / (n - 1)), eB = Math.pow(aB, -1 / (n - 1));
    const den = eA + eB;
    return [1 - eA / den, 1 - eB / den, 0];
  }

  const panels = [
    { kind: 'pg', name: 'standard PG', canvas: canvases[0] },
    { kind: 'rosa', name: 'ROSA+Max', canvas: canvases[1] },
  ];
  const Jof = kind => {                             // landscape objective; rebuilt on every rho / n change
    const aA = 1 - state.aB, aB = state.aB, n = state.n;
    if (kind === 'pg') return p => aA * p[0] + aB * p[1];
    const fA = lut1(q => aA * (1 - Math.pow(1 - q, n)));
    const fB = lut1(q => aB * (1 - Math.pow(1 - q, n)));
    return p => fA(p[0]) + fB(p[1]);
  };
  const wOf = kind => {                             // dJ/dp per action
    const aA = 1 - state.aB, aB = state.aB, n = state.n;
    return kind === 'pg'
      ? () => [aA, aB, 0]
      : p => [aA * n * Math.pow(1 - p[0], n - 1), aB * n * Math.pow(1 - p[1], n - 1), 0];
  };

  /* ten fixed starts, upper simplex (same on both panels).
   * barycentric [🍎, 🍓, 🥦]; p_🥦 ≥ 0.60; left/right pairs, off the A–B midline.
   * GD + heavy-ball, lr = 0.15, μ = 0.5, 1200 steps (a faster schedule than
   * #flow-sgdm-tr-widget so the runs settle quickly after each slider move). */
  const STARTS = [
    [0.14, 0.06, 0.80],
    [0.06, 0.14, 0.80],
    [0.22, 0.08, 0.70],
    [0.08, 0.22, 0.70],
    [0.18, 0.10, 0.72],
    [0.10, 0.18, 0.72],
    [0.26, 0.12, 0.62],
    [0.12, 0.26, 0.62],
    [0.24, 0.16, 0.60],
    [0.15, 0.23, 0.62],
  ];
  const run = { steps: 0 };
  const RTOT = 1200, RLR = 0.15, RMU = 0.5, RPF = 8;
  const g = new Float64Array(3);
  function resetRuns() {
    run.steps = 0;
    const dpr = window.devicePixelRatio || 1;
    for (const p of panels) {
      p.runs = STARTS.map(p0 => ({
        theta: new Float64Array(p0.map(Math.log)),
        cur: p0.slice(),
        opt: makeSgdm(3, RLR, RMU),
      }));
      for (const layer of [p.halo, p.core]) {
        const c = layer.getContext('2d');
        c.setTransform(1, 0, 0, 1, 0, 0);
        c.clearRect(0, 0, layer.width, layer.height);
        c.scale(dpr, dpr);
        c.lineJoin = 'round'; c.lineCap = 'round';
      }
    }
  }

  function stepRuns(iters) {
    for (let it = 0; it < iters; it++) {
      const a = 0.2 + 0.75 * (run.steps / RTOT);
      for (const p of panels) {
        const hc = p.halo.getContext('2d'), cc = p.core.getContext('2d');
        hc.strokeStyle = `rgba(22,24,29,${(0.5 * a).toFixed(3)})`; hc.lineWidth = 3.2;
        cc.strokeStyle = `rgba(255,255,255,${a.toFixed(3)})`; cc.lineWidth = 1.4;
        for (const r of p.runs) {
          const w = wOf(p.kind)(r.cur);
          const mean = r.cur[0] * w[0] + r.cur[1] * w[1] + r.cur[2] * w[2];
          for (let k = 0; k < 3; k++) g[k] = r.cur[k] * (w[k] - mean);
          r.opt.step(r.theta, g);
          const nxt = softmax(Array.from(r.theta));
          const [x0, y0] = toXY(r.cur), [x1, y1] = toXY(nxt);
          hc.beginPath(); hc.moveTo(x0, y0); hc.lineTo(x1, y1); hc.stroke();
          cc.beginPath(); cc.moveTo(x0, y0); cc.lineTo(x1, y1); cc.stroke();
          r.cur = nxt;
        }
      }
      run.steps += 1;
    }
  }

  function drawPanel(p) {
    const ctx = p.ctx;
    ctx.clearRect(0, 0, W, H);
    ctx.drawImage(p.bg, 0, 0, W, H);
    drawSimplexFrame(ctx, V);
    const aA = 1 - state.aB;
    if (p.kind === 'pg') {
      if (Math.abs(aA - state.aB) < 0.011) {        // tie: the whole A-B edge is optimal
        ctx.setLineDash([2, 5]); ctx.strokeStyle = '#22d3ee'; ctx.lineWidth = 2.5;
        ctx.beginPath(); ctx.moveTo(...toXY([1, 0, 0])); ctx.lineTo(...toXY([0, 1, 0])); ctx.stroke();
        ctx.setLineDash([]);
      } else {
        const [sx, sy] = toXY(state.aB > aA ? [0, 1, 0] : [1, 0, 0]);
        star(ctx, sx, sy - 2, 8, '#22d3ee', INK);
      }
    } else {
      const [sx, sy] = toXY(pStar());
      star(ctx, sx, sy - 2, 8, '#22d3ee', INK);
    }
    ctx.drawImage(p.halo, 0, 0, W, H);
    ctx.drawImage(p.core, 0, 0, W, H);
    for (const p0 of STARTS) {
      const [x, y] = toXY(p0);
      ctx.beginPath(); ctx.arc(x, y, 2.6, 0, 2 * Math.PI);
      ctx.fillStyle = 'rgba(255,255,255,0.85)'; ctx.fill();
      ctx.strokeStyle = INK; ctx.lineWidth = 1; ctx.stroke();
    }
    for (const r of p.runs) {
      const [x, y] = toXY(r.cur);
      ctx.beginPath(); ctx.arc(x, y, 3.4, 0, 2 * Math.PI);
      ctx.fillStyle = '#fff'; ctx.fill();
      ctx.strokeStyle = INK; ctx.lineWidth = 1.3; ctx.stroke();
    }
    ctx.font = MONO; ctx.fillStyle = MUTED; ctx.textAlign = 'center';
    ctx.fillText(p.name, W / 2, H - 8);
  }

  function resize() {
    const cw = root.querySelector('.panels').clientWidth;
    W = Math.max(230, Math.floor(cw / 2) - 10);
    H = Math.round(W * 0.92);
    const pad = 38;
    V = { A: [pad, H - pad], B: [W - pad, H - pad], C: [W / 2, pad * 1.15] };
    const dpr = window.devicePixelRatio || 1;
    for (const p of panels) {
      p.ctx = setupCanvas(p.canvas, W, H);
      p.halo = document.createElement('canvas');
      p.core = document.createElement('canvas');
      for (const layer of [p.halo, p.core]) {
        layer.width = Math.round(W * dpr); layer.height = Math.round(H * dpr);
      }
    }
    state.dirty = true;
  }

  const refresh = () => {
    for (const p of panels) p.bg = renderSimplexBg(Jof(p.kind), W, H, V, [0, 1]);
    const ps = pStar();
    optEl.textContent = `ROSA optimum π* : 🍎 ${(100 * ps[0]).toFixed(0)}% · 🍓 ${(100 * ps[1]).toFixed(0)}% · 🥦 0%`;
    resetRuns();                                    // landscape changed: re-run from the same starts
  };

  let wake = () => {};
  rhoSlider.addEventListener('input', () => {
    state.aB = parseInt(rhoSlider.value, 10) / 100;
    rhoLabelA.textContent = `ρ(R₁)=${Math.round(100 - 100 * state.aB)}%`;
    rhoLabelB.textContent = `ρ(R₂)=${Math.round(100 * state.aB)}%`;
    state.dirty = true;
    wake();
  });
  nSlider.addEventListener('input', () => {
    state.n = parseInt(nSlider.value, 10);
    nLabel.textContent = `n = ${state.n}`;
    state.dirty = true;
    wake();
  });

  resize();
  watchWidth(root.querySelector('.panels'), () => { resize(); wake(); });
  wake = animateWhileVisible(root, () => {
    if (state.dirty) { refresh(); state.dirty = false; }
    if (run.steps < RTOT) stepRuns(RPF);
    for (const p of panels) drawPanel(p);
    return run.steps < RTOT;               // idle once the runs have converged
  });
}

/* =========================================================================
 * 5. SET FUNCTION BUILDER — success-count function f~ and its induced landscape
 *    Binary rewards: any set function reduces to f~(s), s = # successes among n.
 *    J(p) = 0.5 E[f~(Binom(n, pA))] + 0.5 E[f~(Binom(n, pB))], exact.
 * ========================================================================= */
function initSetFnWidget(id = 'setfn-tr-widget') {
  const root = document.getElementById(id);
  if (!root) return;
  const topHalf = id === 'setfn-tr-widget';   // sample starts on the top half of the simplex (p2 ≥ 0.5)
  const fCanvas = root.querySelector('.f-canvas');
  const sCanvas = root.querySelector('.s-canvas');
  const badgeEl = root.querySelector('.sf-badge');
  const stepsEl = root.querySelector('.step-count');

  const N = 4;
  const NSTART = 10;
  const TOTAL = 1200, PER_FRAME = 8, LR = 0.05, MU = 0.8;
  /* optional third column: reward / entropy traces for every run */
  const traceCanvas = root.querySelector('.trace-canvas');
  const traces = traceCanvas ? makeTraces(traceCanvas, {
    series: [{ name: 'ROSA+f̃', color: C1, alpha: 0.5 }],
    total: TOTAL,
  }) : null;
  const BINOM = [1, 4, 6, 4, 1];
  const BINOM1 = [1, 3, 3, 1];                 // C(n-1, s)
  const E = Math.E;
  const PRESET_F = {
    max: [0, 1, 1, 1, 1],
    softmax: [0, 1 * E / (1 * E + 3), 2 * E / (2 * E + 2), 3 * E / (3 * E + 1), 1],
    mean: [0, 0.25, 0.5, 0.75, 1],
    convex: [0, 0.0625, 0.25, 0.5625, 1],           // (k/n)^2
    // arcsin: [0, 1 / 3, 1 / 2, 2 / 3, 1],            // 2·arcsin√(k/n), normalized to [0,1] (GRPO)
    //log: [0, 1, 2, 3, 4].map(k => Math.log(1 + k) / Math.log(1 + N)), // log(1+k) normalized to [0,1]
  };
  const state = { f: PRESET_F.softmax.slice(), drag: -1, dirty: true };

  /* ---- geometry ---- */
  let FW, FH, SW, SH, V, fctx, sctx, bg, halo, core, jMin = 0, jMax = 1;
  const FP = { l: 30, r: 12, t: 16, b: 26 };
  const fx = s => FP.l + (s / N) * (FW - FP.l - FP.r);
  const fy = v => FH - FP.b - v * (FH - FP.t - FP.b);
  const toXY = p => [
    p[0] * V.A[0] + p[1] * V.B[0] + p[2] * V.C[0],
    p[0] * V.A[1] + p[1] * V.B[1] + p[2] * V.C[1],
  ];

  function expF(p) {                    // E[f~(Binom(N, p))]
    let e = 0;
    for (let s = 0; s <= N; s++) {
      e += BINOM[s] * Math.pow(p, s) * Math.pow(1 - p, N - s) * state.f[s];
    }
    return e;
  }
  /* d/dq E[f~(Binom(n,q))] = n E[Δf~(Binom(n-1,q))] */
  function dExpF(q) {
    let e = 0;
    for (let s = 0; s < N; s++) {
      e += BINOM1[s] * Math.pow(q, s) * Math.pow(1 - q, N - 1 - s) * (state.f[s + 1] - state.f[s]);
    }
    return N * e;
  }

  /* Dirichlet(1,1,1) = uniform on the simplex, mixed toward the barycenter
   * so log-logits stay finite. Retry until the set covers both sides of the
   * A–B median and some broccoli mass — otherwise convex/mean basins vanish. */
  function sampleSpread(rnd) {
    const mix = 0.08;
    const draw = () => {
      const starts = [];
      for (let i = 0; i < NSTART; i++) {
        const g = [0, 1, 2].map(() => -Math.log(1e-9 + rnd()));
        const s = g[0] + g[1] + g[2];
        const u = [g[0] / s, g[1] / s, g[2] / s];   // uniform on the full simplex
        if (topHalf) {
          /* fold into the top-half sub-triangle (p2 ≥ 0.5); mix toward its
           * centroid (1/6, 1/6, 2/3) so log-logits stay finite */
          const p = [0.5 * u[1], 0.5 * u[2], 0.5 + 0.5 * u[0]];
          starts.push(p.map((v, k) => (1 - mix) * v + mix * (k === 2 ? 2 / 3 : 1 / 6)));
        } else {
          starts.push(u.map(v => (1 - mix) * v + mix / 3));
        }
      }
      return starts;
    };
    for (let t = 0; t < 24; t++) {
      const starts = draw();
      const left = starts.some(p => p[0] > p[1] + 0.12);
      const right = starts.some(p => p[1] > p[0] + 0.12);
      const top = starts.some(p => p[2] > 0.35);
      if (left && right && top) return starts;
    }
    return draw();
  }

  function drawF() {
    const ctx = fctx;
    ctx.clearRect(0, 0, FW, FH);
    ctx.strokeStyle = HAIR; ctx.lineWidth = 1;
    for (const v of [0, 0.5, 1]) {
      ctx.beginPath(); ctx.moveTo(FP.l, fy(v) + 0.5); ctx.lineTo(FW - FP.r, fy(v) + 0.5); ctx.stroke();
    }
    ctx.font = '10px "JetBrains Mono", monospace'; ctx.fillStyle = FAINT;
    ctx.textAlign = 'right';
    ctx.fillText('1', FP.l - 5, fy(1) + 3);
    ctx.fillText('0', FP.l - 5, fy(0) + 3);
    ctx.textAlign = 'center';
    for (let s = 0; s <= N; s++) ctx.fillText(String(s), fx(s), FH - 15);
    ctx.fillStyle = MUTED; ctx.textAlign = 'left';
    ctx.fillText('f̃(s) · drag the dots', FP.l, 10);
    ctx.fillStyle = FAINT; ctx.textAlign = 'center';
    ctx.fillText('s = successes among n = 4', (FP.l + FW - FP.r) / 2, FH - 3);
    // polyline
    ctx.strokeStyle = INK; ctx.lineWidth = 1.8; ctx.lineJoin = 'round';
    ctx.beginPath();
    for (let s = 0; s <= N; s++) {
      if (s === 0) ctx.moveTo(fx(s), fy(state.f[s])); else ctx.lineTo(fx(s), fy(state.f[s]));
    }
    ctx.stroke();
    // dots
    for (let s = 0; s <= N; s++) {
      ctx.beginPath(); ctx.arc(fx(s), fy(state.f[s]), s === 0 ? 4 : 5.5, 0, 2 * Math.PI);
      ctx.fillStyle = s === 0 ? FAINT : C1; ctx.fill();
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.stroke();
    }
  }

  function paintBg() {
    const dpr = window.devicePixelRatio || 1;
    const pw = Math.round(SW * dpr), ph = Math.round(SH * dpr);
    bg = document.createElement('canvas');
    bg.width = pw; bg.height = ph;
    const octx = bg.getContext('2d');
    const img = octx.createImageData(pw, ph);
    const [ax, ay] = V.A, [bx, by] = V.B, [cx, cy] = V.C;
    const det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy);
    const vals = new Float64Array(pw * ph).fill(NaN);
    const ef = lut1(expF);                // tabulate E[f~] once per repaint instead of per pixel
    jMin = Infinity; jMax = -Infinity;
    for (let py = 0; py < ph; py++) {
      for (let px = 0; px < pw; px++) {
        const x = px / dpr, y = py / dpr;
        const l1 = ((by - cy) * (x - cx) + (cx - bx) * (y - cy)) / det;
        const l2 = ((cy - ay) * (x - cx) + (ax - cx) * (y - cy)) / det;
        const l3 = 1 - l1 - l2;
        if (l1 < 0 || l2 < 0 || l3 < 0) continue;
        const v = 0.5 * ef(l1) + 0.5 * ef(l2);
        vals[py * pw + px] = v;
        if (v < jMin) jMin = v;
        if (v > jMax) jMax = v;
      }
    }
    // colors renormalized to this f~'s own range, so a compressed objective
    // (e.g. softmax, max J ~ .68) still uses the full colormap
    const span = jMax - jMin + 1e-12;
    for (let i = 0; i < vals.length; i++) {
      const v = vals[i];
      if (Number.isNaN(v)) continue;
      let [r, g, b] = plasma((v - jMin) / span);
      if (v >= jMax - 0.004) { r = 0.35 * r + 0.65 * 34; g = 0.35 * g + 0.65 * 211; b = 0.35 * b + 0.65 * 238; }
      img.data[4 * i] = r; img.data[4 * i + 1] = g; img.data[4 * i + 2] = b; img.data[4 * i + 3] = 255;
    }
    octx.putImageData(img, 0, 0);
  }

  function drawS() {
    const ctx = sctx;
    ctx.clearRect(0, 0, SW, SH);
    if (bg) ctx.drawImage(bg, 0, 0, SW, SH);
    drawSimplexFrame(ctx, V);
    const dragging = state.drag > 0;
    if (!dragging) {
      ctx.drawImage(halo, 0, 0, SW, SH);
      ctx.drawImage(core, 0, 0, SW, SH);
    }
    for (const p0 of starts) {
      const [x, y] = toXY(p0);
      ctx.beginPath(); ctx.arc(x, y, 2.6, 0, 2 * Math.PI);
      ctx.fillStyle = 'rgba(255,255,255,0.85)'; ctx.fill();
      ctx.strokeStyle = INK; ctx.lineWidth = 1; ctx.stroke();
    }
    if (!dragging) {
      for (const run of runs) {
        const [x, y] = toXY(run.cur);
        ctx.beginPath(); ctx.arc(x, y, 3.4, 0, 2 * Math.PI);
        ctx.fillStyle = '#fff'; ctx.fill();
        ctx.strokeStyle = INK; ctx.lineWidth = 1.3; ctx.stroke();
      }
    }
  }

  function updateBadge() {
    const d = [], d2 = [];
    for (let s = 0; s < N; s++) d.push(state.f[s + 1] - state.f[s]);
    for (let s = 1; s < N; s++) d2.push(d[s] - d[s - 1]);
    const tri = (arr, strictCmp, weakCmp) =>
      arr.every(strictCmp) ? 'strictly' : arr.every(weakCmp) ? 'weakly' : 'not';
    const inc = tri(d, x => x > 1e-3, x => x > -1e-9);
    const cav = tri(d2, x => x < -1e-3, x => x < 1e-9);
    badgeEl.innerHTML =
      `f̃ increasing: <b class="${inc === 'not' ? 'warn' : 'okk'}">${inc}</b> · ` +
      `concave: <b class="${cav === 'not' ? 'warn' : 'okk'}">${cav}</b>`;
  }

  /* ---- live SGD + heavy-ball on exact ∇J (same as the flow-sgdm widget) ---- */
  let seed = 20260831, starts = [], runs = [], steps = 0;
  const g = new Float64Array(3);
  function resetRuns() {
    steps = 0;
    const dpr = window.devicePixelRatio || 1;
    if (traces) traces.reset(starts.length);
    runs = starts.map((p0, ri) => {
      if (traces) traces.push(0, ri, 0, p0);
      return {
        theta: new Float64Array(p0.map(Math.log)),
        cur: p0.slice(),
        opt: makeSgdm(3, LR, MU),
      };
    });
    for (const layer of [halo, core]) {
      if (!layer) continue;
      const c = layer.getContext('2d');
      c.setTransform(1, 0, 0, 1, 0, 0);
      c.clearRect(0, 0, layer.width, layer.height);
      c.scale(dpr, dpr);
      c.lineJoin = 'round'; c.lineCap = 'round';
    }
  }
  function resample() {
    starts = sampleSpread(mulberry32(seed++));
    resetRuns();
  }
  function stepRuns(iters) {
    for (let it = 0; it < iters; it++) {
      const a = 0.2 + 0.75 * (steps / TOTAL);
      const hc = halo.getContext('2d'), cc = core.getContext('2d');
      hc.strokeStyle = `rgba(22,24,29,${(0.5 * a).toFixed(3)})`; hc.lineWidth = 3.2;
      cc.strokeStyle = `rgba(255,255,255,${a.toFixed(3)})`; cc.lineWidth = 1.4;
      runs.forEach((run, ri) => {
        const pr = run.cur;
        const w0 = 0.5 * dExpF(pr[0]), w1 = 0.5 * dExpF(pr[1]);
        const mean = pr[0] * w0 + pr[1] * w1;
        g[0] = pr[0] * (w0 - mean);
        g[1] = pr[1] * (w1 - mean);
        g[2] = pr[2] * (0 - mean);
        run.opt.step(run.theta, g);
        const nxt = softmax(Array.from(run.theta));
        const [x0, y0] = toXY(pr), [x1, y1] = toXY(nxt);
        hc.beginPath(); hc.moveTo(x0, y0); hc.lineTo(x1, y1); hc.stroke();
        cc.beginPath(); cc.moveTo(x0, y0); cc.lineTo(x1, y1); cc.stroke();
        run.cur = nxt;
        if (traces) traces.push(0, ri, steps + 1, nxt);
      });
      steps += 1;
    }
  }

  function resize() {
    const panelsEl = root.querySelector('.panels');
    const cw = panelsEl.clientWidth;
    if (traces && cw >= 540) {
      /* three columns: simplex a bit wider than the f̃ / trace plots; shared height */
      const gap = parseFloat(getComputedStyle(panelsEl).gap) || 14.4;
      const inner = cw - 2 * gap;
      FW = Math.floor(inner * 0.30);
      SW = Math.floor(inner * 0.40);
      SH = Math.round(SW * 0.92);
      FH = SH;
      traces.resize(inner - FW - SW - 1, SH);
    } else {
      const half = Math.floor(cw / 2) - 10;
      SW = Math.max(220, Math.min(290, half));
      SH = Math.round(SW * 0.92);
      FW = Math.max(200, Math.min(280, half));
      FH = SH;
      if (traces) traces.resize(cw, 220);        // narrow: traces wrap below
    }
    /* no on-canvas captions; keep only room for the vertex emojis */
    const padX = 26, padB = 22, padT = 22;
    V = { A: [padX, SH - padB], B: [SW - padX, SH - padB], C: [SW / 2, padT] };
    fctx = setupCanvas(fCanvas, FW, FH);
    sctx = setupCanvas(sCanvas, SW, SH);
    const dpr = window.devicePixelRatio || 1;
    halo = document.createElement('canvas');
    core = document.createElement('canvas');
    for (const layer of [halo, core]) {
      layer.width = Math.round(SW * dpr); layer.height = Math.round(SH * dpr);
    }
    state.dirty = true;
    if (starts.length) resetRuns();
  }

  let wake = () => {};                     // assigned by animateWhileVisible below

  /* dragging on the f-canvas: heatmap follows live, trails freeze until pointer-up */
  fCanvas.addEventListener('pointerdown', e => {
    const r = fCanvas.getBoundingClientRect();
    const x = e.clientX - r.left;
    let best = -1, bd = 20;
    for (let s = 1; s <= N; s++) {
      const d = Math.abs(x - fx(s));
      if (d < bd) { bd = d; best = s; }
    }
    if (best > 0) {
      state.drag = best;
      fCanvas.setPointerCapture(e.pointerId);
      root.querySelectorAll('.sf-btn').forEach(b => b.classList.remove('active'));
      applyDrag(e);
    }
  });
  function applyDrag(e) {
    const r = fCanvas.getBoundingClientRect();
    const y = e.clientY - r.top;
    const v = (FH - FP.b - y) / (FH - FP.t - FP.b);
    state.f[state.drag] = Math.min(1, Math.max(0, v));
    state.dirty = true;
  }
  fCanvas.addEventListener('pointermove', e => { if (state.drag > 0) { applyDrag(e); wake(); } });
  const endDrag = () => {
    if (state.drag < 0) return;
    state.drag = -1;
    resetRuns();
    wake();
  };
  fCanvas.addEventListener('pointerup', endDrag);
  fCanvas.addEventListener('pointercancel', endDrag);

  root.querySelectorAll('.sf-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      state.f = PRESET_F[btn.dataset.f].slice();
      root.querySelectorAll('.sf-btn').forEach(x => x.classList.toggle('active', x === btn));
      resetRuns();
      state.dirty = true;
      wake();
    });
  });
  const replayBtn = root.querySelector('.replay-btn');
  if (replayBtn) replayBtn.addEventListener('click', () => { resample(); wake(); });

  resize();
  watchWidth(root.querySelector('.panels'), () => { resize(); wake(); });
  resample();
  wake = animateWhileVisible(root, () => {
    if (state.dirty) { drawF(); paintBg(); updateBadge(); state.dirty = false; }
    if (state.drag < 0 && steps < TOTAL) stepRuns(PER_FRAME);
    drawS();
    if (traces) traces.draw(steps);
    if (stepsEl) {
      stepsEl.textContent =
        `${starts.length} starts · SGD + momentum · step ${Math.min(steps, TOTAL)}/${TOTAL}`;
    }
    return state.drag < 0 && steps < TOTAL;   // idle while dragging (pointermove wakes) or converged
  });
}

/* ---------- boot ---------- */
const bootAll = () => {
  initCalibWidget();
  initAdvWidget('adv-widget-bags');
  initFlowGdmWidget('flow-sgdm-tr-widget');
  initCtrlWidget();
  initSetFnWidget('setfn-tr-widget');
};
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', bootAll);
} else {
  bootAll();
}

})();
