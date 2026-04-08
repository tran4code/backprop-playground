import { useState, useMemo, useEffect, useRef } from 'react';
import katex from 'katex';
import 'katex/dist/katex.min.css';

/* ═══════════════════ CONSTANTS ═══════════════════ */

const COL = {
  blue: '#1E88E5', red: '#E53935', green: '#43A047', orange: '#F57C00',
  purple: '#8E24AA', teal: '#00897B', slate: '#4B5D8E', gray: '#9e9e9e',
  dark: '#1a1a2e', lightGray: '#e0e0e0',
};

const INIT_W = { w1_1: 1, w1_2: -0.5, w2: 2 };
const INIT_I = { x1: 1, x2: -2, y: 0 };

// SVG node positions — scaled up for lecture projection
const N = {
  x1:  { cx: 110, cy: 110 },
  x2:  { cx: 110, cy: 390 },
  h1:  { cx: 420, cy: 250 },
  out: { cx: 720, cy: 250 },
};
const R_IN = 42, R_N = 50;

/* ═══════════════════ MATH ═══════════════════ */

const sigmoid = x => 1 / (1 + Math.exp(-x));
const fmt = (v, d = 3) => +v.toFixed(d);

function computeForward(w, inp) {
  const z1 = w.w1_1 * inp.x1 + w.w1_2 * inp.x2;
  const a1 = sigmoid(z1);
  const z2 = w.w2 * a1;
  const a2 = sigmoid(z2);
  const cost = (inp.y - a2) ** 2;
  return { z1, a1, z2, a2, cost };
}

function computeBackward(w, inp, f) {
  const dC_da2 = -2 * (inp.y - f.a2);
  const da2_dz2 = f.a2 * (1 - f.a2);
  const dC_dz2 = dC_da2 * da2_dz2;
  const dC_dw2 = dC_dz2 * f.a1;
  const dz2_da1 = w.w2;
  const dC_da1 = dC_dz2 * dz2_da1;
  const da1_dz1 = f.a1 * (1 - f.a1);
  const dC_dz1 = dC_da1 * da1_dz1;
  const dC_dw1_1 = dC_dz1 * inp.x1;
  const dC_dw1_2 = dC_dz1 * inp.x2;
  return { dC_da2, da2_dz2, dC_dz2, dC_dw2, dz2_da1, dC_da1, da1_dz1, dC_dz1, dC_dw1_1, dC_dw1_2 };
}

/* ═══════════════════ TEX ═══════════════════ */

function Tex({ math, display = false }) {
  const html = useMemo(
    () => katex.renderToString(math, { displayMode: display, throwOnError: false }),
    [math, display]
  );
  return <span dangerouslySetInnerHTML={{ __html: html }} />;
}

/* ═══════════════════ STEP DEFINITIONS ═══════════════════ */

const STEPS = [
  {
    id: 'forward', title: 'Forward Pass',
    intuition: 'Push inputs through the network to compute a prediction.',
    color: COL.blue,
    getFormula: (f, b, w, i) => [
      `z^1 = w^1_1 \\cdot x_1 + w^1_2 \\cdot x_2`,
      `\\quad = (${fmt(w.w1_1,2)})(${fmt(i.x1,1)}) + (${fmt(w.w1_2,2)})(${fmt(i.x2,1)}) = ${fmt(f.z1)}`,
      `a^1 = \\sigma(${fmt(f.z1)}) = ${fmt(f.a1)}`,
      `z^2 = w^2 \\cdot a^1 = (${fmt(w.w2,2)})(${fmt(f.a1)}) = ${fmt(f.z2)}`,
      `\\hat{y} = a^2 = \\sigma(${fmt(f.z2)}) = ${fmt(f.a2)}`,
    ],
    getResult: (f) => `a² = ${fmt(f.a2)}`,
    hlEdges: ['w1_1', 'w1_2', 'w2'], hlNodes: ['x1', 'x2', 'h1', 'out'],
  },
  {
    id: 'loss', title: 'Compute Loss',
    intuition: 'How far off is our prediction from the target?',
    color: COL.red,
    getFormula: (f, b, w, i) => [
      `C = (y - a^2)^2 = (${fmt(i.y,1)} - ${fmt(f.a2)})^2 = ${fmt(f.cost, 4)}`,
    ],
    getResult: (f) => `C = ${fmt(f.cost, 4)}`,
    hlNodes: ['out'],
  },
  {
    id: 'dC_da2', title: '∂C/∂a²', subtitle: 'Cost → Output activation',
    intuition: 'If the output activation nudges up, does the cost increase or decrease?',
    color: COL.orange,
    getFormula: (f, b, w, i) => [
      `\\frac{\\partial C}{\\partial a^2} = -2(y - a^2)`,
      `\\quad = -2(${fmt(i.y,1)} - ${fmt(f.a2)}) = \\mathbf{${fmt(b.dC_da2, 4)}}`,
    ],
    getResult: (f, b) => fmt(b.dC_da2, 4),
    hlNodes: ['out'],
  },
  {
    id: 'dC_dz2', title: '∂C/∂z²', subtitle: 'Chain rule through sigmoid',
    intuition: 'Multiply by the sigmoid derivative to pass the gradient through the activation.',
    color: COL.orange,
    chainRule: '\\frac{\\partial C}{\\partial z^2} = \\frac{\\partial C}{\\partial a^2} \\cdot \\frac{\\partial a^2}{\\partial z^2}',
    getFormula: (f, b) => [
      `\\sigma'(z^2) = a^2(1-a^2)`,
      `\\quad = ${fmt(f.a2)} \\times ${fmt(1 - f.a2)} = ${fmt(b.da2_dz2, 4)}`,
      `\\frac{\\partial C}{\\partial z^2} = ${fmt(b.dC_da2, 4)} \\times ${fmt(b.da2_dz2, 4)}`,
      `\\quad = \\mathbf{${fmt(b.dC_dz2, 4)}}`,
    ],
    getResult: (f, b) => fmt(b.dC_dz2, 4),
    hlNodes: ['out'],
  },
  {
    id: 'dC_dw2', title: '∂C/∂w²', subtitle: 'Gradient for the output weight',
    intuition: 'This tells us exactly how to adjust w² to reduce the loss.',
    color: COL.teal,
    chainRule: '\\frac{\\partial C}{\\partial w^2} = \\frac{\\partial C}{\\partial z^2} \\cdot \\frac{\\partial z^2}{\\partial w^2}',
    getFormula: (f, b) => [
      `\\frac{\\partial z^2}{\\partial w^2} = a^1 = ${fmt(f.a1)}`,
      `\\frac{\\partial C}{\\partial w^2} = ${fmt(b.dC_dz2, 4)} \\times ${fmt(f.a1)}`,
      `\\quad = \\mathbf{${fmt(b.dC_dw2, 4)}}`,
    ],
    getResult: (f, b) => fmt(b.dC_dw2, 4),
    hlEdges: ['w2'],
  },
  {
    id: 'dC_da1', title: '∂C/∂a¹', subtitle: 'Gradient flows back through w²',
    intuition: 'The gradient "travels" backward along w² to reach the hidden neuron.',
    color: COL.purple,
    chainRule: '\\frac{\\partial C}{\\partial a^1} = \\frac{\\partial C}{\\partial z^2} \\cdot \\frac{\\partial z^2}{\\partial a^1}',
    getFormula: (f, b, w) => [
      `\\frac{\\partial z^2}{\\partial a^1} = w^2 = ${fmt(w.w2, 2)}`,
      `\\frac{\\partial C}{\\partial a^1} = ${fmt(b.dC_dz2, 4)} \\times ${fmt(w.w2, 2)}`,
      `\\quad = \\mathbf{${fmt(b.dC_da1, 4)}}`,
    ],
    getResult: (f, b) => fmt(b.dC_da1, 4),
    hlNodes: ['h1'], hlEdges: ['w2'],
  },
  {
    id: 'dC_dz1', title: '∂C/∂z¹', subtitle: 'Chain rule through hidden sigmoid',
    intuition: 'Same pattern — multiply by the sigmoid derivative at the hidden layer.',
    color: COL.purple,
    chainRule: '\\frac{\\partial C}{\\partial z^1} = \\frac{\\partial C}{\\partial a^1} \\cdot \\frac{\\partial a^1}{\\partial z^1}',
    getFormula: (f, b) => [
      `\\sigma'(z^1) = a^1(1-a^1)`,
      `\\quad = ${fmt(f.a1)} \\times ${fmt(1 - f.a1)} = ${fmt(b.da1_dz1, 4)}`,
      `\\frac{\\partial C}{\\partial z^1} = ${fmt(b.dC_da1, 4)} \\times ${fmt(b.da1_dz1, 4)}`,
      `\\quad = \\mathbf{${fmt(b.dC_dz1, 4)}}`,
    ],
    getResult: (f, b) => fmt(b.dC_dz1, 4),
    hlNodes: ['h1'],
  },
  {
    id: 'dC_dw1', title: '∂C/∂w¹₁ and ∂C/∂w¹₂', subtitle: 'Gradients for the input weights',
    intuition: 'Final step of backprop — now we know how each input weight should change.',
    color: COL.teal,
    getFormula: (f, b, w, i) => [
      `\\frac{\\partial C}{\\partial w^1_1} = \\frac{\\partial C}{\\partial z^1} \\cdot x_1`,
      `\\quad = ${fmt(b.dC_dz1, 4)} \\times ${fmt(i.x1, 1)} = \\mathbf{${fmt(b.dC_dw1_1, 4)}}`,
      `\\frac{\\partial C}{\\partial w^1_2} = \\frac{\\partial C}{\\partial z^1} \\cdot x_2`,
      `\\quad = ${fmt(b.dC_dz1, 4)} \\times (${fmt(i.x2, 1)}) = \\mathbf{${fmt(b.dC_dw1_2, 4)}}`,
    ],
    getResult: (f, b) => `w¹₁: ${fmt(b.dC_dw1_1, 4)}, w¹₂: ${fmt(b.dC_dw1_2, 4)}`,
    hlEdges: ['w1_1', 'w1_2'],
  },
  {
    id: 'update', title: 'Update Weights', subtitle: 'Gradient descent step',
    intuition: 'Nudge each weight opposite to its gradient to reduce the loss.',
    color: COL.green,
    getFormula: (f, b, w, i, lr) => {
      const nw2 = w.w2 - lr * b.dC_dw2;
      const nw1_1 = w.w1_1 - lr * b.dC_dw1_1;
      const nw1_2 = w.w1_2 - lr * b.dC_dw1_2;
      return [
        `w^2_{\\text{new}} = w^2 - \\eta \\cdot \\frac{\\partial C}{\\partial w^2}`,
        `\\quad = ${fmt(w.w2, 3)} - ${fmt(lr, 2)} \\times ${fmt(b.dC_dw2, 4)} = \\mathbf{${fmt(nw2, 4)}}`,
        `w^1_{1,\\text{new}} = w^1_1 - \\eta \\cdot \\frac{\\partial C}{\\partial w^1_1}`,
        `\\quad = ${fmt(w.w1_1, 3)} - ${fmt(lr, 2)} \\times ${fmt(b.dC_dw1_1, 4)} = \\mathbf{${fmt(nw1_1, 4)}}`,
        `w^1_{2,\\text{new}} = w^1_2 - \\eta \\cdot \\frac{\\partial C}{\\partial w^1_2}`,
        `\\quad = ${fmt(w.w1_2, 3)} - ${fmt(lr, 2)} \\times ${fmt(b.dC_dw1_2, 4)} = \\mathbf{${fmt(nw1_2, 4)}}`,
      ];
    },
    getResult: () => 'Apply →',
    hlEdges: ['w1_1', 'w1_2', 'w2'],
  },
];

/* ═══════════════════ SMALL COMPONENTS ═══════════════════ */

function NumInput({ label, value, onChange, disabled, step = 0.1 }) {
  return (
    <div className="input-row">
      <label>{label}</label>
      <input type="number" className="num-input" value={value}
        onChange={e => onChange(+e.target.value)} step={step} disabled={disabled} />
    </div>
  );
}

function SliderInput({ label, value, onChange, min, max, step, disabled }) {
  return (
    <div className="input-row">
      <label>{label}</label>
      <input type="range" className="slider" min={min} max={max} step={step}
        value={value} onChange={e => onChange(+e.target.value)} disabled={disabled} />
      <span className="val">{value.toFixed(2)}</span>
    </div>
  );
}

/* ═══════════════════ INPUT PANEL ═══════════════════ */

function InputPanel({ weights, inputs, lr, onW, onI, onLr, disabled }) {
  const sw = (k, v) => onW(w => ({ ...w, [k]: v }));
  const si = (k, v) => onI(i => ({ ...i, [k]: v }));
  return (
    <div className="controls">
      <h3>Inputs</h3>
      <NumInput label="x₁" value={inputs.x1} onChange={v => si('x1', v)} disabled={disabled} />
      <NumInput label="x₂" value={inputs.x2} onChange={v => si('x2', v)} disabled={disabled} />
      <NumInput label="y" value={inputs.y} onChange={v => si('y', v)} disabled={disabled} />
      <h3>Weights</h3>
      <SliderInput label="w¹₁" value={weights.w1_1} onChange={v => sw('w1_1', v)} min={-5} max={5} step={0.1} disabled={disabled} />
      <SliderInput label="w¹₂" value={weights.w1_2} onChange={v => sw('w1_2', v)} min={-5} max={5} step={0.1} disabled={disabled} />
      <SliderInput label="w²" value={weights.w2} onChange={v => sw('w2', v)} min={-5} max={5} step={0.1} disabled={disabled} />
      <h3>Learning Rate</h3>
      <SliderInput label="η" value={lr} onChange={onLr} min={0.01} max={2} step={0.01} disabled={disabled} />
    </div>
  );
}

/* ═══════════════════ LOSS SPARKLINE ═══════════════════ */

function LossChart({ history, current }) {
  const all = [...history, current];
  if (all.length < 1) return null;
  const w = 200, h = 48, pad = 3;
  const maxC = Math.max(...all, 0.001);
  const pts = all.map((c, i) => {
    const px = pad + (i / Math.max(all.length - 1, 1)) * (w - 2 * pad);
    const py = pad + (1 - c / maxC) * (h - 2 * pad);
    return `${px},${py}`;
  });
  return (
    <svg className="loss-sparkline" width={w} height={h} style={{ background: 'rgba(255,255,255,0.1)', borderRadius: 4 }}>
      {all.length > 1 && <polyline points={pts.join(' ')} fill="none" stroke="#ef9a9a" strokeWidth="2.5" />}
      <circle cx={pts[pts.length - 1].split(',')[0]} cy={pts[pts.length - 1].split(',')[1]} r="4" fill="#E53935" />
    </svg>
  );
}

/* ═══════════════════ NETWORK DIAGRAM ═══════════════════ */

function NetworkDiagram({ weights, inputs, fwd, bwd, step }) {
  const showForward = step >= 0;
  const showLoss = step >= 1;

  // Edge endpoints (line from circle boundary to circle boundary)
  function edge(from, rFrom, to, rTo) {
    const dx = to.cx - from.cx, dy = to.cy - from.cy;
    const a = Math.atan2(dy, dx);
    return {
      x1: from.cx + rFrom * Math.cos(a), y1: from.cy + rFrom * Math.sin(a),
      x2: to.cx - rTo * Math.cos(a), y2: to.cy - rTo * Math.sin(a),
    };
  }

  const e1 = edge(N.x1, R_IN, N.h1, R_N);
  const e2 = edge(N.x2, R_IN, N.h1, R_N);
  const e3 = edge(N.h1, R_N, N.out, R_N);

  function midLabel(e, offset = -14) {
    const mx = (e.x1 + e.x2) / 2, my = (e.y1 + e.y2) / 2;
    const a = Math.atan2(e.y2 - e.y1, e.x2 - e.x1);
    return { x: mx + offset * Math.sin(a), y: my - offset * Math.cos(a) };
  }

  const ml1 = midLabel(e1, -38);
  const ml2 = midLabel(e2, 38);
  const ml3 = midLabel(e3, 38);

  // Edge colors based on step
  function edgeColor(edgeId) {
    const def = STEPS[step];
    if (step < 0) return COL.slate;
    if (def && def.hlEdges && def.hlEdges.includes(edgeId)) return def.color;
    // Show teal for edges whose gradients have already been computed
    if (edgeId === 'w2' && step >= 4) return COL.teal;
    if ((edgeId === 'w1_1' || edgeId === 'w1_2') && step >= 7) return COL.teal;
    if (step === 0) return COL.blue;
    return COL.slate;
  }

  function edgeWidth(edgeId) {
    const def = STEPS[step];
    if (def && def.hlEdges && def.hlEdges.includes(edgeId)) return 5;
    return 3;
  }

  // Node fill colors
  function nodeFill(nodeId) {
    const def = STEPS[step];
    if (step < 0) return '#fff';
    if (def && def.hlNodes && def.hlNodes.includes(nodeId)) {
      return def.color + '22';
    }
    if (showForward) return COL.blue + '12';
    return '#fff';
  }

  function nodeStroke(nodeId) {
    const def = STEPS[step];
    if (step < 0) return COL.slate;
    if (def && def.hlNodes && def.hlNodes.includes(nodeId)) return def.color;
    return COL.slate;
  }

  // KaTeX badge via foreignObject — renders proper stacked fractions
  function fracBadge(tex, value, cx, cy, minStep, color, bw = 210, bh = 52) {
    if (step < minStep) return null;
    const isActive = step === minStep;
    const html = katex.renderToString(
      `${tex} = ${fmt(value, 3)}`,
      { throwOnError: false }
    );
    return (
      <foreignObject key={tex} x={cx - bw / 2} y={cy - bh / 2} width={bw} height={bh}>
        <div xmlns="http://www.w3.org/1999/xhtml" style={{
          background: isActive ? color + '15' : 'rgba(255,255,255,0.97)',
          border: `${isActive ? 2.5 : 1.5}px solid ${isActive ? color : '#ddd'}`,
          borderRadius: 8,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          height: '100%',
          fontSize: 17,
          color,
          fontWeight: isActive ? 700 : 500,
        }} dangerouslySetInnerHTML={{ __html: html }} />
      </foreignObject>
    );
  }

  return (
    <div className="network-container">
      <svg viewBox="0 0 960 600">
        <defs>
          <marker id="arrow" markerWidth="12" markerHeight="9" refX="12" refY="4.5" orient="auto">
            <path d="M0,0 L12,4.5 L0,9" fill={COL.slate} />
          </marker>
          <marker id="arrow-blue" markerWidth="12" markerHeight="9" refX="12" refY="4.5" orient="auto">
            <path d="M0,0 L12,4.5 L0,9" fill={COL.blue} />
          </marker>
          <marker id="arrow-grad" markerWidth="12" markerHeight="9" refX="0" refY="4.5" orient="auto">
            <path d="M12,0 L0,4.5 L12,9" fill={COL.orange} />
          </marker>
        </defs>

        {/* ── Layer labels ── */}
        <text x={N.x1.cx} y={36} textAnchor="middle" fontSize="20" fill="#666" fontWeight="700">Input</text>
        <text x={N.h1.cx} y={36} textAnchor="middle" fontSize="20" fill="#666" fontWeight="700">Hidden</text>
        <text x={N.out.cx} y={36} textAnchor="middle" fontSize="20" fill="#666" fontWeight="700">Output</text>

        {/* ── Edges ── */}
        {[
          { id: 'w1_1', e: e1, ml: ml1, w: weights.w1_1, label: 'w¹₁' },
          { id: 'w1_2', e: e2, ml: ml2, w: weights.w1_2, label: 'w¹₂' },
          { id: 'w2',   e: e3, ml: ml3, w: weights.w2,   label: 'w²' },
        ].map(({ id, e, ml, w, label }) => (
          <g key={id}>
            <line x1={e.x1} y1={e.y1} x2={e.x2} y2={e.y2}
              stroke={edgeColor(id)} strokeWidth={edgeWidth(id)}
              markerEnd={step === 0 ? 'url(#arrow-blue)' : 'url(#arrow)'} />
            {/* Weight label */}
            <rect x={ml.x - 32} y={ml.y - 16} width={64} height={32} rx={6}
              fill="#fff" stroke={edgeColor(id)} strokeWidth={2} />
            <text x={ml.x} y={ml.y + 6} textAnchor="middle" fontSize="20"
              fill={edgeColor(id)} fontWeight="700">
              {fmt(w, 2)}
            </text>
            {/* Edge name */}
            <text x={ml.x} y={ml.y - 22} textAnchor="middle" fontSize="16" fill="#888" fontWeight="500">
              {label}
            </text>
          </g>
        ))}

        {/* ── Input nodes ── */}
        {[
          { id: 'x1', pos: N.x1, label: 'x₁', val: inputs.x1 },
          { id: 'x2', pos: N.x2, label: 'x₂', val: inputs.x2 },
        ].map(({ id, pos, label, val }) => (
          <g key={id}>
            <circle cx={pos.cx} cy={pos.cy} r={R_IN} fill={nodeFill(id)} stroke={nodeStroke(id)} strokeWidth={3} />
            <text x={pos.cx} y={pos.cy - 6} textAnchor="middle" fontSize="18" fill="#555" fontWeight="600">{label}</text>
            <text x={pos.cx} y={pos.cy + 18} textAnchor="middle" fontSize="22" fill={COL.dark} fontWeight="800">{fmt(val, 1)}</text>
          </g>
        ))}

        {/* ── Hidden node ── */}
        <g>
          <circle cx={N.h1.cx} cy={N.h1.cy} r={R_N} fill={nodeFill('h1')} stroke={nodeStroke('h1')} strokeWidth={3} />
          <text x={N.h1.cx} y={N.h1.cy + 8} textAnchor="middle" fontSize="28" fill={COL.dark} fontWeight="700">σ</text>
          {showForward && (
            <>
              <text x={N.h1.cx} y={N.h1.cy - R_N - 20} textAnchor="middle" fontSize="18" fill="#666" fontWeight="500">z¹ = {fmt(fwd.z1)}</text>
              <text x={N.h1.cx} y={N.h1.cy + R_N + 24} textAnchor="middle" fontSize="20" fill={COL.blue} fontWeight="700">a¹ = {fmt(fwd.a1)}</text>
            </>
          )}
        </g>

        {/* ── Output node ── */}
        <g>
          <circle cx={N.out.cx} cy={N.out.cy} r={R_N} fill={nodeFill('out')} stroke={nodeStroke('out')} strokeWidth={3} />
          <text x={N.out.cx} y={N.out.cy + 8} textAnchor="middle" fontSize="28" fill={COL.dark} fontWeight="700">σ</text>
          {showForward && (
            <>
              <text x={N.out.cx} y={N.out.cy - R_N - 20} textAnchor="middle" fontSize="18" fill="#666" fontWeight="500">z² = {fmt(fwd.z2)}</text>
              <text x={N.out.cx} y={N.out.cy + R_N + 24} textAnchor="middle" fontSize="20" fill={COL.blue} fontWeight="700">a² = {fmt(fwd.a2)}</text>
            </>
          )}
        </g>

        {/* ── Loss ── */}
        {showLoss && (
          <g>
            <text x={N.out.cx + 80} y={N.out.cy - 22} textAnchor="start" fontSize="18" fill="#666" fontWeight="600">Loss</text>
            <text x={N.out.cx + 80} y={N.out.cy + 8} textAnchor="start" fontSize="24" fill={COL.red} fontWeight="800">
              C = {fmt(fwd.cost, 4)}
            </text>
            <text x={N.out.cx + 80} y={N.out.cy + 34} textAnchor="start" fontSize="16" fill="#999">
              (y = {fmt(inputs.y, 1)})
            </text>
          </g>
        )}

        {/* ── Backward flow arrows (rendered BEFORE badges so badges paint on top) ── */}
        {step >= 5 && (
          <line x1={N.out.cx - R_N - 6} y1={N.out.cy + 14} x2={N.h1.cx + R_N + 6} y2={N.h1.cy + 14}
            stroke={COL.purple} strokeWidth={2.5} strokeDasharray="8,5" markerEnd="url(#arrow-grad)" opacity={0.5} />
        )}
        {step >= 7 && (
          <>
            <line x1={N.h1.cx - R_N - 6} y1={N.h1.cy - 14} x2={N.x1.cx + R_IN + 6} y2={N.x1.cy + 14}
              stroke={COL.teal} strokeWidth={2.5} strokeDasharray="8,5" opacity={0.4} />
            <line x1={N.h1.cx - R_N - 6} y1={N.h1.cy + 14} x2={N.x2.cx + R_IN + 6} y2={N.x2.cy - 14}
              stroke={COL.teal} strokeWidth={2.5} strokeDasharray="8,5" opacity={0.4} />
          </>
        )}

        {/* ── Gradient badges (foreignObject + KaTeX, rendered LAST = on top) ── */}
        {/* Node gradients — below each neuron */}
        {fracBadge('\\frac{\\partial C}{\\partial a^2}', bwd.dC_da2, N.out.cx, N.out.cy + R_N + 60, 2, COL.orange)}
        {fracBadge('\\frac{\\partial a^2}{\\partial z^2}', bwd.da2_dz2, N.out.cx, N.out.cy + R_N + 116, 3, COL.orange)}
        {fracBadge('\\frac{\\partial C}{\\partial z^2}', bwd.dC_dz2, N.out.cx, N.out.cy + R_N + 172, 3, COL.orange)}
        {fracBadge('\\frac{\\partial C}{\\partial a^1}', bwd.dC_da1, N.h1.cx, N.h1.cy + R_N + 60, 5, COL.purple)}
        {fracBadge('\\frac{\\partial a^1}{\\partial z^1}', bwd.da1_dz1, N.h1.cx, N.h1.cy + R_N + 116, 6, COL.purple)}
        {fracBadge('\\frac{\\partial C}{\\partial z^1}', bwd.dC_dz1, N.h1.cx, N.h1.cy + R_N + 172, 6, COL.purple)}
        {/* Weight gradients — positioned clear of edges */}
        {fracBadge('\\frac{\\partial C}{\\partial w^2}', bwd.dC_dw2, (N.h1.cx + N.out.cx) / 2, 152, 4, COL.teal)}
        {fracBadge('\\frac{\\partial C}{\\partial w^1_1}', bwd.dC_dw1_1, N.x1.cx, 222, 7, COL.teal)}
        {fracBadge('\\frac{\\partial C}{\\partial w^1_2}', bwd.dC_dw1_2, N.x1.cx, 278, 7, COL.teal)}
      </svg>
    </div>
  );
}

/* ═══════════════════ STEP BAR ═══════════════════ */

function StepBar({ step, onNext, onPrev, onRunAll, onReset, onApply, onAutoTrain, autoTraining }) {
  const maxStep = STEPS.length - 1;
  return (
    <div className="step-bar">
      <button onClick={onReset}>Reset</button>
      <button onClick={onPrev} disabled={step <= -1}>◀ Back</button>
      <button onClick={onNext} disabled={step >= maxStep} className="primary">
        {step < 0 ? 'Start ▶' : 'Step ▶'}
      </button>
      <button onClick={onRunAll} disabled={step >= maxStep}>All ▶▶</button>
      {step >= maxStep && (
        <button onClick={onApply} className="apply">Apply Updates & Continue</button>
      )}
      <button onClick={onAutoTrain} className={`train ${autoTraining ? 'active' : ''}`}>
        {autoTraining ? '■ Stop Training' : '⚡ Auto Train'}
      </button>

      {/* Progress dots */}
      <div className="progress-dots" style={{ marginLeft: 8 }}>
        {STEPS.map((s, i) => (
          <div key={s.id} className={`dot ${i < step ? 'done' : ''} ${i === step ? 'active' : ''}`}
            title={s.title} />
        ))}
      </div>

      {step >= 0 && step <= maxStep && (
        <span className="step-label">
          Step {step + 1}/{STEPS.length}: {STEPS[step].title}
        </span>
      )}
    </div>
  );
}

/* ═══════════════════ CHAIN RULE PANEL ═══════════════════ */

function ChainPanel({ step, fwd, bwd, weights, inputs, lr }) {
  if (step < 0) {
    return (
      <div className="chain-panel">
        <p style={{ color: '#888', fontStyle: 'italic', padding: '8px 0' }}>
          Press <strong>Start</strong> to begin the forward pass, then step through backpropagation one gradient at a time.
        </p>
      </div>
    );
  }

  return (
    <div className="chain-panel">
      <h2>Chain Rule Steps</h2>
      {STEPS.map((s, i) => {
        const isActive = i === step;
        const isDone = i < step;
        const isFuture = i > step;
        const cls = `step-card ${isActive ? 'active' : ''} ${isDone ? 'done' : ''} ${isFuture ? 'future' : ''}`;

        return (
          <div key={s.id} className={cls}>
            <div className="step-header">
              <span className="step-num" style={{ background: isDone || isActive ? s.color : '#ccc' }}>
                {isDone ? '✓' : i + 1}
              </span>
              <span className="step-title">{s.title}</span>
              {s.subtitle && <span className="step-subtitle">{s.subtitle}</span>}
              {isDone && s.getResult && (
                <span className="step-result">{s.getResult(fwd, bwd)}</span>
              )}
            </div>

            {isActive && (
              <div className="step-detail">
                <p className="step-intuition">{s.intuition}</p>

                {s.chainRule && (
                  <div className="chain-vis">
                    <Tex math={s.chainRule} display />
                  </div>
                )}

                <div className="step-formulas">
                  {s.getFormula(fwd, bwd, weights, inputs, lr).map((f, j) => (
                    <div key={j} className="formula-line">
                      <Tex math={f} display />
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

/* ═══════════════════ HEADER ═══════════════════ */

function Header({ epoch, lossHistory, currentLoss }) {
  return (
    <div className="header">
      <h1>Backpropagation Playground</h1>
      <div className="header-right">
        {epoch > 0 && <span className="epoch">Epoch: {epoch}</span>}
        <span className="loss">Loss: {currentLoss.toFixed(4)}</span>
        {lossHistory.length > 0 && <LossChart history={lossHistory} current={currentLoss} />}
      </div>
    </div>
  );
}

/* ═══════════════════ MAIN COMPONENT ═══════════════════ */

export default function BackpropPlayground() {
  const [weights, setWeights] = useState({ ...INIT_W });
  const [inputs, setInputs] = useState({ ...INIT_I });
  const [step, setStep] = useState(-1);
  const [lr, setLr] = useState(0.5);
  const [lossHistory, setLossHistory] = useState([]);
  const [autoTraining, setAutoTraining] = useState(false);
  const [epoch, setEpoch] = useState(0);

  const fwd = useMemo(() => computeForward(weights, inputs), [weights, inputs]);
  const bwd = useMemo(() => computeBackward(weights, inputs, fwd), [weights, inputs, fwd]);

  const nextStep = () => setStep(s => Math.min(s + 1, STEPS.length - 1));
  const prevStep = () => setStep(s => Math.max(s - 1, -1));
  const runAll = () => setStep(STEPS.length - 1);

  const reset = () => {
    setStep(-1);
    setAutoTraining(false);
  };

  const fullReset = () => {
    setWeights({ ...INIT_W });
    setInputs({ ...INIT_I });
    setLr(0.5);
    setStep(-1);
    setLossHistory([]);
    setEpoch(0);
    setAutoTraining(false);
  };

  const applyUpdate = () => {
    setLossHistory(h => [...h, fwd.cost]);
    setWeights({
      w1_1: weights.w1_1 - lr * bwd.dC_dw1_1,
      w1_2: weights.w1_2 - lr * bwd.dC_dw1_2,
      w2: weights.w2 - lr * bwd.dC_dw2,
    });
    setEpoch(e => e + 1);
    setStep(-1);
  };

  // Refs for auto-training interval
  const inputsRef = useRef(inputs);
  const lrRef = useRef(lr);
  useEffect(() => { inputsRef.current = inputs; }, [inputs]);
  useEffect(() => { lrRef.current = lr; }, [lr]);

  useEffect(() => {
    if (!autoTraining) return;
    setStep(-1);
    const id = setInterval(() => {
      const inp = inputsRef.current;
      const rate = lrRef.current;
      setWeights(prev => {
        const f = computeForward(prev, inp);
        const b = computeBackward(prev, inp, f);
        setLossHistory(h => {
          const next = [...h, f.cost];
          return next.length > 200 ? next.slice(-200) : next;
        });
        setEpoch(e => e + 1);
        return {
          w1_1: prev.w1_1 - rate * b.dC_dw1_1,
          w1_2: prev.w1_2 - rate * b.dC_dw1_2,
          w2: prev.w2 - rate * b.dC_dw2,
        };
      });
    }, 120);
    return () => clearInterval(id);
  }, [autoTraining]);

  const steppingDisabled = step >= 0 || autoTraining;

  return (
    <div className="playground">
      <Header epoch={epoch} lossHistory={lossHistory} currentLoss={fwd.cost} />
      <div className="main-area">
        <InputPanel
          weights={weights} inputs={inputs} lr={lr}
          onW={setWeights} onI={setInputs} onLr={setLr}
          disabled={steppingDisabled}
        />
        <NetworkDiagram
          weights={weights} inputs={inputs} fwd={fwd} bwd={bwd} step={step}
        />
        <ChainPanel step={step} fwd={fwd} bwd={bwd} weights={weights} inputs={inputs} lr={lr} />
      </div>
      <StepBar
        step={step} onNext={nextStep} onPrev={prevStep}
        onRunAll={runAll} onReset={epoch > 0 ? fullReset : reset}
        onApply={applyUpdate}
        onAutoTrain={() => setAutoTraining(t => !t)}
        autoTraining={autoTraining}
      />
    </div>
  );
}
