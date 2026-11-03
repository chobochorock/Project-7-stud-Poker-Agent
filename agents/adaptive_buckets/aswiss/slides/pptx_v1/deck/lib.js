'use strict';
// Shared helpers for the ASWISS deck: palette, rich-text markup, shapes, the H/L game tree, icons.
const React = require('react');
const RDS = require('react-dom/server');
const sharp = require('sharp');

const W = 13.333, H = 7.5, ML = 0.6, MR = 0.6, CW = W - ML - MR;

// Palette (the original Beamer deck). Blue = forward split, orange = backward split.
const P = {
  ink: '1B2430', muted: '4A5563', blue: '2A6FDB', dblue: '1F5BBF', orange: 'D9622B', dorange: 'B4501F',
  tblue: 'DCE7F8', torange: 'F8E3D7', soft: 'ECEFF3', line: 'C9CED6', km: '7D8794', lkm: 'B8BEC7',
  white: 'FFFFFF', paper: 'F7F6F2', lblue: '8FB3EE', lorange: 'E8A27F', pale: 'F4F6F9',
};
const THEME = {
  name: 'ASWISS',
  // One family for Hangul, Latin and Greek: mixing Calibri with a Hangul font misplaces combining marks (σ̄).
  headFontFace: 'Malgun Gothic',
  bodyFontFace: 'Malgun Gothic',
  colors: {
    dk1: P.ink, lt1: P.white, dk2: P.muted, lt2: P.soft,
    accent1: P.blue, accent2: P.orange, accent3: P.km, accent4: P.dblue, accent5: P.dorange, accent6: P.lblue,
    hlink: P.dblue, folHlink: P.dorange,
  },
};
const COLOR_NAMES = {
  ink: P.ink, muted: P.muted, blue: P.dblue, orange: P.dorange, km: P.km, white: P.white, lblue: P.lblue,
  lorange: P.lorange, b: P.dblue, o: P.dorange,
};

// Rich-text markup -> pptxgenjs runs.
//   **bold**   {{blue:text}} (colour span; names in COLOR_NAMES)   x_{sub}   x^{sup}
//   a paragraph starting with "- " becomes a bullet; "\n" separates paragraphs.
function rich(s, base = {}) {
  const paras = String(s).split('\n');
  const out = [];
  paras.forEach((para, pi) => {
    let bullet = false;
    if (para.startsWith('- ')) { bullet = true; para = para.slice(2); }
    const runs = [];
    const stack = [{ kind: 'root', o: { ...base } }];
    let buf = '';
    const top = () => stack[stack.length - 1].o;
    const flush = () => { if (buf) { runs.push({ text: buf, options: { ...top() } }); buf = ''; } };
    for (let i = 0; i < para.length;) {
      if (para.startsWith('**', i)) { flush(); top().bold = !top().bold; i += 2; continue; }
      if (para.startsWith('{{', i)) {
        const c = para.indexOf(':', i);
        const name = para.slice(i + 2, c);
        if (!(name in COLOR_NAMES)) throw new Error('unknown colour ' + name + ' in ' + para);
        flush(); stack.push({ kind: 'color', o: { ...top(), color: COLOR_NAMES[name] } }); i = c + 1; continue;
      }
      if (para.startsWith('_{', i) || para.startsWith('^{', i)) {
        flush();
        const sub = para[i] === '_';
        const o = { ...top() };
        if (sub) { o.subscript = true; delete o.superscript; } else { o.superscript = true; delete o.subscript; }
        stack.push({ kind: sub ? 'sub' : 'sup', o }); i += 2; continue;
      }
      if (para[i] === '}') {
        const k = stack[stack.length - 1].kind;
        if (k === 'sub' || k === 'sup') { flush(); stack.pop(); i += 1; continue; }
        if (k === 'color' && para[i + 1] === '}') { flush(); stack.pop(); i += 2; continue; }
      }
      buf += para[i]; i += 1;
    }
    flush();
    if (stack.length !== 1) throw new Error('unbalanced markup: ' + para);
    if (!runs.length) runs.push({ text: '', options: { ...base } });
    if (bullet) runs[0].options.bullet = { indent: 18 };
    if (pi < paras.length - 1) runs[runs.length - 1].options.breakLine = true;
    out.push(...runs);
  });
  return out;
}

function mkHelpers(pres) {
  const S = pres.shapes;
  // Text box. o: x,y,w,h, size, color (hex), bold, align, valign, fill, line, margin, spaceAfter, lineSp
  function T(slide, s, o) {
    const base = { fontSize: o.size || 20, color: o.color || P.ink, lang: 'ko-KR' };
    if (o.bold) base.bold = true;
    if (o.italic) base.italic = true;
    const opts = {
      x: o.x, y: o.y, w: o.w, h: o.h, align: o.align || 'left', valign: o.valign || 'top',
      margin: o.margin === undefined ? 0 : o.margin, isTextBox: true, fit: 'none',
      objectName: o.name,
    };
    if (o.spaceAfter) opts.paraSpaceAfter = o.spaceAfter;
    if (o.lineSp) opts.lineSpacingMultiple = o.lineSp;
    if (o.fill) opts.fill = { color: o.fill };
    if (o.line) opts.line = o.line;
    if (o.rect) { opts.shape = S.ROUNDED_RECTANGLE; opts.rectRadius = o.rect; }
    slide.addText(rich(s, base), opts);
  }
  function box(slide, o) {
    const opts = { x: o.x, y: o.y, w: o.w, h: o.h, objectName: o.name };
    opts.fill = o.fill ? { color: o.fill, transparency: o.transparency || 0 } : { type: 'none' };
    opts.line = o.line ? { color: o.line, width: o.lw || 1, dashType: o.dash || 'solid' } : { type: 'none' };
    if (o.radius) { opts.rectRadius = o.radius; slide.addShape(S.ROUNDED_RECTANGLE, opts); }
    else slide.addShape(o.shape || S.RECTANGLE, opts);
  }
  function line(slide, x1, y1, x2, y2, o = {}) {
    const ln = { color: o.color || P.muted, width: o.width || 1.5 };
    if (o.dash) ln.dashType = o.dash;
    if (o.arrow) ln.endArrowType = o.arrow === true ? 'triangle' : o.arrow;
    if (o.begin) ln.beginArrowType = o.begin;
    slide.addShape(S.LINE, {
      x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.max(Math.abs(x2 - x1), 0.0001), h: Math.max(Math.abs(y2 - y1), 0.0001),
      flipH: (x2 < x1) !== (y2 < y1) ? true : false, flipV: false, line: ln, objectName: o.name,
    });
    // pptxgenjs draws from top-left to bottom-right; flipH covers the anti-diagonal. Direction of the arrow:
    // when the line goes upwards (y2<y1) the arrow head belongs at the start of the drawn segment.
  }
  // Arrow from (x1,y1) to (x2,y2) with the head at (x2,y2), any direction.
  function arrow(slide, x1, y1, x2, y2, o = {}) {
    const ln = { color: o.color || P.muted, width: o.width || 2 };
    if (o.dash) ln.dashType = o.dash;
    const goesUp = y2 < y1 - 1e-6, goesLeft = x2 < x1 - 1e-6;
    // Drawn segment runs from its top-left corner (or top-right when flipped) downwards.
    // Head at the drawn end when the target is the lower point; otherwise at the drawn start.
    const headAtEnd = !goesUp && !(Math.abs(y2 - y1) < 1e-6 && goesLeft);
    if (headAtEnd) ln.endArrowType = 'triangle'; else ln.beginArrowType = 'triangle';
    const flipH = Math.abs(y2 - y1) < 1e-6 ? false : (goesLeft !== goesUp);
    slide.addShape(S.LINE, {
      x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.max(Math.abs(x2 - x1), 0.0001), h: Math.max(Math.abs(y2 - y1), 0.0001),
      flipH, line: ln, objectName: o.name,
    });
  }
  // Numbered circle
  function badge(slide, n, x, y, d, fill, size) {
    slide.addText(String(n), {
      x, y, w: d, h: d, shape: S.OVAL, fill: { color: fill }, line: { type: 'none' },
      color: P.white, bold: true, fontSize: size || 16, align: 'center', valign: 'middle', margin: 0,
      objectName: 'badge ' + n,
    });
  }
  // Rounded chip with text
  function chip(slide, s, o) {
    T(slide, s, {
      x: o.x, y: o.y, w: o.w, h: o.h || 0.42, size: o.size || 15, color: o.color || P.ink, bold: o.bold,
      align: o.align || 'center', valign: 'middle', fill: o.fill || P.soft, rect: o.radius || 0.12,
      line: o.line ? { color: o.line, width: 1 } : undefined, name: o.name,
    });
  }
  return { T, box, line, arrow, badge, chip, S };
}

// ---------------------------------------------------------------- the H/L game (proposition H4)
// Chance deals H or L to player 1 (1/2 each); player 2 never sees it.
// P1 decision 1: bet (-10, ends) or check. P2 (info set J): check (showdown H +1, L -1) or bet.
// P1 decision 2 after the bet: call (H +2, L -3) or fold (-1). Payoffs are player 1's.
const DY = 1.05;
const NODE = {
  root: [0, 0], d1H: [-1.6, 1], d1L: [1.6, 1],
  betH: [-2.6, 2], p2H: [-1.05, 2], p2L: [1.05, 2], betL: [2.6, 2],
  chkH: [-2.25, 3], d2H: [-0.75, 3], d2L: [0.75, 3], chkL: [2.25, 3],
  callH: [-1.25, 4], foldH: [-0.35, 4], callL: [0.35, 4], foldL: [1.25, 4],
};
const EDGES = [
  ['root', 'd1H', 'H  ½'], ['root', 'd1L', 'L  ½'],
  ['d1H', 'betH', '베팅'], ['d1H', 'p2H', '체크'], ['d1L', 'p2L', '체크'], ['d1L', 'betL', '베팅'],
  ['p2H', 'chkH', '체크'], ['p2H', 'd2H', '베팅'], ['p2L', 'd2L', '베팅'], ['p2L', 'chkL', '체크'],
  ['d2H', 'callH', '콜'], ['d2H', 'foldH', '폴드'], ['d2L', 'callL', '콜'], ['d2L', 'foldL', '폴드'],
];
const LEAF = { betH: '−10', betL: '−10', chkH: '+1', chkL: '−1', callH: '+2', foldH: '−1', callL: '−3', foldL: '−1' };
const P1 = ['d1H', 'd1L', 'd2H', 'd2L'], P2 = ['p2H', 'p2L'];

// tree(slide, h, {cx, oy, s, ...overlays})
//   buckets: [{nodes, label, kind:'soft'|'orange'|'blue', labelSide:'left'|'right'}]
//   hists:   [{nodes, tag, tagSide:'in'|'left'|'right'}]   dashed boxes (bucket histories)
//   jpill:   'plain' | 'strong' | null
//   edgeLabels: 'all' | 'd2' | 'none'; leaves: true/false; showChanceLabel
function tree(slide, h, o) {
  const s = o.s || 1, cx = o.cx, oy = o.oy;
  const X = (n) => cx + s * NODE[n][0];
  const Y = (n) => oy + s * NODE[n][1] * DY;
  const r = 0.21 * s;
  const fs = (pt) => Math.max(12, Math.round(pt * Math.min(1, s + 0.08)));
  const out = { X, Y, r, s };

  // 1. bucket boxes (filled, behind everything)
  (o.buckets || []).forEach((b) => {
    const xs = b.nodes.map(X), y = Y(b.nodes[0]);
    const padx = 0.4 * s, pady = 0.3 * s;
    const x0 = Math.min(...xs) - padx, x1 = Math.max(...xs) + padx;
    const fill = b.kind === 'orange' ? P.torange : b.kind === 'blue' ? P.tblue : P.soft;
    const ln = b.kind === 'orange' ? P.orange : b.kind === 'blue' ? P.blue : P.km;
    h.box(slide, { x: x0, y: y - pady, w: x1 - x0, h: 2 * pady, fill, line: ln, lw: b.kind === 'soft' ? 1 : 2, radius: 0.1, name: 'bucket ' + (b.name || b.label) });
    if (b.label) {
      const lw = b.labelW || 0.62;
      const side = b.labelSide || 'left';
      const lx = side === 'left' ? x0 - lw - 0.06 : x1 + 0.06;
      h.T(slide, b.label, { x: lx, y: y - 0.2, w: lw, h: 0.4, size: fs(b.labelSize || 17), bold: true, color: b.kind === 'orange' ? P.dorange : b.kind === 'blue' ? P.dblue : P.ink, align: side === 'left' ? 'right' : 'left', valign: 'middle' });
    }
  });
  // 2. info set J
  if (o.jpill) {
    const x0 = X('p2H') - 0.36 * s, x1 = X('p2L') + 0.36 * s, y = Y('p2H');
    h.box(slide, { x: x0, y: y - 0.27 * s, w: x1 - x0, h: 0.54 * s, fill: o.jpill === 'strong' ? P.pale : undefined, line: P.ink, lw: o.jpill === 'strong' ? 2 : 1, dash: 'sysDot', radius: 0.27 * s, name: 'info set J' });
    if (o.jlabel) h.T(slide, o.jlabel, { x: X('p2H') + 0.3 * s, y: y - 0.2, w: X('p2L') - X('p2H') - 0.6 * s, h: 0.4, size: fs(15), color: P.ink, bold: true, align: 'center', valign: 'middle' });
  }
  // 3. edges and edge labels
  EDGES.forEach(([a, b, lab]) => {
    h.line(slide, X(a), Y(a), X(b), Y(b), { color: P.muted, width: 1.5 });
    const show = o.edgeLabels === undefined || o.edgeLabels === 'all' || (o.edgeLabels === 'd2' && (a === 'd2H' || a === 'd2L'));
    if (!show) return;
    const mx = (X(a) + X(b)) / 2, my = (Y(a) + Y(b)) / 2;
    const left = X(b) < X(a);
    const lw = a === 'root' ? 0.75 : 0.62;
    h.T(slide, lab, { x: left ? mx - lw - 0.05 : mx + 0.05, y: my - 0.17, w: lw, h: 0.32, size: fs(14), color: P.muted, align: left ? 'right' : 'left', valign: 'middle' });
  });
  // 4. bucket-history boxes (dashed, no fill)
  (o.hists || []).forEach((k) => {
    const xs = k.nodes.map(X), y = Y(k.nodes[0]);
    const padx = (k.padx || 0.29) * s, pady = 0.24 * s;
    const x0 = Math.min(...xs) - padx, x1 = Math.max(...xs) + padx;
    h.box(slide, { x: x0, y: y - pady, w: x1 - x0, h: 2 * pady, line: k.color || P.ink, lw: 1.5, dash: 'dash', radius: 0.05, name: 'bucket history ' + (k.tag || '') });
    if (k.tag) {
      const tw = k.tagW || 0.9;
      let tx;
      if (k.tagSide === 'right') tx = x1 + 0.05;
      else if (k.tagSide === 'left') tx = x0 - tw - 0.05;
      else tx = (x0 + x1) / 2 - tw / 2; // centred (between the two members)
      h.T(slide, k.tag, { x: tx, y: y - 0.19, w: tw, h: 0.38, size: fs(k.tagSize || 14), color: k.tagColor || P.ink, bold: !!k.tagBold, align: k.tagSide === 'left' ? 'right' : k.tagSide === 'right' ? 'left' : 'center', valign: 'middle' });
    }
  });
  // 5. nodes
  slide.addShape(h.S.OVAL, { x: X('root') - 0.13 * s, y: Y('root') - 0.13 * s, w: 0.26 * s, h: 0.26 * s, fill: { color: P.km }, line: { type: 'none' }, objectName: 'chance node' });
  if (o.chanceLabel !== false) h.T(slide, '우연: 카드 분배', { x: X('root') + 0.22 * s, y: Y('root') - 0.2, w: 2.0, h: 0.4, size: fs(14), color: P.muted, valign: 'middle' });
  P1.forEach((n) => slide.addText('1', { x: X(n) - r, y: Y(n) - r, w: 2 * r, h: 2 * r, shape: h.S.OVAL, fill: { color: P.ink }, line: { type: 'none' }, color: P.white, bold: true, fontSize: fs(14), align: 'center', valign: 'middle', margin: 0, objectName: 'P1 node ' + n }));
  P2.forEach((n) => slide.addText('2', { x: X(n) - r, y: Y(n) - r, w: 2 * r, h: 2 * r, shape: h.S.OVAL, fill: { color: P.white }, line: { color: P.ink, width: 1.5 }, color: P.ink, bold: true, fontSize: fs(14), align: 'center', valign: 'middle', margin: 0, objectName: 'P2 node ' + n }));
  // 6. leaves
  if (o.leaves !== false) Object.entries(LEAF).forEach(([n, v]) => {
    h.T(slide, v, { x: X(n) - 0.32, y: Y(n) - 0.17, w: 0.64, h: 0.34, size: fs(16), bold: true, color: P.ink, align: 'center', valign: 'middle', name: 'payoff ' + n });
  });
  return out;
}

// react-icons -> PNG data string for addImage
async function icon(Comp, color, px = 256) {
  const svg = RDS.renderToStaticMarkup(React.createElement(Comp, { color: '#' + color, size: px }));
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return 'image/png;base64,' + buf.toString('base64');
}

module.exports = { W, H, ML, MR, CW, P, THEME, rich, mkHelpers, tree, NODE, DY, icon };
