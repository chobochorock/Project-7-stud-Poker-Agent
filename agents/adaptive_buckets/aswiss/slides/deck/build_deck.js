'use strict';
// ASWISS presentation (editable PPTX). Usage: node build_deck.js OUT.pptx PLOTDATA_DIR
// Numbers come from the original Beamer deck (aswiss_slides.tex) unless a slide says otherwise.
const fs = require('fs');
const path = require('path');
const pptxgen = require('pptxgenjs');
const fa = require('react-icons/fa6');
const { W, ML, CW, P, THEME, rich, mkHelpers, tree, icon } = require('./lib');
const { postprocess } = require('./post');

const OUT = process.argv[2] || path.join(__dirname, '..', 'aswiss_deck.pptx');
const DATA = process.argv[3] || path.join(__dirname, '..', 'plotdata');

const pres = new pptxgen();
pres.layout = 'LAYOUT_WIDE';
pres.title = 'ASWISS: Abstraction Split WIth Sampling by Strategy';
pres.subject = '학습한 전략의 후회로 추상화를 나누는 방법';
pres.author = 'ASWISS 졸업 논문 연구';
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const h = mkHelpers(pres);
const { T, box, line, arrow, badge, chip } = h;
const S = pres.shapes;

// ---------------------------------------------------------------- layouts
const TITLE_PH = { name: 'title', type: 'title', x: ML, y: 0.34, w: 9.45, h: 0.78, fontSize: 28, bold: true, color: P.ink, valign: 'middle', align: 'left', margin: 0 };
pres.defineSlideMaster({
  title: 'CONTENT', background: { color: P.white }, margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: TITLE_PH, text: '' } },
    { text: { text: 'ASWISS', options: { x: ML, y: 7.02, w: 3, h: 0.3, fontSize: 11, color: P.km, margin: 0 } } },
  ],
  slideNumber: { x: 12.13, y: 7.02, w: 0.6, h: 0.3, fontSize: 11, color: P.km, align: 'right', margin: 0 },
});
pres.defineSlideMaster({
  title: 'APPENDIX', background: { color: P.white }, margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: TITLE_PH, text: '' } },
    { text: { text: 'ASWISS · 부록', options: { x: ML, y: 7.02, w: 3, h: 0.3, fontSize: 11, color: P.km, margin: 0 } } },
  ],
  slideNumber: { x: 12.13, y: 7.02, w: 0.6, h: 0.3, fontSize: 11, color: P.km, align: 'right', margin: 0 },
});
pres.defineSlideMaster({
  title: 'TITLE', background: { color: P.ink },
  objects: [
    { placeholder: { options: { name: 'title', type: 'title', x: 0.8, y: 1.55, w: 7.4, h: 1.15, fontSize: 60, bold: true, color: P.white, valign: 'middle', align: 'left', margin: 0 }, text: '' } },
    { placeholder: { options: { name: 'body', type: 'body', x: 0.8, y: 2.72, w: 7.4, h: 0.5, fontSize: 22, color: P.lblue, valign: 'middle', align: 'left', margin: 0 }, text: '' } },
  ],
});
pres.defineSlideMaster({
  title: 'DARK', background: { color: P.ink },
  objects: [{ placeholder: { options: { ...TITLE_PH, color: P.white }, text: '' } }],
  slideNumber: { x: 12.13, y: 7.02, w: 0.6, h: 0.3, fontSize: 11, color: P.lkm, align: 'right', margin: 0 },
});

let curSection = null;
function section(title) { pres.addSection({ title }); curSection = title; }
function content(title, chipText, master = 'CONTENT') {
  const s = pres.addSlide({ masterName: master, sectionTitle: curSection });
  s.addText(title, { placeholder: 'title' });
  if (chipText) chip(s, chipText, { x: 10.25, y: 0.5, w: 2.48, h: 0.46, size: 14, fill: master === 'DARK' ? '2C3A4F' : P.soft, color: master === 'DARK' ? P.white : P.ink, name: 'question tag' });
  return s;
}
function banner(s, text) {
  T(s, text, { x: ML, y: 1.3, w: CW, h: 0.52, size: 20, bold: false, color: P.ink, fill: P.pale, rect: 0.08, valign: 'middle', margin: [0, 12, 0, 12], name: 'research question' });
}
function caption(s, text, y = 6.55, o = {}) {
  T(s, text, { x: o.x || ML, y, w: o.w || CW, h: o.h || 0.36, size: o.size || 14, color: P.muted, valign: 'top', name: 'caption' });
}
// small table helper; rows of markup strings, first row = header
function table(s, rows, o) {
  const data = rows.map((r, ri) => r.map((c, ci) => ({
    text: rich(c, { fontSize: o.size || 18, color: P.ink, lang: 'ko-KR', bold: ri === 0 || (o.boldCol === ci) }),
    options: {
      fill: { color: ri === 0 ? P.soft : (o.rowFill && o.rowFill[ri]) || P.white },
      align: (o.aligns && o.aligns[ci]) || 'left', valign: 'middle', margin: [0.04, 0.09, 0.04, 0.09],
    },
  })));
  s.addTable(data, { x: o.x, y: o.y, w: o.w, colW: o.colW, rowH: o.rowH, border: { type: 'solid', pt: 0.75, color: P.line }, fontSize: o.size || 18, color: P.ink, objectName: o.name });
}
// stat callout: big number + caption
function stat(s, big, small, o) {
  T(s, big, { x: o.x, y: o.y, w: o.w, h: o.bh || 0.7, size: o.size || 36, bold: true, color: o.color || P.ink, valign: 'bottom' });
  T(s, small, { x: o.x, y: o.y + (o.bh || 0.7) + 0.04, w: o.w, h: o.sh || 0.8, size: o.ssize || 18, color: P.muted });
}
// icon helper (async preloaded)
const ICON = {};

// ---------------------------------------------------------------- chart helpers
const chartText = { catAxisLabelFontFace: '+mn-lt', valAxisLabelFontFace: '+mn-lt', dataLabelFontFace: '+mn-lt', legendFontFace: '+mn-lt', titleFontFace: '+mn-lt', catAxisTitleFontFace: '+mn-lt', valAxisTitleFontFace: '+mn-lt' };
function hbar(s, name, labels, values, colors, o) {
  s.addChart(pres.charts.BAR, [{ name, labels, values }], {
    x: o.x, y: o.y, w: o.w, h: o.h, barDir: 'bar', chartColors: colors, catAxisOrientation: 'maxMin',
    valAxisHidden: true, valAxisMinVal: 0, valAxisMaxVal: o.max, valGridLine: { style: 'none' }, catGridLine: { style: 'none' },
    catAxisLineShow: false, valAxisLineShow: false,
    showValue: true, dataLabelPosition: 'outEnd', dataLabelFormatCode: o.fmt || '0.000', dataLabelFontSize: o.dsize || 16, dataLabelColor: P.ink,
    catAxisLabelFontSize: o.csize || 16, catAxisLabelColor: P.ink, barGapWidthPct: o.gap || 45, showLegend: false,
    objectName: o.objectName, ...chartText,
  });
}
function readDat(file) {
  const lines = fs.readFileSync(file, 'utf8').trim().split('\n');
  const head = lines[0].trim().split(/\s+/);
  const rows = lines.slice(1).map((l) => l.trim().split(/\s+/).map(Number));
  const col = (k) => rows.map((r) => r[head.indexOf(k)]);
  return { col, n: rows.length };
}
function logLine(s, series, colors, o) {
  s.addChart(pres.charts.LINE, series, {
    x: o.x, y: o.y, w: o.w, h: o.h, chartColors: colors, lineSize: 2, lineDataSymbol: 'none',
    valAxisLogScaleBase: 10, valAxisMinVal: o.min, valAxisMaxVal: o.max, valAxisLabelFormatCode: '[<1]0.######;0',
    valAxisLabelFontSize: 13, catAxisLabelFontSize: 13, valAxisLabelColor: P.muted, catAxisLabelColor: P.muted,
    catAxisLabelFrequency: 1, catAxisLabelRotate: 0, valGridLine: { color: 'E3E6EA', size: 0.75 }, catGridLine: { style: 'none' },
    showValAxisTitle: true, valAxisTitle: o.ytitle, valAxisTitleFontSize: 14, valAxisTitleColor: P.muted,
    showCatAxisTitle: true, catAxisTitle: o.xtitle, catAxisTitleFontSize: 14, catAxisTitleColor: P.muted,
    showLegend: true, legendPos: o.legendPos || 'b', legendFontSize: 13, legendColor: P.ink,
    objectName: o.objectName, ...chartText,
  });
}
const fmtIters = (v) => v.toLocaleString('en-US');

// ================================================================= slides
async function build() {
  ICON.check = await icon(fa.FaCircleCheck, P.ink);
  ICON.open = await icon(fa.FaCircleExclamation, P.muted);
  ICON.xmark = await icon(fa.FaCircleXmark, P.km);
  ICON.flag = await icon(fa.FaFlagCheckered, P.ink);
  ICON.arrowR = await icon(fa.FaArrowRight, P.muted);

  // ------------------------------------------------------------ 1. title
  section('도입');
  {
    const s = pres.addSlide({ masterName: 'TITLE', sectionTitle: curSection });
    s.addText('ASWISS', { placeholder: 'title' });
    s.addText('Abstraction Split WIth Sampling by Strategy', { placeholder: 'body' });
    T(s, '학습한 전략의 후회로\n추상화를 스스로 나누는 방법', { x: 0.8, y: 3.45, w: 7.4, h: 1.3, size: 30, color: P.white, bold: true, lineSp: 1.05 });
    T(s, '졸업 논문 연구 · 2인 영합 불완전 정보 게임의 적응형 추상화 · 2026년 11월', { x: 0.8, y: 6.35, w: 9, h: 0.4, size: 16, color: P.lkm });
    // motif: a bucket splits (blue), and a later bucket sends an orange arrow back up
    const bx = 8.85, bw = 3.6;
    box(s, { x: bx, y: 1.55, w: bw, h: 0.95, line: P.lkm, lw: 1.5, radius: 0.12, fill: '243145', name: 'motif bucket' });
    [0, 1, 2, 3].forEach((i) => s.addShape(S.OVAL, { x: bx + 0.45 + i * 0.78, y: 1.82, w: 0.42, h: 0.42, fill: { color: i < 2 ? P.white : '243145' }, line: { color: P.white, width: 1.5 } }));
    arrow(s, bx + 1.0, 2.55, bx + 0.6, 3.35, { color: P.blue, width: 3 });
    arrow(s, bx + 2.6, 2.55, bx + 3.0, 3.35, { color: P.blue, width: 3 });
    T(s, '정방향 분할', { x: bx + 1.05, y: 2.75, w: 1.5, h: 0.4, size: 15, color: P.lblue, align: 'center' });
    box(s, { x: bx - 0.15, y: 3.4, w: 1.65, h: 0.85, line: P.blue, lw: 2, radius: 0.12, fill: '243145' });
    box(s, { x: bx + 2.1, y: 3.4, w: 1.65, h: 0.85, line: P.blue, lw: 2, radius: 0.12, fill: '243145' });
    [0, 1].forEach((i) => s.addShape(S.OVAL, { x: bx + 0.12 + i * 0.7, y: 3.62, w: 0.42, h: 0.42, fill: { color: P.white }, line: { color: P.white, width: 1.5 } }));
    [0, 1].forEach((i) => s.addShape(S.OVAL, { x: bx + 2.37 + i * 0.7, y: 3.62, w: 0.42, h: 0.42, fill: { color: '243145' }, line: { color: P.white, width: 1.5 } }));
    box(s, { x: bx + 0.6, y: 5.0, w: 2.4, h: 0.85, line: P.lkm, lw: 1.5, radius: 0.12, fill: '243145' });
    box(s, { x: bx + 0.78, y: 5.13, w: 2.04, h: 0.59, line: P.lkm, lw: 1.25, dash: 'dash', radius: 0.05 });
    [0, 1].forEach((i) => s.addShape(S.OVAL, { x: bx + 1.15 + i * 0.85, y: 5.22, w: 0.42, h: 0.42, fill: { color: i === 0 ? P.white : '243145' }, line: { color: P.white, width: 1.5 } }));
    line(s, bx + 3.05, 5.42, bx + 3.95, 5.42, { color: P.orange, width: 3 });
    line(s, bx + 3.95, 5.42, bx + 3.95, 2.02, { color: P.orange, width: 3 });
    arrow(s, bx + 3.95, 2.02, bx + 3.65, 2.02, { color: P.orange, width: 3 });
    T(s, '역방향 분할', { x: bx + 2.05, y: 4.45, w: 1.8, h: 0.4, size: 15, color: P.lorange, align: 'right' });
    s.addNotes([
      '오늘 발표할 연구는 ASWISS, Abstraction Split WIth Sampling by Strategy입니다.',
      '한 문장으로 말하면, 미리 정한 특징으로 게임을 묶어 두는 대신, 학습한 전략이 "여기는 서로 다르게 두고 싶다"고 말하는 곳만 나누어 가는 추상화 방법입니다.',
      '오른쪽 그림이 발표 전체의 약속입니다. 파란색은 정방향 분할, 주황색은 역방향 분할입니다. 정방향은 한 묶음 안에서 갈리는 것을 그 자리에서 나누고, 역방향은 뒤에서 발견한 차이를 앞 결정으로 거슬러 올라가 나눕니다.',
      '발표는 여덟 가지 질문에 차례로 답하는 순서로 갑니다. 각 슬라이드 오른쪽 위의 표시가 지금 어느 질문에 답하고 있는지 알려 줍니다.',
    ].join('\n'));
  }

  // ------------------------------------------------------------ 2. information sets
  {
    const s = content('불완전 정보 게임과 정보집합', '질문 1 · 정보집합이란?');
    tree(s, h, { cx: 3.4, oy: 1.62, s: 0.95, jpill: 'strong', jlabel: 'J' });
    T(s, '플레이어 2는 카드를 모른다:\n두 상태가 하나의 정보집합 J', { x: 0.55, y: 6.18, w: 6.0, h: 0.75, size: 16, color: P.ink, align: 'center' });
    const items = [
      '**게임 트리** · 카드 분배(우연)와 두 플레이어의 행동. 끝의 수는 플레이어 1의 보수(칩)',
      '**보는 것 / 못 보는 것** · 플레이어 1(●)은 자기 카드를 본다. 플레이어 2(○)는 행동만 본다',
      '**정보집합 (information set)** · 그 플레이어가 서로 구별할 수 없는 상태들의 묶음',
      '**전략 (strategy)** · 정보집합마다 행동 확률 하나. J의 두 상태에서는 같은 확률로 둔다',
      '**내시 균형 (Nash equilibrium)** · 누구도 혼자 전략을 바꿔 이득을 볼 수 없는 상태',
      '**exploitability** · 상대가 내 전략을 알고 최선으로 대응할 때 균형보다 더 잃는 양. 0이면 균형',
    ];
    T(s, items.join('\n'), { x: 6.95, y: 1.4, w: 5.78, h: 5.15, size: 20, spaceAfter: 8, lineSp: 1.0 });
    caption(s, '정의는 일반 이론의 보충 설명이다.', 6.62, { x: 6.95, w: 5.78 });
    s.addNotes([
      '먼저 불완전 정보 게임이 무엇인지 이 작은 포커로 보겠습니다. 이 게임은 뒤에서 ASWISS의 동작을 설명할 때도 계속 쓰는 예시입니다.',
      '맨 위 회색 점은 우연, 즉 카드 분배입니다. 플레이어 1이 H 또는 L 카드를 절반씩 받습니다. 검은 원은 플레이어 1, 흰 원은 플레이어 2의 차례이고, 맨 아래 숫자는 플레이어 1이 얻는 칩입니다.',
      '핵심은 플레이어 2가 카드를 보지 못한다는 것입니다. 플레이어 1이 H를 들고 체크했는지 L을 들고 체크했는지 구별할 수 없으므로, 점선으로 묶은 두 상태가 플레이어 2에게는 같은 상황입니다. 이렇게 구별할 수 없는 상태의 묶음을 정보집합이라고 합니다.',
      '전략은 상태마다가 아니라 정보집합마다 정합니다. 그래서 J의 두 상태에서 플레이어 2는 같은 확률로 체크하거나 베팅해야 합니다.',
      '내시 균형은 누구도 혼자 전략을 바꿔서 이득을 볼 수 없는 상태입니다. exploitability는 상대가 내 전략을 알고 최선으로 대응했을 때 균형보다 얼마나 더 잃는지를 재는 값으로, 0이면 균형입니다. 이 값이 이후 실험의 기본 지표입니다.',
      '참고로 이 게임에서 균형은 H면 콜, L이면 폴드, 플레이어 2는 체크이고, 플레이어 1의 값은 0입니다.',
    ].join('\n'));
  }

  // ------------------------------------------------------------ 3. why abstraction
  {
    const s = content('추상화: 묶으면 얻는 것과 잃는 것', '질문 2 · 왜 추상화?');
    T(s, '큰 게임은 정보집합이 너무 많다. 세븐 포커에서는 같은 정보집합이 학습 중 거의 다시 나오지 않아 정보집합마다 표를 둘 수 없다.', { x: ML, y: 1.3, w: CW, h: 0.8, size: 20 });
    // left: one table per information set
    T(s, '정보집합마다 표', { x: ML, y: 2.25, w: 3.3, h: 0.4, size: 18, bold: true, align: 'center' });
    const acts = ['콜', '콜', '콜', '콜', '폴드'];
    acts.forEach((a, i) => {
      const y = 2.75 + i * 0.68;
      s.addText(a, { x: 1.0, y, w: 0.95, h: 0.52, shape: S.OVAL, fill: { color: a === '폴드' ? P.soft : P.white }, line: { color: P.ink, width: 1.5 }, color: P.ink, fontSize: 15, bold: true, align: 'center', valign: 'middle', margin: 0 });
      box(s, { x: 2.2, y: y + 0.06, w: 1.2, h: 0.4, fill: P.white, line: P.km, lw: 1 });
      line(s, 2.6, y + 0.06, 2.6, y + 0.46, { color: P.km, width: 0.75 });
      line(s, 3.0, y + 0.06, 3.0, y + 0.46, { color: P.km, width: 0.75 });
    });
    caption(s, '원 안의 글자: 그 정보집합의 최선 행동', 6.2, { x: ML, w: 3.4 });
    arrow(s, 4.05, 4.25, 5.05, 4.25, { color: P.muted, width: 2.5 });
    T(s, '묶는다', { x: 4.0, y: 3.75, w: 1.1, h: 0.4, size: 16, color: P.muted, align: 'center' });
    // right: two buckets
    const rows = [
      { y: 2.25, acts: ['콜', '콜', '콜'], head: '이득', body: '표 3개가 1개. 한 곳에서 배운 것을 나머지가 함께 쓴다.', icon: ICON.check },
      { y: 4.15, acts: ['콜', '폴드'], head: '손실', body: '둘 다 콜이거나 둘 다 폴드. 한쪽은 최선이 아닌 행동을 한다.', icon: ICON.open },
    ];
    rows.forEach((r, ri) => {
      box(s, { x: 5.2, y: r.y, w: 3.85, h: 1.45, fill: P.soft, line: P.km, lw: 1, radius: 0.12, name: 'bucket ' + (ri ? 'B (mixed)' : 'A (same)') });
      T(s, ri ? 'bucket B' : 'bucket A', { x: 5.35, y: r.y + 0.06, w: 2, h: 0.35, size: 15, bold: true, color: P.ink });
      r.acts.forEach((a, i) => s.addText(a, { x: 5.32 + i * 0.92, y: r.y + 0.6, w: 0.82, h: 0.5, shape: S.OVAL, fill: { color: a === '폴드' ? P.soft : P.white }, line: { color: P.ink, width: 1.5 }, color: P.ink, fontSize: 15, bold: true, align: 'center', valign: 'middle', margin: 0 }));
      box(s, { x: 8.12, y: r.y + 0.65, w: 0.8, h: 0.4, fill: P.white, line: P.ink, lw: 1.25 });
      T(s, '표 1개', { x: 8.02, y: r.y + 1.07, w: 1.0, h: 0.3, size: 13, color: P.muted, align: 'center' });
      s.addImage({ data: r.icon, x: 9.3, y: r.y + 0.12, w: 0.42, h: 0.42 });
      T(s, r.head, { x: 9.85, y: r.y + 0.08, w: 2.85, h: 0.5, size: 22, bold: true, valign: 'middle' });
      T(s, r.body, { x: 9.3, y: r.y + 0.6, w: 3.43, h: 1.35, size: 20, color: P.ink });
    });
    T(s, '**추상화(abstraction)**: 정보집합을 bucket으로 묶는 분할. bucket 안의 정보집합은 한 전략을 함께 쓴다. 무엇을 묶느냐가 성능을 정한다.', { x: ML, y: 6.05, w: CW, h: 0.8, size: 20, fill: P.pale, rect: 0.08, margin: [6, 12, 6, 12], valign: 'middle' });
    s.addNotes([
      '왜 추상화가 필요한지부터 보겠습니다. 실제 크기의 세븐 포커는 정보집합이 너무 많아서, 학습하는 동안 같은 정보집합을 다시 만나는 일이 거의 없습니다. 정보집합마다 따로 표를 두면 표마다 배울 경험이 없습니다.',
      '그래서 비슷한 정보집합을 bucket으로 묶고, 묶음 하나가 전략 표 하나를 함께 씁니다. 이것이 추상화입니다.',
      '위의 bucket A처럼 모두 같은 행동을 원하는 정보집합을 묶으면 손해가 없습니다. 표가 줄고, 한 곳에서 배운 것을 나머지가 함께 씁니다.',
      '그러나 아래 bucket B처럼 콜을 원하는 것과 폴드를 원하는 것을 묶으면, 표가 하나뿐이라 둘 중 하나는 최선이 아닌 행동을 하게 됩니다. 이것이 추상화의 손실입니다.',
      '결국 질문은 무엇을 묶고 무엇을 나눌 것인가입니다. 이 결정을 무엇에 근거해 내리느냐가 다음 슬라이드의 주제입니다.',
    ].join('\n'));
  }

  // ------------------------------------------------------------ 4. fixed abstraction
  {
    const s = content('고정 추상화에 빠져 있는 것', '질문 3 · 고정 추상화의 빈틈');
    const steps = [
      ['카드 특징', '예: 손패 강도'],
      ['k-means 군집 × 베팅 요약', '미리 정한 기준'],
      ['bucket 고정', '학습 내내 그대로'],
      ['CFR로 학습', '추상 게임 안에서'],
    ];
    const bw = 2.55, gap = 0.55, y0 = 1.55;
    steps.forEach(([a, b], i) => {
      const x = ML + i * (bw + gap);
      box(s, { x, y: y0, w: bw, h: 1.15, fill: i === 2 ? P.soft : P.white, line: P.km, lw: 1.25, radius: 0.1 });
      T(s, a, { x: x + 0.1, y: y0 + 0.12, w: bw - 0.2, h: 0.5, size: 19, bold: true, align: 'center', valign: 'middle' });
      T(s, b, { x: x + 0.1, y: y0 + 0.62, w: bw - 0.2, h: 0.4, size: 16, color: P.muted, align: 'center', valign: 'middle' });
      if (i < 3) arrow(s, x + bw + 0.06, y0 + 0.57, x + bw + gap - 0.06, y0 + 0.57, { color: P.muted, width: 2 });
    });
    // broken feedback arrow
    const xa = ML + 3 * (bw + gap) + bw / 2, xb = ML + 2 * (bw + gap) + bw / 2;
    line(s, xa, y0 + 1.2, xa, y0 + 1.75, { color: P.km, width: 2, dash: 'dash' });
    line(s, xa, y0 + 1.75, xb, y0 + 1.75, { color: P.km, width: 2, dash: 'dash' });
    arrow(s, xb, y0 + 1.75, xb, y0 + 1.2, { color: P.km, width: 2, dash: 'dash' });
    s.addImage({ data: ICON.xmark, x: (xa + xb) / 2 - 0.22, y: y0 + 1.53, w: 0.44, h: 0.44 });
    T(s, '학습 결과가 묶음을 고치지 않는다', { x: xb - 1.2, y: y0 + 1.95, w: xa - xb + 2.4, h: 0.4, size: 17, color: P.muted, align: 'center' });
    // three gaps
    const gaps = [
      ['묶는 기준이 해를 보지 않는다', '무엇을 합쳐도 되는지는 게임의 해(균형)가 정한다. 카드 특징은 그것을 모른다.'],
      ['전략적 동등성의 보장이 없다', '같은 bucket이 같은 행동을 원한다는 보장이 없다. 다르면 앞 장의 손실이 생긴다.'],
      ['손실을 찾아 고칠 길이 없다', '"여기는 다르게 두고 싶다"는 신호가 나와도 추상화는 바뀌지 않는다.'],
    ];
    gaps.forEach(([a, b], i) => {
      const x = ML + i * 4.13, w = 3.87;
      box(s, { x, y: 4.2, w, h: 2.15, fill: P.pale, radius: 0.1 });
      badge(s, i + 1, x + 0.2, 4.38, 0.45, P.ink, 16);
      T(s, a, { x: x + 0.78, y: 4.33, w: w - 0.9, h: 0.55, size: 19, bold: true, valign: 'middle' });
      T(s, b, { x: x + 0.2, y: 4.98, w: w - 0.4, h: 1.35, size: 19, color: P.ink });
    });
    caption(s, '보충: 고정 추상화에서 학습기가 노리는 것은 추상 게임의 해다. 그것이 원 게임의 균형이라는 보장은 따로 없다.', 6.55);
    s.addNotes([
      '지금까지 큰 게임에서 쓰던 방식은 고정 추상화입니다. 카드 강도 같은 특징을 k-means로 군집하고, 베팅 상황을 요약해서 bucket을 만든 뒤, 그 bucket을 학습 내내 고정해 두고 CFR로 풉니다.',
      '여기에는 세 가지가 빠져 있습니다. 첫째, 묶는 기준이 게임의 해를 보지 않습니다. 무엇을 합쳐도 되는지는 균형에서 같은 행동을 원하느냐로 정해지는데, 카드 특징은 그것을 모릅니다.',
      '둘째, 그래서 같은 bucket 안의 정보집합들이 같은 행동을 원한다는 보장, 즉 전략적 동등성의 보장이 없습니다. 다르면 앞 장에서 본 손실이 생깁니다.',
      '셋째, 학습 도중에 "여기는 다르게 두고 싶다"는 신호가 나와도 추상화를 고칠 통로가 없습니다. 그림의 끊어진 화살표가 그것입니다.',
      'ASWISS는 바로 이 끊어진 화살표를 잇는 방법입니다.',
    ].join('\n'));
  }

  // ------------------------------------------------------------ 5. core idea
  section('방법');
  {
    const s = content('ASWISS의 핵심 아이디어', '질문 4 · 어디를 나누나');
    T(s, 'ASWISS는 미리 정한 특징이 아니라 **학습된 전략의 반사실 후회**를 이용하여 정보집합의 분할 필요성을 판단하고, {{blue:**정방향**}} 및 {{orange:**역방향**}} **분할**로 추상화에 의해 숨겨진 전략적 차이를 복원하려는 방법이다.', { x: ML, y: 1.3, w: CW, h: 1.35, size: 22, fill: P.pale, rect: 0.1, margin: [8, 16, 8, 16], valign: 'middle', lineSp: 1.1 });
    const flow = [
      ['출발', '가장 거친 추상화 P_{0}: 플레이어·행동집합·결정 순번만 구별'],
      ['학습 · 고정', 'CFR+ 계열로 학습하고 평균 전략 σ̄을 고정'],
      ['라벨', 'σ̄의 반사실 후회로 "바꾸고 싶은 행동"을 구한다'],
    ];
    const y = 3.0, bw = 2.42, gap = 0.36;
    flow.forEach(([a, b], i) => {
      const x = ML + i * (bw + gap);
      box(s, { x, y, w: bw, h: 1.95, fill: P.white, line: P.km, lw: 1.25, radius: 0.1 });
      badge(s, i + 1, x + 0.15, y + 0.15, 0.42, P.ink, 15);
      T(s, a, { x: x + 0.65, y: y + 0.12, w: bw - 0.75, h: 0.5, size: 19, bold: true, valign: 'middle' });
      T(s, b, { x: x + 0.18, y: y + 0.7, w: bw - 0.3, h: 1.2, size: 17, color: P.muted });
      if (i < 2) arrow(s, x + bw + 0.05, y + 0.97, x + bw + gap - 0.05, y + 0.97, { color: P.muted, width: 2 });
    });
    const xs = ML + 3 * (bw + gap) + 0.15;
    arrow(s, xs - gap + 0.02, y + 0.97, xs + 0.15, 3.35, { color: P.blue, width: 2.5 });
    arrow(s, xs - gap + 0.02, y + 0.97, xs + 0.15, 4.95, { color: P.orange, width: 2.5 });
    const cw = W - 0.6 - xs - 0.2;
    T(s, '{{blue:**정방향 분할**}}\n한 bucket 안에서 **bucket 이력끼리** 원하는 행동이 다르면 그 bucket을 나눈다', { x: xs + 0.2, y: 2.62, w: cw, h: 1.48, size: 18, fill: P.tblue, rect: 0.1, margin: [6, 10, 6, 10], valign: 'middle' });
    T(s, '{{orange:**역방향 분할**}}\n한 bucket 이력 안에서 **정보집합끼리** 다르면, 정보가 합쳐진 앞 bucket을 나눈다', { x: xs + 0.2, y: 4.22, w: cw, h: 1.48, size: 18, fill: P.torange, rect: 0.1, margin: [6, 10, 6, 10], valign: 'middle' });
    T(s, '다시 학습하고 판정한다. 아무것도 나뉘지 않으면 멈춘다 (**정지 상태**). 다음 장부터 후회 → 자료구조 → 예시 게임 순서로 한 단계씩 본다.', { x: ML, y: 5.95, w: CW, h: 0.8, size: 20 });
    s.addNotes([
      '이것이 이 연구의 핵심 문장입니다. ASWISS는 미리 정한 특징이 아니라, 학습된 전략의 반사실 후회를 이용해서 어떤 정보집합들을 나눠야 하는지 판단하고, 정방향과 역방향 분할로 추상화가 숨긴 전략적 차이를 되살리려는 방법입니다.',
      '절차는 단순합니다. 가장 거친 추상화에서 시작해 학습하고, 그 시점의 평균 전략을 고정합니다. 그 전략 아래에서 후회를 계산해 각 묶음이 "어느 행동으로 바꾸고 싶은지"를 라벨로 붙입니다.',
      '라벨이 갈리는 곳이 분할이 필요한 곳입니다. 갈림이 bucket 안의 이력들 사이에서 보이면 파란색 정방향 분할로 그 bucket을 나누고, 한 이력 안의 정보집합들 사이에 숨어 있으면 주황색 역방향 분할로 정보가 합쳐진 앞 bucket을 나눕니다.',
      '이 문장에 나온 반사실 후회, bucket 이력, 정방향과 역방향을 다음 장부터 하나씩, 같은 예시 게임 위에서 설명하겠습니다.',
    ].join('\n'));
  }

  // ------------------------------------------------------------ 6. regret (CFR intuition)
  {
    const s = content('후회: 그 행동을 했더라면 얼마나 나았을까', '질문 4 · 어디를 나누나');
    T(s, '예시 게임의 결정 2 (체크 → 상대 베팅 → 콜 또는 폴드). 지금 전략은 "항상 콜"이다.', { x: ML, y: 1.3, w: 7.0, h: 0.75, size: 20 });
    table(s, [
      ['카드', '콜의 값', '폴드의 값', '폴드의 후회', '× ½'],
      ['H', '+2', '−1', '−1 − (+2) = −3', '**−1.5**'],
      ['L', '−3', '−1', '−1 − (−3) = **+2**', '**+1.0**'],
    ], { x: ML, y: 2.2, w: 7.0, colW: [0.95, 1.2, 1.3, 2.35, 1.2], rowH: 0.52, size: 19, aligns: ['center', 'center', 'center', 'center', 'center'] });
    T(s, '**후회 (regret)** = 그 행동의 값 − 지금 전략의 값. 양수면 "바꾸고 싶다".\n**× ½** = 상대와 우연이 여기로 데려올 확률 (카드 ½ × 상대 베팅 1). 남의 확률만 곱하므로 **반사실(counterfactual)** 값이라 한다.', { x: ML, y: 3.9, w: 7.0, h: 2.05, size: 20, spaceAfter: 8 });
    // CFR loop (right)
    const lx = 8.05, lw = 4.68;
    T(s, '**CFR**: 후회를 줄여 가는 반복', { x: lx, y: 1.3, w: lw, h: 0.5, size: 20 });
    const loop = ['지금 전략 σ^{t}로 끝까지 계산', '정보집합·행동마다 반사실 후회를 누적', '누적 후회가 양수인 행동을 그 비율로 둔다 (regret matching)', '반복한 전략들의 평균 σ̄이 균형으로 간다'];
    loop.forEach((t, i) => {
      const y = 1.95 + i * 0.98;
      badge(s, i + 1, lx, y + 0.12, 0.44, i === 3 ? P.ink : P.muted, 15);
      T(s, t, { x: lx + 0.6, y, w: lw - 0.6, h: 0.8, size: 19, valign: 'middle', bold: i === 3 });
      if (i < 2) arrow(s, lx + 0.22, y + 0.6, lx + 0.22, y + 1.06, { color: P.km, width: 1.5 });
    });
    line(s, lx - 0.15, 2.07 + 2 * 0.98 + 0.25, lx - 0.15, 2.32, { color: P.km, width: 1.5 });
    arrow(s, lx - 0.15, 2.32, lx - 0.02, 2.32, { color: P.km, width: 1.5 });
    T(s, '보충 · 왜 평균인가: 지금 전략은 서로 쫓고 쫓기며 진동할 수 있다. 2인 영합·완전 기억 게임에서는 평균의 수렴이 보장된다.', { x: lx, y: 5.9, w: lw, h: 0.95, size: 15, color: P.muted });
    T(s, 'r(I, a) = q(I, a) − Σ_{b} σ(I, b) q(I, b)          q(I, a) = Σ_{h∈I} π_{−i}(h) · (h에서 a를 둔 뒤의 기대 보수)', { x: ML, y: 6.05, w: 7.3, h: 0.75, size: 15, color: P.ink, fill: P.pale, rect: 0.06, margin: [4, 10, 4, 10], valign: 'middle' });
    s.addNotes([
      '분할을 판단하는 재료는 후회입니다. 후회는 "그 행동을 했더라면 얼마나 나았을까"입니다.',
      '예시 게임의 두 번째 결정을 보겠습니다. 플레이어 1이 체크했고 상대가 베팅했습니다. 지금 전략은 항상 콜입니다. H를 들고 있으면 콜이 +2, 폴드가 -1이라 폴드의 후회는 -3입니다. L을 들고 있으면 콜이 -3, 폴드가 -1이라 폴드의 후회는 +2, 즉 폴드로 바꾸고 싶습니다.',
      '여기에 상대와 우연이 이 상태로 데려올 확률을 곱합니다. 카드가 나올 확률 1/2, 상대가 베팅할 확률 1입니다. 내가 여기까지 오려 했다고 가정하고 남의 확률만 곱하기 때문에 반사실 값이라고 부릅니다. 그래서 H는 -1.5, L은 +1.0입니다. 이 두 숫자를 잘 기억해 주십시오. 세 장 뒤에 다시 나옵니다.',
      'CFR은 이 후회를 반복해서 누적하고, 누적 후회가 양수인 행동을 그 크기에 비례해 두는 방식으로 전략을 고칩니다. 지금 전략 자체는 진동할 수 있지만, 반복한 전략들의 평균은 2인 영합 완전 기억 게임에서 균형으로 수렴한다는 것이 알려져 있습니다. 그래서 우리가 내놓는 정책도, 분할 판정의 기준도 평균 전략 σ̄입니다.',
      '아래 식은 지금 말한 것을 기호로 쓴 것뿐입니다. 자세한 정의는 부록 A1에 있습니다.',
    ].join('\n'));
  }

  // ------------------------------------------------------------ 7. bucket & bucket history (tree step 0)
  const TREE_POS = { cx: 3.35, oy: 1.62, s: 0.95 };
  const B1 = { nodes: ['d1H', 'd1L'], label: 'B_{1}' }, B2 = { nodes: ['d2H', 'd2L'], label: 'B_{2}' };
  const K1H = { nodes: ['d1H'], tag: 'K^{1}_{H}', tagSide: 'right', tagW: 0.7 }, K1L = { nodes: ['d1L'], tag: 'K^{1}_{L}', tagSide: 'left', tagW: 0.7 };
  const K2 = { nodes: ['d2H', 'd2L'], tag: 'K^{2}', tagW: 0.6 };
  {
    const s = content('bucket과 bucket 이력', '질문 4 · 어디를 나누나');
    tree(s, h, { ...TREE_POS, jpill: 'plain', buckets: [B1, B2], hists: [K1H, K1L, K2] });
    const lx = 6.85, lw = 5.88;
    // legend
    s.addText('1', { x: lx + 0.17, y: 1.42, w: 0.42, h: 0.42, shape: S.OVAL, fill: { color: P.ink }, line: { type: 'none' }, color: P.white, bold: true, fontSize: 14, align: 'center', valign: 'middle', margin: 0 });
    T(s, '**정보집합 I** · 게임이 정한 구별 단위 (여기서는 플레이어 1의 결정점 하나하나)', { x: lx + 0.85, y: 1.35, w: lw - 0.85, h: 0.75, size: 18, valign: 'middle' });
    box(s, { x: lx, y: 2.3, w: 0.75, h: 0.5, fill: P.soft, line: P.km, lw: 1, radius: 0.08 });
    T(s, '**bucket B** · 전략을 함께 쓰는 정보집합 묶음 = 추상 게임의 정보집합', { x: lx + 0.85, y: 2.18, w: lw - 0.85, h: 0.75, size: 18, valign: 'middle' });
    box(s, { x: lx, y: 3.13, w: 0.75, h: 0.5, line: P.ink, lw: 1.5, dash: 'dash', radius: 0.05 });
    T(s, '**bucket 이력 K** · (지나온 bucket, 둔 행동, 새 관측). 과거는 bucket 번호로만 기억한다', { x: lx + 0.85, y: 3.01, w: lw - 0.85, h: 0.75, size: 18, valign: 'middle' });
    T(s, '**σ̄** · bucket마다 하나인 평균 전략 (판정 때 고정)\n**λ(K)** · K에 모은 후회가 가장 큰 행동 = 라벨', { x: lx, y: 3.9, w: lw, h: 0.85, size: 18 });
    T(s, 'B_{1}에는 이력이 둘: K^{1}_{H} = (카드 H), K^{1}_{L} = (카드 L)\nB_{2}에는 이력이 하나: K^{2} = (B_{1}, 체크, 상대 베팅)\n→ 카드는 B_{1}에 합쳐져 B_{2}에서는 H와 L이 구별되지 않는다', { x: lx, y: 4.85, w: lw, h: 1.55, size: 19, fill: P.pale, rect: 0.08, margin: [6, 10, 6, 10], valign: 'middle' });
    caption(s, '출발점 P_{0}: (플레이어, 행동집합, 자기 결정 순번)이 같은 정보집합을 하나로 묶는 가장 거친 분할', 6.5, { x: lx, w: lw, h: 0.45 });
    s.addNotes([
      '이제 ASWISS가 쓰는 자료구조를 같은 게임 위에서 보겠습니다. 세 가지 기호를 구별해 주십시오.',
      '검은 원 하나하나가 플레이어 1의 정보집합입니다. 회색으로 칠한 상자가 bucket, 즉 전략 표를 함께 쓰는 묶음입니다. 점선 상자가 bucket 이력입니다.',
      '처음에는 가장 거친 분할 P0에서 시작합니다. 같은 플레이어, 같은 행동집합, 같은 결정 순번이면 하나로 묶습니다. 그래서 첫 결정의 H와 L이 B1 하나로, 두 번째 결정의 H와 L이 B2 하나로 묶입니다.',
      'bucket 이력은 그 정보집합에 오기까지 지나온 bucket 번호, 둔 행동, 그리고 새로 관측한 것으로 만든 열쇠입니다. 첫 결정에서는 카드가 새 관측이라 H와 L의 이력이 다릅니다. 그래서 B1 안에는 이력이 두 개 있습니다.',
      '그런데 두 번째 결정에서는 과거를 bucket 번호로만 기억합니다. 지나온 것은 B1, 체크, 상대 베팅으로 H와 L이 똑같습니다. 카드는 이미 B1에 합쳐졌기 때문입니다. 그래서 B2 안에는 이력이 하나뿐이고, 그 안에 H와 L 두 정보집합이 들어 있습니다.',
      '정리하면, 전략은 bucket 단위로 공유하고, 라벨은 bucket 이력 단위로 모읍니다. 한 bucket 안에 서로 다른 행동을 원하는 이력이 있을 수도 있고, 한 이력 안에 서로 다른 행동을 원하는 정보집합이 있을 수도 있습니다. 앞의 것을 정방향이, 뒤의 것을 역방향이 다룹니다.',
    ].join('\n'));
  }

  // ------------------------------------------------------------ 8. forward split (tree step 1)
  {
    const s = content('정방향 분할: 이력끼리 라벨이 갈리면 나눈다', '질문 4 · 어디를 나누나');
    tree(s, h, { ...TREE_POS, jpill: 'plain', buckets: [B1, B2], hists: [{ ...K1H, tag: 'λ = 체크', tagW: 0.95, tagColor: P.dblue, tagBold: true }, { ...K1L, tag: 'λ = 체크', tagW: 0.95, tagColor: P.dblue, tagBold: true }, K2] });
    chip(s, '공통 라벨 → 유지', { x: 5.55, y: 2.18, w: 1.55, h: 0.4, size: 13, fill: P.tblue, color: P.dblue, bold: true });
    const lx = 7.25, lw = 5.48;
    T(s, '**규칙** · 한 bucket 안의 이력들이 공통 라벨을 하나도 갖지 않으면 라벨별로 자식 bucket을 만든다.', { x: lx, y: 1.3, w: lw, h: 1.0, size: 20 });
    T(s, '라벨이 갈리는 경우 (설명용 예시 값)', { x: lx, y: 2.32, w: lw, h: 0.35, size: 15, color: P.muted });
    table(s, [
      ['이력', '폴드', '콜', '레이즈', '라벨'],
      ['K_{1}', '−0.40', '+0.70', '+0.10', '{{blue:콜}}'],
      ['K_{2}', '−0.20', '+0.35', '−0.05', '{{blue:콜}}'],
      ['K_{3}', '+0.55', '−0.30', '−0.60', '{{blue:폴드}}'],
    ], { x: lx, y: 2.7, w: lw, colW: [0.9, 1.1, 1.1, 1.18, 1.2], rowH: 0.36, size: 16, aligns: ['center', 'center', 'center', 'center', 'center'] });
    // split figure
    const fy = 4.38;
    box(s, { x: lx, y: fy, w: 1.75, h: 1.5, fill: P.soft, line: P.km, lw: 1, radius: 0.1 });
    T(s, 'B', { x: lx + 0.1, y: fy + 0.04, w: 0.5, h: 0.32, size: 15, bold: true });
    ['K_{1} 콜', 'K_{2} 콜', 'K_{3} 폴드'].forEach((t, i) => T(s, t, { x: lx + 0.18, y: fy + 0.38 + i * 0.36, w: 1.4, h: 0.32, size: 14, align: 'center', valign: 'middle', line: { color: P.ink, width: 1, dashType: 'dash' } }));
    arrow(s, lx + 1.85, fy + 0.75, lx + 2.55, fy + 0.35, { color: P.blue, width: 2.5 });
    arrow(s, lx + 1.85, fy + 0.75, lx + 2.55, fy + 1.2, { color: P.blue, width: 2.5 });
    T(s, '**자식 1 (콜)** = {K_{1}, K_{2}}', { x: lx + 2.65, y: fy + 0.05, w: 2.75, h: 0.55, size: 16, fill: P.tblue, rect: 0.08, valign: 'middle', margin: [0, 8, 0, 8] });
    T(s, '**자식 2 (폴드)** = {K_{3}}', { x: lx + 2.65, y: fy + 0.9, w: 2.75, h: 0.55, size: 16, fill: P.tblue, rect: 0.08, valign: 'middle', margin: [0, 8, 0, 8] });
    T(s, '같은 이력은 절대 갈라지지 않는다 · 자식은 표를 새로 시작한다', { x: lx, y: 5.98, w: lw, h: 0.4, size: 15, color: P.muted });
    T(s, '왼쪽 게임: B_{1}의 두 이력 모두 "체크" → 유지 (옳다)', { x: lx, y: 6.38, w: lw, h: 0.45, size: 17, bold: true, color: P.dblue });
    s.addNotes([
      '첫 번째 분할 규칙인 정방향 분할입니다. 평균 전략을 고정하고, bucket 이력마다 후회를 모아 라벨, 즉 가장 바꾸고 싶은 행동을 붙입니다.',
      '오른쪽 표는 설명용 예시입니다. 한 bucket B 안에 이력이 세 개 있는데 K1과 K2는 콜을, K3는 폴드를 원합니다. 셋이 함께 원하는 라벨이 없으므로 B를 라벨별로 나눕니다. 콜을 원하는 K1, K2가 한 자식으로, 폴드를 원하는 K3가 다른 자식으로 갑니다.',
      '두 가지를 지킵니다. 같은 이력은 절대 둘로 갈라지지 않고 늘 한 bucket에 있습니다. 그리고 새 자식 bucket은 표를 새로 시작합니다.',
      '왼쪽 예시 게임에 적용해 보겠습니다. 첫 결정의 B1에는 이력이 H와 L 두 개입니다. 베팅하면 -10이라 두 이력 모두 체크가 최선이고, 라벨이 같으므로 나누지 않습니다. 첫 결정만 보면 이것은 옳은 판단입니다.',
      '문제는 두 번째 결정입니다. 다음 장에서 보겠습니다.',
    ].join('\n'));
  }

  // ------------------------------------------------------------ 9. forward failure (tree step 2)
  {
    const s = content('정방향 분할만으로는 멈출 수 있다 (명제 H4)', '질문 5 · 정방향의 한계');
    const t = tree(s, h, { ...TREE_POS, jpill: 'plain', buckets: [B1, B2], hists: [K1H, K1L, { ...K2, tag: 'λ = 콜', tagW: 0.7, tagBold: true }] });
    // fold regrets under the fold leaves
    chip(s, '후회 −1.5', { x: t.X('foldH') - 0.62, y: t.Y('foldH') + 0.26, w: 1.05, h: 0.38, size: 14, fill: P.soft, color: P.ink, bold: true });
    chip(s, '후회 +1.0', { x: t.X('foldL') - 0.43, y: t.Y('foldL') + 0.26, w: 1.05, h: 0.38, size: 14, fill: P.torange, color: P.dorange, bold: true });
    // right panel: cancellation bars
    const lx = 7.15, lw = 5.58;
    T(s, '결정 2에서 폴드의 반사실 후회 (앞의 표)', { x: lx, y: 1.32, w: lw, h: 0.4, size: 17, color: P.muted });
    const zx = lx + 3.4, u = 1.05, rows = [['카드 H', -1.5, P.km], ['카드 L', 1.0, P.orange], ['K^{2} 합', -0.5, P.ink]];
    rows.forEach(([lab, v, c], i) => {
      const y = 1.85 + i * 0.62;
      T(s, lab, { x: lx, y, w: 1.0, h: 0.46, size: 17, bold: i === 2, valign: 'middle' });
      box(s, { x: v < 0 ? zx + v * u : zx, y: y + 0.06, w: Math.abs(v) * u, h: 0.34, fill: c });
      T(s, (v > 0 ? '+' : '−') + Math.abs(v).toFixed(1), { x: v < 0 ? zx + v * u - 0.68 : zx + v * u + 0.08, y, w: 0.62, h: 0.46, size: 17, bold: true, align: v < 0 ? 'right' : 'left', valign: 'middle' });
    });
    line(s, zx, 1.8, zx, 3.72, { color: P.ink, width: 1 });
    T(s, 'L은 폴드를 원하지만(+1.0) 같은 이력의 H와 더해져 사라진다. 라벨은 "콜"이고 B_{2}에는 이력이 하나뿐이라 나눌 것이 없다 → **정지**', { x: lx, y: 3.72, w: lw, h: 1.15, size: 19 });
    T(s, '그런데 L이 폴드로 바꾸면 1을 더 얻는다\n→ **NashConv = 1**\n{{muted:NashConv = 각 플레이어가 혼자 바꿔 얻을 수 있는 이득의 합}}', { x: lx, y: 4.92, w: lw, h: 1.25, size: 18, fill: P.torange, rect: 0.08, margin: [5, 10, 5, 10], valign: 'middle' });
    T(s, 'Leduc에서도 정방향만 (+ 트리 유지) 쓰면 P_{0}에서 한 번도 나누지 못하고 exploitability 2.93에 머물렀다.', { x: lx, y: 6.24, w: lw, h: 0.7, size: 16, color: P.muted });
    s.addNotes([
      '정방향 분할만으로는 멈출 수 있다는 것이 명제 H4입니다. 같은 게임의 두 번째 결정을 보겠습니다.',
      '앞에서 계산한 폴드의 반사실 후회가 여기 다시 나옵니다. H는 -1.5, L은 +1.0입니다. L은 폴드로 바꾸고 싶어 합니다.',
      '그런데 두 번째 결정에서는 H와 L이 같은 bucket 이력 K2에 들어 있습니다. 라벨은 이력 단위로 후회를 더해서 정하므로 -1.5 더하기 +1.0, 즉 -0.5가 되고, 라벨은 콜입니다. L의 양의 후회가 H의 음의 후회에 상쇄되어 보이지 않습니다.',
      'B2에는 이력이 하나뿐이니 정방향 분할은 나눌 것이 없습니다. 첫 결정의 B1은 앞 장에서 본 것처럼 나눌 이유가 없습니다. 그래서 알고리즘은 여기서 멈춥니다.',
      '하지만 이 상태는 균형이 아닙니다. L을 든 플레이어 1이 폴드로 바꾸면 1을 더 얻으므로 NashConv가 1입니다. 정확한 학습기와 정확한 라벨을 써도 이렇게 멈추는 게임이 있다는 것이 명제 H4입니다.',
      '이것은 작은 예시만의 문제가 아니었습니다. Leduc 홀덤에서 정방향만 쓰고 트리 구조를 유지하면 가장 거친 분할에서 한 번도 나누지 못하고 exploitability 2.93에 머물렀습니다.',
    ].join('\n'));
  }

  // ------------------------------------------------------------ 10. backward split (tree step 3)
  {
    const s = content('역방향 분할: 정보가 합쳐진 앞 bucket을 나눈다', '질문 6 · 역방향의 해결');
    const pos = { cx: 3.4, oy: 1.62, s: 0.95 };
    const t = tree(s, h, {
      ...pos, jpill: 'plain',
      buckets: [{ nodes: ['d1H'], label: 'B^{H}_{1}', kind: 'orange' }, { nodes: ['d1L'], label: 'B^{L}_{1}', kind: 'orange', labelSide: 'left' }, { nodes: ['d2H', 'd2L'] }],
      hists: [{ nodes: ['d2H'], padx: 0.27 }, { nodes: ['d2L'], padx: 0.27 }],
    });
    // labels of the two information sets in K2
    chip(s, 'λ = 콜', { x: (t.X('callH') + t.X('foldH')) / 2 - 0.55, y: t.Y('callH') + 0.27, w: 1.1, h: 0.38, size: 14, fill: P.tblue, color: P.dblue, bold: true });
    chip(s, 'λ = 폴드', { x: (t.X('callL') + t.X('foldL')) / 2 - 0.55, y: t.Y('callL') + 0.27, w: 1.1, h: 0.38, size: 14, fill: P.tblue, color: P.dblue, bold: true });
    // forward split hint inside B2
    line(s, t.X('root'), t.Y('d2H') - 0.36, t.X('root'), t.Y('d2H') + 0.36, { color: P.blue, width: 2.5, dash: 'dash' });
    // backward arrow: from B2 (right) up to B1^L
    const xr = t.X('betL') + 0.45, yb = t.Y('d2L') + 0.42, yt = t.Y('d1L');
    line(s, t.X('d2L') + 0.45, yb, xr, yb, { color: P.orange, width: 3 });
    line(s, xr, yb, xr, yt, { color: P.orange, width: 3 });
    arrow(s, xr, yt, t.X('d1L') + 0.5, yt, { color: P.orange, width: 3 });
    badge(s, 1, t.X('d2H') - 0.38 * 0.95 - 0.52, t.Y('d2H') - 0.21, 0.42, P.orange, 15);
    badge(s, 2, xr - 0.21, (yb + yt) / 2 - 0.21, 0.42, P.orange, 15);
    badge(s, 3, t.X('root') - 0.21, t.Y('d1H') - 0.21, 0.42, P.orange, 15);
    badge(s, 4, t.X('root') - 0.19, t.Y('d2H') - 0.74, 0.38, P.blue, 14);
    const lx = 7.75, lw = 4.98;
    const steps = [
      [P.orange, '진단', '이력 K^{2} 안의 정보집합끼리 라벨이 갈린다: H는 콜, L은 폴드 (H_{K} > 0)'],
      [P.orange, '거슬러 오르기', '이력이 달랐는데 같은 bucket에 묶인 앞 결정 → 결정 1의 B_{1} (보조정리 H3)'],
      [P.orange, '역방향 분할', 'B_{1}을 B^{H}_{1}, B^{L}_{1}로 나눈다. 정보집합은 그대로 두고, 뒤 결정의 이력이 갈라진다'],
      [P.blue, '다음 판정', '결정 2의 이력이 둘로 갈라져, 정방향 분할이 B_{2}를 콜과 폴드로 나눈다'],
    ];
    steps.forEach(([c, a, b], i) => {
      const y = 1.35 + i * 1.3;
      badge(s, i + 1, lx, y + 0.04, 0.44, c, 15);
      T(s, `**${a}**\n${b}`, { x: lx + 0.6, y, w: lw - 0.6, h: 1.22, size: 18 });
    });
    caption(s, '7-stud 구현은 정보집합이 반복되지 않아, 직전 결정의 bucket 이력으로 묶어 진단한다 (교차 검정). 부록 A5.', 6.6, { x: ML, w: CW, size: 14 });
    s.addNotes([
      '이 문제를 푸는 것이 역방향 분할입니다. 네 단계로 보겠습니다.',
      '1번, 진단입니다. 라벨을 이력 단위로만 보지 않고, 이력 안의 정보집합별 후회도 봅니다. K2 안에서 H는 콜, L은 폴드를 원하니 서로 공통 라벨이 없습니다. 인증식으로 말하면 이력 안 상쇄 H_K가 0보다 큽니다.',
      '2번, 거슬러 오르기입니다. 두 정보집합의 과거를 거꾸로 따라가면서, 이력은 달랐는데 같은 bucket에 들어간 결정을 찾습니다. 이 게임에서는 첫 결정입니다. 첫 결정에서는 카드가 달라 이력이 달랐는데 B1이라는 같은 bucket에 묶였고, 그 뒤로는 구별되지 않았습니다. 이런 지점이 늘 존재한다는 것이 보조정리 H3입니다. bucket 이력에 직전 bucket이 들어 있기 때문에 이 지점을 찾을 수 있습니다.',
      '3번, 그 bucket B1을 카드별로 나눕니다. 주황색 상자가 그것입니다. 정보집합을 옮기는 것이 아니라 앞 bucket만 쪼갭니다.',
      '4번, 그러면 두 번째 결정의 이력이 B1-H를 지나온 것과 B1-L을 지나온 것으로 갈라집니다. 이제 H의 이력은 콜, L의 이력은 폴드를 원하므로, 다음 판정에서 정방향 분할이 B2를 나눌 수 있습니다. 파란 점선이 나뉠 자리입니다.',
      '7-stud처럼 큰 게임에서는 정보집합이 반복되지 않아 정보집합별 후회를 쓸 수 없으므로, 직전 결정의 bucket 이력으로 묶어서 같은 진단을 하는 약한 판정을 씁니다. 그 차이는 뒤에서 한계로 다시 말씀드리겠습니다.',
    ].join('\n'));
  }

  // ------------------------------------------------------------ 11. before / after
  {
    const s = content('같은 게임, 두 결과: 정방향만 vs 역방향 포함', '질문 6 · 역방향의 해결');
    const sc = 0.78;
    chip(s, '정방향만', { x: 0.95, y: 1.32, w: 4.9, h: 0.46, size: 18, fill: P.tblue, color: P.dblue, bold: true });
    chip(s, '정방향 + 역방향', { x: 7.4, y: 1.32, w: 4.9, h: 0.46, size: 18, fill: P.torange, color: P.dorange, bold: true });
    tree(s, h, { cx: 3.4, oy: 2.1, s: sc, jpill: 'plain', edgeLabels: 'd2', chanceLabel: false, buckets: [B1, B2], hists: [K1H, K1L, { ...K2, tag: 'λ = 콜', tagW: 0.7, tagBold: true }] });
    const t = tree(s, h, {
      cx: 9.85, oy: 2.1, s: sc, jpill: 'plain', edgeLabels: 'd2', chanceLabel: false,
      buckets: [{ nodes: ['d1H'], label: 'B^{H}_{1}', kind: 'orange' }, { nodes: ['d1L'], label: 'B^{L}_{1}', kind: 'orange', labelSide: 'right' }, { nodes: ['d2H'], label: '콜', kind: 'blue', labelW: 0.42, labelSize: 15 }, { nodes: ['d2L'], label: '폴드', kind: 'blue', labelSide: 'right', labelW: 0.6, labelSize: 15 }],
      hists: [{ nodes: ['d2H'], padx: 0.25 }, { nodes: ['d2L'], padx: 0.25 }],
    });
    line(s, 6.62, 1.45, 6.62, 6.8, { color: P.line, width: 1 });
    T(s, '**정지 · NashConv = 1**\nL의 폴드 선호가 H와 상쇄되어 보이지 않는다', { x: 0.75, y: 5.85, w: 5.6, h: 1.0, size: 18, fill: P.soft, rect: 0.08, margin: [4, 10, 4, 10], valign: 'middle', align: 'center' });
    T(s, '**균형 · 원 게임 값 0** (H 콜, L 폴드, 상대 체크)\n판정 두 번: 역방향(B_{1}) → 정방향(B_{2})', { x: 6.95, y: 5.85, w: 5.78, h: 1.0, size: 18, fill: P.torange, rect: 0.08, margin: [4, 10, 4, 10], valign: 'middle', align: 'center' });
    s.addNotes([
      '같은 게임에서 두 방법이 도착하는 곳을 나란히 놓았습니다.',
      '왼쪽은 정방향만 쓴 경우입니다. B1과 B2가 처음 그대로이고, B2 안의 이력은 하나라 라벨이 콜입니다. 여기서 멈추고, L이 폴드로 바꾸면 1을 더 얻으므로 NashConv는 1입니다. 플레이어 1의 값은 -0.5입니다.',
      '오른쪽은 역방향 분할을 더한 경우입니다. 첫 판정에서 역방향 분할이 B1을 카드별로 나누고, 다음 판정에서 정방향 분할이 B2를 콜과 폴드로 나눕니다. 판정 두 번 뒤에 플레이어 1은 H면 콜, L이면 폴드를 하게 되고, 그러면 플레이어 2는 체크하는 것이 낫습니다. 이것이 원 게임의 균형이고 플레이어 1의 값은 0입니다.',
      '중요한 점은 오른쪽도 정보집합 하나하나를 다 나눈 것이 아니라는 점입니다. 필요한 곳, 즉 숨은 차이가 있던 두 bucket만 나눴습니다.',
      '이 동작이 일반적으로 성립한다는 것을 보이는 것이 다음 이론 부분입니다.',
    ].join('\n'));
  }

  // ------------------------------------------------------------ 12. algorithm flow
  {
    const s = content('알고리즘 흐름', '질문 4–6 · 전체 절차');
    const bw = 2.62, gap = 0.55, bh = 1.3, y1 = 1.45, y2 = 3.55;
    const colX = (c) => ML + c * (bw + gap);
    const stepBox = (n, c, y, a, b, col, fill) => {
      const x = colX(c);
      box(s, { x, y, w: bw, h: bh, fill: fill || P.white, line: col || P.km, lw: col ? 2 : 1.25, radius: 0.1 });
      badge(s, n, x + 0.14, y + 0.14, 0.42, col || P.ink, 15);
      T(s, a, { x: x + 0.65, y: y + 0.1, w: bw - 0.75, h: 0.5, size: 20, bold: true, valign: 'middle' });
      T(s, b, { x: x + 0.15, y: y + 0.66, w: bw - 0.3, h: 0.56, size: 16, color: col ? P.ink : P.muted, valign: 'middle' });
    };
    stepBox(1, 0, y1, '출발', '가장 거친 추상화 P_{0}');
    stepBox(2, 1, y1, '학습', 'CFR+ / MCCFR+ (표본)');
    stepBox(3, 2, y1, '고정', '그 시점의 평균 전략 σ̄');
    stepBox(4, 3, y1, '라벨', 'bucket 이력마다 λ(K)');
    for (let c = 0; c < 3; c++) arrow(s, colX(c) + bw + 0.06, y1 + bh / 2, colX(c + 1) - 0.06, y1 + bh / 2, { color: P.muted, width: 2 });
    arrow(s, colX(3) + bw / 2, y1 + bh + 0.04, colX(3) + bw / 2, y2 - 0.04, { color: P.muted, width: 2 });
    stepBox(5, 3, y2, '정방향 분할', 'bucket 안의 이력끼리 갈리면 라벨별로', P.blue, P.tblue);
    stepBox(6, 2, y2, '역방향 분할', '이력 안에서 갈리면 합쳐진 앞 bucket을', P.orange, P.torange);
    arrow(s, colX(3) - 0.06, y2 + bh / 2, colX(2) + bw + 0.06, y2 + bh / 2, { color: P.muted, width: 2 });
    arrow(s, colX(2) - 0.06, y2 + bh / 2, colX(1) + bw + 0.04, y2 + bh / 2, { color: P.muted, width: 2 });
    s.addText('나뉜 것이\n있나?', { x: colX(1), y: y2 - 0.05, w: bw, h: bh + 0.1, shape: S.DIAMOND, fill: { color: P.white }, line: { color: P.ink, width: 1.5 }, color: P.ink, fontSize: 18, bold: true, align: 'center', valign: 'middle', margin: 0, lang: 'ko-KR' });
    // yes: back up to step 2 (step 7 = relearn on the new partition)
    arrow(s, colX(1) + bw / 2, y2 - 0.03, colX(1) + bw / 2, y1 + bh + 0.04, { color: P.ink, width: 2.5 });
    T(s, '예 → **⑦ 다시 학습** (새 분할에서)', { x: colX(1) + bw / 2 + 0.12, y: y1 + bh + 0.2, w: 4.6, h: 0.4, size: 17, valign: 'middle' });
    // no: to step 8
    arrow(s, colX(1) + 0.02, y2 + bh / 2, colX(0) + bw + 0.06, y2 + bh / 2, { color: P.muted, width: 2 });
    T(s, '아니오', { x: colX(0) + bw + 0.0, y: y2 + bh / 2 - 0.46, w: 0.6, h: 0.36, size: 14, color: P.muted, align: 'center' });
    box(s, { x: colX(0), y: y2, w: bw, h: bh, fill: P.soft, line: P.ink, lw: 1.5, radius: 0.1 });
    badge(s, 8, colX(0) + 0.14, y2 + 0.14, 0.42, P.ink, 15);
    T(s, '정지 상태', { x: colX(0) + 0.65, y: y2 + 0.1, w: bw - 0.75, h: 0.5, size: 20, bold: true, valign: 'middle' });
    s.addImage({ data: ICON.flag, x: colX(0) + 0.18, y: y2 + 0.74, w: 0.36, h: 0.36 });
    T(s, 'σ̄을 출력한다', { x: colX(0) + 0.65, y: y2 + 0.66, w: bw - 0.8, h: 0.56, size: 16, valign: 'middle' });
    T(s, '**주의**: 정지 상태는 "더 나눌 곳을 찾지 못한 상태"이지, 그 자체로 원 게임의 내시 균형이 아니다. 균형이 되려면 학습기가 bucket 후회를 ε 아래로 만들어야 한다 (정리 H5).', { x: ML, y: 5.2, w: CW, h: 1.15, size: 19, fill: P.torange, rect: 0.08, margin: [6, 12, 6, 12], valign: 'middle' });
    caption(s, '실험에서는 판정 문턱 없이 epoch마다(반복 수 두 배 간격) 판정했다.', 6.5);
    s.addNotes([
      '지금까지의 내용을 하나의 절차로 모으면 이렇습니다.',
      '1번, 가장 거친 추상화 P0에서 출발합니다. 2번, CFR+로 학습합니다. 큰 게임에서는 표본을 쓰는 MCCFR+입니다. 3번, 그 시점의 평균 전략을 고정합니다. 4번, 고정한 전략 아래에서 bucket 이력마다 라벨을 계산합니다.',
      '5번, 한 bucket 안의 이력들이 공통 라벨을 갖지 않으면 정방향 분할로 나눕니다. 6번, 한 이력 안의 정보집합들이 갈리면 역방향 분할로 정보가 합쳐진 앞 bucket을 나눕니다.',
      '나뉜 것이 있으면 7번, 새 분할에서 다시 학습하고 2번으로 돌아갑니다. 아무것도 나뉘지 않으면 8번, 멈추고 평균 전략을 내놓습니다. 이것을 정지 상태라고 부릅니다.',
      '여기서 꼭 강조하고 싶은 것은, 정지 상태가 자동으로 원 게임의 내시 균형은 아니라는 점입니다. 정지 상태는 더 나눌 곳을 찾지 못했다는 뜻입니다. 균형이라고 말하려면 학습기가 bucket 후회를 충분히 작게 만들어야 한다는 조건이 필요하고, 그것이 정리 H5의 가정입니다. 실험에서는 문턱 없이 epoch마다 판정했습니다.',
    ].join('\n'));
  }

  // ------------------------------------------------------------ 13. pseudo-code & definitions
  {
    const s = content('의사코드와 정의', '질문 4–6 · 전체 절차');
    box(s, { x: ML, y: 1.35, w: 6.75, h: 5.45, fill: P.pale, radius: 0.1, name: 'pseudo-code panel' });
    const code = [
      [0, '**P** ← P_{0}'],
      [0, '**repeat**'],
      [1, 'σ̄ ← Learn(P)            {{muted:▹ CFR+ / MCCFR+}}'],
      [1, '**for each** bucket 이력 K:'],
      [2, 'λ(K) ← argmax_{a} r^{σ̄}(K, a)  또는 없음'],
      [1, '{{blue:**정방향**}}: 이력들이 공통 라벨을 갖지 않는'],
      [2, 'bucket을 라벨별로 나눈다'],
      [1, '{{orange:**역방향**}}: 멤버의 라벨이 갈리는 K마다'],
      [2, '정보가 합쳐진 지점의 bucket을 나눈다'],
      [0, '**until** 아무것도 나뉘지 않음'],
      [0, '**return** σ̄'],
    ];
    code.forEach(([ind, t], i) => T(s, t, { x: ML + 0.3 + ind * 0.5, y: 1.55 + i * 0.47, w: 6.3 - ind * 0.5, h: 0.45, size: 19, valign: 'middle' }));
    const lx = 7.65, lw = 5.08;
    const defs = [
      ['P_{0}', '(플레이어, 행동집합, 자기 결정 순번)이 같은 정보집합을 하나로'],
      ['Learn', '이론: ρ_{P} ≤ ε를 만드는 학습기로 가정한다. 실험: 문턱 없이 epoch마다 판정'],
      ['라벨 λ(K)', '분할 시점의 σ̄을 고정하고, 그 아래 순간 반사실 후회 r^{σ̄}(K, a) = Σ_{I∈K} r^{σ̄}(I, a)가 가장 큰 행동. 모두 0 이하면 "없음"'],
      ['정지 상태', '아무것도 나뉘지 않은 판정: 모든 G_{B} = 0이고 모든 H_{K} = 0'],
    ];
    let y = 1.35;
    defs.forEach(([a, b]) => {
      const hgt = b.length > 60 ? 1.75 : 1.15;
      T(s, `**${a}**\n${b}`, { x: lx, y, w: lw, h: hgt, size: 18 });
      y += hgt + 0.08;
    });
    s.addNotes([
      '같은 절차를 의사코드로 쓰면 이렇습니다. 논문에서 증명하는 기본 형태입니다.',
      'P를 P0로 두고, 반복마다 현재 분할에서 학습해 평균 전략을 얻습니다. 모든 bucket 이력에 라벨을 붙이고, 정방향과 역방향 분할을 합니다. 아무것도 나뉘지 않으면 멈추고 평균 전략을 돌려줍니다.',
      '오른쪽은 각 줄의 정의입니다. 이론에서는 Learn을, bucket 후회를 ε 아래로 만드는 학습기로 가정합니다. 실험에서는 그런 문턱을 두지 않고 epoch마다 판정했습니다. 이 차이는 마지막 실험에서 다시 다룹니다.',
      '라벨은 분할 시점의 평균 전략을 고정하고, 그 아래의 순간 반사실 후회를 이력 안에서 더해 가장 큰 행동입니다. 모두 0 이하면 지금 전략이 최선이라는 뜻에서 "없음"입니다.',
      '정지 상태는 아무것도 나뉘지 않은 판정이고, 다음 장의 인증식에서 두 상쇄 항 G와 H가 모두 0인 상태와 같습니다.',
      '분할은 정보집합을 옮기지 않고 bucket만 쪼개므로, 분할 횟수는 정보집합 수를 넘지 못합니다. 이것이 유한 종료입니다.',
    ].join('\n'));
  }

  // ------------------------------------------------------------ 14. certificate
  section('이론');
  {
    const s = content('인증 부등식: 세 항이 각각 무엇을 막는가', '질문 7 · 보장의 조건');
    T(s, '정리 H1 (어떤 lifting된 전략 σ에도 성립)', { x: ML, y: 1.3, w: CW, h: 0.4, size: 16, color: P.muted, align: 'center' });
    const fy = 1.8, fh = 0.95;
    T(s, 'NashConv(σ) ≤', { x: 0.9, y: fy, w: 3.25, h: fh, size: 32, bold: true, align: 'right', valign: 'middle' });
    T(s, 'ρ_{P}(σ)', { x: 4.35, y: fy, w: 2.2, h: fh, size: 32, bold: true, align: 'center', valign: 'middle', fill: P.soft, rect: 0.1 });
    T(s, '+', { x: 6.58, y: fy, w: 0.5, h: fh, size: 32, bold: true, align: 'center', valign: 'middle' });
    T(s, 'Σ_{B} G_{B}', { x: 7.1, y: fy, w: 2.3, h: fh, size: 32, bold: true, align: 'center', valign: 'middle', fill: P.tblue, rect: 0.1, color: P.dblue });
    T(s, '+', { x: 9.43, y: fy, w: 0.5, h: fh, size: 32, bold: true, align: 'center', valign: 'middle' });
    T(s, 'Σ_{K} H_{K}', { x: 9.95, y: fy, w: 2.3, h: fh, size: 32, bold: true, align: 'center', valign: 'middle', fill: P.torange, rect: 0.1, color: P.dorange });
    const cards = [
      [P.soft, P.ink, 'bucket 후회 ρ_{P}', 'bucket이 스스로 느끼는 후회의 합', '학습기 (CFR+)가 줄인다'],
      [P.tblue, P.dblue, 'bucket 안 상쇄 G_{B}', '한 bucket 안에서 서로 다른 이력끼리 상쇄되어 사라진 후회', '정방향 분할이 0으로'],
      [P.torange, P.dorange, '이력 안 상쇄 H_{K}', '한 이력 안에서 정보집합끼리 상쇄되어 사라진 후회', '역방향 분할이 0으로'],
    ];
    cards.forEach(([f, c, a, b, who], i) => {
      const x = ML + i * 4.13, w = 3.87, y = 3.05;
      box(s, { x, y, w, h: 2.05, fill: f, radius: 0.1 });
      T(s, a, { x: x + 0.2, y: y + 0.12, w: w - 0.4, h: 0.45, size: 20, bold: true, color: c });
      T(s, b, { x: x + 0.2, y: y + 0.58, w: w - 0.4, h: 0.9, size: 18 });
      T(s, '→ ' + who, { x: x + 0.2, y: y + 1.5, w: w - 0.4, h: 0.42, size: 18, bold: true, color: c });
    });
    T(s, '**정지 상태** (G = 0, H = 0)에서는 NashConv ≤ ρ_{P}. 학습기가 ρ_{P} ≤ ε를 주면 NashConv ≤ ε (정리 H5, 조건부). 표본 라벨이면 인증에 + 4Σ_{K} η_{K} (정리 H6).', { x: ML, y: 5.3, w: 6.35, h: 1.5, size: 18, fill: P.pale, rect: 0.08, margin: [6, 10, 6, 10], valign: 'middle' });
    T(s, '**예시 게임 (명제 H4의 정지점)**\nH_{K²} = max(0, −1.5) + max(0, +1.0) − max(0, −0.5) = 1 = NashConv\n→ 남은 것은 전부 이력 안 상쇄였다', { x: 7.15, y: 5.3, w: 5.58, h: 1.5, size: 18, fill: P.torange, rect: 0.08, margin: [6, 10, 6, 10], valign: 'middle' });
    s.addNotes([
      '이론의 중심은 이 인증 부등식, 정리 H1입니다. 원 게임에서의 NashConv가 세 항의 합으로 위에서 막힙니다. 어떤 lifting된 전략, 즉 bucket마다 하나의 전략을 쓰는 어떤 전략에도 성립합니다.',
      '첫째 항 ρ_P는 bucket 후회입니다. bucket을 하나의 정보집합처럼 보았을 때 느끼는 후회로, 학습기가 줄이는 몫입니다.',
      '둘째 항 G_B는 한 bucket 안에서 서로 다른 이력끼리 상쇄되어 사라진 후회입니다. 라벨이 갈리는 bucket을 나누는 정방향 분할이 이 항을 0으로 만듭니다.',
      '셋째 항 H_K는 한 이력 안에서 정보집합끼리 상쇄되어 사라진 후회입니다. 라벨 판정으로는 볼 수 없고, 역방향 분할이 이 항을 0으로 만듭니다. 학습기, 정방향, 역방향이 세 항을 하나씩 맡는 구조입니다.',
      '앞의 예시 게임에 넣어 보면, 정지점에서 H_K는 0 더하기 1 빼기 0으로 1이고, 실제 NashConv도 1입니다. 남은 오차가 전부 이력 안 상쇄였다는 뜻입니다.',
      '정지 상태에서는 두 상쇄 항이 0이므로 NashConv는 bucket 후회 이하가 됩니다. 그래서 학습기가 bucket 후회를 ε 아래로 만들어 준다면 원 게임의 NashConv도 ε 이하라는 것이 정리 H5입니다. 다만 이것은 조건부입니다. 표본으로 라벨을 추정하면 추정 오차 항이 더해집니다.',
    ].join('\n'));
  }

  // ------------------------------------------------------------ 15. what is proven
  {
    const s = content('무엇이 증명되었고, 무엇이 조건부인가', '질문 7 · 보장의 조건');
    const rows = [
      ['증명 · 부등식', P.ink, P.white, '**H1 인증**: NashConv ≤ ρ_{P} + Σ G_{B} + Σ H_{K}  ·  **H6 표본 판정**: 좋은 사건 위에서 거짓 분할이 없고 인증에 + 4Σ_{K} η_{K}'],
      ['증명 · 유한 종료', P.ink, P.white, '**H2**: 분할은 많아야 N − |P_{0}|번 (N = 정보집합 수). 라벨이 추정이어도 같다'],
      ['증명 · 반례와 동치', P.ink, P.white, '**H4**: 정방향만으로 NashConv = 1에서 멈추는 게임이 있다  ·  **H7**: 모든 bucket의 부모(직전 bucket, 행동)가 하나 ⇔ 추상화의 완전 기억'],
      ['조건부 정리', P.muted, P.white, '**H5**: 학습기가 ρ_{P} ≤ ε를 주고 라벨이 정확하면, 정방향 + 역방향은 NashConv ≤ ε에서 멈춘다'],
      ['실험적 수렴', P.lkm, P.ink, 'Leduc 21/21 정지 상태, mini-stud ε = 0.2에서 정지 상태 도달, Kuhn 수렴. 관찰한 설정에서만 말할 수 있다 (증명 아님)'],
      ['미해결', P.white, P.ink, '**학습기 조건**: CFR 평균은 자기가 가지 않는 곳의 후회 때문에 ρ_{P} ≤ ε를 맞추지 못한다 (추상화 없는 Leduc도 ρ/2 = 0.0855에서 멈춤)  ·  표본 역방향 분할의 보장'],
    ];
    let y = 1.35;
    const HS = [0.86, 0.62, 0.86, 0.72, 0.86, 0.88];
    rows.forEach(([lab, f, c, body], ri) => {
      const hgt = HS[ri];
      T(s, lab, { x: ML, y: y + 0.08, w: 2.55, h: hgt - 0.16, size: 17, bold: true, color: c, fill: f, rect: 0.08, align: 'center', valign: 'middle', line: f === P.white ? { color: P.ink, width: 1.25, dashType: 'dash' } : undefined });
      T(s, body, { x: ML + 2.75, y, w: CW - 2.75, h: hgt, size: 18, valign: 'middle' });
      y += hgt + 0.07;
    });
    caption(s, '명칭은 증명 초안(ASWISS_HISTORY_PROOF_KO.md)을 따른다. 정확한 가정과 진술은 부록 A2–A6.', 6.6);
    s.addNotes([
      '이론 결과를 상태별로 정리했습니다. 무엇이 증명이고 무엇이 조건부인지 구별하는 것이 중요합니다.',
      '증명된 것은 네 줄입니다. 인증 부등식 H1과, 라벨을 표본으로 추정할 때의 버전 H6. 분할 횟수가 정보집합 수를 넘지 못한다는 유한 종료 H2. 정방향만으로는 NashConv 1에서 멈추는 게임이 있다는 반례 H4. 그리고 모든 bucket이 부모를 하나만 가지는 트리 구조가 추상화의 완전 기억과 같다는 H7입니다.',
      '원 게임의 균형에 대한 보장은 H5 하나이고, 이것은 조건부입니다. 학습기가 bucket 후회를 ε 아래로 만들고 라벨이 정확하다면, 정방향과 역방향을 함께 쓴 ASWISS는 NashConv ε 이하에서 멈춥니다.',
      '실험적 수렴은 증명이 아니라 관찰입니다. Leduc의 21개 출발점 모두, mini-stud의 ε 0.2 실행이 정지 상태에 도달했다는 것은 그 설정에서의 사실입니다.',
      '미해결로 남은 것이 학습기 조건입니다. 표준 CFR의 평균 전략은 자기가 가지 않는 곳의 전략을 거의 고치지 않아서, 추상화가 전혀 없는 Leduc에서도 bucket 후회의 절반이 0.0855에서 멈춥니다. 그때 exploitability는 이미 0.000067로 아주 작은데도 그렇습니다. 그래서 H5의 가정을 CFR이 만족한다고 말할 수 없습니다. 또 표본으로 판정하는 역방향 분할에 대한 보장도 아직 없습니다.',
    ].join('\n'));
  }

  // ------------------------------------------------------------ 16. experiment design
  section('실험');
  {
    const s = content('실험 설계: 다섯 가지 질문', '질문 8 · 검증');
    table(s, [
      ['', '연구 질문', '게임 (크기)', '지표'],
      ['A', '역방향 분할이 bucket 이력 단위의 손실을 되찾는가?', 'mini-stud (정보집합 51,114)', 'exploitability (정확)'],
      ['B', '어떤 출발점에서도 정지 상태에 도달하는가?', 'Leduc (288), Kuhn', 'exploitability (정확)'],
      ['C', '트리 구조(완전 기억)는 무엇을 돕는가?', 'Leduc, 고정 추상화 96 bucket', 'D/2, exploitability'],
      ['D', '큰 게임에서 고정 추상화(k-means)와 비교하면?', '7-stud 헤즈업 (5구부터)', 'LBR (근사 하한)'],
      ['E', '판정 문턱 ε가 도움이 되는가?', '7-stud 5.11M roots', 'LBR (근사 하한)'],
    ], { x: ML, y: 1.35, w: CW, colW: [0.55, 5.5, 3.35, 2.73], rowH: 0.58, size: 18, aligns: ['center', 'left', 'left', 'left'], boldCol: 0 });
    T(s, '**exploitability (정확)**\n원 게임에서 최선 대응에 잃는 양 (= NashConv / 2). 0이면 균형. 게임 트리 전체를 계산할 수 있는 작은 게임에서만 잴 수 있다.', { x: ML, y: 5.0, w: 5.95, h: 1.8, size: 18, fill: P.pale, rect: 0.08, margin: [6, 10, 6, 10], valign: 'middle' });
    T(s, '**LBR (local best response)**\n큰 게임용 근사. 최선 대응을 일부만 찾으므로 exploitability의 **하한**이다. 실제 값은 더 클 수 있다. 낮을수록 좋다.', { x: 6.78, y: 5.0, w: 5.95, h: 1.8, size: 18, fill: P.pale, rect: 0.08, margin: [6, 10, 6, 10], valign: 'middle' });
    s.addNotes([
      '실험은 다섯 가지 질문에 답하도록 짰습니다.',
      'A는 mini-stud에서, 역방향 분할이 bucket 이력을 쓰면서 생긴 손실을 되찾는지 봅니다. B는 Leduc과 Kuhn에서, 가장 거친 분할이든 무작위 추상화든 어떤 출발점에서도 정지 상태에 도달하는지 봅니다. C는 트리 구조, 즉 추상화의 완전 기억이 무엇을 돕는지 봅니다.',
      'D와 E는 실제 크기의 7-stud입니다. D는 고정 k-means 추상화와의 비교, E는 이론의 학습기 조건을 판정 문턱으로 흉내 냈을 때 도움이 되는지입니다.',
      '지표를 구별해 주십시오. A부터 C까지는 게임 트리 전체를 계산할 수 있어서 정확한 exploitability를 씁니다. 7-stud는 그럴 수 없어서 LBR을 씁니다. LBR은 최선 대응을 일부만 찾는 근사라서 exploitability의 하한입니다. 실제 exploitability는 더 클 수 있으니, LBR 숫자를 exploitability와 같은 것으로 읽으면 안 됩니다.',
    ].join('\n'));
  }

  // ------------------------------------------------------------ 17. mini-stud
  {
    const s = content('A. mini-stud: 역방향 분할이 손실을 되찾았다', '질문 8 · 검증');
    banner(s, '질문 A · 역방향 분할이 bucket 이력 단위의 손실을 되찾는가?');
    hbar(s, 'mini-stud', ['k-means 고정 추상화', 'bucket 이력, 정방향만', 'ASWISS (이력 + 역방향)', '정보집합 단위 (참고)'], [0.3476, 0.0929, 0.0277, 0.0211], [P.km, P.blue, P.orange, P.lkm], { x: ML, y: 2.0, w: 7.6, h: 3.9, max: 0.43, fmt: '0.0000', csize: 17, dsize: 17, objectName: 'mini-stud exploitability' });
    caption(s, '정보집합 51,114개 · 4,096만 반복 · seed 3개 평균 · 평균 정책의 정확한 exploitability (낮을수록 좋음) · 비교군마다 bucket 수가 다르다 (부록 A8)', 6.0, { w: 7.6, h: 0.75 });
    stat(s, '3.4배', '역방향 분할을 더하자 0.0929 → 0.0277', { x: 8.55, y: 1.95, w: 4.18, size: 44, color: P.dorange, bh: 0.8, sh: 0.45 });
    stat(s, '정지 상태 도달', 'ε = 0.2 비교군: 2,900만–3,050만 반복에 G = 0, H = 0. 실제 0.034–0.038 ≤ 인증 0.19–0.20', { x: 8.55, y: 3.35, w: 4.18, size: 26, bh: 0.55, sh: 1.1 });
    stat(s, '0.0211', '정보집합 단위 라벨이 조금 더 낮지만, 큰 게임에서는 정보집합이 반복되지 않아 쓸 수 없다', { x: 8.55, y: 5.15, w: 4.18, size: 26, color: P.muted, bh: 0.55, sh: 1.1 });
    s.addNotes([
      '첫 번째 질문입니다. mini-stud는 정보집합이 51,114개라 모두 열거할 수 있어서 정확한 exploitability를 잴 수 있습니다. 4,096만 반복, seed 세 개의 평균이고 낮을수록 좋습니다.',
      '맨 위 회색이 고정 k-means 추상화로 0.3476입니다. 파란색이 bucket 이력에 정방향만 쓴 것으로 0.0929, 주황색이 역방향 분할까지 넣은 ASWISS로 0.0277입니다. 역방향 분할을 더하자 3.4배 낮아졌습니다. 앞의 예시 게임에서 본 이력 안 상쇄가 실제 게임에서도 손실의 큰 부분이었다는 뜻입니다.',
      '맨 아래 연한 회색은 정보집합 단위로 라벨을 매긴 비교용 변형입니다. 조금 더 낮지만, 큰 게임에서는 같은 정보집합이 거의 다시 나오지 않아 쓸 수 없는 방법입니다.',
      '오른쪽 가운데는 이론의 정지 상태가 실제로 나오는지 본 결과입니다. 학습기 조건을 ε 0.2의 문턱으로 강제한 비교군에서, 2,900만에서 3,050만 반복 사이에 G와 H가 모두 0인 판정이 나왔고, 그때 실제 값 0.034에서 0.038이 인증값 0.19에서 0.20 아래에 있었습니다.',
      '주의할 점은 비교군마다 최종 bucket 수가 다르다는 것입니다. 같은 bucket 수의 비교가 아니므로 부록 A8에 수를 적어 두었습니다.',
    ].join('\n'));
  }

  // ------------------------------------------------------------ 18. Leduc
  {
    const s = content('B. Leduc: 어떤 출발점에서도 정지 상태에 도달', '질문 8 · 검증');
    banner(s, '질문 B · 가장 거친 분할이든 무작위 추상화든, 정지 상태와 낮은 exploitability에 도달하는가?');
    const d = readDat(path.join(DATA, 'leduc_conv.dat'));
    const labels = d.col('iters').map((v, i) => (i % 3 === 0 ? fmtIters(v) : ''));
    const ser = [
      ['추상화 없는 CFR+', 'ref'], ['ASWISS, 무작위 출발 20개 (중앙값)', 'aswiss'], ['ASWISS, P₀에서', 'p0'],
      ['같은 bucket 수의 k-means (중앙값)', 'km'], ['정방향만 + 트리 유지, P₀에서', 'fwdtree'],
      ['ASWISS 최소', 'aswissmin'], ['ASWISS 최대', 'aswissmax'], ['k-means 최소', 'kmmin'], ['k-means 최대', 'kmmax'],
    ].map(([name, k]) => ({ name, labels, values: d.col(k) }));
    logLine(s, ser, [P.ink, P.orange, P.orange, P.km, P.blue, P.lorange, P.lorange, P.lkm, P.lkm], { x: ML, y: 1.95, w: 7.75, h: 4.6, min: 0.00001, max: 10, ytitle: '평균 정책 exploitability (로그 축)', xtitle: 'CFR+ 반복 수 (두 배 간격 epoch의 끝)', objectName: 'Leduc convergence' });
    caption(s, '정확한 CFR+ · 정보집합 288 · 옅은 선: 무작위 출발 20개(및 짝지은 k-means)의 최소·최대', 6.58, { w: 7.75 });
    stat(s, '21 / 21', 'P₀와 무작위 출발 20개 모두 정지 상태 (G = 0, H = 0)', { x: 8.7, y: 1.95, w: 4.03, size: 40, color: P.dorange, bh: 0.75, sh: 0.75 });
    stat(s, '0.000067', '65.5만 반복 뒤 exploitability. 추상화 없는 CFR+와 같다', { x: 8.7, y: 3.55, w: 4.03, size: 34, bh: 0.65, sh: 0.75 });
    stat(s, '203–270', '최종 bucket 수 (288개 중). 같은 수의 k-means는 0.14–0.88', { x: 8.7, y: 5.05, w: 4.03, size: 34, bh: 0.65, sh: 0.75 });
    caption(s, '이 설정에서 관찰한 결과이며, 수렴의 증명은 아니다.', 6.58, { x: 8.7, w: 4.03 });
    s.addNotes([
      '두 번째 질문은 출발점과 상관없이 정지 상태에 도달하느냐입니다. Leduc 홀덤은 정보집합이 288개라 정확한 CFR+와 정확한 exploitability를 쓸 수 있습니다. 두 축 모두 로그 축입니다.',
      '검은 선이 추상화 없는 CFR+, 즉 이 학습량에서 도달할 수 있는 기준입니다. 주황색 실선이 무작위 추상화 20개에서 출발한 ASWISS의 중앙값, 주황색 점선이 가장 거친 P0에서 출발한 ASWISS입니다. 처음에는 거친 추상화 때문에 높지만, 분할이 진행되면서 검은 선에 붙습니다.',
      '회색은 ASWISS와 같은 bucket 수로 만든 k-means 고정 추상화입니다. 0.1 아래로 내려가지 않습니다. 파란 선은 정방향만 쓰고 트리 구조를 유지한 경우로, 앞에서 말한 대로 2.93에서 움직이지 않습니다.',
      '정리하면 21개 출발점 모두 정지 상태에 도달했고, 65.5만 반복 뒤 exploitability 0.000067로 추상화 없는 CFR+와 같았습니다. 최종 bucket은 288개 중 203에서 270개로, 정보집합을 다 나누지 않고도 같은 값에 왔습니다. 같은 수의 k-means는 0.14에서 0.88이었습니다.',
      '다만 이것은 이 게임, 이 설정에서 관찰한 결과입니다. 수렴을 증명한 것은 아닙니다.',
    ].join('\n'));
  }

  // ------------------------------------------------------------ 19. tree structure
  {
    const s = content('C. 트리 구조와 역방향 분할의 역할', '질문 8 · 검증');
    banner(s, '질문 C · 트리 구조(각 bucket의 부모가 하나 = 완전 기억)는 무엇을 돕고, 무엇을 돕지 않는가?');
    const d = readDat(path.join(DATA, 'leduc_gain.dat'));
    const labels = d.col('iters').map((v, i) => (i % 3 === 0 ? fmtIters(v) : ''));
    const ser = [
      ['트리 (부모 bucket 하나)', 'tree'], ['자유 (과거를 무작위로 섞음)', 'free'], ['추상화 없음', 'ref'],
      ['트리 최소', 'treemin'], ['트리 최대', 'treemax'], ['자유 최소', 'freemin'], ['자유 최대', 'freemax'],
    ].map(([name, k]) => ({ name, labels, values: d.col(k) }));
    logLine(s, ser, [P.ink, P.km, P.lkm, '8A93A3', '8A93A3', 'C9CED6', 'C9CED6'], { x: ML, y: 1.95, w: 6.9, h: 4.45, min: 0.000001, max: 1, ytitle: 'D/2: 추상 게임에서의 이탈 이득 (로그 축)', xtitle: 'CFR+ 반복 수', objectName: 'Leduc tree vs free gain' });
    caption(s, 'D = bucket 하나만 바꿔 얻는 이득의 합 (0이면 추상 게임을 푼 것) · Leduc, 고정 무작위 추상화 96 bucket, seed 10개 · 옅은 선: 최소·최대', 6.45, { w: 6.9, h: 0.5, size: 13 });
    const lx = 7.85, lw = 4.88;
    table(s, [
      ['65.5만 반복', '트리', '자유'],
      ['D/2 (추상 게임)', '**0.00008**', '0.26'],
      ['exploitability (원 게임)', '1.81', '1.74'],
    ], { x: lx, y: 1.98, w: lw, colW: [2.68, 1.2, 1.0], rowH: 0.46, size: 17, aligns: ['left', 'center', 'center'] });
    T(s, '**학습기**: 트리면 CFR+가 추상 게임을 푼다 (D → 0). 자유 추상화는 멈춘다.\n**원 게임**: 그래도 트리가 낫지 않았다 (1.81 대 1.74). 추상 게임 성능 ≠ 원 게임 성능.\n{{orange:**분할의 수렴**}}: 정방향만 + 트리 유지는 P₀에서 2.93에 멈췄다. 역방향이 있으면 트리 여부와 상관없이 21/21 정지 상태.', { x: lx, y: 3.5, w: lw, h: 3.35, size: 18, spaceAfter: 8 });
    s.addNotes([
      '세 번째 질문은 트리 구조의 역할입니다. 트리 구조란 각 bucket이 부모, 즉 직전 bucket과 그때 둔 행동을 하나만 갖는 추상화이고, 이것이 추상화의 완전 기억과 같다는 것이 H7입니다.',
      '그래프의 세로축 D는 추상 게임 안에서 bucket 하나만 바꿔 얻을 수 있는 이득의 합입니다. 0이면 추상 게임을 푼 것입니다. 검은 선이 트리 추상화로, 추상화 없는 경우처럼 0으로 내려갑니다. 회색은 과거를 무작위로 섞은 자유 추상화로, 0.26 근처에서 멈춥니다. 트리 구조는 학습기를 확실히 돕습니다.',
      '그런데 원 게임 exploitability는 트리가 1.81, 자유가 1.74로 트리가 낫지 않았습니다. 추상 게임을 잘 푸는 것과 원 게임에서 좋은 전략인 것은 다른 이야기입니다.',
      '그리고 분할 과정의 수렴은 트리로 해결되지 않습니다. 명제 H4의 반례도 트리 추상화였고, Leduc에서 정방향만 쓰고 트리를 유지하면 2.93에서 멈췄습니다. 역방향 분할이 있으면 트리 여부와 상관없이 21개 모두 정지 상태에 도달했습니다.',
      '그래서 직전 bucket을 기억하는 것의 역할은 추상 게임을 완전 기억으로 만드는 데 있다기보다, 정보가 합쳐진 지점을 찾아 역방향 분할을 가능하게 하는 데 있다고 봅니다.',
    ].join('\n'));
  }

  // ------------------------------------------------------------ 20. 7-stud
  {
    const s = content('D. 7-stud: 부분 구현과 완성형 시범을 나눠 본다', '질문 8 · 검증');
    banner(s, '질문 D · 실제 크기의 게임에서 고정 추상화(k-means)와 비교하면?');
    chip(s, '100M roots · 역방향 없는 부분 구현', { x: ML, y: 2.02, w: 5.95, h: 0.48, size: 17, fill: P.tblue, color: P.dblue, bold: true });
    chip(s, '5.11M roots · ASWISS 완성형 시범', { x: 6.78, y: 2.02, w: 5.95, h: 0.48, size: 17, fill: P.torange, color: P.dorange, bold: true });
    hbar(s, 'LBR100M', ['이력 + 방법 B (861,446)', '이력 + 구간 라벨 (1,274,437)', 'k-means 1024 (145,410)', 'k-means 256 (37,249)'], [1.060, 1.023, 1.282, 1.603], [P.blue, P.lblue, P.km, P.lkm], { x: ML, y: 2.55, w: 5.95, h: 2.5, max: 2.95, csize: 15, dsize: 15, gap: 40, objectName: '7-stud 100M LBR' });
    hbar(s, 'LBR5M', ['ASWISS 완성형 (56,949)', '역방향 없음 = 방법 B (31,645)', 'k-means 1024 (141,452)'], [1.575, 1.805, 1.223], [P.orange, P.blue, P.km], { x: 6.78, y: 2.55, w: 5.95, h: 2.0, max: 2.95, csize: 15, dsize: 15, gap: 40, objectName: '7-stud 5.11M LBR' });
    T(s, '방법 B − k-means 1024 = **−0.22** [−0.41, −0.04] (같은 deal로 짝지은 차이). 이 실행에는 역방향 분할이 없다.', { x: ML, y: 5.1, w: 5.95, h: 1.0, size: 18 });
    T(s, '역방향 효과 **−0.23 ± 0.21**. 그러나 k-means 1024보다 아직 **+0.35 ± 0.19** 높다. 완성형의 100M은 아직 돌리지 않았다.', { x: 6.78, y: 4.62, w: 5.95, h: 1.2, size: 18 });
    T(s, '평균 정책 LBR (낮을수록 좋음, ± 95% 구간) · 괄호: bucket 수 · LBR은 exploitability의 근사 하한이다 · PolicyLBR 64 particles, 20,000 pair, seed 1 · 헤즈업, 5구부터 학습, 4구 버림·공개는 휴리스틱 · 두 패널은 학습량·bucket 수·버전이 달라 서로 비교하지 않는다', { x: ML, y: 6.12, w: CW, h: 0.8, size: 14, color: P.muted, fill: P.pale, rect: 0.06, margin: [4, 10, 4, 10], valign: 'middle' });
    s.addNotes([
      '네 번째 질문은 실제 크기의 7-stud입니다. 이 게임은 exploitability를 정확히 잴 수 없어서 LBR로 비교합니다. LBR은 exploitability의 근사 하한이고 낮을수록 좋습니다. 같은 deal, 64 particles, 20,000 pair로 평가했고 seed는 하나입니다.',
      '두 결과를 반드시 나눠서 봐야 합니다. 왼쪽은 1억 roots까지 돌린 실행인데, 이 버전에는 역방향 분할이 없습니다. bucket 이력과 방법 B 라벨을 쓴 비교군이 1.060으로, k-means 1024의 1.282보다 낮았고, 같은 deal로 짝지은 차이는 -0.22, 95% 구간이 -0.41에서 -0.04로 0을 포함하지 않았습니다. 다만 bucket 수가 86만 대 14.5만으로 다릅니다.',
      '오른쪽은 역방향까지 넣은 완성형의 511만 roots 시범입니다. 역방향 분할은 1.805를 1.575로 낮췄지만, 같은 학습량의 k-means 1024의 1.223보다는 아직 0.35 높습니다. 그래서 완성형 ASWISS가 k-means보다 낫다고 말할 수 없습니다. 완성형의 1억 roots 실행은 아직 하지 않았습니다.',
      '두 패널은 학습량, bucket 수, 알고리즘 버전이 모두 다르므로 서로 비교하지 않습니다. 패널 안에서도 bucket 수가 다르다는 점을 함께 봐야 합니다. 비율로 말하면 왼쪽 방법 B는 k-means 1024의 0.83배, 오른쪽 완성형은 1.29배입니다.',
    ].join('\n'));
  }

  // ------------------------------------------------------------ 21. gate
  {
    const s = content('E. 판정 문턱: 엄격할수록 나빠졌다', '질문 8 · 검증');
    banner(s, '질문 E · 학습기 조건을 흉내 내어, 후회가 작아질 때까지 판정을 미루면 좋아지는가?');
    hbar(s, 'gate', ['문턱 없음 · 매 epoch 판정 (56,949)', '반복 후회 ≤ 20 (29,698)', 'ρ ≤ 10 (5,210)', 'ρ ≤ 3 (956)', 'ρ ≤ 1 (343)', 'k-means 1024 (141,452)'], [1.575, 1.862, 2.479, 3.317, 4.030, 1.223], [P.orange, P.lorange, P.lorange, P.lorange, P.lorange, P.km], { x: ML, y: 1.98, w: 7.5, h: 4.2, max: 4.9, csize: 16, dsize: 16, gap: 35, objectName: 'gate LBR' });
    caption(s, '7-stud 5.11M roots · 평균 정책 LBR, 20,000 pair (낮을수록 좋음) · 괄호: 마지막 bucket 수', 6.3, { w: 7.5 });
    const lx = 8.45, lw = 4.28;
    T(s, '**엄격할수록 나빴다**\n문턱을 기다린 비교군은 모두 매 epoch 판정보다 LBR이 높았다. 기다리는 동안 bucket이 수십–수백 개에 묶여 있었다.\n**후회가 작다 ≠ 정책이 좋다**\nε = 1은 bucket 36개, ρ = 0.86에서 통과했지만 LBR은 7.5로 같은 시점 k-means의 7.6배였다.\n**문턱의 뜻**\n이론에서 ε는 학습기 조건(ρ_{P} ≤ ε)이지 성능 지표가 아니다. 실험에서는 매 epoch 판정했다.', { x: lx, y: 1.98, w: lw, h: 4.9, size: 18, spaceAfter: 3 });
    s.addNotes([
      '마지막 질문은 판정 문턱입니다. 이론의 정리 H5는 학습기가 bucket 후회를 ε 아래로 만든 뒤에 판정한다고 가정합니다. 그래서 7-stud에서 이 조건을 문턱으로 걸고, 후회가 문턱 아래로 내려올 때까지 판정을 미뤄 보았습니다.',
      '결과는 반대였습니다. 주황색 막대가 문턱 없이 매 epoch 판정한 ASWISS로 1.575이고, 문턱을 엄격하게 걸수록 2.479, 3.317, 4.030으로 나빠졌습니다. 기다리는 동안 bucket이 수십에서 수백 개에 묶여 있었기 때문입니다.',
      '또 하나 중요한 관찰은 후회가 작다고 정책이 좋은 것은 아니라는 점입니다. ε 1 비교군은 bucket 36개, 후회 0.86에서 문턱을 통과했지만 LBR은 7.5로 같은 시점 k-means의 7.6배였습니다. 거친 추상화에서는 학습기가 bucket 후회를 줄여도, 인증식의 나머지 두 상쇄 항이 커서 정책이 좋아지지 않습니다.',
      '그래서 ε는 언제 판정할지의 조건, 즉 증명을 위한 학습기 조건이지 성능 지표가 아닙니다. 실험에서는 매 epoch 판정하는 쪽이 이 학습량 안에서 가장 좋았습니다. 이 차이는 이론과 실험 절차 사이에 남은 간격이기도 합니다.',
    ].join('\n'));
  }

  // ------------------------------------------------------------ 22. verified vs not
  section('정리');
  {
    const s = content('검증된 것과 아직 아닌 것', '질문 8 · 검증');
    T(s, '확인된 것', { x: ML, y: 1.3, w: 5.9, h: 0.5, size: 22, bold: true });
    T(s, '아직 아닌 것', { x: 6.83, y: 1.3, w: 5.9, h: 0.5, size: 22, bold: true });
    const ok = [
      ['증명', '인증 부등식(H1), 유한 종료(H2), 정방향만의 반례(H4), 표본 판정(H6), 트리 ⇔ 완전 기억(H7)'],
      ['Kuhn · Leduc', 'P₀와 무작위 출발 모두 정지 상태, 추상화 없는 CFR+와 같은 값 (관찰)'],
      ['mini-stud', '역방향 분할로 0.0929 → 0.0277 (seed 3개)'],
      ['7-stud 100M', '역방향 없는 bucket 이력(방법 B)이 k-means 1024보다 LBR 0.22 낮음 (seed 1)'],
    ];
    const no = [
      ['원 게임 보장', 'H5의 학습기 조건을 CFR 평균이 만족한다는 증명이 없다'],
      ['완성형의 큰 게임 성능', '5.11M에서 k-means 1024보다 +0.35 높다. 100M은 미실행'],
      ['표본 역방향 분할', '7-stud는 직전 결정의 이력으로 묶는 약한 판정. 일반 보장 없음'],
      ['평가의 한계', 'LBR은 근사 하한, seed 1개, 4구 버림·공개 휴리스틱'],
      ['문턱과 절차', '이론의 학습기 조건(ρ_{P} ≤ ε)과 실험 절차(매 epoch 판정)가 다르다'],
    ];
    const col = (list, x, ic, step) => list.forEach(([a, b], i) => {
      const y = 1.92 + i * step;
      s.addImage({ data: ic, x, y: y + 0.05, w: 0.38, h: 0.38 });
      T(s, `**${a}** · ${b}`, { x: x + 0.52, y, w: 5.38, h: step - 0.06, size: 18 });
    });
    col(ok, ML, ICON.check, 1.2);
    col(no, 6.83, ICON.open, 0.97);
    line(s, 6.6, 1.4, 6.6, 6.85, { color: P.line, width: 1 });
    s.addNotes([
      '검증된 것과 아직 아닌 것을 정직하게 나누겠습니다.',
      '확인된 것입니다. 이론에서는 인증 부등식, 유한 종료, 정방향만으로는 멈추는 반례, 표본 판정, 트리와 완전 기억의 동치가 증명되었습니다. 실험에서는 Kuhn과 Leduc에서 모든 출발점이 정지 상태에 도달해 추상화 없는 CFR+와 같은 값을 냈고, mini-stud에서 역방향 분할이 손실을 3.4배 줄였고, 7-stud 1억 roots에서는 역방향 없는 bucket 이력이 k-means 1024보다 LBR이 낮았습니다.',
      '아직 아닌 것입니다. 첫째, 원 게임 균형의 보장은 학습기 조건에 달려 있는데, 표준 CFR 평균이 그 조건을 만족한다는 증명이 없습니다. 둘째, 역방향까지 넣은 완성형은 큰 게임에서 아직 k-means보다 나쁘고, 1억 roots 실행을 하지 않았습니다. 셋째, 7-stud의 역방향 분할은 정보집합별 표본이 없어서 직전 결정의 이력으로 묶는 약한 판정이고, 일반적인 보장이 없습니다.',
      '넷째, 7-stud의 평가는 LBR이라는 근사 하한이고, seed가 하나이며, 4구의 버림과 공개는 휴리스틱입니다. 다섯째, 이론은 학습기 조건을 만족한 뒤 판정한다고 가정하지만, 실험은 문턱 없이 epoch마다 판정했습니다.',
    ].join('\n'));
  }

  // ------------------------------------------------------------ 23. conclusion
  {
    const s = content('정리와 다음 단계', null, 'DARK');
    T(s, '학습된 전략의 반사실 후회로 나눌 곳을 정하고, {{lblue:**정방향**}}(bucket 안)과 {{lorange:**역방향**}}(이력 안) 분할로 추상화가 숨긴 전략적 차이를 되살린다.', { x: ML, y: 1.3, w: CW, h: 1.25, size: 22, color: P.white, fill: '2C3A4F', rect: 0.1, margin: [8, 14, 8, 14], valign: 'middle' });
    const sum = [
      '작은 게임: 어떤 출발점에서도 정지 상태, 추상화 없는 CFR+와 같은 값',
      '큰 게임: bucket 이력은 고정 k-means를 넘었다 (역방향 없는 100M). 완성형은 시범 단계',
      '이론: 인증과 종료는 증명. 원 게임 균형은 학습기 조건 아래에서만',
    ];
    T(s, '요약', { x: ML, y: 2.85, w: 5.9, h: 0.5, size: 22, bold: true, color: P.lblue });
    T(s, sum.map((t) => '- ' + t).join('\n'), { x: ML, y: 3.4, w: 5.9, h: 3.3, size: 19, color: P.white, spaceAfter: 10 });
    const nx = [
      '7-stud에서 역방향 분할을 넣은 ASWISS의 100M 실행',
      '카드 휴리스틱(power 군집) 제거: 카드를 한 장씩 관측으로',
      '학습기 조건을 경로 밖 후회 없이 다시 쓰기 (도달 가중 이득 D)',
      '표본 학습기 MCCFR+의 수렴 보장',
    ];
    T(s, '다음 단계', { x: 6.83, y: 2.85, w: 5.9, h: 0.5, size: 22, bold: true, color: P.lorange });
    nx.forEach((t, i) => {
      badge(s, i + 1, 6.83, 3.45 + i * 0.82, 0.42, P.orange, 15);
      T(s, t, { x: 7.4, y: 3.38 + i * 0.82, w: 5.33, h: 0.75, size: 18, color: P.white, valign: 'middle' });
    });
    s.addNotes([
      '정리하겠습니다. ASWISS는 학습된 전략의 반사실 후회로 나눌 곳을 정하고, bucket 안의 차이는 정방향 분할로, 이력 안에 숨은 차이는 역방향 분할로 되살리는 방법입니다.',
      '작은 게임에서는 어떤 출발점에서도 정지 상태에 도달해 추상화 없는 CFR+와 같은 값을 냈습니다. 큰 게임에서는 bucket 이력이라는 아이디어가 고정 k-means를 넘었지만, 그것은 역방향 분할이 없는 버전이었고, 완성형은 아직 시범 단계입니다. 이론에서는 인증과 종료가 증명되었고, 원 게임 균형은 학습기 조건 아래에서만 보장됩니다.',
      '다음 단계는 네 가지입니다. 완성형을 7-stud에서 1억 roots까지 돌리는 것, 카드 쪽 휴리스틱을 없애는 것, 학습기 조건을 CFR이 실제로 만족할 수 있는 형태로 다시 쓰는 것, 그리고 표본 학습기의 수렴 보장입니다.',
      '감사합니다. 자세한 정리와 실험 설정은 부록에 있습니다.',
    ].join('\n'));
  }

  // ================================================================= appendix
  section('부록');
  const app = (n, title) => content(title, '부록 A' + n, 'APPENDIX');
  {
    const s = app(1, 'A1. 기호와 정의');
    const eq = [
      '**반사실 가치** q^{σ}(I, a) = Σ_{h∈I} π^{σ}_{−i}(h) Σ_{z⊒ha} π^{σ}(ha, z) u_{i}(z)',
      '**순간 후회** r^{σ}(I, a) = q^{σ}(I, a) − Σ_{b} σ(I, b) q^{σ}(I, b),   r^{σ}(I, 없음) = 0',
      '**모은 후회** r(K) = Σ_{I∈K} r(I),   r(B) = Σ_{K⊆B} r(K)',
      '**ψ와 라벨 집합** ψ(x) = max(0, max_{a} x_{a}),   M(x) = argmax_{ℓ∈A∪{없음}} x_{ℓ}.  bucket B가 일치: ∩_{K⊆B} M(r(K)) ≠ ∅',
      '**세 양** ρ_{P}(σ) = Σ_{B} ψ(r(B)),   G_{B} = Σ_{K⊆B} ψ(r(K)) − ψ(r(B)),   H_{K} = Σ_{I∈K} ψ(r(I)) − ψ(r(K))   (ψ가 준선형이라 G, H ≥ 0)',
      '**bucket 이력** κ_{P}(I) = (i, A(I), b(I^{1}), a^{1}, …, b(I^{m}), a^{m}, o(I)).  7-stud 구현: (b(I^{m}), a^{m}, o(I)) (직전 bucket만)',
      '**허용 분할**: 모든 bucket이 이력류(같은 bucket 이력의 정보집합 묶음)의 합집합.  **P_{0}**: (플레이어, 행동집합, 자기 결정 순번)으로 묶음',
      '**NashConv**(σ) = Σ_{i} (max_{σ′_{i}} u_{i}(σ′_{i}, σ_{−i}) − u_{i}(σ)).  실험의 exploitability = NashConv / 2',
    ];
    T(s, eq.join('\n'), { x: ML, y: 1.35, w: CW, h: 5.2, size: 17, spaceAfter: 10 });
    caption(s, '출처: 증명 초안 ASWISS_HISTORY_PROOF_KO.md §2. G는 완전 기억 유한 2인 영합 전개형 게임, σ는 lifting된 행동 전략(σ(I) = σ(b(I))).', 6.55);
    s.addNotes('본문에서 말로 설명한 양들의 정확한 정의입니다. 라벨 판정과 인증식이 모두 이 정의 위에 서 있습니다. ψ는 후회 벡터에서 "바꿔서 얻을 수 있는 최대 이득"이고, 이것이 준선형이기 때문에 두 상쇄 항 G와 H가 항상 0 이상입니다. 실험에서 말한 exploitability는 NashConv의 절반입니다.');
  }
  {
    const s = app(2, 'A2. 정리 H1 (인증)과 정리 H2 (유한 종료)');
    T(s, '**보조정리 H5 (NashConv 분해)** · β_{i}를 σ_{−i}에 대한 순수 최선 응답이라 하면\nNashConv(σ) = Σ_{i} Σ_{I∈𝓘_{i}} π^{β_{i}}_{i}(I) r^{σ}(I, β_{i}(I)) ≤ Σ_{I} ψ(r^{σ}(I))\n**보조정리 H6 (두 단계 라벨 항등식)** · 허용 분할과 임의의 σ에 대해\nΣ_{I} ψ(r(I)) = ρ_{P}(σ) + Σ_{B} G_{B} + Σ_{K} H_{K},  그리고 G_{B} = 0 ⇔ B가 일치, H_{K} = 0 ⇔ ∩_{I∈K} M(r(I)) ≠ ∅\n**정리 H1 (인증)** · 임의의 허용 분할과 lifting된 σ에 대해\nNashConv(σ) ≤ ρ_{P}(σ) + Σ_{B} G_{B} + Σ_{K} H_{K}.   모든 bucket이 일치하면 NashConv(σ) ≤ ρ_{P}(σ) + Σ_{K} H_{K}\n**정리 H2 (유한 종료)** · 분할(역방향 포함)은 모든 표본 경로에서 많아야 N − |P_{0}|번이다.\n증명: 분할마다 bucket 수가 엄격히 늘고, bucket 수는 N을 넘지 못한다. 라벨이 정확한지와 무관하다.', { x: ML, y: 1.35, w: CW, h: 5.2, size: 17, spaceAfter: 6 });
    caption(s, '출처: 증명 초안 §5. H1은 H5와 H6을 잇는 것으로 증명된다. 상태: [증명].', 6.55);
    s.addNotes('인증식이 어떻게 나오는지의 골격입니다. 먼저 NashConv를 최선 응답 경로 위의 정보집합별 후회의 합으로 쓰고, 그 합을 bucket 후회와 두 상쇄 항으로 정확히 나눕니다. 그래서 부등식은 첫 단계에서만 생깁니다. 유한 종료는 분할이 bucket을 쪼개기만 하고 정보집합을 옮기지 않는다는 데서 바로 나옵니다.');
  }
  {
    const s = app(3, 'A3. 조건부 결과와 학습기 조건');
    T(s, '**ε-오라클 가정** · 매 라운드 학습기가 ρ_{P}(σ̄) ≤ ε인 lifting된 프로필을 낸다. 라벨은 정확히 계산한다.\n**정리 H3 (역방향 없이)** · 이 가정 아래 분할 라운드는 많아야 N − |P_{0}|번이고, 멈춘 출력은 NashConv ≤ ε + Σ_{K} H_{K}. 모든 H_{K} = 0이면 NashConv ≤ ε.\n**정리 H5 (정방향 + 역방향)** · 같은 가정 아래, 멈춘 출력은 NashConv ≤ ε. ε = 0이면 원 게임의 정확한 균형.\n증명: 멈춤 = 정방향 분할 없음(G_{B} = 0) + 역방향 분할 없음(H_{K} = 0). 정리 H1에 넣는다.\n**오라클이 낼 프로필의 존재** · P_{0}에서 나뉜 분할은 같은 bucket의 두 정보집합이 한 경로 위에 있지 않으므로 ρ_{P} = 0인 프로필이 있다. 그러나 H_{K} = 0까지 함께 만족한다는 뜻은 아니다 (명제 H4).\n**학습기 조건 [미해결]** · 표준 CFR 평균은 이 가정을 만족하지 않는다. 자기가 가지 않는 정보집합의 평균 전략이 거의 고쳐지지 않아 그곳의 후회가 남는다. 추상화 없는 Leduc에서도 CFR+ 평균의 ρ/2가 0.0855에서 멈췄다 (그때 exploitability 0.000067).', { x: ML, y: 1.35, w: CW, h: 5.3, size: 17, spaceAfter: 9 });
    caption(s, '출처: 증명 초안 §6, §8, §11과 원본 슬라이드의 이론 표. 정리 H3은 원본 슬라이드에는 없고 증명 초안에만 있다.', 6.6);
    s.addNotes('원 게임에 대한 보장이 정확히 어떤 가정 위에 있는지 적었습니다. 핵심은 ε-오라클, 즉 학습기가 bucket 후회를 ε 아래로 만든다는 가정입니다. 역방향이 없으면 이력 안 상쇄가 남을 수 있고, 역방향이 있으면 그것까지 0이 되어 NashConv ε 이하가 됩니다. 그러나 표준 CFR 평균은 이 가정을 만족하지 않으므로, 이 결과를 그대로 CFR에 적용할 수는 없습니다. 이것이 가장 큰 미해결 문제입니다.');
  }
  {
    const s = app(4, 'A4. 합쳐진 지점, 역방향 규칙, 트리 구조');
    T(s, '**보조정리 H3 (정보가 합쳐진 지점)** · I ≠ I′이고 κ(I) = κ(I′)이면 어떤 j ≤ m에서\nI^{j} ≠ I′^{j},  κ(I^{j}) ≠ κ(I′^{j}),  b(I^{j}) = b(I′^{j}).  즉 j번째 결정에서 이력은 달랐는데 같은 bucket에 들어갔다.\n**역방향 분할 규칙** · 이력류 K의 멤버들이 최대 라벨을 공유하지 않으면(H_{K} > 0) 대표 라벨이 다른 두 멤버 I, I′을 고르고, 지점 j의 bucket b(I^{j})에서 κ(I^{j})를 떼어 자식 하나로 만든다. 그다음 라운드부터 I와 I′의 이력이 달라진다 (보조정리 H1).\n구현(mini-stud·Leduc): 쌍은 첫 멤버와 라벨이 다른 멤버, 지점은 이력이 다른 가장 가까운 앞 결정. 역방향으로만 나뉜 자식은 부모의 표를 복사한다.\n**보조정리 H7 (트리 = 추상화의 완전 기억)** · P가 P_{0}를 세분할 때, 모든 bucket의 멤버가 같은 (직전 자기 bucket, 행동)을 가지는 것과 같은 자기 과거 열 전체를 가지는 것은 동치다. 이때 CFR(+)는 추상 게임을 푼다. 그래도 정방향만으로는 멈출 수 있다 (명제 H4의 게임도 트리다).', { x: ML, y: 1.35, w: CW, h: 5.3, size: 17, spaceAfter: 9 });
    caption(s, '출처: 증명 초안 §4, §8. 직전 bucket만 쓰는 bucket 이력(7-stud)에서도 H3은 j₁ = max{j : κ(I^j) ≠ κ(I′^j)}로 성립한다.', 6.6);
    s.addNotes('역방향 분할이 가능한 이유는 보조정리 H3입니다. 같은 이력에 들어간 두 정보집합은 과거 어딘가에서 이력은 달랐는데 같은 bucket에 들어간 지점이 반드시 있습니다. 그 지점의 bucket을 나누면 이후 두 정보집합의 이력이 갈라집니다. 트리 구조와 완전 기억의 동치인 H7도 여기 적었습니다. 트리는 학습기의 수렴을 돕지만 분할의 수렴을 보장하지는 않습니다.');
  }
  {
    const s = app(5, 'A5. 표본 판정 (정리 H6)과 구현의 차이');
    T(s, '**설정** · 라운드 k에서 σ̄을 고정하고, 두 좌석 모두 σ̄로 두는 외부 샘플링 순회를 n_{k}번 한다. 순회마다 방문한 정보집합의 표본 후회를 이력별로 더해 n_{k}로 나눈 것이 r̂(K).\n**집중** · 한 순회에서 이력류 K의 멤버는 많아야 하나 방문되므로 r̂(K)는 불편 추정량이다. 확률 1 − δ 이상에서 모든 k, K, a에 대해\n|r̂(K, a) − r^{σ̄}(K, a)| ≤ η^{(k)}_{K} = Δ_{K} √( (2 / n_{k}) ln( π² N_{A} (k + 1)² / (3δ) ) )\n**정리 H6** · 여유 라벨(최대에서 2η 이내)로 판정하면, 좋은 사건 위에서 ① 참으로 일치하는 bucket은 나뉘지 않고 ② 나뉠 때 각 이력은 참 최대 라벨과 4η 이내인 자식으로 가며 ③ 분할 없는 라운드의 출력은 NashConv ≤ ρ_{P} + 4Σ_{K} η_{K} + Σ_{K} H_{K} ④ 분할 수는 많아야 N − |P_{0}|.\n**7-stud 구현과의 차이** · 순회 수에 상한(epoch당 256만 roots)이 있어 η가 0으로 가지 않는다. 여유 라벨 대신 z = 2 정규 근사 검정을 써서 거짓 분할이 있을 수 있다 (종료는 깨지지 않지만 bucket이 는다). 역방향은 정보집합 대신 직전 결정의 이력으로 묶고, 두 세부 이력이 서로 자기 라벨을 유의하게 선호할 때만 나눈다 (교차 검정). 일반 보장은 [미해결].', { x: ML, y: 1.35, w: CW, h: 5.3, size: 16, spaceAfter: 9 });
    caption(s, '출처: 증명 초안 §9, §8과 원본 슬라이드의 역방향 설명. 상태: H6 [증명], 표본 역방향 [미해결].', 6.6);
    s.addNotes('큰 게임에서는 라벨을 정확히 계산할 수 없어 표본으로 추정합니다. 이론적으로는 순회 수를 늘리면 추정 오차 η가 0으로 가고, 인증식에 4Ση가 더해질 뿐입니다. 하지만 실제 7-stud 구현은 순회 수에 상한이 있고, 여유 라벨 대신 정규 근사 검정을 쓰며, 역방향 분할은 정보집합 대신 직전 결정의 이력으로 묶는 약한 판정을 씁니다. 이 부분은 보장이 없는 실험적 선택입니다.');
  }
  {
    const s = app(6, 'A6. 명제 H4 게임의 계산');
    table(s, [
      ['위치', '계산 (σ*: 플레이어 1은 체크 후 콜, 플레이어 2는 베팅)', '결론'],
      ['결정 1, 카드 H', '베팅의 후회 ½(−10 − 2) = −6', '라벨 {체크, 없음}'],
      ['결정 1, 카드 L', '베팅의 후회 ½(−10 + 3) = −3.5', '라벨 {체크, 없음} → B₁ 일치'],
      ['결정 2, 카드 H', '폴드의 후회 ½(−1 − 2) = −1.5', ''],
      ['결정 2, 카드 L', '폴드의 후회 ½(−1 + 3) = +1.0', ''],
      ['결정 2, 이력 K²', '합 −0.5', '라벨 {콜, 없음} → 나뉘지 않음'],
      ['플레이어 2 (J)', '베팅으로 0.5를 더 얻는다', '후회 없음'],
      ['결과', 'ρ_{P}(σ*) = 0, 모든 bucket 일치, NashConv(σ*) = 1 = H_{K²}', '정방향만으로는 정지'],
      ['원 게임 균형', 'H 콜, L 폴드, 플레이어 2 체크. 플레이어 1의 값 0 (σ*에서는 −0.5)', '역방향 → 정방향 두 판정 뒤 도달'],
    ], { x: ML, y: 1.35, w: CW, colW: [2.4, 6.4, 3.33], rowH: 0.5, size: 16 });
    caption(s, '출처: 증명 초안 §7–§8. 덜 극단적인 경우(베팅 보상 −1)에는 L의 첫 결정에서 베팅이 나아 B₁이 정방향으로 나뉘고 문제가 저절로 풀린다.', 6.45, { h: 0.5 });
    s.addNotes('본문의 예시 게임을 숫자로 끝까지 계산한 표입니다. 첫 결정에서는 두 카드 모두 체크가 압도적이라 나뉠 이유가 없고, 두 번째 결정에서는 H와 L이 같은 이력이라 L의 양의 후회가 상쇄됩니다. 그 결과 bucket 후회가 0이고 모든 bucket이 일치하는데 NashConv는 1입니다. 남은 오차는 정확히 이력 안 상쇄와 같습니다.');
  }
  {
    const s = app(7, 'A7. 실험 설정');
    table(s, [
      ['', 'mini-stud', 'Kuhn · Leduc', '7-stud'],
      ['게임', '정보집합 51,114', 'Leduc 정보집합 288 (rank만)', '헤즈업, 5구부터 학습'],
      ['학습기', 'MCCFR+ (외부 샘플링), 라벨은 정확 계산', '정확한 전체 트리 CFR+', 'MCCFR+ (외부 샘플링), 라벨은 표본'],
      ['학습량', '4,096만 반복', '65만 5,350반복', '100M roots / 5.11M roots'],
      ['seed', '3', 'P₀ + 무작위 출발 20 (Leduc)', '1'],
      ['판정', 'epoch마다 (두 배 간격)', 'epoch마다', 'epoch마다, 라벨 표본 상한'],
      ['지표', '평균 정책 exploitability', '평균 정책 exploitability, D/2', '평균 정책 LBR (64 particles, 20,000 pair)'],
      ['비교군', 'k-means 고정, 정보집합 단위, 이력 정방향만', '같은 bucket 수의 k-means, 정방향만 + 트리 유지', 'k-means 256 / 1024, 방법 B, 구간 라벨'],
    ], { x: ML, y: 1.35, w: CW, colW: [1.5, 3.35, 3.65, 3.63], rowH: 0.56, size: 15 });
    caption(s, '출처: 원본 슬라이드와 aswiss/README_KO.md, aswiss/small/README_KO.md, aswiss/stud7/README_KO.md. 4구 버림·공개는 휴리스틱(power 군집).', 6.4, { h: 0.5 });
    s.addNotes('실험별 설정을 한 표에 모았습니다. 작은 게임은 정확한 CFR+와 정확한 exploitability, 7-stud는 표본 학습기와 LBR이라는 점이 가장 큰 차이입니다. 7-stud는 seed가 하나라는 점도 한계로 함께 기억해 주십시오.');
  }
  {
    const s = app(8, 'A8. 비교 조건의 차이 (같은 조건의 비교가 아닌 것)');
    table(s, [
      ['mini-stud 비교군 (4,096만 반복)', 'bucket 수 (seed 1/2/3)'],
      ['정보집합 단위, 매 epoch', '12,683 / 12,839 / 12,888'],
      ['bucket 이력, 정방향만', '12,895 / 13,132 / 13,158'],
      ['bucket 이력 + 역방향 (ASWISS)', '15,745 / 16,304 / 15,811'],
      ['카드 k-means 고정', '7,051'],
    ], { x: ML, y: 1.35, w: 6.2, colW: [3.3, 2.9], rowH: 0.48, size: 15 });
    T(s, '**7-stud에서 다른 것**\n- 100M과 5.11M은 학습량과 버전(역방향 유무)이 다르다\n- 같은 패널 안에서도 bucket 수가 다르다 (예: 86만 대 14.5만)\n- 5.11M에서는 k-means 256도 측정했지만 여기서는 더 강한 k-means 1024만 보였다\n**맞춘 것**: 같은 deal, LBR seed, 64 particles, 20,000 pair로 평가했고 평균 정책끼리 비교했다\n**비율 (원본 값으로 계산)**: 100M 방법 B / k-means 1024 = 0.83, 5.11M 완성형 / k-means 1024 = 1.29', { x: 7.05, y: 1.35, w: 5.68, h: 5.0, size: 15, spaceAfter: 6 });
    T(s, '**mini-stud의 대가**: 역방향 분할 뒤 H_{K} = 0이 되었을 때 bucket 이력이 정보집합의 99.4–99.8%를 구별했다. 이 게임에서 H_{K} = 0을 얻으려면 이력이 사실상 정보집합을 다 기억해야 했다. 큰 게임에서도 그런지는 미해결.', { x: ML, y: 4.0, w: 6.2, h: 1.9, size: 15, fill: P.pale, rect: 0.08, margin: [6, 10, 6, 10], valign: 'middle' });
    caption(s, '원본 PDF에 없던 수: mini-stud bucket 수(aswiss/README_KO.md)와 99.4–99.8%(증명 초안 §8). 비율은 원본 값에서 계산했다.', 6.4, { h: 0.5 });
    s.addNotes('공정한 비교를 위해 조건이 다른 부분을 모았습니다. mini-stud에서는 비교군마다 최종 bucket 수가 다르고, ASWISS가 가장 많은 bucket을 썼습니다. 또 역방향 분할로 이력 안 상쇄를 0으로 만들었을 때 bucket 이력이 정보집합을 거의 다 구별했다는 점은 이 방법의 대가로 기억해야 합니다. 7-stud에서는 두 실행의 학습량과 버전이 다르고, 같은 실행 안에서도 bucket 수가 다릅니다. 평가 조건은 같은 deal과 같은 LBR 설정으로 맞췄습니다.');
  }
  {
    const s = app(9, 'A9. 판정 문턱 실험의 세부');
    T(s, '**mini-stud (정확 계산, 1만 반복마다 검사)**\n- ε = 0.2 + 역방향: 29–30.5M에 "문턱 통과 + 모든 bucket 일치 + H_{K} = 0" 판정 (세 seed). 인증 ρ/2 = 0.19–0.20, 실제 NashConv/2 = 0.034–0.038\n- 역방향이 없으면 같은 상태에서 H_{K}/2 ≈ 0.40이 남아 인증 0.59–0.60, 실제 값도 두 배\n- CFR 평균은 계속 바뀌므로 학습을 이어 가면 라벨이 다시 갈려 분할이 이어졌다\n- ε = 0.05, 0.1은 4천만 반복 동안 판정이 5–8번뿐. ε/√t 문턱은 고정 분할에서 ρ도 t^{−1/2}로 줄어 추상화를 얼린다\n**7-stud (표본, 5.11M roots, seed 1)**\n- ε = 1: bucket 36개, ρ̂ = 0.86에서 통과했지만 LBR 7.5 (같은 시점 k-means 1024의 7.6배)\n- 자식이 많이 생기면 ρ가 크게 뛴다 (149 → 956 bucket 판정 직후 2.40 → 54.0)\n- 학습 순회의 표본 후회 합은 오라클 조건을 대신하지 못한다', { x: ML, y: 1.35, w: CW, h: 5.3, size: 16, spaceAfter: 6 });
    caption(s, '출처: 원본 슬라이드(ε = 1, 정지 상태 도달)와 증명 초안 §6. 원본 PDF에 없던 수: 0.40, 0.59–0.60, 5–8번, 2.40 → 54.0.', 6.6);
    s.addNotes('본문의 문턱 실험을 더 자세히 적었습니다. mini-stud에서는 문턱을 ε 0.2로 두었을 때 이론의 정지 상태가 실제로 나왔고, 실제 값이 인증값 아래에 있었습니다. 그러나 문턱이 작으면 거의 판정하지 못했고, 시간에 따라 줄어드는 문턱은 추상화를 얼렸습니다. 7-stud에서는 문턱이 정책을 오히려 나쁘게 했습니다.');
  }
  {
    const s = app(10, 'A10. 원본 이후 작업 (참고): 라벨용 혼합 평균');
    T(s, '{{orange:원본 PDF에 없는 내용이다. 발표 본문에서는 쓰지 않았다.}}', { x: ML, y: 1.3, w: CW, h: 0.45, size: 16 });
    T(s, '**문제** · 평균 전략은 자기가 거의 가지 않게 된 곳에서 초반 값에 멈춘다. 정책에는 영향이 없지만(실현 계획의 질량이 0), 자주 가는 정보집합의 라벨이 그 아래의 멈춘 값으로 계산되어 "그 행동이 사실 더 좋다"는 분할 신호를 놓친다.\n**규칙 (--avg-fill plain)** · 판정에 쓰는 전략만 바꾼다. 평균 전략으로 자기 도달확률이 10^{−4} 이상인 곳은 평균 전략, 그보다 작은 곳은 그 bucket의 현재 전략들의 단순 평균. 내놓는 정책은 그대로다.\n**지금까지의 결과** · 명제 B4(a) 반례: 평균 전략만으로는 끝까지 안 나뉘었고(NashConv 1), 혼합 평균으로는 150반복에 나뉘어 0으로 수렴. Leduc 21개 출발점: 정책과 bucket 수는 그대로, 판정에 쓴 평균의 ρ_{P}/2가 0.090 → 0.00029. mini-stud: 차이 없음 (0.0277 → 0.0271).\n**한계** · 이력 안 상쇄(명제 H4)는 풀지 못한다. 문턱 10^{−4} 한 값만 시험했다. 증명은 없다. 7-stud 100M 비교(hist_bbf 대 hist_bb)는 진행 중이다.', { x: ML, y: 1.85, w: CW, h: 4.75, size: 17, spaceAfter: 12 });
    caption(s, '출처: aswiss/AVG_FILL_KO.md. 필요 없으면 이 장은 지워도 본문에 영향이 없다.', 6.6);
    s.addNotes('질의응답에 대비한 참고 장입니다. 원본 발표 이후에 한 작업으로, 학습기 조건 문제 중 경로 밖 부분을 라벨 계산에서만 우회하는 방법입니다. 내놓는 정책은 바꾸지 않습니다. 작은 게임에서는 경로 밖 라벨 때문에 멈췄던 실행이 풀렸고, 7-stud에서의 효과는 1억 roots 실행으로 확인하는 중입니다.');
  }

  await pres.writeFile({ fileName: OUT });
  await postprocess(OUT, THEME);
  console.log('wrote', OUT);
}

build().catch((e) => { console.error(e); process.exit(1); });
