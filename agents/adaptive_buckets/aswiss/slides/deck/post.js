'use strict';
// Post-processing of the pptxgenjs output (things the library does not expose):
//  - theme colours (as the pptx skill's apply_theme) and an East Asian theme font for Hangul
//  - per-series line width / dash and hidden legend entries on the line charts
//  - per-point data-label text ("1.060 ± 0.143") on the 7-stud bar charts
const fs = require('fs');
const JSZip = require(require.resolve('jszip', { paths: [require.resolve('pptxgenjs')] }));

const EA_FONT = '맑은 고딕';
const SLOTS = ['dk1', 'lt1', 'dk2', 'lt2', 'accent1', 'accent2', 'accent3', 'accent4', 'accent5', 'accent6', 'hlink', 'folHlink'];

// line series styling, by series name
const LINE_STYLE = {
  '추상화 없는 CFR+': { w: 2.25 },
  'ASWISS, 무작위 출발 20개 (중앙값)': { w: 3.5 },
  'ASWISS, P₀에서': { w: 2.5, dash: 'dash' },
  '같은 bucket 수의 k-means (중앙값)': { w: 3 },
  '정방향만 + 트리 유지, P₀에서': { w: 3 },
  'ASWISS 최소': { w: 1, hide: true }, 'ASWISS 최대': { w: 1, hide: true },
  'k-means 최소': { w: 1, hide: true }, 'k-means 최대': { w: 1, hide: true },
  '트리 (부모 bucket 하나)': { w: 3.5 },
  '자유 (과거를 무작위로 섞음)': { w: 3.5 },
  '추상화 없음': { w: 2, dash: 'dash' },
  '트리 최소': { w: 1, hide: true }, '트리 최대': { w: 1, hide: true },
  '자유 최소': { w: 1, hide: true }, '자유 최대': { w: 1, hide: true },
};
// custom per-point data labels, by series name
const POINT_LABELS = {
  LBR100M: ['1.060 ± 0.143', '1.023 ± 0.140', '1.282 ± 0.130', '1.603 ± 0.156'],
  LBR5M: ['1.575 ± 0.146', '1.805 ± 0.156', '1.223 ± 0.136'],
};

const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

function patchChart(xml) {
  const hidden = [];
  xml = xml.replace(/<c:ser>[\s\S]*?<\/c:ser>/g, (ser) => {
    const name = (ser.match(/<c:tx>[\s\S]*?<c:v>([^<]*)<\/c:v>/) || [])[1];
    const idx = (ser.match(/<c:idx val="(\d+)"\/>/) || [])[1];
    if (!name) return ser;
    const st = LINE_STYLE[name];
    if (st) {
      ser = ser.replace(/(<c:spPr>\s*<a:ln )w="\d+"/, `$1w="${Math.round(st.w * 12700)}"`);
      if (st.dash) ser = ser.replace(/(<c:spPr>[\s\S]*?)<a:prstDash val="[a-zA-Z]+"\/>/, `$1<a:prstDash val="${st.dash}"/>`);
      if (st.hide) hidden.push(idx);
    }
    const labels = POINT_LABELS[name];
    if (labels) {
      const rPr = (sz) => `<a:rPr lang="ko-KR" sz="${sz}" b="0"><a:solidFill><a:srgbClr val="1B2430"/></a:solidFill><a:latin typeface="+mn-lt"/><a:ea typeface="+mn-ea"/></a:rPr>`;
      const dl = labels.map((t, i) => `<c:dLbl><c:idx val="${i}"/><c:tx><c:rich><a:bodyPr wrap="none"/><a:lstStyle/><a:p><a:r>${rPr(1500)}<a:t>${esc(t)}</a:t></a:r></a:p></c:rich></c:tx><c:dLblPos val="outEnd"/><c:showLegendKey val="0"/><c:showVal val="1"/><c:showCatName val="0"/><c:showSerName val="0"/><c:showPercent val="0"/><c:showBubbleSize val="0"/></c:dLbl>`).join('');
      if (!/<c:dLbls>/.test(ser)) throw new Error('no dLbls in series ' + name);
      ser = ser.replace('<c:dLbls>', '<c:dLbls>' + dl);
    }
    return ser;
  });
  if (hidden.length) {
    const entries = hidden.map((i) => `<c:legendEntry><c:idx val="${i}"/><c:delete val="1"/></c:legendEntry>`).join('');
    xml = xml.replace(/(<c:legendPos val="[a-z]+"\/>)/, `$1${entries}`);
  }
  return xml;
}

async function postprocess(file, theme) {
  const zip = await JSZip.loadAsync(fs.readFileSync(file));
  const tp = 'ppt/theme/theme1.xml';
  let th = await zip.file(tp).async('string');
  const scheme = `<a:clrScheme name="${theme.name}">` + SLOTS.map((k) => `<a:${k}><a:srgbClr val="${theme.colors[k]}"/></a:${k}>`).join('') + '</a:clrScheme>';
  th = th.replace(/<a:clrScheme\b[\s\S]*?<\/a:clrScheme>/, scheme)
    .replace(/(<a:(?:theme|fontScheme)\b[^>]*?\bname=")[^"]*"/g, `$1${theme.name}"`)
    .replace(/<a:ea typeface=""\/>/g, `<a:ea typeface="${EA_FONT}"/>`);
  if (!th.includes(scheme)) throw new Error('theme colour scheme not replaced');
  zip.file(tp, th);
  for (const name of Object.keys(zip.files)) {
    if (/^ppt\/charts\/chart\d+\.xml$/.test(name)) zip.file(name, patchChart(await zip.file(name).async('string')));
  }
  fs.writeFileSync(file, await zip.generateAsync({ type: 'nodebuffer', compression: 'DEFLATE' }));
}

module.exports = { postprocess };
