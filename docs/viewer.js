const el = id => document.getElementById(id);
let records = [];
let selected;
let initialId;
function cycle(row) {
  return row.cycle_id == null ? 'History #' + row.id : 'Cycle ' + row.cycle_id;
}
function reason(row) {
  const message = row.error || row.reasons?.join(' · ') || 'No reason recorded';
  return message[0].toUpperCase() + message.slice(1);
}
function badge(target, result) {
  target.textContent = result;
  target.dataset.result = result;
}
// Same colours and captions as the station window (src/inspection/draw.py).
const COLOR = {sure: '#3cc83c', unsure: '#ffc800', foreign: '#ff00ff', ignored: '#969696',
  ok: '#3cc83c', mismatch: '#e63c3c', doubt: '#ffc800'};
let band = {low: 0.2, high: 0.7};
function iou(a, b) {
  const w = Math.min(a[2], b[2]) - Math.max(a[0], b[0]), h = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
  if (w <= 0 || h <= 0) return 0;
  const area = r => (r[2] - r[0]) * (r[3] - r[1]);
  return w * h / (area(a) + area(b) - w * h);
}
function tag(ctx, text, x, y, color, size) {
  const top = Math.max(0, y - size - 6);
  ctx.fillStyle = color;
  ctx.fillRect(x, top, ctx.measureText(text).width + 10, size + 6);
  ctx.fillStyle = '#000';
  ctx.fillText(text, x + 5, top + size);
}
function draw() {
  const frame = el('frame'), canvas = el('overlay');
  canvas.width = frame.naturalWidth;
  canvas.height = frame.naturalHeight;
  if (!canvas.width || !selected || !el('boxes').checked) return;
  const ctx = canvas.getContext('2d');
  const size = Math.round(canvas.width / 64), thick = Math.max(2, Math.round(canvas.width / 400));
  ctx.font = `bold ${size}px ui-monospace, monospace`;
  ctx.lineWidth = thick;
  const label = selected.label;
  if (label?.box) {
    const color = COLOR[label.status] || COLOR.doubt;
    ctx.strokeStyle = color;
    ctx.lineWidth = thick + 2;
    ctx.beginPath();
    label.box.forEach(([x, y], i) => i ? ctx.lineTo(x, y) : ctx.moveTo(x, y));
    ctx.closePath();
    ctx.stroke();
    ctx.lineWidth = thick;
    const [x, y] = label.box.reduce((a, b) => a[1] >= b[1] ? a : b);
    const text = `label ${label.text} ${(label.conf ?? 1).toFixed(2)}`;
    tag(ctx, text, Math.min(x, canvas.width - ctx.measureText(text).width - 10), Math.min(y + size + 10, canvas.height), color, size);
  }
  // Boxes on the same object (e.g. a coin read as "other" and as a weak "nut") share one caption.
  const shown = (selected.detections || []).filter(d => d.in_bowl !== false || d.label_ambiguous);
  const colorOf = d => d.in_bowl === false ? COLOR.ignored : d.cls === 'other' ? COLOR.foreign
    : d.conf >= band.high ? COLOR.sure : COLOR.unsure;
  const rank = d => d.in_bowl === false ? 0 : d.cls === 'other' ? 3 : d.conf >= band.high ? 1 : 2;
  const groups = [];
  for (const d of shown) {
    const g = groups.find(g => iou(g[0].box, d.box) > 0.5);
    g ? g.push(d) : groups.push([d]);
  }
  for (const g of groups) {
    g.sort((a, b) => rank(b) - rank(a));
    for (const d of [...g].reverse()) {
      const [x, y, right, bottom] = d.box;
      ctx.strokeStyle = colorOf(d);
      ctx.strokeRect(x, y, right - x, bottom - y);
    }
    const caption = g.map(d => d.label_ambiguous ? 'check label area' : `${d.cls} ${d.conf.toFixed(2)}`).join(' / ');
    tag(ctx, caption, g[0].box[0], g[0].box[1], colorOf(g[0]), size);
  }
  for (const u of selected.unknown_objects || []) {
    const [x, y, right, bottom] = u.box;
    ctx.strokeStyle = COLOR.foreign;
    ctx.lineWidth = thick + 1;
    ctx.strokeRect(x - 6, y - 6, right - x + 12, bottom - y + 12);
    tag(ctx, 'unknown', x - 6, y - 6, COLOR.foreign, size);
  }
}
function partLine(row) {
  // One line per part: found / expected, coloured like the boxes. Unsure parts are named with their confidence.
  const counts = row.counts, wrap = document.createElement('div');
  wrap.className = 'parts';
  if (!Object.keys(counts?.spec || {}).length) return wrap;
  const dets = (row.detections || []).filter(d => d.in_bowl !== false);
  const items = Object.entries(counts.spec).filter(([part]) => part !== 'other').map(([part, n]) => [part, n, counts.sure[part] || 0,
    dets.filter(d => d.cls === part && d.conf < band.high).map(d => d.conf)]);
  const foreign = dets.filter(d => d.cls === 'other').length + (row.unknown_objects || []).length;
  if (foreign) items.push(['foreign object', 0, foreign, []]);
  for (const [part, n, sure, unsure] of items) {
    const chip = document.createElement('span');
    chip.className = 'part';
    // like decision.py: too many sure, or too few even counting the unsure = wrong; unsure could change it = unsure
    chip.dataset.state = sure > n || sure + unsure.length < n ? 'wrong' : unsure.length ? 'unsure' : 'ok';
    const name = document.createElement('b');
    name.textContent = part;
    chip.append(name, part === 'foreign object' ? ` ${sure}` : ` ${sure} of ${n}` + (unsure.length ? ` + ${unsure.length} unsure (${unsure.map(c => c.toFixed(2)).join(', ')})` : ''));
    wrap.append(chip);
  }
  return wrap;
}
function legend() {
  const p = document.createElement('p');
  p.className = 'legend';
  for (const [color, text] of [[COLOR.sure, `sure (≥ ${band.high})`], [COLOR.unsure, `unsure (${band.low}–${band.high}) → REVIEW`],
    [COLOR.foreign, 'foreign object'], [COLOR.ignored, 'ignored (kit label)']]) {
    const key = document.createElement('span'), dot = document.createElement('i');
    dot.style.background = color;
    key.append(dot, text);
    p.append(key);
  }
  return p;
}
function display() {
  selected = records.find(r => '#inspection-' + r.id === location.hash) || records.find(r => r.id === initialId) || records[0];
  if (!selected) return;
  const row = selected;
  if (el('filter').value && el('filter').value !== row.result) {
    el('filter').value = '';
    filterList();
  }
  el('cycle').textContent = cycle(row);
  el('stamp').textContent = row.source_photo || row.time.slice(0,19).replace('T',' ');
  el('truth').hidden = !row.truth;
  el('truth').textContent = row.truth ? 'On the tray: ' + row.truth : '';
  badge(el('verdict'),row.result);
  el('reason').textContent = reason(row);
  const frame = el('frame');
  el('frameLegend').textContent = row.label?.located_on_replay ? 'Label outline from OCR replay'
    : row.label?.box ? 'Printed kit tag located by OCR' : 'Station camera';
  frame.hidden = !row.image;
  el('noFrame').hidden = Boolean(row.image);
  el('overlay').hidden = !row.image;
  if (row.image) {
    frame.alt = 'Tray photographed for ' + cycle(row);
    frame.src = row.image;
  } else frame.removeAttribute('src');
  el('facts').replaceChildren();
  for (const [key,value] of [['PLC recipe',row.kit_id],['Label read',row.label?.text],
    ['Trigger to result',row.ms_total == null ? null : Math.round(row.ms_total)+' ms']]) {
    const pair = document.createElement('div'), term = document.createElement('dt'), detail = document.createElement('dd');
    term.textContent = key; detail.textContent = value || '—'; pair.append(term,detail); el('facts').append(pair);
  }
  if (row.label?.passes != null) {
    const detail = document.createElement('div');
    detail.innerHTML = '<dt>OCR</dt><dd></dd>';
    detail.lastChild.textContent = row.label.passes + (row.label.passes === 1 ? ' pass · ' : ' passes · ') + Math.round(row.label.ms) + ' ms';
    el('facts').append(detail);
  }
  el('counts').replaceChildren(partLine(row), legend());
  const list = el('records');
  for (const a of list.children) {
    const active = a.hash === '#inspection-' + row.id;
    a.setAttribute('aria-current', String(active));
    if (active) {
      const top = a.getBoundingClientRect().top - list.getBoundingClientRect().top;
      if (top < 0) list.scrollTop += top;
      else if (top + a.offsetHeight > list.clientHeight) list.scrollTop += top + a.offsetHeight - list.clientHeight;
    }
  }
}
function filterList() {
  for (const a of el('records').children) a.hidden = Boolean(el('filter').value && a.dataset.result !== el('filter').value);
}
function summarize() {
  const totals = new Map();
  for (const row of records) totals.set(row.result, (totals.get(row.result) || 0) + 1);
  const times = records.map(r => r.ms_total).filter(Number.isFinite).sort((a,b) => a-b);
  const metrics = [[records[0].source_photo ? 'Replay cycles' : 'Records', records.length]];
  for (const verdict of ['PASS','FAIL','REVIEW','FAULT']) {
    if (verdict !== 'FAULT' || totals.has(verdict)) metrics.push([verdict, totals.get(verdict) || 0]);
    if (totals.has(verdict)) el('filter').add(new Option(verdict,verdict));
  }
  if (times.length) {
    const mid = Math.floor(times.length / 2);
    const median = (times[mid] + times[Math.floor((times.length-1)/2)]) / 2;
    metrics.push(['Median cycle time', Math.round(median) + ' ms']);
  }
  for (const [name,value] of metrics) {
    const cell = document.createElement('div'), term = document.createElement('dt'), detail = document.createElement('dd');
    term.textContent = name; detail.textContent = value; cell.append(term,detail); el('summary').append(cell);
  }
  el('summary').hidden = false;
}
async function start() {
  try {
    const response = await fetch('inspections.json',{cache:'no-store'});
    if (!response.ok) throw Error();
    const data = await response.json();
    el('title').textContent = data.source;
    records = data.records;
    initialId = data.initial_id;
    band = data.band || band;
    el('evaluation').hidden = !data.description;
    el('intro').textContent = data.description || (initialId ? 'One recorded bench run. Select a cycle to see what the station found.'
      : 'Latest saved station records. Refresh to load new cycles.');
    if (!records.length) { el('status').textContent = 'No inspections saved yet. Run a station cycle, then refresh.'; return; }
    summarize();
    for (const row of records) {
      const a = document.createElement('a'), head = document.createElement('div'), title = document.createElement('span'),
        status = document.createElement('span'), message = document.createElement('p');
      a.href = '#inspection-'+row.id; a.className = 'record'; a.dataset.result = row.result; head.className = 'recordhead'; status.className = 'badge';
      title.textContent = cycle(row); badge(status,row.result); message.textContent = reason(row);
      head.append(title,status); a.append(head,message); el('records').append(a);
      a.addEventListener('click',() => {if (matchMedia('(max-width:760px)').matches) el('layout').scrollIntoView();});
    }
    el('status').hidden = true; el('layout').hidden = false; display();
  } catch { el('status').textContent = 'Inspection history could not be loaded. Refresh to retry.'; }
}
el('filter').onchange = () => {
  filterList();
  if (el('filter').value && selected.result !== el('filter').value)
    location.hash = el('records').querySelector('a:not([hidden])').hash;
};
el('frame').onload = draw;
el('frame').onerror = () => {el('frame').hidden = true; el('overlay').hidden = true; el('noFrame').hidden = false;};
el('boxes').onchange = draw;
window.addEventListener('hashchange',display);
start();
