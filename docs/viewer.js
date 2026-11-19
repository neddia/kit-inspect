const el = id => document.getElementById(id);
let records = [];
let selected;
let initialId;
let replay = false;

// Same colours and captions as the station window (src/inspection/draw.py).
const COLOR = {sure: '#3cc83c', unsure: '#ffc800', foreign: '#ff00ff', ignored: '#969696',
  ok: '#3cc83c', mismatch: '#e63c3c', doubt: '#ffc800'};
let band = {low: 0.2, high: 0.7};

function cycle(row) {
  return row.cycle_id == null ? 'History #' + row.id : (replay ? 'Tray ' : 'Cycle ') + row.cycle_id;
}
function sentence(text) {
  return text ? text[0].toUpperCase() + text.slice(1) : text;
}
function reasons(row) {
  if (row.result === 'FAIL' && row.label?.status === 'mismatch')
    return ['Wrong kit for this order: PLC ordered ' + row.kit_id + ', tag says ' + row.label.text];
  if (row.error) return [row.error];
  return (row.reasons || ['no reason recorded']).map(sentence);
}
function badge(target, result) {
  target.textContent = result;
  target.dataset.result = result;
}
// Was the station right? Only known when the tray's truth was recorded.
function correct(row) {
  if (!row.expected) return null;
  return row.expected === 'PASS' ? row.result === 'PASS' : row.result !== 'PASS';
}
function verdictNote(row) {
  const ok = correct(row);
  if (ok === null) return '';
  if (!ok) return row.expected === 'PASS' ? 'Station wrong: a good kit was failed' : 'Station wrong: a faulty kit passed';
  if (row.expected === 'PASS') return 'Station right: released';
  return row.result === 'REVIEW' ? 'Station right: held for an operator' : 'Station right: held';
}

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
  const size = Math.round(canvas.width / 56), thick = Math.max(2, Math.round(canvas.width / 400));
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
    const text = `tag ${label.text} ${(label.conf ?? 1).toFixed(2)}`;
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
    const name = d => d.cls === 'other' ? 'foreign' : d.cls;
    const caption = g.map(d => d.label_ambiguous ? 'check tag area' : `${name(d)} ${d.conf.toFixed(2)}`).join(' / ');
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
function legend() {
  const p = el('legend');
  p.replaceChildren();
  for (const [color, text] of [[COLOR.sure, 'sure'], [COLOR.unsure, 'unsure → REVIEW'],
    [COLOR.foreign, 'foreign object'], [COLOR.ok, 'kit tag (OCR)']]) {
    const key = document.createElement('span'), dot = document.createElement('i');
    dot.style.borderColor = color;
    key.append(dot, text);
    p.append(key);
  }
}

function display() {
  selected = records.find(r => '#inspection-' + r.id === location.hash) || records.find(r => r.id === initialId) || records[0];
  if (!selected) return;
  const row = selected;
  if (el('filter').value && el('filter').value !== row.result) {
    el('filter').value = '';
    filterList();
  }
  // The banner: what the station window shows above the camera picture.
  badge(el('verdict'), row.result);
  el('banner').dataset.result = row.result;
  const released = row.result === 'PASS';
  el('plc').textContent = 'PLC: ' + (released ? 'RELEASE' : 'HOLD');
  el('plc').dataset.released = String(released);
  el('reasons').replaceChildren(...reasons(row).map(text => {
    const li = document.createElement('li'); li.textContent = text; return li;
  }));
  const facts = [cycle(row), row.source_photo ? 'photo ' + row.source_photo : row.time.slice(0, 19).replace('T', ' '),
    'order ' + (row.kit_id || '—'), 'tag ' + (row.label?.text || 'not read'),
    row.ms_total == null ? null : Math.round(row.ms_total) + ' ms'].filter(Boolean);
  el('facts').textContent = facts.join(' · ');

  const ok = correct(row);
  el('truth').hidden = !row.truth;
  el('truth').replaceChildren();
  if (row.truth) {
    const head = document.createElement('b'), note = document.createElement('span');
    head.textContent = 'Checked by eye: ';
    note.className = 'check';
    note.dataset.ok = String(ok);
    note.textContent = verdictNote(row);
    el('truth').append(head, row.truth + ' ', note);
  }

  const frame = el('frame');
  frame.hidden = !row.image;
  el('noFrame').hidden = Boolean(row.image);
  el('overlay').hidden = !row.image;
  if (row.image) {
    frame.alt = 'Camera photo for ' + cycle(row) + ': ' + (row.truth || row.result);
    frame.src = row.image;
  } else frame.removeAttribute('src');

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
  // Live station history only: counts over a hand-picked replay would mean nothing.
  const totals = new Map();
  for (const row of records) totals.set(row.result, (totals.get(row.result) || 0) + 1);
  const times = records.map(r => r.ms_total).filter(Number.isFinite).sort((a,b) => a-b);
  const metrics = [['Records', records.length]];
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
    replay = Boolean(data.description);          // the published replay, not a live station's history
    document.body.classList.toggle('replay', replay);
    el('about').hidden = !replay;
    el('filterLabel').hidden = replay;
    el('listTitle').textContent = replay ? 'Trays' : 'Cycle history';
    el('title').textContent = data.source;
    records = data.records;
    initialId = data.initial_id;
    band = data.band || band;
    el('intro').textContent = data.description || 'Latest saved station records. Refresh to load new cycles.';
    if (!records.length) { el('status').textContent = 'No inspections saved yet. Run a station cycle, then refresh.'; return; }
    if (!replay) summarize();
    legend();
    for (const row of records) {
      const a = document.createElement('a'), head = document.createElement('div'), title = document.createElement('span'),
        status = document.createElement('span'), message = document.createElement('p');
      a.href = '#inspection-'+row.id; a.className = 'record'; a.dataset.result = row.result; head.className = 'recordhead'; status.className = 'badge';
      title.textContent = cycle(row); badge(status,row.result); message.textContent = reasons(row).join(' · ');
      head.append(title,status); a.append(head,message);
      const ok = correct(row);
      if (ok !== null) {
        const mark = document.createElement('p');
        mark.className = 'check'; mark.dataset.ok = String(ok);
        mark.textContent = ok ? 'station right' : row.expected === 'PASS' ? 'station wrong: good kit failed' : 'station wrong: faulty kit passed';
        a.append(mark);
      }
      el('records').append(a);
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
