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
function draw() {
  const frame = el('frame'), canvas = el('overlay');
  canvas.width = frame.naturalWidth;
  canvas.height = frame.naturalHeight;
  if (!canvas.width || !selected || !el('boxes').checked) return;
  const ctx = canvas.getContext('2d');
  const boxes = (selected.detections || []).filter(d => d.in_bowl !== false || d.label_ambiguous);
  boxes.push(...(selected.unknown_objects || []));
  const labelBox = selected.label?.box;
  if (labelBox) {
    ctx.strokeStyle = '#71f2ed';
    ctx.lineWidth = 4;
    ctx.beginPath();
    labelBox.forEach(([x,y], i) => i ? ctx.lineTo(x,y) : ctx.moveTo(x,y));
    ctx.closePath();
    ctx.stroke();
  }
  // A recipe mismatch ends the inspection before part counts are evaluated.
  if (selected.label?.status === 'mismatch') return;
  ctx.font = 'bold 18px monospace';
  for (const item of boxes) {
    const [x,y,right,bottom] = item.box;
    const label = item.label_ambiguous ? 'check label area' : item.cls || 'unknown';
    ctx.strokeStyle = ctx.fillStyle = '#ffce6c';
    ctx.lineWidth = 3;
    ctx.strokeRect(x,y,right-x,bottom-y);
    const top = Math.max(0,y-22);
    ctx.fillRect(x,top,ctx.measureText(label).width+9,22);
    ctx.fillStyle = '#15262d';
    ctx.fillText(label,x+4,top+17);
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
  el('cycle').textContent = cycle(row);
  el('stamp').textContent = row.time.slice(0,19).replace('T',' ');
  badge(el('verdict'),row.result);
  el('reason').textContent = reason(row);
  const frame = el('frame');
  el('frameLegend').textContent = row.label?.located_on_replay ? 'Label outline from OCR replay'
    : row.label?.box ? 'Kit label outlined in cyan' : 'Station camera';
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
  const counts = row.counts;
  el('counts').replaceChildren();
  if (Object.keys(counts?.spec || {}).length) {
    const table = document.createElement('table');
    table.innerHTML = '<caption>Part count</caption><thead><tr><th>Part</th><th>Expected</th><th>Confirmed</th><th>Uncertain</th></tr></thead>';
    const body = table.createTBody();
    for (const [part, n] of Object.entries(counts.spec)) {
      const tr = body.insertRow();
      for (const value of [part, n, counts.sure[part] || 0, counts.unsure[part] || 0]) tr.insertCell().textContent = value;
    }
    el('counts').append(table);
  } else if (row.label?.status === 'mismatch') {
    el('counts').textContent = 'The wrong kit label failed this inspection before the part-count check.';
  }
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
    el('title').textContent = data.source;
    records = data.records;
    initialId = data.initial_id;
    el('intro').textContent = initialId ? 'One recorded bench run. Select a cycle to see what the station found.'
      : 'Latest saved station records. Refresh to load new cycles.';
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
