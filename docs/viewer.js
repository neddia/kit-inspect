const el = id => document.getElementById(id);
let records = [];
let selected;
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
  const boxes = [...(selected.detections || []), ...(selected.unknown_objects || [])];
  ctx.font = 'bold 18px monospace';
  for (const item of boxes) {
    const [x,y,right,bottom] = item.box;
    const label = item.cls || 'unknown';
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
  selected = records.find(r => '#inspection-' + r.id === location.hash) || records[0];
  if (!selected) return;
  const row = selected;
  el('cycle').textContent = cycle(row);
  el('stamp').textContent = row.time.slice(0,19).replace('T',' ');
  badge(el('verdict'),row.result);
  el('reason').textContent = reason(row);
  const frame = el('frame');
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
  const counts = row.counts;
  el('counts').textContent = Object.entries(counts?.spec || {}).map(([part,n]) =>
    part + ': ' + (counts.sure?.[part] || 0) + ' confirmed / ' + n + ' expected').join(' · ');
  for (const a of el('records').children) a.setAttribute('aria-current',String(a.hash === '#inspection-'+row.id));
}
async function start() {
  try {
    const response = await fetch('inspections.json',{cache:'no-store'});
    if (!response.ok) throw Error();
    const data = await response.json();
    el('title').textContent = data.source;
    records = data.records;
    if (!records.length) { el('status').textContent = 'No inspections saved yet. Run a station cycle, then refresh.'; return; }
    for (const row of records) {
      const a = document.createElement('a'), head = document.createElement('div'), title = document.createElement('span'),
        status = document.createElement('span'), message = document.createElement('p');
      a.href = '#inspection-'+row.id; a.className = 'record'; head.className = 'recordhead'; status.className = 'badge';
      title.textContent = cycle(row); badge(status,row.result); message.textContent = reason(row);
      head.append(title,status); a.append(head,message); el('records').append(a);
      a.addEventListener('click',() => {if (matchMedia('(max-width:760px)').matches) el('layout').scrollIntoView();});
    }
    el('status').hidden = true; el('layout').hidden = false; display();
  } catch { el('status').textContent = 'Inspection history could not be loaded. Refresh to retry.'; }
}
el('frame').onload = draw;
el('frame').onerror = () => {el('frame').hidden = true; el('overlay').hidden = true; el('noFrame').hidden = false;};
el('boxes').onchange = draw;
window.addEventListener('hashchange',display);
start();
