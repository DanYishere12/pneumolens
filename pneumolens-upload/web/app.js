import {ScanViewer} from './viewer.mjs';
import {recordedRequest} from './recorded.mjs';

const $ = id => document.getElementById(id);
const labels = {tp:'True positive',tn:'True negative',fp:'False positive',fn:'False negative'};
const state = {model:'baseline', target:'PNEUMONIA', mode:'compare', filter:'all', offset:0, cases:[], caseId:null, upload:null, uploadId:0, meta:null, result:null, request:0, galleryRequest:0};
let controller;
let sourceRevision = 0;
let recordedData;
const cache = new Map();
const titleCase = label => label ? label[0] + label.slice(1).toLowerCase() : 'Not provided';
const percent = value => value == null ? '—' : `${(value * 100).toFixed(1)}%`;
const currentModel = () => state.meta?.models.find(model => model.id === state.model);

const viewer = new ScanViewer($('scan'), $('viewport'), $('divider'), zoom => {
  $('zoom-label').value = `${Math.round(zoom * 100)}%`;
  $('zoom-out').disabled = zoom <= 1; $('zoom-in').disabled = zoom >= 4;
  $('scan-hint').textContent = zoom > 1 ? 'Drag or use arrow keys to pan' : state.mode === 'compare' ? 'Drag the divider to compare' : 'Use + to look closer';
}, comparison => { $('compare').value = comparison; });

function notice(message) { $('notice').textContent = message || ''; $('notice').hidden = !message; }
async function request(url, options) {
  if (recordedData && url.startsWith('/api/')) return recordedRequest(recordedData, url, options);
  const response = await fetch(url, options);
  let data;
  try { data = await response.json(); } catch { throw new Error('The local server did not return a valid response.'); }
  if (!response.ok) throw new Error(data.error || 'Unable to complete the request.');
  return data;
}
function loading(value) {
  $('loading').hidden = !value;
  $('viewport').setAttribute('aria-busy', String(value));
  $('download').disabled = value || !state.result;
  if (value) {
    $('prediction').textContent = state.meta?.recorded ? 'Loading example…' : 'Analyzing…'; $('score').textContent = '—';
    $('truth').textContent = '—'; $('outcome-badge').textContent = '—'; $('score-fill').style.width = '0%';
  }
}
function setMode(mode) {
  state.mode = mode; viewer.setMode(mode);
  document.querySelectorAll('[data-mode]').forEach(button => { const active = button.dataset.mode === mode; button.classList.toggle('active',active); button.setAttribute('aria-pressed',String(active)); });
  $('comparison-control').hidden = mode !== 'compare';
  $('left-view-label').hidden = mode === 'overlay'; $('right-view-label').hidden = mode === 'original';
  $('scan-hint').textContent = viewer.zoom > 1 ? 'Drag or use arrow keys to pan' : mode === 'compare' ? 'Drag the divider to compare' : 'Use + to look closer';
}
async function analyze(reset = false) {
  if (!state.caseId && !state.upload) return;
  const sequence = ++state.request;
  controller?.abort(); controller = new AbortController(); viewer.invalidate();
  state.result = null; loading(true); notice('');
  if (reset) viewer.reset();
  const key = `${state.model}:${state.caseId || `upload-${state.uploadId}`}:${state.target}`;
  const payload = {model:state.model,target:state.target,...(state.upload ? {image:state.upload.data,filename:state.upload.name} : {case_id:state.caseId})};
  try {
    let result = cache.get(key);
    if (!result) {
      result = await request('/api/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload),signal:controller.signal});
      if (sequence !== state.request) return;
      cache.set(key,result);
      if (cache.size > 8) cache.delete(cache.keys().next().value);
    }
    if (sequence !== state.request) return;
    const displayed = await viewer.load(result);
    if (!displayed || sequence !== state.request) return;
    state.result = result;
    $('scan-name').textContent = result.filename;
    $('image-size').textContent = `${result.source_size[0]} × ${result.source_size[1]}`;
    $('prediction').textContent = titleCase(result.prediction);
    $('score').textContent = (result.score * 100).toFixed(1);
    $('score-fill').style.width = `${result.score * 100}%`;
    $('threshold-tick').style.left = `${result.threshold * 100}%`;
    $('threshold-label').textContent = `Decision threshold ${Math.round(result.threshold * 100)}%`;
    $('truth').textContent = titleCase(result.label);
    const correct = !result.label || result.label === result.prediction;
    $('outcome-badge').textContent = !result.label ? 'Unlabeled upload' : correct ? 'Matches label' : 'Mismatch';
    $('outcome-badge').classList.toggle('error',!correct);
    $('target-caption').textContent = `${result.target} CLASS`;
    $('right-view-label').textContent = `GRAD-CAM · ${result.target}`;
    if (!result.has_activation) notice('No positive Grad-CAM activation for this class. This does not establish a normal scan.');
  } catch (error) {
    if (sequence !== state.request || error.name === 'AbortError') return;
    viewer.clear();
    notice(error.message); $('prediction').textContent = 'Analysis unavailable';
    $('scan-name').textContent = 'Select another case or try again';
  } finally {
    if (sequence === state.request) loading(false);
  }
}
function selectCase(caseId, scroll = false) {
  sourceRevision++;
  state.caseId = caseId; state.upload = null; $('upload').value = '';
  document.querySelectorAll('.case-card').forEach(card => { const active = card.dataset.id === caseId; card.classList.toggle('selected',active); card.setAttribute('aria-pressed',String(active)); });
  analyze(true);
  if (scroll) $('explorer-page').scrollIntoView({behavior:window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth'});
}
function renderCard(item) {
  const card = document.createElement('button'); card.className = 'case-card'; card.dataset.id = item.id;
  card.setAttribute('aria-label',`${labels[item.outcome]}: ${item.filename}. Dataset ${titleCase(item.label)}. Pneumonia score ${percent(item.score)}.`);
  card.setAttribute('aria-pressed',String(state.caseId === item.id)); card.classList.toggle('selected',state.caseId === item.id);
  const picture = document.createElement('div'); picture.className = 'case-image';
  const image = document.createElement('img'); image.src = item.thumbnail || `/api/thumbnail/${item.id}?model=${encodeURIComponent(state.model)}`; image.alt = `Chest X-ray example, dataset label ${titleCase(item.label)}`; image.loading = 'lazy';
  const badge = document.createElement('span'); badge.className = `case-outcome ${item.outcome === 'fp' || item.outcome === 'fn' ? 'error' : ''}`; badge.textContent = labels[item.outcome];
  picture.append(image,badge);
  const content = document.createElement('div'); content.className = 'case-text';
  const name = document.createElement('div'); name.className = 'case-name'; name.textContent = item.filename;
  const row = document.createElement('div'); row.className = 'row';
  const label = document.createElement('strong'); label.textContent = titleCase(item.label);
  const score = document.createElement('span'); score.textContent = percent(item.score); score.title = 'Pneumonia model score';
  row.append(label,score); content.append(name,row); card.append(picture,content);
  card.addEventListener('click',() => selectCase(item.id,true)); return card;
}
async function loadGallery(selectFirst = false) {
  const sequence = ++state.galleryRequest;
  const source = sourceRevision;
  $('gallery').setAttribute('aria-busy','true');
  $('previous').disabled = $('next').disabled = true;
  try {
    const data = await request(`/api/cases?model=${encodeURIComponent(state.model)}&outcome=${state.filter}&offset=${state.offset}&limit=12`);
    if (sequence !== state.galleryRequest) return;
    state.cases = data.cases; $('gallery').replaceChildren(...data.cases.map(renderCard));
    $('gallery-count').textContent = `${data.total} ${state.meta.recorded ? (data.total === 1 ? 'recorded example' : 'recorded examples') : state.filter === 'all' ? 'test images' : labels[state.filter].toLowerCase() + (data.total === 1 ? '' : 's')}`;
    $('page-count').textContent = data.total ? `${state.offset + 1}–${state.offset + data.cases.length} of ${data.total} cases` : 'No cases in this category';
    $('previous').disabled = state.offset === 0; $('next').disabled = state.offset + 12 >= data.total;
    if (selectFirst && source === sourceRevision && data.cases.length) selectCase(data.cases[0].id);
  } catch (error) { if (sequence === state.galleryRequest) notice(error.message); }
  finally { if (sequence === state.galleryRequest) $('gallery').setAttribute('aria-busy','false'); }
}
function setFilter(filter, choose = true) {
  if (!state.meta) return;
  state.filter = filter; state.offset = 0;
  document.querySelectorAll('[data-filter]').forEach(button => { const active = button.dataset.filter === filter; button.classList.toggle('active',active); button.setAttribute('aria-pressed',String(active)); });
  return loadGallery(choose);
}
function showPage(page) {
  document.querySelectorAll('.page').forEach(section => { section.hidden = section.id !== `${page}-page`; });
  document.querySelectorAll('[data-page]').forEach(button => { const active = button.dataset.page === page; button.classList.toggle('active',active); if (active) button.setAttribute('aria-current','page'); else button.removeAttribute('aria-current'); });
  if (page === 'explorer') requestAnimationFrame(() => viewer.render());
  window.scrollTo({top:0,behavior:'instant'});
}
function renderEvaluation() {
  const model = currentModel(); if (!model) return;
  const metrics = model.metrics;
  const items = [['sensitivity','PNEUMONIA RECALL','Pneumonia cases detected'],['specificity','NORMAL RECALL','Normal cases correctly classified'],['accuracy','ACCURACY','Correct classifications'],['roc_auc','AUROC','Ranking across thresholds']];
  $('evaluation-metrics').replaceChildren(...items.map(([key,label,detail]) => { const card = document.createElement('div'); card.className = 'metric-card'; const eyebrow = document.createElement('p'); eyebrow.className = 'eyebrow'; eyebrow.textContent = label; const value = document.createElement('strong'); value.textContent = key === 'roc_auc' ? metrics[key].toFixed(3) : percent(metrics[key]); const caption = document.createElement('span'); caption.textContent = detail; card.append(eyebrow,value,caption); return card; }));
  document.querySelectorAll('[data-outcome]').forEach(button => { button.querySelector('strong').textContent = model.counts[button.dataset.outcome]; });
  document.querySelectorAll('[data-count]').forEach(span => { span.textContent = (model.gallery_counts || model.counts)[span.dataset.count]; });
  const table = document.createElement('table'); table.className = 'comparison-table';
  const header = table.createTHead().insertRow();
  for (const text of ['Test metric',...state.meta.models.map(m => m.id === 'baseline' ? 'Baseline' : 'Fine-tuned')]) { const th = document.createElement('th'); th.textContent = text; header.append(th); }
  const body = table.createTBody();
  for (const [key,label] of [['accuracy','Accuracy'],['sensitivity','Pneumonia recall'],['specificity','Normal recall'],['roc_auc','AUROC'],['average_precision','Average precision'],['brier_score','Brier score ↓']]) {
    const row = body.insertRow(); row.insertCell().textContent = label;
    for (const candidate of state.meta.models) row.insertCell().textContent = ['roc_auc','average_precision','brier_score'].includes(key) ? candidate.metrics[key].toFixed(3) : percent(candidate.metrics[key]);
  }
  $('model-comparison').replaceChildren(table);
  $('comparison-note').textContent = state.meta.models.length > 1 ? 'Both use the same 624-image test partition and fixed 0.50 decision threshold. Fine-tuning was selected using validation loss. Test reuse makes this exploratory. Brier score measures squared probability error (lower is better); it worsened despite higher accuracy.' : 'The frozen-feature baseline uses a fixed 0.50 threshold. A second evaluated model is not available yet.';
}
function switchModel(model) {
  state.model = model; $('model').value = $('evaluation-model').value = model;
  renderEvaluation();
  if (state.caseId || state.upload) analyze(false);
  setFilter('all',!state.caseId && !state.upload);
}

document.querySelectorAll('[data-page]').forEach(button => button.addEventListener('click',() => showPage(button.dataset.page)));
document.querySelectorAll('[data-mode]').forEach(button => button.addEventListener('click',() => setMode(button.dataset.mode)));
document.querySelectorAll('[data-filter]').forEach(button => button.addEventListener('click',() => setFilter(button.dataset.filter)));
document.querySelectorAll('[data-outcome]').forEach(button => button.addEventListener('click',() => { showPage('explorer'); setFilter(button.dataset.outcome); }));
$('gallery-reset').addEventListener('click',() => setFilter('all',false));
$('previous').addEventListener('click',() => { state.offset = Math.max(0,state.offset - 12); loadGallery(); });
$('next').addEventListener('click',() => { state.offset += 12; loadGallery(); });
$('model').addEventListener('change',event => switchModel(event.target.value));
$('evaluation-model').addEventListener('change',event => switchModel(event.target.value));
$('target').addEventListener('change',event => { state.target = event.target.value; analyze(false); });
$('opacity').addEventListener('input',event => { $('opacity-value').value = `${event.target.value}%`; viewer.setOpacity(Number(event.target.value) / 100); });
$('compare').addEventListener('input',event => viewer.setComparison(Number(event.target.value)));
$('zoom-in').addEventListener('click',() => viewer.setZoom(viewer.zoom + .25));
$('zoom-out').addEventListener('click',() => viewer.setZoom(viewer.zoom - .25));
$('reset-view').addEventListener('click',() => viewer.reset());
document.querySelectorAll('.upload-trigger').forEach(button => button.addEventListener('click',() => $('upload').click()));
$('upload').addEventListener('change',async event => {
  const file = event.target.files[0]; if (!file) return;
  const source = ++sourceRevision;
  if (file.size > 10 * 1024 * 1024) { notice('Choose a JPEG or PNG under 10 MB.'); event.target.value = ''; return; }
  if (!['image/jpeg','image/png'].includes(file.type)) { notice('Only 8-bit JPEG and PNG exports are supported.'); event.target.value = ''; return; }
  const reader = new FileReader();
  reader.onload = () => { if (source !== sourceRevision) return; state.upload = {data:String(reader.result).split(',')[1],name:file.name}; state.uploadId++; state.caseId = null; document.querySelectorAll('.case-card').forEach(card => { card.classList.remove('selected'); card.setAttribute('aria-pressed','false'); }); analyze(true); };
  reader.onerror = () => notice('Could not read that file. Please choose another image.'); reader.readAsDataURL(file);
});
$('download').addEventListener('click',async () => {
  const result = state.result; if (!result) return;
  const blob = await viewer.export(result); if (!blob) return;
  const url = URL.createObjectURL(blob), link = document.createElement('a'); link.href = url; link.download = `pneumolens-${result.model}-${result.target.toLowerCase()}-overlay.png`; link.click(); setTimeout(() => URL.revokeObjectURL(url),1000);
});

async function start() {
  const source = sourceRevision;
  const uploadButtons = document.querySelectorAll('.upload-trigger');
  uploadButtons.forEach(button => { button.disabled = true; });
  try {
    if (new URLSearchParams(window.location.search).get('demo') === '1') {
      recordedData = await request('/demo/manifest.json');
      if (recordedData.schema_version !== 1 || !recordedData.recorded) throw new Error('Recorded demo format is unsupported.');
    }
    state.meta = await request('/api/meta');
    $('demo-banner').hidden = !state.meta.recorded;
    uploadButtons.forEach(button => { button.hidden = Boolean(state.meta.recorded); button.disabled = false; });
    if (state.meta.recorded) {
      document.querySelector('.brand').href = '/?demo=1';
      $('run-mode').textContent = 'Recorded demo';
      $('gallery-description').textContent = 'A small selection of recorded test cases. Filter counts refer to these examples; Evaluation reports all 624 test images.';
      $('loading-detail').textContent = 'Loading a saved, real model output…';
    }
    $('sanity-report').hidden = !state.meta.sanity_available;
    if (!state.meta.models.length) throw new Error('No evaluated model is available. Complete training and evaluation, then reload.');
    state.model = state.meta.default_model;
    for (const select of [$('model'),$('evaluation-model')]) {
      select.replaceChildren(...state.meta.models.map(model => { const option = document.createElement('option'); option.value = model.id; option.textContent = model.name; return option; }));
      select.value = state.model; select.disabled = false;
    }
    renderEvaluation(); viewer.reset();
    await loadGallery();
    const galleryRequest = state.galleryRequest;
    const model = state.model;
    const featured = await request(`/api/cases?model=${model}&outcome=tp&limit=1`);
    if (source !== sourceRevision || model !== state.model || galleryRequest !== state.galleryRequest) return;
    const initial = featured.cases[0] || state.cases[0]; if (initial) selectCase(initial.id); else loading(false);
  } catch (error) { notice(error.message); loading(false); $('prediction').textContent = 'Model unavailable'; }
}
start();
