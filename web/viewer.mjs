export function blendPixels(original, activation, opacity) {
  if (original.length !== activation.length || original.length % 4 || !Number.isFinite(opacity) || opacity < 0 || opacity > 1) throw new Error('Invalid overlay inputs');
  const result = new Uint8ClampedArray(original.length);
  for (let i = 0; i < original.length; i += 4) {
    const intensity = activation[i] / 255;
    const alpha = intensity * opacity;
    const heat = [255, 255 * .85 * (1 - intensity), 255 * .12 * (1 - intensity)];
    for (let channel = 0; channel < 3; channel++) result[i + channel] = Math.floor(original[i + channel] * (1 - alpha) + heat[channel] * alpha);
    result[i + 3] = 255;
  }
  return result;
}

export function imagePlacement(width, height, imageWidth, imageHeight, zoom, pan = {x: 0, y: 0}) {
  const scale = Math.min((width - 50) / imageWidth, (height - 70) / imageHeight) * zoom;
  const w = imageWidth * scale, h = imageHeight * scale;
  const maxX = Math.max(0, (w - width + 50) / 2), maxY = Math.max(0, (h - height + 70) / 2);
  const x = Math.min(maxX, Math.max(-maxX, pan.x)), y = Math.min(maxY, Math.max(-maxY, pan.y));
  return {x: (width - w) / 2 + x, y: (height - h) / 2 + y, width: w, height: h, pan: {x, y}};
}

const decode = src => new Promise((resolve, reject) => { const image = new Image(); image.onload = () => resolve(image); image.onerror = () => reject(new Error('Could not display this image.')); image.src = src; });

export class ScanViewer {
  constructor(canvas, viewport, divider, onZoom, onCompare) {
    Object.assign(this, {canvas, viewport, divider, onZoom, onCompare, mode: 'compare', opacity: .55, comparison: 50, zoom: 1, pan: {x: 0, y: 0}, revision: 0});
    this.context = canvas.getContext('2d');
    this.base = document.createElement('canvas'); this.heat = document.createElement('canvas');
    new ResizeObserver(() => this.render()).observe(viewport);
    let drag;
    divider.addEventListener('pointerdown', event => { drag = 'divider'; divider.setPointerCapture(event.pointerId); event.preventDefault(); });
    divider.addEventListener('pointermove', event => { if (drag === 'divider') { const box = viewport.getBoundingClientRect(); this.setComparison(100 * (event.clientX - box.left) / box.width); this.onCompare(this.comparison); } });
    divider.addEventListener('pointerup', () => { drag = null; });
    divider.addEventListener('pointercancel', () => { drag = null; });
    canvas.addEventListener('pointerdown', event => { if (this.zoom <= 1) return; drag = {x: event.clientX, y: event.clientY, pan: {...this.pan}}; canvas.setPointerCapture(event.pointerId); });
    canvas.addEventListener('pointermove', event => { if (!drag || drag === 'divider') return; this.pan = {x: drag.pan.x + event.clientX - drag.x, y: drag.pan.y + event.clientY - drag.y}; this.render(); });
    canvas.addEventListener('pointerup', () => { drag = null; });
    canvas.addEventListener('pointercancel', () => { drag = null; });
    viewport.addEventListener('keydown', event => {
      if (event.key === '+' || event.key === '=') this.setZoom(this.zoom + .25);
      else if (event.key === '-') this.setZoom(this.zoom - .25);
      else if (event.key === '0') this.reset();
      else if (this.zoom > 1 && ['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(event.key)) { this.pan.x += event.key === 'ArrowLeft' ? 30 : event.key === 'ArrowRight' ? -30 : 0; this.pan.y += event.key === 'ArrowUp' ? 30 : event.key === 'ArrowDown' ? -30 : 0; this.render(); }
      else return;
      event.preventDefault();
    });
  }
  async load(result) {
    const revision = ++this.revision;
    const [image, cam] = await Promise.all([decode(result.image), decode(result.cam)]);
    if (revision !== this.revision) return false;
    this.image = image;
    this.base.width = this.heat.width = image.width; this.base.height = this.heat.height = image.height;
    const ctx = this.base.getContext('2d'); ctx.drawImage(image, 0, 0);
    this.original = ctx.getImageData(0, 0, image.width, image.height);
    const heat = this.heat.getContext('2d'); heat.drawImage(cam, 0, 0);
    this.activation = heat.getImageData(0, 0, image.width, image.height);
    this.updateHeat(); this.render(); return true;
  }
  invalidate() { this.revision++; }
  clear() { this.invalidate(); this.image = null; this.original = null; this.activation = null; this.render(); }
  updateHeat() {
    if (!this.original) return;
    this.heat.getContext('2d').putImageData(new ImageData(blendPixels(this.original.data, this.activation.data, this.opacity), this.base.width, this.base.height), 0, 0);
  }
  setOpacity(value) { this.opacity = value; this.updateHeat(); this.render(); }
  setComparison(value) { this.comparison = Math.min(100, Math.max(0, value)); this.render(); }
  setMode(mode) { this.mode = mode; this.render(); }
  setZoom(value) { this.zoom = Math.min(4, Math.max(1, value)); this.onZoom(this.zoom); this.render(); }
  reset() { this.pan = {x: 0, y: 0}; this.setZoom(1); }
  render() {
    const w = this.viewport.clientWidth, h = this.viewport.clientHeight;
    if (!w || !h) return;
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.round(w * dpr); this.canvas.height = Math.round(h * dpr);
    const ctx = this.context; ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, h);
    this.divider.hidden = this.mode !== 'compare'; this.divider.style.left = `${this.comparison}%`;
    this.viewport.classList.toggle('zoomed', this.zoom > 1);
    if (!this.image) return;
    const rect = imagePlacement(w, h, this.image.width, this.image.height, this.zoom, this.pan); this.pan = rect.pan;
    ctx.drawImage(this.base, rect.x, rect.y, rect.width, rect.height);
    if (this.mode !== 'original') {
      ctx.save();
      if (this.mode === 'compare') { const split = w * this.comparison / 100; ctx.beginPath(); ctx.rect(split, 0, w - split, h); ctx.clip(); }
      ctx.drawImage(this.heat, rect.x, rect.y, rect.width, rect.height); ctx.restore();
    }
  }
  export(result) {
    // Preserve the full image, independent of the camera zoom, and keep the
    // explanation's context attached when a PNG is shared outside the app.
    const output = document.createElement('canvas');
    const scale = Math.max(1, this.heat.width / 800);
    output.width = this.heat.width;
    output.height = this.heat.height + Math.ceil(116 * scale);
    const ctx = output.getContext('2d');
    ctx.fillStyle = '#0c1013'; ctx.fillRect(0, 0, output.width, output.height);
    ctx.drawImage(this.heat, 0, 0);
    const left = 16 * scale, top = this.heat.height;
    const line = (text, y, color = '#b4c4c6') => { ctx.fillStyle = color; ctx.fillText(text, left, top + y * scale, output.width - 2 * left); };
    ctx.font = `${14 * scale}px sans-serif`;
    line(`PneumoLens · ${result.model_name} · ${result.target} class · opacity ${Math.round(this.opacity * 100)}%`, 25, '#8be3cc');
    ctx.font = `${11 * scale}px sans-serif`;
    line(`Pneumonia score ${(result.score * 100).toFixed(1)}% (uncalibrated) · ${result.recorded ? 'Recorded output' : 'Local inference'}`, 47);
    line('Grad-CAM shows model influence, not disease localization. Research demo.', 68);
    line(result.label ? 'Image: Kermany, Zhang & Goldbaum · CC BY 4.0 · Modified overlay' : 'Source: uploaded image · Modified overlay', 89);
    return new Promise(resolve => output.toBlob(resolve, 'image/png'));
  }
}
