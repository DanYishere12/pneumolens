import test from 'node:test';
import assert from 'node:assert/strict';
import {blendPixels, imagePlacement} from '../web/viewer.mjs';

test('zero activation or zero opacity leaves original pixels unchanged', () => {
  const image = new Uint8ClampedArray([120, 70, 55, 255, 10, 20, 30, 255]);
  assert.deepEqual(blendPixels(image, new Uint8ClampedArray(8), .55), image);
  assert.deepEqual(blendPixels(image, new Uint8ClampedArray(8).fill(255), 0), image);
});
test('full positive activation and full opacity produce the defined red', () => {
  assert.deepEqual([...blendPixels([22, 44, 66, 255], [255, 255, 255, 255], 1)], [255, 0, 0, 255]);
  assert.throws(() => blendPixels([0], [0], .5));
  assert.throws(() => blendPixels([0,0,0,255], [0,0,0,255], 2));
});
test('shared placement preserves aspect ratio and clamps pan', () => {
  const fit = imagePlacement(850, 470, 800, 400, 1, {x:999, y:999});
  assert.equal(fit.width / fit.height, 2);
  assert.deepEqual(fit.pan, {x:0,y:0});
  assert.equal(fit.x + fit.width / 2, 425);
  assert.equal(fit.y + fit.height / 2, 235);
  const zoom = imagePlacement(850, 470, 800, 400, 2, {x:999,y:-999});
  assert.equal(zoom.pan.x, 400); assert.equal(zoom.pan.y, -200);
});
