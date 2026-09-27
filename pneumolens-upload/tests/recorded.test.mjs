import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {recordedRequest} from '../web/recorded.mjs';

const packageData = JSON.parse(readFileSync(new URL('../web/demo/manifest.json', import.meta.url)));
const readAsset = path => readFileSync(new URL(`../web${path}`, import.meta.url));
const dimensions = buffer => [buffer.readUInt32BE(16), buffer.readUInt32BE(20)];

test('recorded assets match hashes, model provenance, and image geometry', () => {
  assert.equal(packageData.recorded, true);
  assert.equal(JSON.stringify(packageData).includes('/Users/'), false);
  for (const [path, digest] of Object.entries(packageData.assets_sha256)) {
    assert.equal(createHash('sha256').update(readAsset(`/demo/${path}`)).digest('hex'), digest);
  }
  for (const model of packageData.models) {
    const rows = packageData.cases[model.id];
    assert.equal(Object.values(model.counts).reduce((a, b) => a + b), model.metrics.n);
    assert.equal(Object.values(model.gallery_counts).reduce((a, b) => a + b), rows.length);
    assert(rows.length < model.metrics.n, 'the selected gallery is distinct from full evaluation');
    for (const row of rows) {
      const variants = packageData.analyses[model.id][row.id];
      for (const [target, result] of Object.entries(variants)) {
        assert.equal(result.target, target);
        assert.equal(result.recorded, true);
        assert.equal(result.checkpoint_sha256, model.metrics.checkpoint_sha256);
        assert(Math.abs(result.score - row.score) < 1e-5);
        assert.deepEqual(dimensions(readAsset(result.image)), [result.width, result.height]);
        assert.deepEqual(dimensions(readAsset(result.cam)), [result.width, result.height]);
      }
      assert.equal(variants.NORMAL.score, variants.PNEUMONIA.score);
      assert.equal(variants.NORMAL.prediction, variants.PNEUMONIA.prediction);
      assert.notDeepEqual(readAsset(variants.NORMAL.cam), readAsset(variants.PNEUMONIA.cam));
    }
  }
});

test('recorded filters expose selected errors without implying full-test coverage', () => {
  for (const model of packageData.models) {
    for (const outcome of ['tp', 'tn', 'fp', 'fn']) {
      const page = recordedRequest(packageData, `/api/cases?model=${model.id}&outcome=${outcome}&limit=1`);
      assert.equal(page.total, model.gallery_counts[outcome]);
      assert.equal(page.cases.length, Math.min(1, page.total));
      if (page.cases.length) assert.equal(page.cases[0].outcome, outcome);
    }
  }
});

test('recorded mode rejects uploads, unknown cases, and invalid pagination', () => {
  const analyze = payload => recordedRequest(packageData, '/api/analyze', {body: JSON.stringify(payload)});
  assert.throws(() => analyze({model: 'baseline', image: 'uploaded-bytes'}), /Uploads require/);
  assert.throws(() => analyze({model: 'baseline', case_id: 'unknown', target: 'NORMAL'}), /not in the recorded demo/);
  assert.throws(() => recordedRequest(packageData, '/api/cases?offset=-1'), /Invalid pagination/);
  const row = packageData.cases.baseline[0];
  assert.equal(analyze({model: 'baseline', case_id: row.id, target: 'NORMAL'}).recorded, true);
});
