// A read-only adapter for exported real predictions; never runs inference.
export function recordedRequest(packageData, url, options = {}) {
  const route = new URL(url, 'http://localhost');
  if (route.pathname === '/api/meta') return packageData;
  const modelId = route.searchParams.get('model') || 'baseline';
  if (route.pathname === '/api/cases') {
    const rows = packageData.cases[modelId];
    if (!rows) throw new Error('Unknown recorded model.');
    const outcome = route.searchParams.get('outcome') || 'all';
    if (!['all', 'tp', 'tn', 'fp', 'fn'].includes(outcome)) throw new Error('Unknown outcome.');
    const offset = Number(route.searchParams.get('offset') || 0);
    const limit = Number(route.searchParams.get('limit') || 12);
    if (!Number.isInteger(offset) || offset < 0 || !Number.isInteger(limit) || limit < 1 || limit > 48) throw new Error('Invalid pagination.');
    const filtered = rows.filter(row => outcome === 'all' || row.outcome === outcome);
    return {total: filtered.length, offset, cases: filtered.slice(offset, offset + limit)};
  }
  if (route.pathname === '/api/analyze') {
    const payload = JSON.parse(options.body || '{}');
    if (!payload || typeof payload !== 'object' || payload.image) throw new Error('Uploads require the full inference app.');
    if (!packageData.models.some(model => model.id === payload.model)) throw new Error('Unknown recorded model.');
    const result = packageData.analyses[payload.model]?.[payload.case_id]?.[payload.target];
    if (!result?.recorded) throw new Error('This example or explanation is not in the recorded demo.');
    return result;
  }
  throw new Error('This action requires the full inference app.');
}
