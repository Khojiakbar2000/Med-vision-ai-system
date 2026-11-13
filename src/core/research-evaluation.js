export function validateEvaluationResults(value) {
  if (!value || value.schema !== 'medvision-evaluation-results-v1' || !Array.isArray(value.cases) || value.cases.length > 5000 || !value.summary || !value.datasetProvenance) {
    throw new TypeError('INVALID_RESEARCH_RESULTS');
  }
  for (const row of value.cases) {
    if (!row || typeof row.case !== 'string' || !/^case_\d{6}$/.test(row.case) || !['success', 'failed'].includes(row.status)) throw new TypeError('INVALID_RESEARCH_RESULTS');
    if (row.status === 'failed' && row.errorCode !== undefined && (typeof row.errorCode !== 'string' || !/^[A-Z0-9_]{1,64}$/.test(row.errorCode))) throw new TypeError('INVALID_RESEARCH_RESULTS');
    if (row.status === 'success') {
      if (!Number.isFinite(row.inferenceMs)) throw new TypeError('INVALID_RESEARCH_RESULTS');
      for (const region of ['whole_tumor', 'tumor_core', 'enhancing_tumor']) {
        const metrics = row.metrics?.[region];
        if (!metrics || !Number.isFinite(metrics.dice) || !Number.isFinite(metrics.iou)) throw new TypeError('INVALID_RESEARCH_RESULTS');
      }
    }
  }
  return value;
}

export function pairedResearchDeltas(baseline, candidate) {
  const left = new Map(baseline.cases.filter(row => row.status === 'success').map(row => [row.case, row]));
  const right = new Map(candidate.cases.filter(row => row.status === 'success').map(row => [row.case, row]));
  const keys = [...left.keys()].filter(key => right.has(key)).sort();
  return keys.map(caseId => {
    const delta = { case: caseId };
    for (const region of ['whole_tumor', 'tumor_core', 'enhancing_tumor']) {
      for (const metric of ['dice', 'iou']) {
        delta[`${region}.${metric}`] = right.get(caseId).metrics[region][metric] - left.get(caseId).metrics[region][metric];
      }
    }
    delta.inferenceMs = right.get(caseId).inferenceMs - left.get(caseId).inferenceMs;
    const baselineMemory = left.get(caseId).peakDeviceMemoryBytes;
    const candidateMemory = right.get(caseId).peakDeviceMemoryBytes;
    delta.peakDeviceMemoryBytes = Number.isFinite(baselineMemory) && Number.isFinite(candidateMemory)
      ? candidateMemory - baselineMemory
      : null;
    return delta;
  });
}
