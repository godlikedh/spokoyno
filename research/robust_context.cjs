/* Fixed frozen AudioSet categories and before/event/after context features. */
'use strict';
(() => {
  const roundEven = (value) => {
    const lo = Math.floor(value);
    return value - lo === 0.5 ? (lo % 2 === 0 ? lo : lo + 1) : Math.round(value);
  };
  function extract(embeddings, contentPositions, head) {
    if (embeddings.length !== contentPositions.length * 3072) throw new Error('context embedding dimensions differ');
    const classes = new Map(head.indices.map((v, i) => [v, i])),
      rows = [];
    let names;
    for (let row = 0; row < contentPositions.length; row++) {
      const vectors = [0, 1, 2].map((i) => embeddings.subarray(row * 3072 + i * 1024, row * 3072 + (i + 1) * 1024));
      const scores = [],
        squared = [0, 0, 0];
      for (let part = 0; part < 3; part++) {
        const logits = Float64Array.from(head.bias);
        for (let f = 0; f < 1024; f++) {
          const value = vectors[part][f];
          squared[part] += value * value;
          for (let c = 0; c < logits.length; c++) logits[c] += value * head.kernel[f][c];
        }
        scores.push(logits.map((v) => 1 / (1 + Math.exp(-v))));
      }
      const norms = squared.map(Math.sqrt);
      const dot = (a, b) => {
        let value = 0;
        for (let i = 0; i < 1024; i++) value += a[i] * b[i];
        return value;
      };
      const distance = (dotProduct, an, bn) =>
        an < 1e-12 && bn < 1e-12 ? 0 : 1 - Math.max(-1, Math.min(1, dotProduct / Math.max(an * bn, 1e-12)));
      const be = dot(vectors[0], vectors[1]),
        ea = dot(vectors[1], vectors[2]),
        ba = dot(vectors[0], vectors[2]);
      const values = {
        ctx_boundary_opening: roundEven(contentPositions[row] / 0.05) <= 20 ? 1 : 0,
        ctx_embedding_before_event_distance: distance(be, norms[0], norms[1]),
        ctx_embedding_event_after_distance: distance(ea, norms[1], norms[2]),
        ctx_embedding_before_after_distance: distance(ba, norms[0], norms[2]),
        ctx_embedding_context_event_distance: distance(
          (be + ea) / 2,
          Math.sqrt(Math.max(0, (squared[0] + squared[2] + 2 * ba) / 4)),
          norms[1]
        ),
        ctx_embedding_event_before_log_norm: Math.log((norms[1] + 1e-6) / (norms[0] + 1e-6)),
        ctx_embedding_event_after_log_norm: Math.log((norms[1] + 1e-6) / (norms[2] + 1e-6))
      };
      const groups = {};
      for (const [name, indices] of Object.entries(head.groups)) {
        const p = scores.map((s) => Math.max(...indices.map((i) => s[classes.get(i)])));
        groups[name] = p;
        values[`ctx_semantic_${name}_event`] = p[1];
        values[`ctx_semantic_${name}_rise`] = p[1] - p[0];
        values[`ctx_semantic_${name}_fall`] = p[1] - p[2];
      }
      for (const name of ['vocal_alarm', 'noise'])
        values[`ctx_semantic_${name}_introduced`] = groups[name][1] * (1 - groups[name][0]);
      for (const name of ['music', 'singing'])
        values[`ctx_semantic_${name}_persistent`] = Math.min(groups[name][0], groups[name][1]);
      names = Object.keys(values).sort();
      rows.push(names.map((n) => values[n]));
    }
    return { names, rows };
  }
  if (typeof module !== 'undefined') module.exports = { extract };
  else globalThis.SpokoynoContext = { extract };
})();
