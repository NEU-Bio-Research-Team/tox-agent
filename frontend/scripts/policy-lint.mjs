#!/usr/bin/env node
/**
 * Redesign plan section 13: the frontend's own guard against the aggregate
 * verdict re-entering through the UI layer after the backend validator
 * (prohibited_claims.py) already refuses to let a model write it. Grep is
 * deliberately blunt — a false positive here costs one line of review, a
 * false negative reintroduces exactly the SCI-02 violation this project
 * spent an ADR ruling out.
 */
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, extname } from 'node:path';

const SRC_DIR = new URL('../src', import.meta.url).pathname;
const EXTENSIONS = new Set(['.ts', '.tsx']);

const FORBIDDEN = [
  { pattern: /\brisk_level\b/i, reason: 'risk_level is a removed field from the pre-rebuild agent layer (SCI-02/ADR-0002)' },
  { pattern: /overall toxicity/i, reason: 'aggregate verdict language (SCI-02)' },
  { pattern: /aggregate (toxicity|score|verdict)/i, reason: 'aggregate verdict language (SCI-02)' },
  { pattern: /toxicity score/i, reason: 'implies a single combined score across endpoints (SCI-02)' },
  { pattern: /hit.?count/i, reason: 'Tox21 active-assay count as a severity signal (SCI-05)' },
  { pattern: /\bassay_hits\b/i, reason: 'removed field from the pre-rebuild agent layer (SCI-05)' },
  { pattern: /\bfusion_result\b/i, reason: 'removed field that combined baseline + evidence into one label' },
];

function collectFiles(dir) {
  const out = [];
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    const info = statSync(full);
    if (info.isDirectory()) out.push(...collectFiles(full));
    else if (EXTENSIONS.has(extname(full))) out.push(full);
  }
  return out;
}

// A line that negates the phrase ("Không có aggregate toxicity score" / "no
// overall toxicity score") is stating the product's own invariant, not
// violating it. This is the same negation-blindness problem the backend's
// own prohibited_claims.py documents for `_CLINICAL_OVERREACH`/
// `_AGGREGATE_VERDICT` (progress doc section 4.7) — worth a comment here so
// nobody "fixes" it back into a false positive later.
// No `\b` around the Vietnamese alternatives: JS's ASCII-only `\b` never
// matches next to a diacritic like "ô"/"ó", so it silently fails to find a
// boundary at all next to those words.
const NEGATION = /(không có|không phải|không (?:tồn tại|dùng)|\bno \b|\bnot a\b|\bdoes not\b|\bdoesn'?t\b|\bnever\b)/i;

let violations = 0;
for (const file of collectFiles(SRC_DIR)) {
  const text = readFileSync(file, 'utf8');
  const lines = text.split('\n');
  for (const { pattern, reason } of FORBIDDEN) {
    lines.forEach((line, index) => {
      if (pattern.test(line) && !NEGATION.test(line)) {
        violations += 1;
        console.error(`${file}:${index + 1}: ${line.trim()}\n  → ${reason}`);
      }
    });
  }
}

if (violations > 0) {
  console.error(`\npolicy-lint: ${violations} violation(s) found.`);
  process.exit(1);
}
console.log('policy-lint: clean.');
