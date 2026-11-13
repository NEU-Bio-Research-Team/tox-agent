#!/usr/bin/env node
/**
 * Design-system ratchet (docs/plans/frontend-design-unification-2026-11-13.md, P0).
 *
 * Counts the patterns the redesign is retiring: colour literals, raw Tailwind
 * palette classes, inline colour styles, arbitrary pixel type sizes and the
 * legacy token aliases. The counts are recorded in `style-baseline.json`; this
 * check fails when any count goes UP, and `--update` records a lower baseline
 * after a cleanup. It never asks for a count to reach zero, so it can ship
 * before the migration is finished.
 */
import { readdirSync, readFileSync, statSync, writeFileSync, existsSync } from 'node:fs';
import { join, extname, relative } from 'node:path';

const ROOT = new URL('..', import.meta.url).pathname;
const SRC = join(ROOT, 'src');
const BASELINE = join(ROOT, 'scripts', 'style-baseline.json');

// Files whose colours are not ours to tokenise: the Google brand mark.
const ALLOW = new Set(['features/auth/FirebaseSignIn.tsx']);
const SKIP_DIR = new Set(['api']);

const PALETTE =
  '(?:slate|gray|zinc|neutral|stone|red|orange|amber|yellow|lime|green|emerald|teal|cyan|sky|blue|indigo|violet|purple|fuchsia|pink|rose)';

const RULES = {
  hexLiterals: /#[0-9a-fA-F]{3,8}\b/g,
  rgbLiterals: /\b(?:rgba?|hsla?)\(/g,
  rawPaletteClasses: new RegExp(
    `\\b(?:bg|text|border|ring|from|to|via|fill|stroke|divide|outline|shadow)-${PALETTE}-[0-9]{2,3}\\b`,
    'g',
  ),
  inlineColorStyles: /\b(?:color|backgroundColor|borderColor|background):\s*['"`]/g,
  arbitraryTextPx: /text-\[[0-9.]+px\]/g,
  legacyAliases: /var\(--(?:text|text-muted|text-faint|border|surface-alt|bg|accent-[a-z-]+)\)/g,
};

function walk(dir, out = []) {
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    const info = statSync(full);
    if (info.isDirectory()) {
      if (!SKIP_DIR.has(entry)) walk(full, out);
    } else if (['.ts', '.tsx'].includes(extname(full)) && !/\.test\.tsx?$/.test(full) && !full.endsWith('.d.ts')) {
      out.push(full);
    }
  }
  return out;
}

function measure() {
  const counts = Object.fromEntries(Object.keys(RULES).map((key) => [key, 0]));
  for (const file of walk(SRC)) {
    if (ALLOW.has(relative(SRC, file))) continue;
    const text = readFileSync(file, 'utf8');
    for (const [key, pattern] of Object.entries(RULES)) {
      counts[key] += (text.match(pattern) ?? []).length;
    }
  }
  return counts;
}

const current = measure();

if (process.argv.includes('--update')) {
  writeFileSync(BASELINE, `${JSON.stringify(current, null, 2)}\n`);
  console.log('style-lint: baseline written', current);
  process.exit(0);
}

if (!existsSync(BASELINE)) {
  console.error('style-lint: no baseline; run `node scripts/style-lint.mjs --update`.');
  process.exit(1);
}

const baseline = JSON.parse(readFileSync(BASELINE, 'utf8'));
const regressions = Object.keys(RULES).filter((key) => current[key] > (baseline[key] ?? 0));
const improved = Object.keys(RULES).filter((key) => current[key] < (baseline[key] ?? 0));

for (const key of Object.keys(RULES)) {
  console.log(`  ${key.padEnd(20)} ${String(current[key]).padStart(5)}  (baseline ${baseline[key] ?? 0})`);
}
if (regressions.length) {
  console.error(`style-lint: FAILED, grew: ${regressions.join(', ')}. Use a design token or a shared component instead.`);
  process.exit(1);
}
if (improved.length) console.log(`style-lint: below baseline for ${improved.join(', ')}; run with --update to lock it in.`);
console.log('style-lint: clean.');
