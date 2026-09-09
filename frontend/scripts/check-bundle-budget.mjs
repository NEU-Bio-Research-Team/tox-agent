import { gzipSync } from 'node:zlib';
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join, relative } from 'node:path';

const DIST = new URL('../dist/', import.meta.url);
const BUDGET_BYTES = 500 * 1024;
const CHECKED_EXTENSIONS = new Set(['.js', '.css', '.json']);

function filesUnder(directory) {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const path = join(directory, entry.name);
    return entry.isDirectory() ? filesUnder(path) : [path];
  });
}

const distPath = DIST.pathname;
if (!existsSync(distPath)) {
  throw new Error('Missing dist/; run vite build before checking the bundle budget.');
}

const oversized = [];
for (const file of filesUnder(distPath)) {
  const extension = file.slice(file.lastIndexOf('.'));
  if (!CHECKED_EXTENSIONS.has(extension)) continue;
  const gzipBytes = gzipSync(readFileSync(file)).byteLength;
  if (gzipBytes > BUDGET_BYTES) oversized.push({ file: relative(distPath, file), gzipBytes });
}

if (oversized.length > 0) {
  const detail = oversized.map(({ file, gzipBytes }) => `${file}: ${gzipBytes} bytes gzip`).join('\n');
  throw new Error(`Bundle budget exceeded (${BUDGET_BYTES} bytes gzip per asset):\n${detail}`);
}

console.log(`bundle-budget: every JS/CSS/JSON asset is within ${BUDGET_BYTES} bytes gzip.`);
