import '@testing-library/jest-dom/vitest';
import { cleanup } from '@testing-library/react';
import { afterEach, beforeEach, expect, vi } from 'vitest';

// React Testing Library only auto-cleans when Vitest's globals are enabled,
// and they are not here. Without this, every render in a file accumulates in
// the same document and the second test asking for a placeholder gets
// "multiple elements found" — a failure that looks like a component bug.
afterEach(cleanup);

// jsdom implements the <canvas> element but not its 2D context, so anything
// that measures text or draws a figure logs "Not implemented:
// HTMLCanvasElement.prototype.getContext" to stderr and then behaves as though
// the call returned nothing. P2-4 of the 2026-09-13 audit: that noise was
// indistinguishable from a real error in the test output, and it hid the
// question of whether the component actually copes with a missing context.
//
// The stub is deliberately minimal and deliberately honest — it records the
// calls and returns plausible geometry, so a component that depends on real
// pixel output fails a test rather than passing against a lie.
const canvasContext = () =>
  ({
    canvas: typeof document === 'undefined' ? null : document.createElement('canvas'),
    clearRect: vi.fn(),
    fillRect: vi.fn(),
    strokeRect: vi.fn(),
    beginPath: vi.fn(),
    closePath: vi.fn(),
    moveTo: vi.fn(),
    lineTo: vi.fn(),
    arc: vi.fn(),
    fill: vi.fn(),
    stroke: vi.fn(),
    save: vi.fn(),
    restore: vi.fn(),
    translate: vi.fn(),
    scale: vi.fn(),
    rotate: vi.fn(),
    setTransform: vi.fn(),
    drawImage: vi.fn(),
    putImageData: vi.fn(),
    createLinearGradient: vi.fn(() => ({ addColorStop: vi.fn() })),
    measureText: vi.fn((text: string) => ({ width: text.length * 7 })),
    fillText: vi.fn(),
    strokeText: vi.fn(),
    getImageData: vi.fn(() => ({ data: new Uint8ClampedArray(4) })),
  }) as unknown as CanvasRenderingContext2D;

// Guarded: some suites declare `@vitest-environment node` because they test
// pure reducers and pagination helpers that never touch a DOM. There is no
// HTMLCanvasElement there, and reaching for one would fail the whole file at
// collection — which is worse than the noise this stub exists to remove.
if (typeof HTMLCanvasElement !== 'undefined') {
  HTMLCanvasElement.prototype.getContext = vi.fn(
    (type: string) => (type === '2d' ? canvasContext() : null),
  ) as unknown as HTMLCanvasElement['getContext'];
  HTMLCanvasElement.prototype.toDataURL = vi.fn(
    () => 'data:image/png;base64,',
  ) as unknown as HTMLCanvasElement['toDataURL'];
}

// An unexpected console.error fails the test that produced it.
//
// React reports a great deal through console.error and nothing else: an
// invalid prop type, a key warning, a state update outside act(), an error
// boundary catching a throw. Left on stderr, those read as noise and
// accumulate until nobody reads the output at all — which is the state the
// audit found the suite in.
//
// Escaping it is deliberate and per-test: call `expectConsoleError(pattern)`
// in a test that means to provoke one. Nothing is globally silenced, because a
// global allow-list is how the noise comes back.
const ALLOWED_MESSAGE_PATTERNS: RegExp[] = [
  // Nothing is allowed by default. Add a pattern here only with a comment
  // saying why it cannot be fixed, and a date by which it will be.
];

let expectedPatterns: RegExp[] = [];
let unexpected: string[] = [];
let realConsoleError: typeof console.error;

/**
 * Declare that this test expects a console.error matching `pattern`.
 * Anything else it logs still fails the test.
 */
export function expectConsoleError(pattern: RegExp): void {
  expectedPatterns.push(pattern);
}

beforeEach(() => {
  expectedPatterns = [];
  unexpected = [];
  realConsoleError = console.error;
  console.error = (...args: unknown[]) => {
    const message = args
      .map((arg) => (arg instanceof Error ? arg.message : String(arg)))
      .join(' ');
    const allowed =
      ALLOWED_MESSAGE_PATTERNS.some((pattern) => pattern.test(message)) ||
      expectedPatterns.some((pattern) => pattern.test(message));
    if (!allowed) {
      unexpected.push(message);
    }
    realConsoleError(...(args as Parameters<typeof console.error>));
  };
});

afterEach(() => {
  console.error = realConsoleError;
  const logged = unexpected;
  unexpected = [];
  expect(
    logged,
    `this test logged ${logged.length} unexpected console.error call(s). ` +
      'Fix the cause, or call expectConsoleError(/pattern/) if the test means ' +
      'to provoke it.',
  ).toEqual([]);
});
