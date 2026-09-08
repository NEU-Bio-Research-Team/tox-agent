import '@testing-library/jest-dom/vitest';
import { cleanup } from '@testing-library/react';
import { afterEach } from 'vitest';

// React Testing Library only auto-cleans when Vitest's globals are enabled,
// and they are not here. Without this, every render in a file accumulates in
// the same document and the second test asking for a placeholder gets
// "multiple elements found" — a failure that looks like a component bug.
afterEach(cleanup);
