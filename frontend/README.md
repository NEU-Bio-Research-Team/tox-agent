# ToxAgent frontend

The browser product: React 18, Vite, TypeScript, Tailwind. It talks only to the
control plane's `/v1` API (proxied to `127.0.0.1:8000` in development).

## Commands

```bash
npm ci
npm run dev          # Vite dev server
npm run typecheck    # tsc -b
npm run lint         # ESLint, then the product policy lint (scripts/policy-lint.mjs)
npm test             # vitest
npm run build        # production build + bundle budget (scripts/check-bundle-budget.mjs)
npm run test:e2e     # Playwright
```

`make lint`, `make typecheck` and `make test SERVICE=frontend` at the
repository root run the same commands.

## Layout

Code is grouped by feature, not by file type. A feature owns its page, its
components, hooks, store slices and helpers; code two or more features use lives
in `shared/`.

```text
src/
  main.tsx, App.tsx, router.tsx   entry point, providers, routes
  app/            application shell: the workspace frame (rail, header, palette), error boundary
  features/
    workbench/    the session page: transcript, answers, artifacts, run
                  inspector, report blocks, investigation board, event store
    analysis/     prediction panels shared by the workbench and quick predict
    composer/     message composer, image upload, structure editor, SMILES parsing
    quick-predict/ the session-less prediction page
    sessions/     session list, its query hook and date grouping
    settings/     section list, account, AI connections, build info
    auth/         token and PKCE login, the route guard
    landing/      the landing page, one file per section
    docs/         the documentation frame and its markdown articles (content/*.md)
  shared/
    ui/           shadcn/Radix primitives and the design-system components
                  (Callout, StatusPill, Segmented, DataRow, EmptyState, PageHeader…)
    api/          HTTP client, endpoints, response types
    events/       SSE connection and the session event bus
    lib/          labels, preferences, query client, AI profile
    hooks/        hooks with no feature of their own
  styles/, assets/, test/
```

A feature may import from `shared/` and from another feature's public
components (`workbench` and `quick-predict` both use `analysis` and
`composer`); `shared/` never imports from `features/`.

## Design system

One visual language across every route (plan: `docs/plans/frontend-design-unification-2026-11-13.md`).

- **Tokens** live in `src/styles/theme.css`: neutrals (`ink`, `line`, `surface*`), brand
  (`brand*`, `link`), status (`success|warning|danger|info` with `-soft`, `-border`,
  `-ink`), radii and the type scale. Each has a dark value. Use the semantic utilities
  (`text-ink-secondary`, `bg-brand-soft`, `border-line`, `bg-warning-soft`), not hex values,
  raw Tailwind palette classes or arbitrary pixel sizes.
- **Two frames.** The *workspace* frame (`app/shell/WorkspaceLayout`, rail plus page via
  `WorkspacePage`) holds Phân tích nhanh, Sessions, Cài đặt and the session workbench. The
  *docs* frame (`features/docs`) is public and has its own top bar. The landing page keeps its own.
- **Components, not inline styles.** Status is drawn with `Callout` and `StatusPill`. A
  prediction's class label is `warning` or `neutral`, never `danger` (ADR-0002;
  `features/analysis/labelTone.ts`).
- **Guardrails.** `npm run lint` includes `scripts/style-lint.mjs`, a ratchet: counts of colour
  literals, inline colour styles and legacy aliases are recorded in `scripts/style-baseline.json`
  and may only go down (`node scripts/style-lint.mjs --update` after a cleanup).
  `e2e/visual.spec.ts` opens every route in light and dark, at desktop and phone width.
- **Docs content** is markdown in `features/docs/content/`. A `> [!NOTE] Title` quotation becomes a
  `Callout`; a fenced `card-anatomy` block draws the annotated result card.
