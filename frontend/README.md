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
  app/            application shell: sidebar, header, layout, error boundary
  features/
    workbench/    the session page: transcript, answers, artifacts, run
                  inspector, report blocks, investigation board, event store
    analysis/     prediction panels shared by the workbench and quick predict
    composer/     message composer, image upload, structure editor, SMILES parsing
    quick-predict/ the session-less prediction page
    sessions/     session list, its query hook and date grouping
    settings/     connection and AI-provider settings
    auth/         token and PKCE login, the route guard
    landing/      landing and about pages, one file per landing section
  shared/
    ui/           shadcn/Radix primitives
    api/          HTTP client, endpoints, response types
    events/       SSE connection and the session event bus
    lib/          labels, preferences, query client, AI profile
    hooks/        hooks with no feature of their own
  styles/, assets/, test/
```

A feature may import from `shared/` and from another feature's public
components (`workbench` and `quick-predict` both use `analysis` and
`composer`); `shared/` never imports from `features/`.
