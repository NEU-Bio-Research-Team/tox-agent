# Frontend design unification plan

Date: 2026-11-13. Branch at time of audit: `fix/chatgpt-model-picker` (clean).

Status: **implemented on branch `feat/frontend-redesign`, not committed.** The
audit and proposal are sections 1-9; what was built, what differs from the plan and
what is left is section 12.

Revision 2 (2026-11-13, later the same day): the owner confirmed the colour tone and
asked for a layout and UX redesign of Sessions, Predict, Settings, About and
Documents. Section 6 is that redesign. Revision 3 (same day): the owner approved
the plan ("OK. Thực hiện luôn") and it was built; decisions D2-D9 below are
recorded as adopted on that instruction.

## 1. Goal

One visual language across every route: the same colour tokens, the same
typography, the same shape and spacing rules, and a consistent page frame. A
person moving from the landing page to Quick Analyze to a session should not feel
they changed product.

## 2. Method

1. Code audit of `frontend/src` (109 non-test `.tsx` files, 5 stylesheets): token
   definitions, token consumption, colour literals, fonts, radii, primitives,
   container widths, contrast arithmetic.
2. Visual audit: Chromium (Playwright 1.52, 1440x900 and 390x844) against the Vite
   dev server, light and dark, 14 scenarios, with the API served from the
   repository's own e2e fixtures (`frontend/e2e/fixtures/`). Pages covered:
   landing (9 scroll slices), about, sign-in gate, sessions, analyze (empty and
   with result), documents, settings, and the workbench in five states (published
   answer, release states, answer audit, evidence viewer, case board, withheld
   run). Not covered: states that need a live backend (real streaming, a running
   run, ChatGPT OAuth, batch results, structure drawing, image upload).
3. Reference research (revision 2): public pages of SwissADME, ProTox 3.0, Stripe
   Docs, Vercel Docs, Linear Docs and Perplexity were opened in Chromium and
   screenshotted. ADMETlab 3.0 did not load from this machine and, like the
   conventions for settings pages, comes from web search only. Logged-in products
   were not accessible. Section 6.1 marks what was seen and what was not.

## 3. Findings

### 3.1 Three page frames, three palettes, four accents

| Surface | Routes | Frame | Palette | Display type |
|---|---|---|---|---|
| Marketing | `/` | own sticky header + own `Footer` (`landing/sections/Footer.tsx`) | indigo `#1E0368` / `#2D0A5E` / `#0D0021` hard-coded in class strings | `Cal Sans`, `Climate Crisis`, `Orbitron` (never loaded) |
| Public tool | `/analyze`, `/predict`, `/documents`, `/about` | `Navbar` + `app/shell/Footer` | slate neutrals + blue `#3b82f6` (`.legacy-ui`) on analyze and documents, blue-to-green gradient title, cyan "Rate us!" `#22d3ee`; `/about` is a blue navbar over a purple page body | Inter |
| Workspace | `/sessions`, `/settings`, `/s/:id/**` | `AppSidebar` + `WorkspaceHeader` | purple `#7743e8` (`theme.css`) | Inter |
| Sign-in gate | any guarded route without a token | none | purple | Inter |

The sidebar's "Phân tích nhanh" link goes to `/predict`, an alias of `/analyze`, so
one click swaps the entire frame and palette.

`legacy-ui.css` was added on 2026-11-12 (commit `0c6f5ac`) to keep the production
site's look on the analysis page and navigation bar. This plan reverses that
decision on purpose; see decision D1.

### 3.2 The token layer exists but is bypassed

`theme.css` defines a complete purple system (`--ink*`, `--line*`, `--surface*`,
`--purple-50..700`, `--success/--warning/--danger/--info`, radii, shadows) plus a
block of "compatibility aliases". Consumption in non-test TSX:

| Token family | References |
|---|---|
| `var(--text*)` (alias of `--ink*`) | 472 |
| `var(--border)` | 108 |
| `var(--accent-blue/red/green/yellow)` (aliases) | 195 |
| `var(--surface)` / `var(--surface-alt)` | 93 |
| `var(--success)`, `var(--warning)`, `var(--danger)`, `var(--info)` | **0** |
| `var(--ink*)`, `var(--line)`, `var(--purple-*)` (the new names) | 29 |
| `var(--radius-control)`, `var(--radius-card)` | **0** |

Consequences:

- The aliases are redefined inside `.legacy-ui`. `var(--accent-blue)` is purple on
  the workbench and blue on analyze/documents/navbar; the same source line renders
  two colours. A portal (popover, sheet, dialog) leaves the subtree and must repeat
  the class (`Navbar.tsx:197,264`, `ReleaseNotesModal.tsx:10`).
- 741 inline `style={{ ... }}` objects across 109 files. Top: `InvestigationBoard`
  57, `QuickPredictPage` 45, `Navbar` 33, `StageLineage` 32, `EvidencePassages` 32,
  `RunTrace` 25, `AnswerAuditViewer` 22, `ModelPicker` 21, `AiConnectionsCard` 20,
  `DocumentsPage` 19. Hover, focus and dark variants cannot be expressed inline, so
  each of these is a place where states drift.
- `rounded-answer` is used once and defined nowhere (a silent no-op).

### 3.3 Colour literals outside the tokens

- Navbar cyan pill `#0891b2 / #22d3ee / #083344` + `rgba(8,145,178,.28)` shadow,
  twice (desktop and mobile).
- Landing: `#1E0368` (about 25 uses), `#2D0A5E`, `#0D0021`, `rgba(26,0,77,.3)`,
  `rgba(255,255,255,.3)`.
- `QuickPredictPage` / `StructureImagePanel`: tinted status boxes built from
  `rgba(34,197,94,.08)`, `rgba(239,68,68,.08)`, `rgba(245,158,11,.08)`,
  `rgba(59,130,246,.08)` and `rgba(239,68,68,.35)` borders (10 sites). Those are the
  legacy palette's RGB values, so they would not follow a palette change.
- `#ffffff` backgrounds behind molecule depictions (4 sites). Intentional (the
  depictions are black-on-transparent) and should be one named token.
- `ExplanationLegend.tsx`: `#D73027 / #1A9850 / #BDBDBD`. A scientific diverging
  scale; keep it, but name it and check it for colour-vision deficiency.
- `ModelPicker.tsx`: avatar tint from `hsl(${hue} ...)`.
- `FirebaseSignIn.tsx`: Google brand SVG colours. Correct as is.

### 3.4 Typography

- `fonts.css` only declares `local("Inter Variable")`, `local("Inter")`,
  `local("JetBrains Mono")`. No font file ships in the bundle (`dist/assets` has no
  `woff/ttf`). On this machine Inter happens to be installed system-wide
  (`/usr/share/fonts/opentype/inter`), so screenshots here show Inter. A visitor to
  the public site without Inter gets the OS UI font, so the look differs per
  visitor.
- `font-['Cal_Sans']` (36 uses), `font-['Climate_Crisis']` (12), `font-['Orbitron']`
  (2) are never declared, loaded or installed. The browser falls back to the default
  serif. The landing page renders in Times New Roman: headline, navigation, buttons,
  tech-stack tiles (confirmed in `01-landing-*.png`).
- Sizes: 310 `text-xs` and 166 `text-sm` against 10 `text-base`; 24 uses of
  `text-[11px]`, 7 of `text-[13px]`, plus `12.5px`, `11.5px`, `10px`, `12px`,
  `32px`. There is no declared scale, so 11, 12, 12.5 and 13px all occur for
  "caption".

### 3.5 Shape, spacing, frame

- Radii: `rounded-md` 73, `rounded-lg` 56, `rounded-full` 47, `rounded-xl` 40, plus
  `2xl/3xl`. `--radius-control` (12px), `--radius-card` (18px) and
  `--radius-floating` (24px) are defined and unused, so cards are 12, 16 and 18px
  depending on the file.
- Page content widths differ per page: Sessions about 770px, Settings about 670px,
  About about 720px, Analyze about 975px, Documents about 880px plus a left rail. They
  are `max-w-2xl/3xl/6xl/7xl` and `max-w-[960px]/[1480px]` chosen file by file.
- Two `Footer` components (`app/shell/Footer.tsx` 13 lines, `landing/sections/
  Footer.tsx` 58 lines) with different background, type and content.
- Sessions page has no visible page title of its own: only the 14px label in the
  workspace header and a muted one-line description, while Analyze has a 48px
  display title and Documents a card header.

### 3.6 Primitives and patterns

- Native controls inside a shadcn UI: 5 `<select>` (`CaseChanges`,
  `InvestigationBoard`, `AiConnectionsCard`, `ExplainPanel`, `QuickPredictPage`)
  against 22 uses of the shared `Select`; 6 `<details>` with the browser triangle;
  45 raw `<button>` against 51 `<Button>`. Result: the Settings provider select is a
  native control with the OS chevron while the composer's model picker uses the
  shared `Select`; native controls render differently per OS.
- There is no shared callout. Release banner (outlined, transparent), limitation
  block (filled with an amber left bar), validation message (filled tint),
  withheld banner (outlined amber), unavailable banner (outlined neutral), error
  boxes (red tint) are separate hand-written patterns.
- Inspector panel at 332px: `TabsList` overflows (the last tab, "Ki...", is clipped
  in `13-wb-run-withheld`), and long identifiers (`run_3333...`, `ans_1111...`) run
  past the panel edge instead of truncating or wrapping.
- Links in answers are raw underlined purple, with no shared link style.

### 3.7 Dark mode

- The logo (`assets/logo-tox.png`, navy) is nearly invisible on the dark navbar and
  the dark sidebar.
- `--purple-50 ... --purple-700` have no dark counterparts. Any component that
  uses `--purple-50` as a soft background (the preview's "Rate us!" pill) glows
  white-lilac in dark.
- `--accent-blue` (purple-600) is used for both fills and text links. As link text
  on `--surface-solid` dark it is 3.1:1.
- `.legacy-ui` dark `--text-faint #475569` on `--surface #161b27` is 2.3:1.

### 3.8 Contrast (WCAG 2.x, computed)

| Pair | Ratio | Verdict at 12-14px |
|---|---|---|
| `--ink-tertiary #918a9c` on white | 3.32 | fail (AA needs 4.5) |
| `--ink-tertiary` on `--surface-muted` | 3.09 | fail |
| legacy `--text-faint #94a3b8` on white | 2.56 | fail |
| legacy `--accent-green #22c55e` text on white | 2.28 | fail |
| legacy `--accent-yellow #f59e0b` text on white | 2.15 | fail |
| legacy `--accent-red #ef4444` text on white | 3.76 | fail |
| white on legacy `--accent-blue #3b82f6` button | 3.68 | fail |
| `--purple-600 #7743e8` link on white | 5.60 | pass |
| `--purple-600` link on dark `--surface-solid` | 3.11 | fail |
| `--ink-secondary #5f5a68` on white | 6.67 | pass |
| `--success #16865c` on white | 4.57 | pass (barely) |
| `--warning #b36a00` on white | 4.22 | fail by a hair |

The new purple system is mostly sound; the legacy palette and the tertiary ink are
the main offenders. 11px and 12px text at 3:1 is where "faint" labels sit (caption
rows in the result cards, `QUICK EXAMPLES`, `MOLECULE PREVIEW`).

### 3.9 Language

The product sets `<html lang="vi">`, but chrome mixes both: "Session mới",
"Phiên mới", "Tất cả sessions", "Cài đặt", "Giới thiệu" in the sidebar against
"Sessions", "Settings", "About", "Documents", "Analyze", "Rate us!" in the navbar.
The same destination is named differently in each frame (Settings / Cài đặt,
About / Giới thiệu). "Session mới" and "Phiên mới" name the same concept in two
words.

### 3.10 Product constraint the redesign must respect

ADR 0002 and `scripts/policy-lint.mjs` forbid an aggregate verdict, and the About
page states that no screen merges the measurements "into one label, one colour or
one number". Therefore:

- Status colours are semantic per measurement (a blocker badge for hERG, a row
  state for a Tox21 assay), never a page-level or card-level traffic light.
- "Not run" and "failed" are never drawn in the danger colour ("not run is not a
  negative result"); they use neutral or info.
- Any new status component takes its tone from a single measurement's own field.

## 4. Decisions

| # | Decision | Status | Recommendation | Why |
|---|---|---|---|---|
| D1 | Brand axis: purple, blue, or keep the split | **Resolved 2026-11-13** ("tone màu là ổn" after seeing the purple preview) | Purple | The purple system is complete and the landing indigo is the same hue family. Retires the blue palette, the blue-to-green title and the cyan pill. Reverses `0c6f5ac`. |
| D2 | Landing: re-skin fully, or keep the dark hero as a brand moment | Adopted 2026-11-13 | Keep the composition, move it onto tokens (`--brand-deep`), drop unloaded fonts | Consistency here is shared tokens and type, not identical layout. |
| D3 | Display type | Adopted 2026-11-13 | Inter 700/800 and JetBrains Mono, self-hosted | No new asset or licence; shown in the preview. Cal Sans, Climate Crisis and Orbitron need a licence check first. |
| D4 | Frames | **Adopted and built** | Two frames: Workspace (rail) and Docs; `/analyze` moves into Workspace, no longer optional | Removes the frame jump from the sidebar link and the second navigation. |
| D5 | Language rule | Adopted 2026-11-13 | Chrome in Vietnamese, identifiers in English, one glossary; docs articles in Vietnamese | `lang="vi"` already says so. The mockup copy is a draft for the owner to review. |
| D6 | Rewrite or rebuild | Adopted 2026-11-13 | Rebuild the presentation route by route, keep API, events, hooks, auth and tests (section 6.6) | Tests do not assert style; logic is not where the findings are; live site auto-deploys from `main`. |
| D7 | About page | Adopted 2026-11-13 | Fold into Docs as "Bốn ràng buộc của ToxAgent"; `/about` redirects | The page is four statements of policy; they belong with the docs, with a pointer to where each shows in the UI. |
| D8 | Where do the version badge, release notes, "Rate us!" and GitHub go once the navbar leaves the app pages | Adopted 2026-11-13 | Settings "Về bản này", Docs header, landing footer | The app frame stays free of promotion and build metadata. |
| D9 | "Lưu vào session mới" on Analyze creates a session and submits the molecule | Adopted 2026-11-13 | Yes, through the existing session and message endpoints | Gives a one-click path from a throwaway result to an audited session. No new API. |

## 5. Target design language

Three layers: primitives (raw values), semantic tokens (what a thing means),
component styles (what a component uses). Components consume only semantic tokens.

### 5.1 Colour

- Neutrals: keep `--ink`, `--ink-secondary`, `--line`, `--surface*`, `--canvas*`.
  Raise `--ink-tertiary` to at least 4.5:1 on `--surface-muted` (preview used
  `#6b6577`, about 5.2:1 on white). Keep three ink levels only.
- Brand: `--purple-50..900` primitives with explicit dark values, and semantic
  `--primary`, `--primary-soft`, `--primary-soft-ink`, `--link`, `--focus`. Link and
  fill become different tokens (a lighter link tone in dark; white-on-fill stays at
  or above 4.5:1).
- Status: `--success/--warning/--danger/--info`, each with `-soft` (background),
  `-border` and `-ink` (text) variants for light and dark. Replaces every
  `rgba(...,0.08)` tint and every `var(--accent-*)` use.
- `--brand-deep` / `--brand-deep-2` for the landing gradient (values from the
  current hard-coded indigo, so nothing moves visually).
- Named utility colours: `--depiction-bg` (white), the explanation scale
  (`--attr-positive/-negative/-neutral`).
- Alias block in `theme.css` is kept during migration with a documented map, then
  deleted.

### 5.2 Typography

- Self-hosted woff2 for Inter variable and JetBrains Mono variable, `font-display:
  swap`, preload the Inter file. Remove `local()`-only declarations and the three
  unloaded families.
- One scale: 11 (caption, only for non-essential metadata), 12, 14, 16, 18, 20, 24,
  and a display step. Express as tokens in `@theme` and ban `text-[Npx]` in the
  lint ratchet.

### 5.3 Shape and elevation

- Adopt `--radius-control` 12, `--radius-card` 18, `--radius-floating` 24, mapped
  into Tailwind's `rounded-md/lg/xl`. Delete `rounded-answer` or define it.
- Shadows: `--shadow-float` for floating surfaces only; no per-file box-shadows.

### 5.4 Layout

- Three container tokens: prose (about 720px), page (about 1120px), wide
  (about 1280px). Each route picks one; the choice is recorded in the route table.
- One `PageHeader` (title, description, actions) used by Sessions, Settings,
  Documents, Analyze.
- One `Footer` with a marketing variant.

### 5.5 Components to add or replace

`Callout` (tone = neutral | info | success | warning | danger, with icon, title,
body), `StatusBadge`, `Select` (replace 5 native), `Disclosure` (replace 6
`<details>`), scroll-safe `TabsList`, `Logo` (dual-tone SVG or two assets), and a
`truncate-id` pattern for run/answer/evidence ids. The full list, extended for the redesigned
pages, is in 6.5.

## 6. Page redesign (added 2026-11-13)

Owner feedback after the first audit: the colour tone is fine; the real problem is
that Sessions, Predict (Analyze), Settings, About and Documents do not share a
layout or a UX approach, and a rebuild is acceptable. This section is the design
for those five pages plus the frames that hold them. A visual mockup of every
screen is on the design canvas (section 8).

### 6.1 What was learned from other platforms

Public pages were opened in Chromium on 2026-11-13 and screenshotted; logged-in
products were not accessible. "Verified" below means seen on screen in this
session; "search only" means taken from search results and not seen.

| Platform | What it does | Taken for ToxAgent | Rejected |
|---|---|---|---|
| SwissADME (swissadme.ch), verified | The home page is the tool: structure sketcher on the left, a SMILES list on the right, "Fill with an example", "Clear", "Run!". Dated chrome and a news banner above the tool. | Draw and paste are two ways into one screen; examples and clear sit next to the input. | Chrome and banners that push the tool down. |
| ProTox 3.0 (tox.charite.de/protox3), verified | A long prose and image landing page before the tool; the tool is a tab. Search results say each endpoint carries a confidence score and the site adds an overall toxicity radar (search only). | Per-endpoint confidence is the right unit. | The overall radar (ADR-0002) and any marketing page between the user and the tool. |
| ADMETlab 3.0, search only (the site did not load from this machine) | Table-first results with symbolic probability bands, uncertainty estimates, CSV export, a per-molecule detail view. | Dense table for Tox21 and batch; show uncertainty and limits; export. | Not evaluated visually. |
| Stripe Docs, verified | Search with a `/` hint and "Ask AI" in the top bar, product tabs, collapsible left rail, two-column pages with a sticky side pane, numbered steps. | Docs frame: search first, article column, right-hand "on this page". | Code-first density. |
| Vercel Docs, verified | Left nav grouped under small labels, `⌘K` search, a hero with three buttons, card grid. | Group labels, `⌘K`, previous/next at the foot of an article. | The hero for a small docs set. |
| Linear Docs, verified | Dense grouped left nav with icons, footer links (Docs, Developers, Learn, Contact), card grid of popular articles; dark parity. | Dense grouped nav; the docs site works in both themes. | Card grid as the only entry. |
| Perplexity, verified | Left rail: New, nav items, Recents, account at the bottom. An input-first start page with a prompt, a model picker inside the input, and two example cards beneath. | The rail anatomy; an empty state that already contains the input and examples. | A single global input for everything; the "Computer" and "Teammates" upsells. |
| Settings pages in general, search only | Left section navigation on larger apps; label-and-description on the left with the control on the right; connected accounts and API keys as their own sections, with create, scope, revoke and last-used. | The same row pattern; connections as rows with status and actions. | Not verified against ChatGPT, Claude or Linear screens. |

### 6.2 Design principles

1. **One frame per purpose, never two navigations.** Workspace (left rail) for
   Analyze, Sessions, Settings and the session workbench; Docs (top bar, grouped
   nav, right-hand table of contents) for documentation and principles; the
   landing page keeps its own frame.
2. **The tool is the page.** Analyze opens with the input and with a description
   of what will come back. Nothing marketing sits between the user and the input.
3. **Design every state.** Empty, loading, error, partial and populated are drawn
   for each page; the empty state teaches (what each measurement is) instead of
   leaving a blank pane.
4. **Honesty is persistent UI.** "No combined score", "probability not
   calibrated", "result not saved", "ClinTox not served here" are visible
   components, not footnotes.
5. **One component per job.** Callout, StatusPill, Segmented, Switch,
   DataRow, ProviderTile, EndpointCard, AssayTable, PageHeader, EmptyState.
   Hand-built tinted boxes and inline styles stop.
6. **Machine strings are monospace** (SMILES, ids, model ids, `observation_id`);
   prose is the UI sans.
7. **Tone follows ADR-0002.** A positive model label ("blocker", "active") uses
   the warning tone, a negative label is neutral, "not run" and "unavailable" are
   neutral or info; none of them uses the danger tone, and no card or page carries
   a combined colour.
8. **Language rule (D5).** Interface chrome in Vietnamese, one glossary
   (Phân tích nhanh, Sessions, Tài liệu, Cài đặt, Session mới), technical
   identifiers in English.

### 6.3 Frames and information architecture

| Route | Today | Proposed frame and page |
|---|---|---|
| `/` | Marketing frame | Marketing frame, on tokens, shared type and one footer |
| `/analyze`, `/predict` | Public navbar, blue palette | **Workspace frame**, "Phân tích nhanh" (one route, `/predict` stays an alias) |
| `/sessions`, `/s/:id/**`, `/settings` | Workspace (sidebar) | Workspace frame |
| `/documents` | Public navbar | **Docs frame**, `/docs` and `/docs/:slug`; `/documents` redirects |
| `/about` | Public navbar | Folded into Docs as the article "Bốn ràng buộc của ToxAgent"; `/about` redirects to `/docs/principles` |
| Sign-in gate | Plain centred card | Centred card with the logo and the same tokens (no frame) |

Workspace rail (264px, collapsible to an icon rail, a sheet on phones): logo and
`⌘K` search; "Session mới"; Phân tích nhanh, Sessions, Tài liệu; "Gần đây" (running
sessions first, at most five, link to all); Cài đặt; account chip and theme toggle.
The sidebar stops showing date groups; full history lives on the Sessions page.
Top bar inside the content area: breadcrumb on the left, page actions on the right.

Items the old navbar carried and where they go (decision D8): version, Beta chip,
build time and release notes move to Settings under "Về bản này" (a row with the
release-notes dialog); "Rate us!" and GitHub move to the Docs header and the
landing footer; Sessions, About, Settings and Documents become the rail and docs
nav above.

### 6.4 Page specifications

**Sessions** (mockup: "Sessions")
- Today: card list of rows, search that only covers loaded rows, running state
  duplicated against the sidebar, no page title of its own.
- New: page title and one-line purpose; "Đang chạy" strip for sessions with an
  active run; toolbar with search, a Tất cả / Đang chạy filter and a sort control;
  one dense list with date group headers (Hôm nay, Hôm qua, Tuần này), each row
  title, one-line preview, run count (monospace), relative time, running pill;
  footer "Đang hiển thị N trong M session đã tải" and "Tải thêm". Search says what
  it covers ("trong N session đã tải").
- Data: everything comes from `SessionListRow` (title, preview, run_count,
  updated_at, active_run, status). No new API. Row actions (rename, copy id) reuse
  the existing rename mutation and id copy. No delete is drawn because no delete
  endpoint exists in `shared/api/endpoints.ts`.
- States to build: loading skeleton rows, error with retry, empty (three example
  prompts plus a link to Phân tích nhanh), filtered-empty.

**Phân tích nhanh / Predict** (mockups: "trạng thái trống", "có kết quả")
- Today: one long column; input card, then preview, then results far below the
  fold; endpoints and models hidden under "Advanced options"; mixed English and
  Vietnamese; native selects; a blue-to-green title.
- New: two panes. Left, a fixed-width input card: segmented SMILES / Vẽ / Ảnh,
  Một chất / Danh sách switch, SMILES field with inline validation, example chips,
  structure preview, "Phép đo" as selectable rows with the model name and a
  disabled row for a measurement the deployment does not serve (ClinTox), advanced
  options collapsed to one line, and a primary "Phân tích" button (`⌘+Enter`).
  Right, results: before a run, one ghost card per measurement saying what it will
  show, plus the "no combined score" callout; after a run, an applicability pill
  that stays visible, one EndpointCard per measurement (hERG: probability, a bar
  with the threshold tick, threshold, threshold source, model, collapsible
  explanation; Tox21: assay table with mini bars and label, expandable to all 12),
  a persistent limitations callout, and a save bar ("Kết quả này chưa được lưu":
  "Lưu vào session mới" and "Hỏi agent về kết quả").
- Batch: the right pane becomes one table, one row per molecule and one column
  per measurement, with no total column. Compare mode: one column per model.
- Mobile: panes stack; the "Phân tích" button is sticky at the bottom.
- Logic is kept: `quickPredict`, `quickPredictBatch`, `quickPredictCompare`,
  `quickPredictCapabilities`, endpoint selection preference. New behaviour to
  confirm (D9): "Lưu vào session mới" creates a session and submits the molecule
  through the existing session message endpoint.
- The 658-line `QuickPredictPage` is split into `InputPanel`, `EndpointPicker`,
  `ResultsPane`, `EndpointCard`, `AssayTable`, `SaveBar`, with the state hook kept.
  `QuickPredictPage.test.tsx` and `e2e/quick-predict.spec.ts` stay as the safety
  net; only selectors that move are updated.

**Cài đặt / Settings** (mockup: "Cài đặt")
- Today: three stacked cards, a native provider select, a blank-looking add form,
  "Nâng cao" as a browser disclosure triangle, English labels (Control plane,
  Access token) beside Vietnamese ones.
- New: left section navigation (Kết nối AI, Tài khoản, Hiển thị, Nâng cao) and
  label-left / control-right rows. Kết nối AI: connected connections as rows with
  a status pill (Sẵn sàng, Kết nối lỗi), Kiểm tra and Xoá, and an inline remedy
  for a failed one ("Đăng nhập lại"); "Thêm kết nối" as provider tiles (OpenAI,
  DeepSeek, OpenAI-compatible; the list comes from the providers endpoint) that
  open the form for that provider, with API key or ChatGPT Plus/Pro as a segmented
  choice and the existing privacy sentence. Tài khoản: control plane, status, the
  token (masked) and a separated "Ngắt kết nối". Hiển thị: theme as Sáng / Tối.
  Nâng cao: the two existing switches with their existing descriptions. "Về bản
  này": version, build, release notes.
- Logic is kept: `AiConnectionsCard` state machine and the ChatGPT device flow;
  only its presentation is rebuilt.

**Tài liệu / Docs** (mockups: "Đọc kết quả", "Nguyên tắc")
- Today: the navbar plus a rail with one document and a hand-rolled table of
  contents of scroll buttons; every section is a bordered card; English text.
- New: Docs frame (logo and "Tài liệu", search with `⌘K`, theme toggle, "Mở ứng
  dụng"); grouped left navigation (Bắt đầu, Hiểu kết quả, Làm việc với agent,
  Nguyên tắc); article column about 720px with breadcrumb, lede, prose without
  card wrappers, callouts for the invariants, one annotated figure ("anatomy of a
  result card"), previous/next; right-hand "Trên trang này" with scroll-spy.
- Content as markdown files rendered with the `react-markdown` and `remark-gfm`
  already in the bundle, so articles are edited as text and can be translated; the
  annotated figure and callouts are small components referenced from markdown.
- About is not a separate page any more: its four statements become the article
  "Bốn ràng buộc của ToxAgent", each with a "Bạn thấy ở đâu" line that points to
  where the constraint shows in the UI.
- The article text in the mockup is a Vietnamese draft derived from the current
  English guide and the About page; the owner reviews it (D5).

### 6.5 Components to build

`AppShell` (Rail, TopBar), `DocsShell` (header, NavGroup, Toc), `PageHeader`,
`Segmented`, `Switch` (restyle the existing one), `Callout` (tone: neutral, info,
success, warning, danger), `StatusPill`, `DataRow` (label and description with a
control), `KeyValueGrid`, `EmptyState`, `EndpointCard`, `AssayTable`,
`ConnectionRow`, `ProviderTile`, `Mono` (text style), and the existing shared
`Select` in place of the five native selects.

### 6.6 Rebuild or rewrite

Recommendation: **rebuild the presentation, keep the engine** (a strangler
approach, route by route). Reasons, all measured above:

- No unit or e2e test asserts a colour, style or class, and the e2e "stories"
  encode the scientific release states; they are the safety net and they survive
  a presentation rebuild.
- The problems found are in frames, layout, inline styles and primitives. The API
  client, the SSE event bus, the query hooks, the auth flow and the preference
  store are not part of any finding.
- A from-scratch rewrite of 109 components would re-derive logic that already
  enforces no-aggregate and the release banners (ADR-0002), and a push to `main`
  that touches `frontend/**` deploys the live site.

Order: foundations, frame, the two low-risk pages (Sessions, Settings), Analyze,
Docs, landing, then the workbench. The workbench (57 inline styles in
`InvestigationBoard` alone, the transcript, the artifacts panel, the run
inspector) is the core product surface and is **not designed here**: it inherits
the new frame and tokens in this plan and gets its own design plan afterwards.

## 7. Phases

Each phase ships on its own, is verified before the next starts, and is merged to
a branch checked on the `main-preview` channel before any live deploy. Size is
relative (S: about a day, M: a few days, L: a week or more).

| Phase | Content | Main files | Verified by | Size |
|---|---|---|---|---|
| P0 Guardrails | Commit the screenshot harness as an e2e visual spec (light and dark, desktop and mobile, the ten routes) and take baselines of the current UI. Add `scripts/style-lint.mjs` to `npm run lint` as a ratchet: counts of hex/rgb literals, raw palette classes, inline colour styles and `text-[Npx]` may only go down. | `e2e/visual.spec.ts`, `scripts/style-lint.mjs`, `package.json` | `npm run lint`, `npx playwright test visual --workers=1` | S |
| P1 Foundations | Semantic and dark tokens (5.1), contrast fixes, self-hosted fonts, delete `.legacy-ui`, dual-tone logo, `--brand-deep`. New primitives: `Callout`, `StatusPill`, `Segmented`, `Switch` restyle, `DataRow`, `EmptyState`, `Mono`; replace the five native selects with `Select`. | `styles/*.css`, `shared/ui/*` | vitest for primitives; contrast script; visual diff review | M |
| P2 Workspace frame | `AppShell` (rail and top bar) replacing the navbar on app pages and the current sidebar; recents behaviour; account chip and theme toggle; the sign-in gate on tokens; route `/analyze` under the frame. | `app/shell/*`, `router.tsx`, `features/auth/RequireToken.tsx` | e2e route tests; visual baseline; keyboard order | M |
| P3 Sessions and Settings | Rebuild both pages per 6.4 on the new frame. Settings keeps the `AiConnectionsCard` logic and the ChatGPT flow. Add "Về bản này". | `features/sessions/*`, `features/settings/*` | `AiConnectionsCard.test.tsx`; visual baseline; manual connect/test/remove with a test key | M |
| P4 Analyze | Split `QuickPredictPage` into the panes and cards of 6.4; keep every request unchanged; endpoints promoted out of "Advanced"; save bar and handoff (D9); batch and compare layouts. | `features/quick-predict/*`, `features/analysis/*` | `QuickPredictPage.test.tsx`, `e2e/quick-predict.spec.ts`, phone-width check | L |
| P5 Docs | `DocsShell`; markdown content under `features/docs/content/`; search; scroll-spy table of contents; the annotated figure; redirects for `/about` and `/documents`; Vietnamese text reviewed by the owner (D5). | `features/docs/*`, `router.tsx`, delete `AboutPage`, `DocumentsPage` | route tests for redirects; link check; visual baseline | M |
| P6 Landing | Landing on tokens and the shared type scale; one `Footer`; remove the 50 unloaded-font references. | `features/landing/**`, `app/shell/Footer.tsx` | visual baseline at 1440 and 390 | M |
| P7 Workbench on the new frame | Re-seat the session page in the frame, move its inline colours to tokens and `Callout`/`StatusPill`, fix the clipped inspector tabs and overflowing ids. A full workbench redesign is a separate plan. | `features/workbench/**` | `e2e/stories.spec.ts`, `e2e/workbench.spec.ts`; ratchet lowered | L |
| P8 Dark and accessibility | Review every route in dark, add an automated axe check to the visual spec, close remaining contrast failures, delete the compatibility alias block in `theme.css`. | any | axe clean on the ten routes; alias block gone | M |

Order matters: P1 and P2 carry the most pixels for the least logic; P3 is the
lowest-risk proof of the new frame; P4 is the largest single piece of logic and
goes after the frame is stable.

## 8. Previews (throwaway, not in the repository)

### 8.1 Token-level copy

Applied to a copy of `frontend/` under the session scratchpad, then re-captured
with the same harness:

1. `legacy-ui.css` emptied (class becomes a no-op).
2. `--ink-tertiary` `#918a9c` to `#6b6577`; added `--brand-deep`.
3. Cyan "Rate us!" replaced by a purple-outline pill (desktop and mobile).
4. Analyze title gradient blue-to-green replaced by purple-700 to purple-500.
5. Dark mode: logo inverted to white through a CSS filter (stop-gap; the real fix is
   a dual-tone asset).
6. Landing: `Cal Sans`, `Climate Crisis`, `Orbitron` replaced by Inter 700/800.

Observed: the navbar, analyze page, documents and about pick up the purple system
with no other edits (the aliases already pointed there); the landing stops being
serif; the dark logo returns. Still wrong after the preview, and in scope for P1 (tokens) and P2 (frame):
the "Rate us!" pill in dark (no dark counterpart for `--purple-50`), the dark link
contrast, and the frame jump between `/analyze` and `/sessions`.

### 8.2 Design canvas (revision 2)

Six screens on a private design canvas: https://claude.ai/artifact/UkQNRWktBt9rzLXUnfj2Qt

- Workspace frame: Sessions; Phân tích nhanh empty; Phân tích nhanh with a result;
  Cài đặt.
- Docs frame: "Đọc kết quả" and "Bốn ràng buộc của ToxAgent" (the About
  replacement).

Facts about the mockups, so they are not mistaken for finished work:

- Light theme only. Dark values are specified in section 5.1 but not drawn.
- Phone width is handled by wrapping flex layouts and was not checked.
- The logo is the repository's `logo-tox.png`; the aspirin structure is a crop of
  the structure the running app drew in the audit screenshots.
- Values come from the e2e fixtures where they exist (hERG 73,0% with threshold
  0,5; SR-MMP 0,800; NR-AR 0,120). The other Tox21 rows, session titles and
  times are illustrative.
- Vietnamese copy is a draft derived from the current screens; the owner reviews it.
- Not drawn: loading and error states (except one failed connection in
  Settings), batch and compare results, the draw and image input tabs, the
  ChatGPT device flow, the release-notes dialog, the sign-in gate, the landing page
  and every workbench screen.
- The canvas was published without a visual check of the rendered page. Review it
  and report anything clipped or misaligned.

## 9. Risks

- **Reverses a deliberate, one-day-old decision** (D1). Existing users of the
  production analyze page will see a colour change.
- **Deploy coupling.** A push to `main` that touches `frontend/**` deploys the live
  site (`frontend-hosting.yml`) when `TOXAGENT_API_ORIGIN` is set. Work stays on a
  branch and is checked on the `main-preview` channel first; the live deploy is the
  owner's command.
- **Test blast radius is small.** No unit or e2e test asserts a colour, style or
  utility class (`toHaveStyle`/`toHaveClass` count: 0); they assert roles and text.
  Changing labels in P5 (D5) is the exception and needs the e2e text selectors
  updated together.
- **Laptop memory.** Visual runs use one Chromium, `--workers=1`, one Vite server,
  `nice -n 10`; the machine has about 2 GB free while the public stack is up.
- Fonts: self-hosting adds about 100-300 KB; `check-bundle-budget.mjs` must be
  checked after P1.
- **Scope.** Section 6 is a rebuild of five pages and two frames; P4 (Analyze) is
  the largest. Phases are independent so the project can stop after any of them
  with a coherent product.
- **Copy.** The mockup text is a draft. Changing visible labels (D5) also changes
  the Vietnamese strings that the e2e tests select on; update both together.
- **Behaviour change (D9).** "Lưu vào session mới" is new behaviour built on
  existing endpoints; confirm before building.
- **Promotion removed from the app frame (D8).** "Rate us!" and GitHub leave the
  app chrome; confirm that is acceptable for the product's goals.

## 10. What is not covered

Live streaming states, a run in progress, ChatGPT/OAuth connection UI, structure
drawing and image upload, batch and compare results, the release-notes modal and
the account popover, keyboard focus order, and screen-reader behaviour. P6 covers
some of these; the rest need a live backend.

Revision 2 adds: loading and error states of most redesigned pages, dark and
phone mockups, batch and compare layouts, the workbench, the landing page and the
sign-in gate. The first audit covered the current workbench visually; its redesign
is a separate plan.

## 11. Sources

Opened and screenshotted on 2026-11-13:
[SwissADME](https://www.swissadme.ch/),
[ProTox 3.0](https://tox.charite.de/protox3/),
[Stripe Docs](https://docs.stripe.com/payments/quickstart),
[Vercel Docs](https://vercel.com/docs),
[Linear Docs](https://linear.app/docs),
[Perplexity](https://www.perplexity.ai/).

Read through search results only (not seen on screen):
[ADMETlab 3.0 paper](https://academic.oup.com/nar/article/52/W1/W422/7642316),
[ProTox 3.0 FAQ](https://tox.charite.de/protox3/?site=faq),
[SaaS settings UX patterns](https://www.saasui.design/blog/saas-profile-account-ux-patterns),
[settings UI design](https://setproduct.com/blog/settings-ui-design).
Statements about those four are the search snippets' words, not verified on the
live pages.

## 12. Implementation status (2026-11-13)

### 12.1 Phases

| Phase | Status | What exists |
|---|---|---|
| P0 Guardrails | Done | `scripts/style-lint.mjs` ratchet in `npm run lint` with `scripts/style-baseline.json`; `e2e/visual.spec.ts` (9 routes x light/dark x desktop/phone, no sideways scroll, one `h1`, screenshot attached). |
| P1 Foundations | Done | Semantic and dark tokens, type scale (`text-2xs`, `text-13`, `text-15`), self-hosted Inter and JetBrains Mono (Latin, Latin-ext, Vietnamese, about 207 KB, fetched by unicode range), `.legacy-ui` deleted, alias block deleted, 844 alias uses and the cyan, blue and indigo literals migrated, undefined tokens `--border-subtle` and `--shadow-purple` fixed. New primitives: `Callout`, `StatusPill`, `Segmented`, `DataRow`/`DataList`, `EmptyState`, `NativeSelect`, `PageHeader`, `Logo`, `CommandDialog`. |
| P2 Workspace frame | Done | New rail (`AppSidebar`), `WorkspacePage`, Ctrl/Cmd+K palette over pages and loaded sessions, `AccountMenu`, `ThemeToggle`; Navbar and the old Footer deleted; `/analyze` and `/predict` render inside the frame; the route error page no longer needs the Navbar. |
| P3 Sessions and Settings | Done | Sessions: running strip, filter, sort, grouped dense list, loading, error, empty and no-match states. Settings: section list with scroll-spy, label/control rows, provider tiles, connection rows with a failed-state remedy, theme control, "Về bản này" with release notes. `AiConnectionsCard` logic and tests kept. |
| P4 Analyze | Done, with gaps below | Two panes, measurement picker, advanced options, per-measurement cards, applicability pill, limitations from `required_limitations`, save bar ("Lưu vào session mới", "Hỏi agent về kết quả"), Ctrl+Enter. Shared result cards (`EndpointCard`, `Tox21AssayTable`, `ApplicabilityChip`, `EndpointUnavailableCard`) are restyled for the workbench too. |
| P5 Docs | Done | Docs frame, seven Vietnamese markdown articles, search (Ctrl/Cmd+K), scroll-spy contents, annotated result card, previous/next; `/about` redirects to `/docs/principles`, `/documents` and `/docs` to `/docs/intro`. `AboutPage` and `DocumentsPage` deleted. |
| P6 Landing | Done | Colours on `brand-deep` tokens, unloaded display fonts removed (Inter 700/800 instead), dead footer links (`/research`, `/contact`) replaced by Documentation, Principles, Quick Predict and Feedback. |
| P7 Workbench | Done for the answer, verification and source surfaces (12.5); the rest is on tokens only | The workbench renders in the new frame and on the new tokens; its answer and checking surfaces were redesigned in plain language (revision 4, section 12.5). The investigation board, report blocks and run trace are restyled by token only. |
| P8 Dark and accessibility | Partial | Every route reviewed in dark and at phone width by screenshot; alias block deleted. No automated axe check was added. |

### 12.2 Where the build differs from the plan

- **Native selects.** `CaseChanges`, `InvestigationBoard`, `ExplainPanel` and the model
  choice in Analyze use the new styled `NativeSelect`, not the Radix `Select`: the e2e
  stories drive them with `selectOption`. The Settings provider select became tiles.
- **Sessions rows** are links only. Rename and copy-id were not added to the list because
  `renameSession` needs `expected_version`, which the list row does not carry; both stay in
  the session header.
- **Batch results** are still one block per molecule, not one table with a column per
  measurement.
- **Tox21 table** keeps the model's fixed task order and shows all 12 rows. The mockup
  sorted by probability and collapsed to five; that was dropped because a ranking or a hidden
  row conflicts with SCI-05 in `Tox21AssayTable`.
- **Thresholds in the UI** use the existing `toFixed` formats (`73.00%`, `0.800`), not the
  localized `73,0%` of the mockup.
- **Palette** also searches the sessions already loaded in the rail, not the server.
- **Dark logo** is a CSS filter (`.dark .logo-mark`); a dual-tone asset is still needed.
- **Display font** on the landing is Inter 700/800 (D3); the landing copy stays English.
- **`--border`** is kept as a token because shadcn's `border-border` needs it.

### 12.3 Verification

Run from `frontend/` on 2026-11-13 after the last change:

| Check | Result |
|---|---|
| `npm run typecheck` | clean |
| `npm run lint` (eslint, policy-lint, style-lint) | clean |
| `npx vitest run` | 49 files, 296 tests passed (baseline before the work: 46 files, 281 tests) |
| `npx playwright test --workers=1` | 52 passed: 20 existing (stories and workbench included), 32 visual |
| `npm run build` and bundle budget | passed; largest asset `canvas_editor` 357 KB gzip, unchanged |

Style ratchet, before and after:

| Pattern | Before | After |
|---|---|---|
| hex colour literals | 58 | 0 |
| `rgb()`/`hsl()` literals | 19 | 6 (framer-motion shadows on the landing) |
| raw Tailwind palette classes | 23 | 0 |
| inline colour styles | 838 | 191 |
| `text-[Npx]` | 40 | 0 |
| legacy token aliases | 844 | 0 |

Screenshots of Sessions, Analyze, Settings and Docs in light, dark and at 390px were
reviewed by eye against the mockup canvas.

### 12.4 Not done, and not verified

- **Workbench, remaining parts** (investigation board, report blocks, run trace and inspector
  tabs, activity-screening card): on the new frame and tokens only, not redesigned. See 12.5.
- **Not exercised against a live backend:** "Lưu vào session mới" and "Hỏi agent về kết quả"
  (covered by an e2e test with mocked endpoints), the ChatGPT sign-in flow in its new layout,
  image recognition, streaming runs, batch and compare in a real deployment.
- **Automated accessibility check** (axe) and keyboard-order review.
- **Copy review.** The Vietnamese UI text and the seven articles are drafts for the owner.
- Branch `feat/frontend-redesign` is **not committed and not pushed**; a push to `main` that
  touches `frontend/**` would deploy the live site (see section 9).
- `LinkedIn` in the landing footer still points at `https://linkedin.com` (pre-existing placeholder).

### 12.5 Workbench redesign (revision 4, 2026-11-13)

Request: fewer technical words, more things drawn instead of described, easier for a
person who is not a developer. Applied to the surfaces a reader meets most.

| Surface | Before | After |
|---|---|---|
| Answer status (`ReleaseBanner`) | Title and detail in the system's own terms (release policy, review, claim, hard-only) | One sentence in plain words plus a short step track: **Viết nháp, Đối chiếu nguồn, Công bố**, each marked done, not met, not reached or skipped. A withheld answer shows where it stopped. "Khớp nguồn" is never worded as "đúng về khoa học". |
| A person's and the assistant's messages | Same bubble shape | Assistant messages sit on a card behind the product mark; a person's stay a purple bubble. |
| A number in an answer (`ClaimChip` hover) | `claim 1 · kind`, `field_path`, `transform`, truncated observation id | "Con số này đến từ đâu", the point in words, links "Xem kết quả gốc của mô hình" and "Xem đoạn trích · Bảng 1, hàng 3, ô 2". The technical fields appear only when Run Details is on in Settings. |
| Answer checking panel (`AnswerAuditViewer`) | Answer id, `candidate 1/2 sha256`, claim cards with `kind`, `transform`, `field_path` | "Cách câu trả lời này được kiểm tra": the status track, each point with a check mark and its sources as buttons, the limitations, "Hành trình của câu trả lời". Ids, hash and per-point technical fields moved into a closed "Chi tiết kỹ thuật". |
| Journey of the draft (`StageLineage`) | Policy name, `validator cứng`, `Review vòng`, claim keys | "Bản nháp đầu tiên", "Sau khi sửa lỗi định dạng", "Đối chiếu nguồn lần 1": counts as "1 khớp nguồn, 1 chưa có nguồn hỗ trợ"; each point marked Giữ nguyên, Đã viết lại, Đã bỏ or Mới thêm. Policy, disposition and codes in "Chi tiết kỹ thuật". |
| Model result viewer (`ObservationArtifact`) | Observation id, `model_projection`, `canonical_payload`, schema | "Kết quả gốc của mô hình": the quoted value pulled out as a large number, limitations in words, full data and ids behind two closed disclosures. |
| Source viewer (`EvidenceArtifact`) | Status, tier, identifiers, SHA-256 at the top | Title, provider, year and authors on top; status, identifiers and hash in "Chi tiết kỹ thuật". |
| Fallback answer | "ĐÁP ÁN DỰ PHÒNG" | "Bản tóm tắt tự động" with one sentence on why it exists. |
| Empty session | A heading and three example sentences | Three example cards with an icon and a caption (analyse one molecule, ask about a risk, look for signals) and a three-step line, "Bạn hỏi, Mô hình đo, Đối chiếu nguồn". |
| Activity-screening chips | `EGFR IC50\|wild_type · độc tính chưa chạy` | `EGFR IC50 · bản gốc · độc tính chưa chạy` |
| Small copy | "Đang gọi predictor", "Artifacts", "Run đang chạy" | "Đang chạy mô hình dự đoán", "Bảng kết quả", "Câu hỏi đang xử lý" |

Rules kept: no combined verdict or colour (ADR-0002); "chưa chạy" and "không có sẵn" are
never drawn as a negative result; an external source's text stays labelled as external
content; the technical view still exists and is one click or one setting away.

Tests changed with the wording: `ReleaseBanner`, `StageLineage`, `AnswerRenderer`,
`ArtifactUnavailable`, `EvidenceArtifact` and `SubjectSelector` unit tests, and the story
specs (`e2e/stories.spec.ts`). A step-track test was added for the banner.

Not redesigned: the investigation board and case changes, report blocks and the report
progress timeline, the run trace and the run inspector tabs (developer-facing), the
activity-screening card, and the composer. They carry the new tokens and frame.

