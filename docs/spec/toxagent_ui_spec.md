# ToxAgent — Đặc Tả Giao Diện Người Dùng (UI Layout Specification)

> **Dự án:** ToxAgent — Hệ thống phân tích độc tính thuốc đa tác nhân AI
> **Sự kiện:** GDGoC Vietnam 2026 Hackathon Demo
> **Ngày:** 31/03/2026
> **Phiên bản:** 1.0

---

## Mục Lục

1. [Hệ thống thiết kế (Design System)](#1-hệ-thống-thiết-kế)
2. [Trang Index (`/`)](#2-trang-index)
   - 2.1 Navbar
   - 2.2 Hero / Input Section
   - 2.3 Agent Progress Panel
   - 2.4 Quick Verdict Card
3. [Trang Report (`/report`)](#3-trang-report)
   - 3.1 Report Header
   - 3.2 Section 1: Clinical Toxicity
   - 3.3 Section 2: Mechanism Profiling
   - 3.4 Section 3: Structural Explanation
   - 3.5 Section 4: Literature Context
   - 3.6 Section 5: AI Recommendations
4. [Điều hướng giữa các trang](#4-điều-hướng-giữa-các-trang)
5. [Responsive (Thiết kế tương thích)](#5-responsive)
6. [Export / Download](#6-export--download)
7. [Trạng thái trống, lỗi, loading](#7-trạng-thái-trống-lỗi-loading)

---

## 1. Hệ Thống Thiết Kế

### 1.1 Bảng màu

ToxAgent dùng **dark mode làm mặc định** với khả năng toggle sang light mode. Bảng màu mang phong cách "lab notebook lâm sàng" — tối, chính xác, khoa học.

#### Dark Mode (Mặc định)

| Vai trò               | Hex       | OKLCH tương đương      | Dùng cho                                   |
|-----------------------|-----------|------------------------|--------------------------------------------|
| `--bg`                | `#0f1117` | `oklch(12% 0.02 255)`  | Nền chính toàn trang                       |
| `--surface`           | `#161b27` | `oklch(16% 0.03 255)`  | Card, panel, container                     |
| `--surface-alt`       | `#1e2535` | `oklch(20% 0.04 255)`  | Nền input, bảng                            |
| `--border`            | `#2a3348` | `oklch(25% 0.04 255)`  | Viền card, divider                         |
| `--text`              | `#e2e8f0` | `oklch(90% 0.01 250)`  | Văn bản chính                              |
| `--text-muted`        | `#94a3b8` | `oklch(65% 0.02 250)`  | Nhãn phụ, caption                          |
| `--text-faint`        | `#475569` | `oklch(40% 0.03 250)`  | Placeholder, tertiary                      |
| `--accent-green`      | `#22c55e` | `oklch(72% 0.19 145)`  | Non-toxic, safe, success                   |
| `--accent-red`        | `#ef4444` | `oklch(63% 0.24 27)`   | Toxic, danger, error                       |
| `--accent-yellow`     | `#f59e0b` | `oklch(76% 0.17 70)`   | Warning, moderate risk                     |
| `--accent-blue`       | `#3b82f6` | `oklch(60% 0.20 260)`  | CTA chính, link, agent running             |
| `--accent-blue-muted` | `#1e3a5f` | `oklch(25% 0.07 255)`  | Nền highlight agent đang chạy              |

#### Light Mode

| Vai trò               | Hex       | Dùng cho                                   |
|-----------------------|-----------|---------------------------------------------|
| `--bg`                | `#f8f9fb` | Nền chính                                   |
| `--surface`           | `#ffffff` | Card, panel                                 |
| `--surface-alt`       | `#f1f5f9` | Nền input, bảng                             |
| `--border`            | `#e2e8f0` | Viền                                        |
| `--text`              | `#0f172a` | Văn bản chính                               |
| `--text-muted`        | `#475569` | Nhãn phụ                                    |
| `--text-faint`        | `#94a3b8` | Placeholder                                 |
| (accent không đổi)    | —         | Giữ nguyên các màu accent                   |

#### Nguyên tắc mã màu ngữ nghĩa

```
p_toxic >= 0.7  →  --accent-red    (TOXIC)
p_toxic 0.3–0.7 →  --accent-yellow (UNCERTAIN)
p_toxic < 0.3   →  --accent-green  (NON-TOXIC)
```

### 1.2 Typography

| Vai trò              | Font                | Size   | Weight | Ghi chú                          |
|----------------------|---------------------|--------|--------|----------------------------------|
| Display/Hero         | Inter               | 48px   | 700    | Tên app, tiêu đề lớn             |
| Page Heading (H1)    | Inter               | 28px   | 700    | Tiêu đề trang                    |
| Section Heading (H2) | Inter               | 20px   | 600    | Tiêu đề section                  |
| Sub-heading (H3)     | Inter               | 16px   | 600    | Nhãn card, tiêu đề nhỏ           |
| Body                 | Inter               | 15px   | 400    | Văn bản chính                    |
| Body small           | Inter               | 13px   | 400    | Caption, label phụ               |
| Data value           | JetBrains Mono      | 14px   | 500    | Giá trị số, SMILES, score        |
| SMILES/Code          | JetBrains Mono      | 13px   | 400    | Chuỗi SMILES, ID hóa học         |
| Badge/Chip           | Inter               | 11px   | 700    | Uppercase tracking-wide          |

Load fonts qua CDN:
```html
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
```

### 1.3 Spacing & Radius

```
spacing unit: 4px (base)
  xs: 4px   sm: 8px   md: 16px   lg: 24px   xl: 40px   2xl: 64px

border-radius:
  sm: 6px    md: 10px    lg: 16px    xl: 24px    full: 9999px
```

### 1.4 Elevation (Shadow)

```
level-1: 0 1px 3px rgba(0,0,0,0.4)             -- card nền
level-2: 0 4px 16px rgba(0,0,0,0.5)            -- panel nổi
level-3: 0 8px 32px rgba(0,0,0,0.6), glow      -- verdict card, modal
glow-green: 0 0 20px rgba(34,197,94,0.25)      -- safe result
glow-red:   0 0 20px rgba(239,68,68,0.25)      -- toxic result
```

### 1.5 Motion & Animation

```
transition-fast:   150ms ease-out   -- hover, focus
transition-base:   250ms ease-out   -- expand, collapse
transition-slow:   400ms ease-out   -- page transition, modal

Agent progress pulse: infinite 1.5s ease-in-out (opacity 0.4 → 1 → 0.4)
Score bar: fill 600ms cubic-bezier(0.34, 1.56, 0.64, 1)  -- spring
Gauge sweep: 800ms ease-out
Count-up animation: 1000ms linear  -- số liệu xuất hiện
```

---

## 2. Trang Index (`/`)

### Tổng quan bố cục trang Index

```
┌─────────────────────────────────────────────────────────┐
│  NAVBAR                                           [100%] │
│  Logo | ToxAgent    [About] [GitHub] [Docs]  [☀/☾]     │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  HERO / INPUT SECTION                           [100%]  │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Tagline + Description                          │   │
│  │  ┌──────────────────────────────────┐  [Scan]  │   │
│  │  │  SMILES input (monospace)        │  [CTA]   │   │
│  │  └──────────────────────────────────┘          │   │
│  │  [Validator feedback bar]                       │   │
│  │  [Example molecules: Caffeine | Aspirin | ...]  │   │
│  │  [Advanced options: threshold slider ▼]         │   │
│  └─────────────────────────────────────────────────┘   │
│                                                          │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  AGENT PROGRESS PANEL        (xuất hiện sau submit)     │
│  ┌──────────────────────────────────────────────────┐  │
│  │  [InputValidator ✓]                              │  │
│  │        │                                         │  │
│  │    ┌───┴────────────────────┐                   │  │
│  │    │                        │                   │  │
│  │  [ScreeningAgent ●]   [ResearcherAgent ●]       │  │
│  │    │                        │                   │  │
│  │    └───────────┬────────────┘                   │  │
│  │                │                                 │  │
│  │          [WriterAgent ○]                        │  │
│  │                                                  │  │
│  │  [Streaming log preview]                        │  │
│  └──────────────────────────────────────────────────┘  │
│                                                          │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  QUICK VERDICT CARD          (xuất hiện sau hoàn tất)   │
│  ┌──────────────────────────────────────────────────┐  │
│  │  [Verdict Badge]  [p_toxic Gauge]  [Top Risk]   │  │
│  │                  [→ Xem Báo Cáo Đầy Đủ]         │  │
│  └──────────────────────────────────────────────────┘  │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

### 2.1 Navbar

#### Bố cục

```
┌──────────────────────────────────────────────────────────┐
│ [◈] ToxAgent          [About] [GitHub] [Docs]   [☾/☀]   │
└──────────────────────────────────────────────────────────┘
```

**Container:** `max-width: 1200px`, auto-centered, `padding: 0 24px`
**Height:** `64px` (desktop), `56px` (mobile)
**Position:** `sticky top-0`, `z-index: 100`
**Background:** `--bg` với `backdrop-filter: blur(12px)` + `border-bottom: 1px solid --border`

#### Các thành phần

| Phần tử            | Mô tả chi tiết                                                                                               |
|--------------------|--------------------------------------------------------------------------------------------------------------|
| **Logo mark**      | Icon phân tử 6 cạnh (hexagon) màu `--accent-blue`, SVG inline, `24×24px`. Hover: rotate 30° (200ms).         |
| **App name**       | "ToxAgent" — Inter 700, `18px`, `--text`. Khoảng cách `8px` từ logo.                                        |
| **Nav links**      | "About", "GitHub" (external ↗), "Docs" — Inter 500, `14px`, `--text-muted`. Hover: `--text` + underline.    |
| **Theme toggle**   | Circular icon button `32×32px`, `border-radius: full`. Trạng thái dark → icon mặt trăng ☾; light → mặt trời ☀. Tooltip: "Chuyển sang chế độ sáng/tối". Transition: 300ms rotate + opacity. |
| **Separator**      | `|` divider giữa nav links và theme toggle, màu `--border`.                                                   |

#### Mobile Navbar (`< 768px`)

```
┌─────────────────────────────┐
│ [◈] ToxAgent        [≡] [☾] │
└─────────────────────────────┘
```

Nav links collapse vào hamburger menu `[≡]`. Drawer slide-in từ phải, `280px` width.

---

### 2.2 Hero / Input Section

#### Bố cục tổng thể

```
┌──────────────────────────────────────────────────────────┐
│                                                          │
│   DRUG TOXICITY ANALYSIS                                 │
│   Phân tích độc tính thuốc bằng AI đa tác nhân          │
│   [caption: Nhập chuỗi SMILES hoặc tên phân tử]         │
│                                                          │
│   ┌────────────────────────────────────────┐ ┌───────┐  │
│   │  CC(=O)Oc1ccccc1C(=O)O_               │ │ Scan  │  │
│   └────────────────────────────────────────┘ └───────┘  │
│   ✓ SMILES hợp lệ — Aspirin (MW: 180.16)                │
│                                                          │
│   Ví dụ nhanh:                                           │
│   [☕ Caffeine] [💊 Aspirin] [⚠ Penicillin] [☣ Taxol]   │
│                                                          │
│   ▼ Tùy chọn nâng cao                                    │
│     Ngưỡng độc tính: [══════●══] 0.5                    │
│     Chế độ phân tích: ● Đầy đủ  ○ Nhanh                 │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

**Container:** `max-width: 720px`, centered, `padding: 80px 24px 48px`
**Background:** Subtle gradient từ `--bg` xuống `--surface` (radial tỏa từ trung tâm).

#### 2.2.1 Tagline & Description

- **Tagline:** "Drug Toxicity Analysis" — Inter 700, `42px` (desktop) / `28px` (mobile), `--text`. Gradient text: `from --accent-blue to --accent-green` (CSS `background-clip: text`).
- **Subtitle:** "Phân tích độc tính thuốc bằng hệ thống AI đa tác nhân" — Inter 400, `16px`, `--text-muted`. `margin-top: 8px`.
- **Caption:** "Nhập chuỗi SMILES hoặc tên phân tử để bắt đầu phân tích" — Inter 400, `14px`, `--text-faint`. `margin-top: 4px`.

#### 2.2.2 SMILES Input Field

**Cấu trúc:**

```
[Input wrapper — border + glow on focus]
│
├── [Icon phân tử — left prefix, 20px, --text-faint]
├── [Input text — JetBrains Mono, 14px, --text]
│   placeholder: "CC(=O)Oc1ccccc1C(=O)O  hoặc  aspirin"
└── [Clear button × — right suffix, xuất hiện khi có nội dung]
```

**Thuộc tính input:**
- `type="text"`, `spellcheck="false"`, `autocomplete="off"`, `autocorrect="off"`
- `font-family: JetBrains Mono`, `font-size: 14px`, `letter-spacing: 0.02em`
- Height: `52px` (desktop), `48px` (mobile)
- Background: `--surface-alt`
- Border: `1.5px solid --border`
- Border-radius: `10px`
- Focus state: border → `--accent-blue`, box-shadow: `0 0 0 3px rgba(59,130,246,0.25)`

**CTA Button "Scan" (Phân tích):**
- Đặt bên phải input, liền kề hoặc inline (desktop: inline, mobile: full-width bên dưới)
- Height: `52px`, `padding: 0 28px`
- Background: `--accent-blue`, hover: darken 10%
- Font: Inter 600, `15px`, white
- Border-radius: `10px`
- Icon trái: ⚡ hoặc 🔬 SVG icon `16px`

#### 2.2.3 Máy trạng thái CTA Button

```
┌─────────────────────────────────────────────────────────┐
│  TRẠNG THÁI        │  UI Expression                    │
├────────────────────┼───────────────────────────────────┤
│  IDLE              │  [⚡ Phân tích]  bg: --accent-blue │
│  DISABLED (empty)  │  [⚡ Phân tích]  opacity: 0.4,     │
│                    │  cursor: not-allowed               │
│  DISABLED (invalid)│  [⚡ Phân tích]  opacity: 0.4,     │
│                    │  tooltip: "SMILES không hợp lệ"   │
│  LOADING (valid.)  │  [⟳ Đang kiểm tra...]  spinner    │
│  RUNNING           │  [● Đang phân tích...]  pulse dot  │
│                    │  button disabled, bg: --border     │
│  DONE              │  [✓ Hoàn tất]  bg: --accent-green  │
│                    │  → tự chuyển IDLE sau 2s           │
│  ERROR             │  [✕ Có lỗi xảy ra]  bg: --accent-red│
└─────────────────────────────────────────────────────────┘
```

#### 2.2.4 SMILES Validator Feedback

**Vị trí:** Nằm ngay bên dưới input field, `margin-top: 6px`.
**Hiển thị:** Inline, không phải modal/toast.
**Debounce:** 400ms sau khi người dùng ngừng gõ.

```
Trạng thái VALID:
  ┌─────────────────────────────────────────────────────┐
  │  ✓  SMILES hợp lệ — Aspirin · MW: 180.16 · C9H8O4  │
  └─────────────────────────────────────────────────────┘
  Màu: --accent-green, background: rgba(34,197,94,0.08)
  Font: Inter 13px, border-left: 2px solid --accent-green

Trạng thái INVALID:
  ┌───────────────────────────────────────────────────────┐
  │  ✕  SMILES không hợp lệ — Kiểm tra lại ký tự tại vị  │
  │     trí 7: ký tự 'X' không được nhận dạng.            │
  └───────────────────────────────────────────────────────┘
  Màu: --accent-red, background: rgba(239,68,68,0.08)
  Font: Inter 13px, border-left: 2px solid --accent-red

Trạng thái CHECKING:
  ┌────────────────────────────────┐
  │  ⟳  Đang xác minh SMILES...  │
  └────────────────────────────────┘
  Màu: --text-faint, animation: pulse

Trạng thái EMPTY (không hiện):
  — (ẩn hoàn toàn)
```

**Dữ liệu hiển thị khi valid** (lấy từ PubChem lookup nhanh hoặc RDKit):
- Tên phân tử (nếu nhận ra)
- MW (Molecular Weight)
- Formula phân tử

#### 2.2.5 Example Molecule Buttons

**Bố cục:** Flex row, `gap: 8px`, `flex-wrap: wrap`. Tiêu đề nhỏ: "Ví dụ nhanh:" — Inter 12px, `--text-faint`, uppercase.

```
[☕ Caffeine]  [💊 Aspirin]  [⚗ Ethanol]  [⚠ Benzene]  [💉 Taxol]
```

**Style mỗi button:**
- Background: `--surface-alt`
- Border: `1px solid --border`
- Border-radius: `full` (pill)
- Font: Inter 500, `12px`, `--text-muted`
- Padding: `6px 14px`
- Hover: background `--surface`, border `--accent-blue`, color `--text`
- Click: tự điền SMILES vào input + trigger validation

**Molecule data map:**

| Label      | SMILES                                    |
|------------|-------------------------------------------|
| Caffeine   | `Cn1cnc2c1c(=O)n(c(=O)n2C)C`             |
| Aspirin    | `CC(=O)Oc1ccccc1C(=O)O`                  |
| Ethanol    | `CCO`                                     |
| Benzene    | `c1ccccc1`                               |
| Taxol      | `CC1=C2C(C(=O)C3(C(CC4C(...))...)O3)...` |

#### 2.2.6 Advanced Options (Accordion)

**Trigger:** `▼ Tùy chọn nâng cao` — Inter 13px, `--text-muted`. Click để toggle expand/collapse (250ms ease).

**Nội dung khi mở:**

```
┌──────────────────────────────────────────────────────┐
│  Ngưỡng độc tính (Toxicity Threshold)                │
│  0.0 ────────────●──────── 1.0                       │
│                  0.5                                  │
│                                                       │
│  Chế độ phân tích:                                   │
│  ● Full (ScreeningAgent + ResearcherAgent + Writer)  │
│  ○ Quick (ScreeningAgent only)                        │
│                                                       │
│  [?] Giải thích ngưỡng                               │
└──────────────────────────────────────────────────────┘
```

**Threshold Slider:**
- Range: 0.0 – 1.0, step: 0.05, default: 0.5
- Track background: gradient từ `--accent-green` (trái) → `--accent-yellow` (giữa) → `--accent-red` (phải)
- Thumb: `16px` circle, `--accent-blue`, shadow level-2
- Giá trị hiển thị: JetBrains Mono `14px` bên phải

---

### 2.3 Agent Progress Panel

**Điều kiện hiển thị:** Xuất hiện với `slide-down + fade-in` (400ms) ngay sau khi user nhấn "Phân tích".

#### Bố cục tổng thể

```
┌────────────────────────────────────────────────────────┐
│  Pipeline Phân Tích                         [✕ hủy]   │
│                                                        │
│  ●──── InputValidator                    ✓ 0.3s       │
│         │                                              │
│    ┌────┴────────────────────────┐                    │
│    │                             │                    │
│  ●  ScreeningAgent          ●  ResearcherAgent        │
│  └─ [███████░░░] 70%          └─ [█████░░░░] 50%      │
│    [GNN forward pass…]           [Fetching PubMed…]   │
│    │                             │                    │
│    └────────────┬────────────────┘                    │
│                 │                                      │
│               ○  WriterAgent                          │
│               [Đang chờ…]                             │
│                                                        │
│ ──────────────────────────────────────────────────    │
│  [LOG]  14:23:01  InputValidator: SMILES validated    │
│  [LOG]  14:23:01  ScreeningAgent: starting GNN...     │
│  [LOG]  14:23:02  ResearcherAgent: querying PubChem.. │
└────────────────────────────────────────────────────────┘
```

**Container style:**
- Background: `--surface`, border: `1px solid --border`, border-radius: `16px`
- Padding: `24px`
- `max-width: 720px`, centered
- Margin: `24px auto`
- Level-2 shadow

#### 2.3.1 Agent Node Component

Mỗi tác nhân (agent) được biểu diễn bằng một **Agent Node**, gồm:

```
[Trạng thái icon] [Tên agent]  [Thời gian / badge trạng thái]
[Progress bar — chỉ khi running]
[Dòng log ngắn — chỉ khi running]
```

**Bảng trạng thái Agent Node:**

| Trạng thái | Icon           | Màu icon    | Progress bar | Background node           |
|------------|----------------|-------------|--------------|---------------------------|
| `pending`  | ○ (vòng rỗng)  | `--border`  | Ẩn           | transparent               |
| `running`  | ● (pulse dot)  | `--accent-blue` | Hiển thị, animated | `--accent-blue-muted` |
| `done`     | ✓ (checkmark)  | `--accent-green` | Ẩn (filled 100% → fade) | transparent  |
| `error`    | ✕ (x mark)     | `--accent-red` | Ẩn           | `rgba(239,68,68,0.08)` |

**Pulse dot animation (trạng thái running):**
```css
@keyframes pulse-dot {
  0%, 100% { opacity: 0.4; transform: scale(0.8); }
  50%       { opacity: 1;   transform: scale(1.2); }
}
animation: pulse-dot 1.5s ease-in-out infinite;
```

**Progress bar:**
- Width: `100%` của node container
- Height: `4px`
- Background: `--border`
- Fill: `--accent-blue`, animated với `transition: width 300ms ease`
- Progress được cập nhật từ streaming events của backend

#### 2.3.2 Sơ đồ Pipeline Flow

Sử dụng **SVG connector lines** để nối các agent nodes:

```
InputValidator (top, full width)
      │  (vertical line, --border)
      ├──────────────┐
      │              │
ScreeningAgent   ResearcherAgent
(parallel, 50%   (parallel, 50%
 width each)      width each)
      │              │
      └──────────────┘
             │
        WriterAgent
```

**Connector lines:**
- `stroke: --border`, `stroke-width: 1.5px`, `stroke-dasharray: 4 4` khi target agent `pending`
- `stroke: --accent-blue` (solid) khi target agent `running` hoặc `done`
- Animation: dash draw từ source → target khi agent chuyển sang `running`

**Parallel indicator:**
- Giữa InputValidator và 2 agent song song có label nhỏ: "PARALLEL" — Inter 10px, uppercase, `--text-faint`, letter-spacing: 0.1em

#### 2.3.3 Streaming Preview

**Vị trí:** Phía dưới pipeline diagram, cách `16px`.
**Style:** Terminal-like, monospace.

```
┌──────────────────────────────────────────────────────┐
│  [timestamp] [agent] message                         │
│  ────────────────────────────────────────────────    │
│  14:23:01  InputValidator  ✓ SMILES validated        │
│  14:23:02  ScreeningAgent  ⟳ Loading GNN model...    │
│  14:23:02  ResearcherAgent ⟳ Querying PubChem CID... │
│  14:23:04  ScreeningAgent  ✓ GNN forward pass done   │
│  14:23:05  ResearcherAgent ⟳ Fetching 5 papers...    │
└──────────────────────────────────────────────────────┘
```

- Background: `--bg` (dark nền đen)
- Font: JetBrains Mono `12px`, `--text-muted`
- Max height: `120px`, overflow-y: auto, scroll to bottom tự động
- Timestamp: `--text-faint`; agent name: `--accent-blue` (running) / `--accent-green` (done); message: `--text-muted`
- Mỗi dòng log `fade-in` từ dưới lên (100ms)

#### 2.3.4 Kết quả trung gian streaming (Intermediate Results)

Khi từng agent hoàn thành, hiện **mini-preview card** bên trong Agent Node:

**ScreeningAgent done preview:**
```
┌─────────────────────────────────────┐
│  ✓ ScreeningAgent  · 3.2s           │
│  p_toxic: 0.73  · Label: TOXIC      │
│  Top risk: SR-HSE                   │
└─────────────────────────────────────┘
```

**ResearcherAgent done preview:**
```
┌─────────────────────────────────────┐
│  ✓ ResearcherAgent · 5.1s           │
│  5 papers found · CID: 2244         │
│  10 bioassay records                │
└─────────────────────────────────────┘
```

- Style: `--surface-alt`, border `1px solid --border`, border-radius `8px`, padding `10px 12px`
- Số liệu: JetBrains Mono `13px`, bold
- `slide-down` animation 200ms khi xuất hiện

---

### 2.4 Quick Verdict Card

**Điều kiện hiển thị:** Xuất hiện sau khi **WriterAgent** hoàn tất. Animation: `scale(0.95) → scale(1)` + `fade-in` (400ms). Agent Progress Panel vẫn giữ nguyên phía trên.

#### Bố cục

```
┌────────────────────────────────────────────────────────┐
│  KẾT QUẢ PHÂN TÍCH NHANH                              │
│                                                        │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────┐  │
│  │  VERDICT     │  │  p_toxic     │  │ Top Risk   │  │
│  │              │  │              │  │            │  │
│  │  ⚠ TOXIC    │  │   [Gauge]    │  │ SR-HSE     │  │
│  │              │  │    0.73      │  │ score:0.89 │  │
│  └──────────────┘  └──────────────┘  └────────────┘  │
│                                                        │
│  [───────────────── Xem Báo Cáo Đầy Đủ ─────────────→]│
└────────────────────────────────────────────────────────┘
```

**Container:**
- Background: `--surface`
- Border: `1.5px solid` — màu theo verdict (`--accent-red` / `--accent-green` / `--accent-yellow`)
- Box-shadow: `glow-red` hoặc `glow-green` (theo verdict)
- Border-radius: `16px`, padding: `28px`
- `max-width: 720px`, centered

#### 2.4.1 Verdict Badge (Card 1/3)

```
┌──────────────────────┐
│   ⚠  TOXIC          │    ← khi p_toxic >= 0.7
│   Độ tin cậy: 89%   │
└──────────────────────┘

┌──────────────────────┐
│   ✓  NON-TOXIC      │    ← khi p_toxic < 0.3
│   Độ tin cậy: 94%   │
└──────────────────────┘

┌──────────────────────┐
│   ?  UNCERTAIN      │    ← khi 0.3 <= p_toxic < 0.7
│   Độ tin cậy: 72%   │
└──────────────────────┘
```

**Badge styling:**
- Font: Inter 700, `20px`, uppercase
- Icon: SVG, `24px`, bên trái
- Màu text + background:
  - TOXIC: text `--accent-red`, bg `rgba(239,68,68,0.12)`
  - NON-TOXIC: text `--accent-green`, bg `rgba(34,197,94,0.12)`
  - UNCERTAIN: text `--accent-yellow`, bg `rgba(245,158,11,0.12)`
- Confidence: Inter 400, `13px`, `--text-muted`, `margin-top: 4px`
- Card padding: `20px`, border-radius: `12px`
- Border: `1px solid` (màu tương ứng, opacity 0.3)

#### 2.4.2 p_toxic Gauge (Card 2/3)

Dạng **Semicircle Gauge** (nửa vòng tròn):

```
        0.73
    ___/‾‾‾\___
   /   TOXIC   \
  ●─────────────●
 0.0            1.0
   NON  WARN  TOXIC
```

**Đặc tả gauge:**
- SVG semicircle, `120px` width, `60px` height
- Track: `stroke: --border`, `stroke-width: 8px`
- Fill arc: gradient dọc theo arc: `--accent-green` → `--accent-yellow` → `--accent-red`
- Needle/indicator: tam giác nhỏ tại vị trí p_toxic, màu `--text`
- Giá trị: JetBrains Mono `24px`, bold, căn giữa bên dưới arc — màu theo ngưỡng
- Animation: sweep từ 0 → giá trị thực, 800ms `ease-out`
- Label: "p_toxic", Inter 11px, `--text-faint`, uppercase

#### 2.4.3 Top Risk Task (Card 3/3)

```
┌─────────────────────┐
│  Rủi ro cao nhất    │
│  ─────────────────  │
│  SR-HSE             │
│  ████████░ 0.89     │
│  (Stress Response - │
│   Heat Shock Elem.) │
└─────────────────────┘
```

- Tên task: Inter 600, `16px`, `--text`
- Mini bar: width tỉ lệ với score, màu `--accent-red`
- Score: JetBrains Mono `14px`
- Description: Inter 12px, `--text-muted`

#### 2.4.4 CTA "Xem Báo Cáo Đầy Đủ"

- **Bố cục:** Full-width button, bên dưới 3 cards, `margin-top: 20px`
- **Style:** 
  - Background: gradient `--accent-blue` → `oklch(55% 0.2 200)`
  - Height: `52px`, border-radius: `10px`
  - Font: Inter 600, `16px`, white
  - Icon: `→` bên phải, `translate-x: 4px` khi hover (200ms)
- **Interaction:** Click → navigate to `/report` (với molecule data trong state/query params)

---

## 3. Trang Report (`/report`)

### Tổng quan bố cục trang Report

```
┌──────────────────────────────────────────────────────────┐
│  NAVBAR (sticky)                                         │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  REPORT HEADER                                  [100%]  │
│  [Tên phân tử | SMILES | Verdict badge | Timestamp]     │
│  [← Phân tích mới] [↓ Tải PDF]                          │
│                                                          │
├─────────────────┬────────────────────────────────────────┤
│                 │                                        │
│   SIDEBAR       │   MAIN CONTENT                        │
│   [Navigation   │                                        │
│    nội bộ]      │  §1 Clinical Toxicity                 │
│                 │  §2 Mechanism Profiling               │
│   [280px]       │  §3 Structural Explanation            │
│                 │  §4 Literature Context                │
│                 │  §5 AI Recommendations                │
│                 │                                        │
└─────────────────┴────────────────────────────────────────┘
```

**Layout chính:** `display: grid`, `grid-template-columns: 280px 1fr` (desktop).
**Main content:** `max-width: 860px`, padding `40px`.
**Sidebar:** `position: sticky`, `top: 64px` (dưới navbar), height: `calc(100vh - 64px)`, overflow-y: auto.

---

### 3.1 Report Header

```
┌──────────────────────────────────────────────────────────┐
│  [← Phân tích mới]                    [↓ Tải PDF]       │
│                                                          │
│  Aspirin                              ⚠ TOXIC           │
│  Acetylsalicylic Acid                 Confidence: 89%   │
│  CC(=O)Oc1ccccc1C(=O)O                                  │
│  CID: 2244 · MW: 180.16 · C9H8O4                        │
│                                                          │
│  31/03/2026, 14:23:07 ICT                                │
└──────────────────────────────────────────────────────────┘
```

**Thành phần:**

| Phần tử              | Style                                                            |
|----------------------|------------------------------------------------------------------|
| **Tên phân tử**      | Inter 700, `32px`, `--text`                                      |
| **Tên IUPAC/alt**    | Inter 400, `16px`, `--text-muted`. Italics nếu là tên khoa học. |
| **SMILES string**    | JetBrains Mono `13px`, `--text-faint`, `word-break: break-all`  |
| **Metadata inline**  | CID · MW · Formula — Inter 13px, `--text-muted`, dấu `·` phân cách |
| **Verdict badge**    | Same style như Quick Verdict Card §2.4.1, size lớn hơn           |
| **Timestamp**        | Inter 12px, `--text-faint`                                       |
| **← Back button**    | Ghost button, Inter 500 `14px`, `--text-muted`                   |
| **↓ Tải PDF**        | Outlined button, `--accent-blue`, border `1px solid`, icon download |

**Divider:** `border-bottom: 1px solid --border`, `padding-bottom: 32px`, `margin-bottom: 40px`.

---

### 3.2 Sidebar Navigation

```
┌──────────────────────┐
│  Nội dung báo cáo    │
│  ────────────────    │
│  ● Clinical Toxicity  │  ← active (highlight --accent-blue)
│  ○ Mechanism          │
│  ○ Structural         │
│  ○ Literature         │
│  ○ AI Recommendations │
│                       │
│  ─────────────────    │
│  THÔNG TIN NHANH     │
│  p_toxic  [gauge sm] │
│  Label    TOXIC       │
│  MW       180.16      │
│  Formula  C9H8O4      │
└──────────────────────┘
```

- Scroll-spy: tự update active item khi scroll qua section
- Smooth scroll khi click
- **Quick info panel:** mini gauge + key stats, luôn hiển thị
- Mobile: sidebar ẩn, thay bằng horizontal scroll tab bar ở đầu content

---

### 3.3 Section 1: Clinical Toxicity

**Dữ liệu từ ScreeningAgent:**
- `p_toxic: float (0-1)`
- `label: "TOXIC" | "NON-TOXIC"`
- `confidence: float (0-1)`

#### Bố cục

```
§1 Clinical Toxicity
────────────────────────────────────────────

┌─────────────────┐  ┌──────────────────────┐
│                 │  │                      │
│  [GAUGE lớn]   │  │  Toxicity Label       │
│    0.73         │  │  ┌────────────────┐  │
│   p_toxic       │  │  │  ⚠ TOXIC      │  │
│                 │  │  └────────────────┘  │
│  [Color zone]   │  │                      │
│  ■ Non-toxic    │  │  Confidence          │
│  ■ Warning      │  │  ████████████░ 89%   │
│  ■ Toxic        │  │                      │
└─────────────────┘  │  Threshold used      │
                     │  0.50                │
                     └──────────────────────┘
```

**Grid:** `display: grid`, `grid-template-columns: 1fr 2fr`, `gap: 24px`.

**Gauge lớn:**
- Dạng vòng tròn đầy đủ (full circle donut gauge), `200px × 200px`
- Phần tô màu: arc từ 0 đến p_toxic × 360°
- Color: `--accent-green` (0–0.3) → `--accent-yellow` (0.3–0.7) → `--accent-red` (0.7–1.0)
- Giá trị ở trung tâm: JetBrains Mono `36px`, bold, màu theo ngưỡng
- Label "p_toxic" bên dưới giá trị: Inter 12px, uppercase, `--text-faint`
- Animation: sweep 800ms

**Confidence Bar:**
- Label: "Độ tin cậy" + giá trị `%` — Inter 14px
- Progress bar: height `8px`, border-radius `full`
- Màu: `--accent-blue`
- Width: `confidence × 100%`, animated 600ms

**Interpretation text:**
- 1–2 câu tóm tắt tự động: "Phân tử này có xác suất độc tính cao (73%). Phân loại: **TOXIC** với độ tin cậy 89%."
- Inter 14px, `--text-muted`, italic

---

### 3.4 Section 2: Mechanism Profiling

**Dữ liệu từ ScreeningAgent:**
- `task_scores: {[taskName: string]: float}` — 12 task Tox21
- `active_tasks: string[]` — các task có score > threshold
- `highest_risk_task: string`

**12 Tox21 Tasks:**
```
NR-AR, NR-AR-LBD, NR-AhR, NR-Aromatase, NR-ER, NR-ER-LBD,
NR-PPAR-gamma, SR-ARE, SR-ATAD5, SR-HSE, SR-MMP, SR-p53
```

#### Bố cục

```
§2 Mechanism Profiling
────────────────────────────────────────────

  Highest Risk: SR-HSE (0.89)  Active: 4/12 tasks

  ┌──────────────────────────────────────────────────────┐
  │  HORIZONTAL BAR CHART — 12 Tox21 Tasks               │
  │                                                       │
  │  NR-AR       ██░░░░░░░░  0.21  [non-toxic]           │
  │  NR-AR-LBD   ████░░░░░░  0.42  [warning]             │
  │  NR-AhR      ██████░░░░  0.63  [warning]             │
  │  NR-Aromatase██░░░░░░░░  0.18                        │
  │  NR-ER       ███░░░░░░░  0.31                        │
  │  NR-ER-LBD   ██░░░░░░░░  0.19                        │
  │  NR-PPAR-g   ██░░░░░░░░  0.24                        │
  │  SR-ARE      █████████░  0.81  ★ ACTIVE              │
  │  SR-ATAD5    ███░░░░░░░  0.33                        │
  │  SR-HSE      ██████████  0.89  ★ HIGHEST RISK        │
  │  SR-MMP      ███████░░░  0.71  ★ ACTIVE              │
  │  SR-p53      ████████░░  0.78  ★ ACTIVE              │
  └──────────────────────────────────────────────────────┘

  [Hiện dạng Heatmap]  ←  Toggle button
```

**Horizontal Bar Chart:**
- `display: flex`, `flex-direction: column`, `gap: 6px`
- Mỗi hàng = 1 task:
  ```
  [Task name — 100px fixed width]  [Bar fill]  [Score value]  [Badge]
  ```
- Bar height: `20px`, border-radius `4px`
- Bar màu: gradient `--accent-green → --accent-yellow → --accent-red` theo score
- Active task badge: "★ ACTIVE" — Inter 10px, uppercase, `--accent-yellow`
- Highest risk badge: "★ HIGHEST RISK" — Inter 10px, uppercase, `--accent-red`, bold
- Animate bars: fill từ 0 → giá trị, 600ms spring, staggered 50ms mỗi bar
- Sorted: cao → thấp

**Heatmap View (Toggle):**
- Grid 3×4 (hoặc 4×3) cells
- Mỗi cell = 1 task, màu nền theo score (color scale `--accent-green → --accent-red`)
- Label task + số ở trung tâm cell
- Tooltip khi hover: tên đầy đủ + score + active/inactive

**Active Tasks Summary:**
```
Active tasks (score > 0.5):  4/12

  [SR-HSE]  [SR-ARE]  [SR-p53]  [SR-MMP]
```
- Pill badges, màu `--accent-red` cho mỗi active task
- "Highest risk:" prefix với badge đặc biệt

---

### 3.5 Section 3: Structural Explanation

**Dữ liệu từ ScreeningAgent:**
- `heatmap_base64: string` — PNG heatmap của phân tử
- `top_atoms: [{atom_idx, contribution, symbol}]`
- `top_bonds: [{bond_idx, contribution, atom1, atom2}]`

#### Bố cục

```
§3 Structural Explanation
────────────────────────────────────────────

  ┌─────────────────────────────────────┐
  │                                     │
  │   MOLECULAR HEATMAP                 │
  │   [hình ảnh base64 — 400×300px]     │
  │                                     │
  │   Xanh = ít nguy hiểm               │
  │   Đỏ   = đóng góp độc tính cao      │
  │   [🔍 Phóng to] [↓ Tải hình]        │
  └─────────────────────────────────────┘

  ┌──────────────────────┐  ┌────────────────────────┐
  │  Top Atoms           │  │  Top Bonds             │
  │  ─────────────────   │  │  ──────────────────    │
  │  Idx  Sym  Score     │  │  Bond  Atoms   Score   │
  │   3   O    0.82 ●   │  │   0    O-C     0.79 ●  │
  │   7   N    0.71 ●   │  │   4    C=O     0.65 ●  │
  │   2   C    0.54 ○   │  │   11   C-N     0.48 ○  │
  │  ...                 │  │  ...                   │
  └──────────────────────┘  └────────────────────────┘
```

**Molecular Heatmap image:**
- `<img>` tag với `src="data:image/png;base64,{heatmap_base64}"`
- Max-width: `100%`, border-radius: `12px`
- Border: `1px solid --border`
- Caption: "Màu đỏ biểu thị nguyên tử/liên kết đóng góp nhiều nhất vào độc tính"
- Lightbox: click → overlay full-screen với close button
- Download button: tải heatmap_base64 thành file PNG

**Top Atoms Table:**
- Bố cục: `display: table` hoặc `<table>` styled
- Columns: Atom Index | Symbol | Contribution Score | Visual indicator
- Visual indicator: màu chấm tròn `●` theo score (đỏ nếu cao, vàng nếu trung bình, xanh nếu thấp)
- Score: JetBrains Mono `13px`
- Rows: tối đa 5, "Hiện thêm" nếu có hơn

**Top Bonds Table:**
- Tương tự Top Atoms
- Columns: Bond Index | Atoms (e.g., "C2-O3") | Contribution Score | Visual indicator
- Highlight bond có score cao nhất: background `rgba(239,68,68,0.08)`

---

### 3.6 Section 4: Literature Context

**Dữ liệu từ ResearcherAgent:**
- `pubchem_info: {cid, iupac_name, molecular_formula, molecular_weight, synonyms, description}`
- `literature_papers: [{pmid, title, authors, journal, year, abstract_snippet, url}]` — 5 papers
- `bioassay_data: [{aid, name, activity_outcome, target}]`

#### Bố cục

```
§4 Literature Context
────────────────────────────────────────────

  PubChem Compound Info
  ┌──────────────────────────────────────────────────────┐
  │  CID: 2244    Aspirin                                │
  │  Formula: C9H8O4  ·  MW: 180.16 g/mol               │
  │  IUPAC: 2-(acetyloxy)benzoic acid                    │
  │  Synonyms: Acetylsalicylic acid, Aspirin, ...        │
  │  [Xem trên PubChem ↗]                                │
  └──────────────────────────────────────────────────────┘

  Nghiên cứu liên quan (5 bài báo)
  ┌───────────────────────────────────────────────────────┐
  │  [1] Aspirin toxicity in hepatocytes: a mechanistic   │
  │      study. (2023)                                    │
  │      Smith J., et al. · Journal of Toxicology         │
  │      "Aspirin at high concentrations..."              │
  │      [PMID: 37291847] [Đọc bài báo ↗]                │
  ├───────────────────────────────────────────────────────┤
  │  [2] ...                                              │
  └───────────────────────────────────────────────────────┘

  Bioassay Data
  ┌──────────┬──────────────────────┬──────────┬──────────┐
  │ AID      │ Tên assay            │ Kết quả  │ Target   │
  ├──────────┼──────────────────────┼──────────┼──────────┤
  │ 588342   │ Tox21 HSE pathway    │ ACTIVE ● │ SR-HSE   │
  │ 743219   │ Tox21 MMP pathway    │ ACTIVE ● │ SR-MMP   │
  │ 602387   │ NR-AR nuclear rec.   │ INACTIVE ○│ NR-AR   │
  └──────────┴──────────────────────┴──────────┴──────────┘
```

**PubChem Info Card:**
- Style: `--surface`, border `1px solid --border`, border-radius `12px`, padding `20px`
- Logo PubChem: nhỏ, bên cạnh tên
- Link "Xem trên PubChem ↗": `--accent-blue`, opens new tab

**Literature Paper Cards:**
- Mỗi paper = 1 card trong danh sách (stack dọc, không phải grid)
- Gồm: số thứ tự, tiêu đề bài báo (bold, `--text`), năm, tác giả, tên journal (italic), đoạn trích abstract
- PMID badge: JetBrains Mono `11px`, `--surface-alt`
- "Đọc bài báo ↗" link
- Hover: background `--surface-alt`, transition 150ms

**Bioassay Table:**
- `<table>` responsive
- ACTIVE: badge đỏ `●`; INACTIVE: badge xám `○`
- Kết quả highlight: ACTIVE rows có background `rgba(239,68,68,0.05)`
- Mobile: scroll ngang

---

### 3.7 Section 5: AI Recommendations

**Dữ liệu từ WriterAgent:**
- `final_report: {summary, clinical_interpretation, mechanism_analysis, literature_context, recommendations}`

#### Bố cục

```
§5 AI Recommendations
────────────────────────────────────────────

  ┌──────────────────────────────────────────────────────┐
  │  AI · WriterAgent  ·  Tổng hợp báo cáo              │
  │  ─────────────────────────────────────────────────   │
  │                                                      │
  │  TÓM TẮT ĐIỀU HÀNH                                  │
  │  Phân tử [Aspirin] thể hiện xác suất độc tính cao   │
  │  (p_toxic = 0.73)...                                 │
  │                                                      │
  │  ĐÁNH GIÁ LÂM SÀNG                                  │
  │  Dựa trên điểm số SR-HSE = 0.89, phân tử này...     │
  │                                                      │
  │  PHÂN TÍCH CƠ CHẾ                                    │
  │  Cơ chế độc tính chủ yếu qua con đường...           │
  │                                                      │
  │  BỐI CẢNH VĂN HỌC                                   │
  │  5 nghiên cứu được phân tích cho thấy...             │
  │                                                      │
  │  KHUYẾN NGHỊ                                         │
  │  ⚠ Không dùng trong liều cao cho bệnh nhân...        │
  │  ✓ Tham khảo thêm nghiên cứu SR-HSE...               │
  └──────────────────────────────────────────────────────┘

  [Tạo lại báo cáo ↺]    [Sao chép văn bản □]
```

**Container:**
- Background: gradient nhẹ từ `--surface` sang `--bg`
- Border-left: `3px solid --accent-blue`
- Border-radius: `0 12px 12px 0`
- Padding: `28px 32px`

**AI source indicator:**
- Nhỏ, bên trên: "AI · WriterAgent" — Inter 11px, uppercase, `--text-faint` + icon robot `12px`
- Nhấn mạnh đây là nội dung AI-generated

**Sub-sections trong WriterAgent output:**
Mỗi trong 5 phần `{summary, clinical_interpretation, mechanism_analysis, literature_context, recommendations}` có:
- Label tiêu đề: Inter 700, `13px`, uppercase, `--text-muted`, `letter-spacing: 0.08em`
- Body text: Inter 400, `15px`, `--text`, `line-height: 1.7`
- Phân cách: `margin-bottom: 20px`

**Recommendations đặc biệt:**
- Bullet list với icon:
  - `⚠` cho caution items — màu `--accent-yellow`
  - `✓` cho positive items — màu `--accent-green`
  - `✕` cho contraindications — màu `--accent-red`

**Tái tạo báo cáo:**
- Ghost button `[↺ Tạo lại báo cáo]` — yêu cầu WriterAgent chạy lại
- Icon button `[□ Sao chép]` — copy text to clipboard, feedback "✓ Đã sao chép" (2s)

---

## 4. Điều Hướng Giữa Các Trang

### 4.1 Flow chính

```
[Index /]  →  [Nhập SMILES]  →  [Chạy pipeline]  →  [Quick Verdict]  →  [/report]
                                                              ↑
                                                    [← Phân tích mới]
                                                       (từ /report)
```

### 4.2 State Persistence

Khi chuyển từ `/` → `/report`:
- Dữ liệu phân tích được lưu vào **sessionStorage** hoặc truyền qua **URL query params** (ví dụ: `/report?session=abc123`)
- Không yêu cầu re-run pipeline
- Nếu mở `/report` trực tiếp không có data → redirect về `/` với toast: "Không tìm thấy dữ liệu phân tích. Hãy chạy phân tích mới."

### 4.3 Navigation components

**"← Phân tích mới" (từ Report về Index):**
- Ghost button, trái trên cùng của Report Header
- Không clear session data ngay — có confirm dialog nếu phân tích vừa xong

**"→ Xem Báo Cáo Đầy Đủ" (từ Quick Verdict Card):**
- Animated button, full-width, slide từ dưới lên khi xuất hiện (400ms)
- Click: smooth page transition — fade-out current page (200ms) → navigate → fade-in report (200ms)

**Sidebar Scroll Navigation (trong /report):**
- In-page anchors: `#clinical`, `#mechanism`, `#structural`, `#literature`, `#recommendations`
- Scroll-spy: highlight active section trong sidebar khi scroll
- Smooth scroll `behavior: smooth`

---

## 5. Responsive

### 5.1 Breakpoints

```
mobile:  < 640px   (sm)
tablet:  640–1023px (md)
desktop: ≥ 1024px  (lg+)
```

### 5.2 Index Page — Responsive

| Component            | Desktop                   | Tablet                  | Mobile                     |
|----------------------|---------------------------|-------------------------|----------------------------|
| Navbar               | Logo + links + toggle     | Logo + 2 links + toggle | Logo + hamburger + toggle  |
| Hero tagline         | 42px                       | 32px                    | 24px                       |
| Input + CTA          | Inline (input + button)   | Inline                  | Stacked (input trên, button dưới) |
| Example buttons      | 1 row flex                | 2 rows                  | Scroll ngang               |
| Agent Pipeline       | 3-column horizontal       | 3-column (compact)      | Stacked vertical           |
| Agent nodes (parallel) | Side by side (50% each) | Side by side (compact)  | Stacked (full width)       |
| Quick Verdict        | 3-column grid             | 3-column (compact)      | Stacked vertical (1 column) |
| Gauge (verdict)      | 120px gauge               | 100px gauge             | 90px gauge                 |

### 5.3 Report Page — Responsive

| Component            | Desktop                   | Tablet                  | Mobile                     |
|----------------------|---------------------------|-------------------------|----------------------------|
| Overall layout       | Sidebar (280px) + content | Sidebar collapsed       | No sidebar                 |
| Sidebar              | Sticky left panel         | Hidden (toggle button)  | Horizontal tab bar at top  |
| Report Header        | 2-column (info + verdict) | 2-column (compact)      | Stacked                    |
| §1 Clinical          | 2-column grid             | 2-column                | Stacked                    |
| §2 Bar chart         | Full width                | Full width              | Scroll ngang nếu cần       |
| §2 Heatmap toggle    | Inline toggle             | Inline toggle           | Full width toggle          |
| §3 Heatmap           | 60% width                 | 100%                    | 100%                       |
| §3 Tables            | 2-column side by side     | 2-column                | Stacked, 1 column          |
| §4 Paper cards       | Full width list           | Full width              | Full width                 |
| §4 Bioassay table    | Full table                | Full table              | Scroll ngang               |
| §5 WriterAgent       | Left-bordered panel       | Same                    | No left border (top border instead) |
| Download PDF button  | Top-right header          | Top-right header        | Fixed bottom action bar    |

### 5.4 Mobile cụ thể

**Horizontal tab navigation (thay sidebar trên mobile):**
```
[Clinical] [Mechanism] [Structural] [Literature] [AI Rec.]
```
- Scroll ngang, active tab underline màu `--accent-blue`
- Sticky dưới Report Header khi scroll

**Fixed bottom action bar (mobile):**
```
┌──────────────────────────────────────────────────┐
│  [← Mới]                [↓ PDF]  [↑ Đầu trang]   │
└──────────────────────────────────────────────────┘
```

---

## 6. Export / Download

### 6.1 Nút Download PDF

**Vị trí:**
- Desktop/Tablet: Button outlined tại Report Header, góc trên phải
- Mobile: Fixed bottom action bar (xem §5.4)

**Style:**
```
[↓ Tải báo cáo PDF]
```
- Outlined button: border `1px solid --accent-blue`, text `--accent-blue`, bg transparent
- Hover: bg `rgba(59,130,246,0.08)`
- Font: Inter 500, `14px`
- Icon: download arrow `16px`

### 6.2 PDF Report Content

Nội dung file PDF xuất ra gồm:
1. Cover page: tên app, tên phân tử, SMILES, verdict, timestamp
2. §1 Clinical Toxicity (gauge dạng static image)
3. §2 Mechanism Profiling (bar chart dạng static image)
4. §3 Structural Explanation (heatmap image + tables)
5. §4 Literature Context (danh sách papers + bioassay table)
6. §5 AI Recommendations (full text)
7. Footer: disclaimer AI-generated content, URL nguồn

**Loading state khi export:**
```
[⟳ Đang tạo PDF...]  (spinner, button disabled)
→ [✓ Đã tải xuống]   (2s, rồi về trạng thái ban đầu)
```

### 6.3 Copy Actions

- Nút sao chép SMILES (icon □ nhỏ bên cạnh SMILES string trong header)
- Nút sao chép AI Recommendations text (§3.7)
- Toast feedback: bottom-center, "✓ Đã sao chép vào clipboard" — 2s auto-dismiss

---

## 7. Trạng Thái Trống, Lỗi, Loading

### 7.1 Empty States

**Index — chưa nhập gì:**
- Input: placeholder mờ, validator feedback ẩn
- Agent Panel: ẩn hoàn toàn
- Quick Verdict: ẩn hoàn toàn

**Report — không có session data:**
```
┌─────────────────────────────────────────────┐
│                                             │
│   [◈]                                       │
│   Không tìm thấy dữ liệu phân tích         │
│                                             │
│   Hãy chạy phân tích mới từ trang chủ.     │
│                                             │
│   [→ Về trang chủ]                          │
│                                             │
└─────────────────────────────────────────────┘
```

### 7.2 Error States

**API Error (backend không phản hồi):**
```
Agent Node style: trạng thái "error" (✕ đỏ)
+
┌────────────────────────────────────────────┐
│  ✕  ScreeningAgent — Lỗi kết nối           │
│  Could not reach POST /analyze              │
│  [Thử lại ↺]                              │
└────────────────────────────────────────────┘
```
- "Thử lại" → retry toàn bộ pipeline từ agent bị lỗi
- Toast: `position: fixed; bottom: 24px; right: 24px` — màu `--accent-red`

**SMILES Invalid Error:**
- Feedback inline (§2.2.4) — không dùng toast
- CTA button ở trạng thái DISABLED

**Timeout Error (> 30s không có response):**
- Toast cảnh báo sau 15s: "Phân tích đang mất nhiều thời gian hơn dự kiến..."
- Sau 30s: error state với nút "Hủy và thử lại"

### 7.3 Loading / Skeleton States

Khi `ResearcherAgent` chạy, Section 4 (Literature) hiển thị skeleton:
```
┌──────────────────────────────────────┐
│  [░░░░░░░░░░░░░░░░░] ← skeleton bar │
│  [░░░░░░░░]                          │
└──────────────────────────────────────┘
```

Skeleton style:
```css
background: linear-gradient(90deg, --surface 0%, --surface-alt 50%, --surface 100%);
background-size: 200% 100%;
animation: shimmer 1.5s infinite;
```

Khi `ScreeningAgent` chạy nhưng chưa xong, các section §1–§3 trong `/report` đều hiện skeleton loader.

---

## Phụ Lục A: Component Reference Summary

| Component             | Trang    | Dữ liệu nguồn           | Loại visualisation     |
|-----------------------|----------|-------------------------|------------------------|
| SMILES Input          | Index    | User input               | Text field + validator |
| Agent Pipeline        | Index    | WebSocket/SSE events     | Flow diagram + nodes   |
| Streaming Log         | Index    | SSE log stream           | Terminal output        |
| p_toxic Gauge (sm)    | Index    | `p_toxic`                | Semicircle gauge       |
| p_toxic Gauge (lg)    | Report   | `p_toxic`                | Full circle gauge      |
| Verdict Badge         | Both     | `label`, `p_toxic`       | Color badge            |
| Confidence Bar        | Report   | `confidence`             | Progress bar           |
| Tox21 Bar Chart       | Report   | `task_scores[12]`        | Horizontal bars        |
| Tox21 Heatmap         | Report   | `task_scores[12]`        | Color grid             |
| Active Tasks Chips    | Report   | `active_tasks`           | Pill badges            |
| Molecular Heatmap     | Report   | `heatmap_base64`         | Image (base64)         |
| Top Atoms Table       | Report   | `top_atoms`              | Table + color dots     |
| Top Bonds Table       | Report   | `top_bonds`              | Table + color dots     |
| PubChem Card          | Report   | `pubchem_info`           | Info card              |
| Paper Cards           | Report   | `literature_papers[5]`   | List cards             |
| Bioassay Table        | Report   | `bioassay_data`          | Data table             |
| WriterAgent Narrative | Report   | `final_report{5 sections}`| Structured text       |
| PDF Download          | Report   | All report data          | Export button          |

---

## Phụ Lục B: Color Encoding Quick Reference

```
p_toxic value → màu sắc:
  0.00 – 0.30  ████  --accent-green  (#22c55e)  "NON-TOXIC"
  0.30 – 0.70  ████  --accent-yellow (#f59e0b)  "UNCERTAIN"
  0.70 – 1.00  ████  --accent-red    (#ef4444)  "TOXIC"

Agent states → màu sắc:
  pending   ○  --border        (#2a3348)
  running   ●  --accent-blue   (#3b82f6)  + pulse animation
  done      ✓  --accent-green  (#22c55e)
  error     ✕  --accent-red    (#ef4444)

Tox21 task score → màu bar:
  0.00 – 0.40  --accent-green (low risk)
  0.40 – 0.60  --accent-yellow (moderate)
  0.60 – 1.00  --accent-red (high risk, active)
```

---

*Tài liệu này là đặc tả UI dành cho đội phát triển frontend ToxAgent. Mọi thay đổi về API response schema cần được cập nhật lại trong phần data mapping tương ứng.*

*Version 1.0 — GDGoC Vietnam 2026*
