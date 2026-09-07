# Kế hoạch refine UI — Session Workbench & Quick Predict

## 1. Mục tiêu

Refine hai trải nghiệm FE quan trọng nhất của ToxAgent:

1. **Session Workbench** — `/s/:sessionId` và các artifact sub-route:
   tạo cảm giác tập trung, tự nhiên và “cuốn” như một sản phẩm chat hiện đại;
   giảm chất “dashboard kỹ thuật”, tăng nhịp đọc và cảm giác đối thoại.
2. **Quick Predict** — `/predict`:
   thay form hai cột hiện tại bằng một trải nghiệm phân tích trực quan, có điểm
   nhấn mạnh ngay từ input đến kết quả nhưng vẫn nhanh, stateless và rõ ràng.

Ngôn ngữ thị giác chung là **white-first + luminous purple**: trắng chiếm phần
lớn diện tích; tím chỉ dùng để dẫn mắt, biểu thị tương tác và tạo chiều sâu kiểu
glass/liquid-glass của iOS. Không thay đổi API contract, routing contract, SSE,
query keys, persistence hay logic khoa học trong đợt này.

---

## 2. Audit hiện trạng

### 2.1. Điểm nên giữ

- Session đã có shell ba vùng hợp lý: sidebar, chat, artifact panel.
- Artifact sub-route dùng cùng một component reference, giữ transcript/composer
  và SSE không bị remount khi đổi artifact.
- Desktop đã có resizable artifact panel; tablet/mobile đã dùng Sheet.
- Draft theo session, optimistic pending message, reconnect state và auto-scroll
  đã có logic riêng; đây là phần không nên viết lại khi đổi UI.
- Quick Predict đã hỗ trợ single/batch, vẽ cấu trúc, OCR, endpoint selection,
  expert threshold và dùng chung `AnalysisPanel`.
- Hệ thống dùng Tailwind v4, Radix/shadcn primitives, Lucide và `motion`; không
  cần thêm UI framework.

### 2.2. Vấn đề cần giải quyết

| Khu vực | Hiện trạng | Hướng refine |
|---|---|---|
| Visual system | Xám-xanh và blue accent kiểu admin app | White-first, purple accent, semantic colors độc lập |
| Typography | Nhiều font import, hierarchy chưa thống nhất | Một sans variable cho UI, một mono cho dữ liệu |
| Session header | Border bar có title + raw session ID | Header nhẹ, title rõ; metadata đưa vào popover |
| Empty session | Logo và ba card lớn chiếm nhiều chiều cao | Greeting ngắn + composer là tâm điểm + suggestion chips |
| Composer | Nhiều hàng control, giống form | Một floating composer; progressive disclosure cho tùy chọn |
| Transcript | Assistant cũng nằm trong card có border | Assistant nằm trực tiếp trên canvas; user bubble nhẹ |
| Run/event | Card kỹ thuật xen dày trong hội thoại | Compact status row; mở rộng khi người dùng cần |
| Artifacts | Select đơn, panel khá phẳng | Gallery/picker rõ loại, sticky toolbar, nội dung có hierarchy |
| Quick Predict | Form/result hai cột tĩnh | Input studio phía trên; result dashboard xuất hiện theo flow |
| Result hierarchy | Các card endpoint gần như đồng cấp | Summary strip trước, endpoint details sau, XAI cuối |

---

## 3. Design direction

### 3.1. Tính cách sản phẩm

- **Calm:** nhiều khoảng trắng, ít đường viền, một điểm focus tại một thời điểm.
- **Scientific:** số liệu rõ, alignment chắc, mono chỉ dùng đúng chỗ.
- **Trustworthy:** semantic colors mô tả trạng thái; tím không được dùng để ám
  chỉ “an toàn” hay “độc”.
- **Polished:** radius mềm, glass có kiểm soát, motion ngắn và có mục đích.
- **Not a ChatGPT clone:** học bố cục tập trung và nhịp tương tác, nhưng giữ logo,
  phân tử, endpoint, artifact và auditability làm bản sắc ToxAgent.

### 3.2. Tỷ lệ màu

- 82–88%: trắng và off-white.
- 8–12%: neutral text/border/surface.
- 3–6%: tím cho CTA, focus, active state và glow.
- Semantic green/amber/red chỉ dùng cho trạng thái khoa học hoặc hệ thống.

Không dùng gradient tím làm nền toàn trang. Glow chỉ nằm sau CTA, composer focus,
selected endpoint và result hero; opacity thấp để giao diện vẫn sáng, sạch.

### 3.3. Design tokens đề xuất

Các token semantic thay cho việc rải inline `style={{ ... }}`:

```css
:root {
  --canvas: #ffffff;
  --canvas-subtle: #fbfaff;
  --surface: rgba(255, 255, 255, 0.88);
  --surface-solid: #ffffff;
  --surface-muted: #f7f6fb;
  --surface-hover: #f3f0fb;

  --ink: #17151c;
  --ink-secondary: #5f5a68;
  --ink-tertiary: #918a9c;
  --line: #e9e5ee;
  --line-strong: #dcd5e5;

  --purple-50: #f7f3ff;
  --purple-100: #eee5ff;
  --purple-300: #c8a8ff;
  --purple-500: #8b5cf6;
  --purple-600: #7743e8;
  --purple-700: #6332c5;
  --purple-glow: rgba(139, 92, 246, 0.22);

  --success: #16865c;
  --warning: #b36a00;
  --danger: #d23b4d;
  --info: #3976d8;

  --radius-control: 12px;
  --radius-card: 18px;
  --radius-floating: 24px;
  --shadow-float: 0 12px 40px rgba(45, 28, 72, 0.10);
  --shadow-purple: 0 10px 30px rgba(119, 67, 232, 0.22);
}
```

Yêu cầu token:

- Map lại các alias shadcn như `--primary`, `--ring`, `--sidebar-*` để primitive
  hiện tại tự đồng bộ.
- Giữ alias cũ (`--bg`, `--accent-blue`, v.v.) trong một migration window,
  sau đó đổi component sang token mới và xóa alias cũ có kiểm soát.
- Glass chỉ bật khi browser hỗ trợ `backdrop-filter`; fallback là white solid.
- Dark mode không phải hướng chủ đạo của sprint này nhưng không được vỡ. Có thể
  giữ dark tokens hiện tại và thực hiện một pass riêng sau khi light mode đạt DoD.

### 3.4. Typography

- UI/body: **Inter Variable**, self-host ở `frontend/src/assets/fonts`; fallback
  `Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif`.
- Dữ liệu: **JetBrains Mono Variable** cho SMILES, probability raw, threshold,
  model ID, analysis ID; không dùng mono cho subtitle hoặc navigation.
- Bỏ `Orbitron`, `Climate Crisis`, Cal Sans và remote Google/CDN import khỏi
  bundle của hai route nếu không còn consumer thực tế.
- Scale:

| Token | Size/line-height | Dùng cho |
|---|---:|---|
| `display-sm` | 32/40, 600 | Greeting, Predict hero |
| `title-lg` | 22/30, 600 | Kết quả chính |
| `title-md` | 16/24, 600 | Card/section title |
| `body` | 15/24, 400 | Hội thoại và mô tả |
| `body-sm` | 13/20, 400–500 | Secondary UI |
| `caption` | 12/16, 500 | Status, metadata |
| `data` | 13/20, 500 mono | SMILES và technical values |

Transcript assistant dùng line-height rộng và width đọc tối đa 720–760px. Không
dùng weight 700 tràn lan; 600 là mức nhấn chính.

### 3.5. Shape, elevation và icon

- Control: radius 10–12px; card: 16–18px; composer/floating panel: 22–24px.
- Card mặc định dùng border rất nhẹ, không shadow. Chỉ floating composer,
  popover/dialog và selected result mới có shadow.
- CTA tím dùng gradient rất nhẹ `purple-500 → purple-700`, highlight trắng mờ ở
  mép trên và shadow tím khuếch tán để tạo cảm giác “bóng”.
- Lucide icon 16/18/20px, stroke 1.75; một button chỉ có tối đa một icon chính.
- Focus ring 3px gồm một line tím + halo tím nhạt; không chỉ đổi màu border.

### 3.6. Motion

- Hover/focus: 120–160ms; panel/reflow: 200–260ms; result reveal: 300–420ms.
- Easing: `cubic-bezier(.2,.8,.2,1)`.
- Quick Predict result đi theo thứ tự: summary → endpoint cards → details, lệch
  nhau 40–60ms; không animate con số từ giá trị giả.
- Composer glow chỉ xuất hiện khi focus-within; running state dùng shimmer/pulse
  rất nhẹ, không làm cả card nhấp nháy.
- Tất cả animation phải tắt hoặc rút gọn với `prefers-reduced-motion`.

---

## 4. Session Workbench — `/s/:sessionId`

### 4.1. Information architecture

```text
┌──────────── Sidebar ────────────┬──────────────── Chat canvas ────────────────┬──── Artifacts (khi mở) ────┐
│ Brand          collapse        │ menu   Session title        status  results │ Result picker          close │
│ + New session                  ├───────────────────────────────────────────────┼──────────────────────────────┤
│ Search                         │                                               │ Summary / viewer              │
│ Running                        │         Empty greeting hoặc transcript         │ Scientific details            │
│ Recent sessions                │             max-width 760px                    │ Audit / inspect               │
│                                │                                               │                               │
│ Predict / Settings / About     │     floating composer, max-width 820px         │ sticky local actions          │
└────────────────────────────────┴───────────────────────────────────────────────┴───────────────────────────────┘
```

Nguyên tắc:

- Chat là vùng ưu tiên. Khi artifact đóng, nội dung chat nằm giữa viewport.
- Composer không có border-top full-width; nó nổi trên canvas với một vùng fade
  trắng phía dưới để nội dung không bị cắt đột ngột.
- Artifact panel là “context surface”, không phải một page khác. URL vẫn phản
  ánh selection như hiện tại.
- Desktop sidebar rộng 264px, icon rail 56px. Artifact mặc định 360–420px và
  vẫn resizable trong giới hạn hiện tại.

### 4.2. App sidebar

Refine `WorkspaceLayout`, `AppSidebar`, `SessionRow`:

- Nền `#f8f7fa` hoặc white/70, đường phân cách phải cực nhẹ.
- Brand row cao 52–56px; logo 24px, wordmark 14px/600.
- `Session mới` là row cao 40px, nền white, icon pencil/plus; hover tím nhạt.
- Search ban đầu là control gọn; `/` hoặc `Cmd/Ctrl+K` focus search nếu không
  xung đột với text input.
- Session rows cao tối thiểu 36px, title một dòng. Active row có nền white,
  border inset tím rất mảnh hoặc accent dot; không fill tím đậm.
- Running session có purple pulse 6px và tooltip trạng thái.
- Group label nhỏ, neutral, sticky theo vùng scroll nếu danh sách dài.
- Footer có shortcut rõ cho **Quick Predict**, Sessions, Settings, About. Thêm
  `/predict` vào sidebar để hai workflow chính liên thông trực tiếp.
- Collapsed rail giữ tooltip, active state và keyboard focus; logo trở thành
  nút về home.
- Mobile sidebar vẫn là Sheet; close sau khi chọn session hoặc route.

States cần thiết: normal, hover, keyboard-focus, active, running, disabled,
loading skeleton, empty, search-no-result và pagination-loading.

### 4.3. Workspace header

Refine `WorkspaceHeader`, `ConnectionIndicator`:

- Height 52–56px, nền canvas/80 có blur nhẹ khi transcript scroll bên dưới.
- Chỉ hiện session title trong luồng chính; bỏ raw session ID khỏi dòng subtitle.
  ID, created time và copy action chuyển vào popover khi click title/chevron.
- Trái: mobile/sidebar trigger → title. Phải: connection status dạng icon/dot,
  button `Kết quả` có badge và panel toggle.
- Status bình thường không chiếm nhiều chữ; “Đã kết nối” là tooltip. Chỉ hiện
  label trực tiếp khi reconnecting/offline/error.
- Header không tạo cảm giác toolbar nặng; border chỉ hiện khi content đã scroll.

### 4.4. Empty session

Refine `EmptyStateHero` từ “landing mini” thành onboarding ngay trong conversation:

- Greeting 28–32px: `Bạn muốn phân tích gì?`.
- Một dòng giải thích ngắn, không rotate tagline liên tục vì gây xao nhãng.
- Composer được kéo lên ngay dưới greeting; đây là CTA chính.
- Ba capability hiện thành suggestion tiles nhỏ hoặc chips:
  `Dán SMILES`, `Tải ảnh cấu trúc`, `Vẽ cấu trúc`.
- Thêm 2–3 prompt example có nội dung thật, ví dụ “Phân tích hERG cho aspirin”
  hoặc “Giải thích atom attribution”, nhưng click chỉ prefill, không auto-send.
- Capability unavailable được disabled với lý do trong tooltip; không hardcode
  “Sắp ra mắt” khi backend có capability động.
- Logo chỉ 32–40px hoặc dùng subtle molecular mark, tránh cạnh tranh với input.

### 4.5. Transcript

Refine `Transcript` và các message component:

#### Message layout

- Container đọc tối đa 760px; gap giữa turn 24–32px, gap nội bộ một turn 10–12px.
- Assistant message không có card/border bao ngoài; text nằm trực tiếp trên nền
  với avatar/mark nhỏ ở đầu turn hoặc một left anchor tinh tế.
- User message căn phải, nền neutral-purple rất nhạt hoặc tím đậm có contrast
  chuẩn; ưu tiên bản nhạt để giữ tổng thể white-first. Max width 72–78%.
- SMILES là block mono riêng, có copy button khi hover/focus, wrap an toàn.
- Message actions (copy, retry nếu có, inspect) chỉ hiện hover/focus nhưng vẫn
  keyboard-accessible.

#### Các component cụ thể

- `MessageBubble`: tách variant `user`, `assistant`, `pending`; bỏ inline color.
- `AnswerBlock` / `AnswerRenderer`: typography giống article; heading, list,
  table, code và citation có spacing nhất quán.
- `ClaimChip`: pill tím nhạt/neutral, selected state tím rõ; giữ semantic status
  riêng cho grounded/unsupported.
- `FallbackBadge`: amber neutral, không dùng tím để biểu diễn cảnh báo.
- `LimitationBlock`: callout nền amber rất nhạt với icon, không phải error card.
- `ClarificationCard`: question callout gọn, action button ngay dưới câu hỏi.
- `StructureRecognitionCard`: thumbnail trái, canonical SMILES + confidence,
  primary action `Dùng SMILES này`, secondary `Chỉnh sửa`.
- `AnalysisSystemCard`: compact timeline row; queued/running dùng purple spinner,
  completed dùng green check, failed red. Completed row click được toàn hàng.
- `RunBlock`: collapsed mặc định thành một status row; expandable để xem tool
  calls/timeline, tránh phá nhịp hội thoại.
- `SystemEventCard`: centered microcopy neutral, không mô phỏng message bubble.
- `RecoveryBanner`: amber callout nằm sát run liên quan, có action/chi tiết rõ.
- Pending message giảm opacity nhẹ, có label `Đang gửi…`; không tạo fake result.

#### Scrolling

- Giữ `useStickToBottom` hiện tại.
- `Tin nhắn mới` đổi thành circular glass button hoặc compact pill ngay trên
  composer, có badge số lượng nếu state hỗ trợ về sau.
- Thêm bottom padding bằng chiều cao composer thực tế; không hardcode một giá
  trị dễ vỡ khi chip/image preview làm composer cao lên.

### 4.6. Composer — thành phần chủ đạo

Refine `MessageComposer` thành một surface 1–2 tầng:

```text
╭────────────────────────────────────────────────────────────────────╮
│ [analysis context chip / image preview / SMILES chip nếu có]       │
│ Hỏi về phân tử hoặc dán SMILES…                                    │
│                                                                    │
│  ＋   [Auto ▾]                         [endpoint summary]   [ ↑ ]   │
╰────────────────────────────────────────────────────────────────────╯
   ToxAgent có thể sai; hãy kiểm tra kết quả quan trọng.
```

- Surface white/90, radius 24px, border neutral và soft shadow; focus-within có
  purple ring + glow mờ.
- Textarea auto-grow từ 1 đến tối đa 7–8 dòng; `Enter` gửi, `Shift+Enter` xuống
  dòng. Mobile cần tránh gửi nhầm khi IME composing.
- Nút `+` mở attachment/action menu: ảnh, vẽ cấu trúc, nhập SMILES riêng.
- Không luôn hiển thị một hàng Input SMILES + Select intent + Settings như hiện
  tại. SMILES sau khi nhập trở thành removable chip/preview phía trên textarea.
- Intent selector rút thành `Auto` ở footer; các intent chuyên sâu nằm trong
  dropdown với mô tả một dòng.
- Endpoint selection nằm trong settings popover; footer chỉ hiện summary như
  `hERG + Tox21` khi có SMILES.
- Send button tròn 36px, tím glossy, icon arrow-up; disabled neutral. Running
  state không làm draft biến mất và không giả lập khả năng cancel nếu API chưa có.
- Analysis context chip giữ hành vi hiện tại nhưng style như contextual capsule.
- Staged image có thumbnail, tên, remove; recognition unavailable phải thông báo
  trước khi người dùng đi hết upload flow.
- Footer disclaimer cực nhẹ; chỉ hiện một dòng và không chiếm conversation.

Giữ nguyên: draft per-session, signal từ empty state, expert role enforcement,
lazy-load structure editor, no auto-submit sau OCR và `onSend(): Promise<boolean>`.

### 4.7. Artifact panel và viewers

Refine `ArtifactsPanel`, `ArtifactViewer` và các viewer con:

- Header đồng chiều cao với chat header. `Artifacts` đổi label UI thành `Kết quả`
  (route/data model vẫn giữ tên artifact).
- Picker hỗ trợ group theo loại: Analysis, Run, Answer, Observation, Evidence;
  mỗi option có icon, title ngắn và relative time. Với số lượng ít có thể dùng
  segmented chips; nhiều thì dùng searchable command/select.
- Dòng “đã tải N kết quả…” chuyển vào menu/help tooltip, không chiếm toolbar.
- Viewer có sticky local header chứa type badge, copy ID, open/close và action
  `Hỏi về phân tích này` khi phù hợp.
- Empty panel có một câu hướng dẫn và icon; không đặt dashed card lồng card.
- Desktop giữ resize; handle rộng vùng hit 8–12px nhưng nét nhìn chỉ 1px.
- Tablet mở Sheet tối đa 480px; mobile mở full screen có back button và safe area.

Chi tiết viewer:

- `AnalysisPanel`: thêm result summary ở đầu; molecule và top metrics đặt chung
  một visual block, details đi theo section.
- `EndpointCard`: probability là hero metric; label và threshold nằm gần bar;
  model/source ở collapsible `Chi tiết mô hình`.
- `Tox21AssayTable`: sticky header, row density vừa, search/filter nếu assay dài;
  semantic badge phải có text, không chỉ có màu.
- `ApplicabilityChip`: đổi thành compact status block có tooltip giải thích
  domain applicability.
- `EndpointUnavailableCard`: neutral unavailable state, lý do rõ, không giống lỗi.
- `ExplainPanel`, `AttributionPanel`: collapsed summary trước; chart/atom map chỉ
  render khi mở nếu có lợi cho performance.
- `AtomHighlightDepiction`, `MoleculeDepiction`: white plot surface, label và
  zoom/legend rõ; không để glow tím làm sai cách đọc heatmap.
- `AnswerAuditViewer`, `EvidenceArtifact`, `ObservationArtifact`: thống nhất
  header metadata, section spacing và action copy/open.
- `RunInspectorContent`: tabs `Timeline`, `Validation`, `Runtime`, `Raw JSON` dùng
  sticky tab bar; Raw JSON luôn mono và có copy.
- `RunTimelineTab`: timeline line neutral, semantic dot theo trạng thái.
- `ValidationTab`, `ViolationList`: pass/warn/fail bằng icon + label + color.
- `RuntimeManifestTab`: group key-value theo topic, tránh một bảng dài phẳng.
- `RawJsonTab`: code surface có line wrap toggle và copy/download nếu đã có API.
- `ArtifactUnavailable`: concise empty/error state với retry hoặc back action khi
  hành vi hiện có cho phép.

### 4.8. Responsive rules

| Breakpoint | Sidebar | Chat | Artifact | Composer |
|---|---|---|---|---|
| `<768px` | Off-canvas Sheet | Full width, padding 16px | Full-screen Sheet | Inset 8–12px + safe area |
| `768–1279px` | Icon rail mặc định | Fluid | Right Sheet ≤480px | Max 820px, inset 20px |
| `≥1280px` | 264px / 56px rail | Centered transcript | Resizable 360–520px | Max 820px |

- Khi mobile keyboard mở, composer phải còn nhìn thấy; ưu tiên `100dvh`, safe-area
  inset và không khóa body theo cách làm mất vị trí transcript.
- User bubble tối đa 88% trên mobile.
- Header title không đẩy `Kết quả` ra khỏi viewport; title truncate và metadata
  nằm trong popover.

---

## 5. Quick Predict — `/predict`

### 5.1. Concept: Molecular Result Studio

Quick Predict nên khác Session ở mục tiêu: không giống chat và không giống một
form cấu hình. Nó là một “analysis studio” có flow rõ:

1. Đưa phân tử vào.
2. Chọn phạm vi dự đoán.
3. Chạy phân tích.
4. Đọc overview ngay lập tức.
5. Mở chi tiết/XAI khi cần.

```text
┌────────────────────────── compact product nav ───────────────────────────┐
│ Quick Predict                                        Mở Session / History │
├───────────────────────────────────────────────────────────────────────────┤
│                 Dự đoán độc tính trong vài giây                          │
│          Nhập SMILES, tải ảnh hoặc vẽ cấu trúc                           │
│ ╭────────────────────── floating molecule console ─────────────────────╮ │
│ │ [Single | Batch]                                                     │ │
│ │ SMILES / recognized preview                                         │ │
│ │ [+ Ảnh] [Vẽ]     [hERG] [Tox21] [ClinTox]       [Phân tích →]        │ │
│ ╰───────────────────────────────────────────────────────────────────────╯ │
│ examples / stateless note                                                │
├────────────────────────── result region ─────────────────────────────────┤
│ Molecule summary          Risk overview / endpoint cards                 │
│ Detailed assays           Applicability                                  │
│ XAI / attribution                                                        │
└───────────────────────────────────────────────────────────────────────────┘
```

### 5.2. Page shell và hero

- Dùng compact app navigation phù hợp authenticated route; không dùng full
  marketing footer khiến task flow bị dài. Header có logo, back/workbench,
  history/settings khi cần.
- Max width 1180–1240px, top spacing 56–72px desktop và 24px mobile.
- Hero title 32–40px, một subtitle ngắn. Background trắng; hai radial purple
  glow rất nhạt nằm sau input console, không phủ vào result text.
- Khi đã có result, hero có thể compact bằng transition nhẹ để kết quả tiến gần
  viewport; không làm layout jump mạnh.

### 5.3. Molecule input console

Tách form hiện tại thành các component có trách nhiệm rõ:

- `PredictModeSwitch`: segmented `Một phân tử` / `Hàng loạt`; chuyển mode giữ
  draft riêng cho từng mode và clear field error tương ứng như logic hiện tại.
- `MoleculeInput`: textarea một dòng auto-grow cho single SMILES; mono, copy/paste
  tốt, validation hint ở dưới, không dùng placeholder làm label duy nhất.
- `MoleculeInputActions`: `Tải ảnh`, `Vẽ cấu trúc`; icon + label, touch target 40px.
- `RecognizedStructurePreview`: thumbnail, confidence, editable canonical SMILES,
  replace/remove actions.
- `EndpointSelector`: selectable luminous chips/cards; endpoint không khả dụng
  vẫn nhìn thấy nhưng disabled kèm reason. Không dùng checkbox list trần.
- `AdvancedPredictOptions`: disclosure/popover cho expert threshold. Hiện summary
  nếu có override để tránh cấu hình ẩn mà người dùng quên.
- `PredictCTA`: glossy purple button, rõ state loading; copy `Đang phân tích…`.
- `StatelessNotice`: icon lock/history-off + “Kết quả này không được lưu”; link
  `Cần audit trail? Mở Session`.

Không tự đoán molecule name hoặc render giả trước response. `looksLikeSmiles`
chỉ là hint; backend vẫn là validator chuẩn như hiện tại.

### 5.4. Single-result experience

Sau khi có kết quả, scroll/reveal đến một `<section aria-live="polite">` nhưng
không cưỡng ép scroll nếu người dùng đang tương tác nơi khác.

#### Result overview

- `ResultHeader`: `Kết quả dự đoán`, canonical SMILES, copy và `Phân tích chất khác`.
- `MoleculeSummaryCard`: depiction lớn, canonical SMILES, applicability summary.
- `EndpointSummaryGrid`: hERG, Tox21, ClinTox ở cùng một row desktop; probability,
  label, threshold marker và availability nhìn được trong 3–5 giây.
- Màu kết quả là semantic red/amber/green với text label và icon. Purple chỉ thể
  hiện selection/progress/brand, không mã hóa độc tính.
- Nếu endpoint unavailable, card giữ đúng vị trí để layout không nhảy.

#### Detail

- Tabs/anchor nav: `Tổng quan`, `Tox21 assays`, `Giải thích`, `Chi tiết mô hình`.
- Reuse logic từ `AnalysisPanel` nhưng cung cấp `variant="predict-page"` để có
  composition rộng thay vì xếp một cột như artifact panel.
- `ExplainPanel` đặt sau overview, collapsed mặc định trên mobile.
- Add copy/export chỉ khi có capability thật; không đưa CTA “Lưu” vì route này
  stateless. Có thể đưa CTA `Tiếp tục trong Session` vào backlog, chỉ implement
  khi backend có contract chuyển kết quả an toàn.

### 5.5. Batch experience

Batch không chỉ là textarea lớn rồi lặp lại nhiều `AnalysisPanel`:

- Textarea có line number hoặc ít nhất counter `N phân tử`, max-limit và ví dụ
  định dạng. Parse theo đúng logic backend hiện tại; frontend chỉ preview count.
- Sau submit, dùng `BatchSummary`: tổng số, thành công, lỗi, endpoint đã chạy.
- Kết quả dùng table/list có sticky header:
  molecule/index, truncated SMILES, endpoint summary, status, expand action.
- Expand một row để xem `AnalysisPanel`; không render toàn bộ chart/XAI của mọi
  molecule ngay lập tức.
- Error rows nằm đúng vị trí input, có reason dễ đọc và copy raw SMILES; không
  gom một khối lỗi dài ở đầu trang như hiện tại.
- Mobile chuyển mỗi row thành compact result card.

### 5.6. Loading, empty và error states

- Initial empty: giữ hero/input là focus, bên dưới có ba mini feature statements,
  không dựng một result placeholder card lớn.
- Loading: CTA spinner + progress copy theo bước chỉ khi backend thật sự cung cấp
  bước; nếu không chỉ nói `Đang phân tích…`. Result skeleton giữ layout ổn định.
- Field error: nằm sát input, icon + text, `aria-describedby`.
- Global/network error: inline banner trong console có retry, không chỉ toast.
- Partial result: endpoint thành công vẫn hiển thị; endpoint unavailable/error có
  card riêng và nguyên nhân.
- Empty endpoints: CTA disabled và có helper rõ “Chọn ít nhất một endpoint”.

### 5.7. Responsive rules

- Desktop: console tối đa 900px; result overview 12-column grid, molecule 4 cột,
  endpoint summary 8 cột.
- Tablet: console full width; molecule và endpoints chia 5/7 hoặc stack nếu hẹp.
- Mobile: hero trái hàng, console radius 18px, action wrap hai hàng; result stack;
  CTA full-width hoặc sticky bottom chỉ trong form viewport, không che kết quả.
- Batch table chuyển card list dưới 768px; không horizontal-scroll dữ liệu chính.

---

## 6. Shared component architecture

Không tạo hai design system riêng. Chia thành ba lớp:

### 6.1. Foundation primitives

- `AppButton`: variants `primary-gloss`, `secondary`, `ghost`, `danger`.
- `AppInput`, `AppTextarea`: size, focus ring, error/helper contract thống nhất.
- `Surface`: `plain`, `card`, `floating`, `glass`.
- `StatusBadge`: semantic status, luôn có text/icon.
- `Metric`: label/value/unit/trend hoặc threshold.
- `EmptyState`, `InlineAlert`, `Skeleton`, `IconButton`, `CopyButton`.
- `Tooltip`, `Popover`, `Sheet`, `Dialog`, `Select` tiếp tục dùng Radix hiện có.

Không nhất thiết wrapper mọi primitive ngay lập tức. Nếu shadcn primitive hiện
tại đáp ứng đủ, cập nhật variant/token tại chỗ để tránh abstraction thừa.

### 6.2. Scientific shared components

- `MoleculePreview`
- `SmilesValue`
- `EndpointBadge` / `EndpointSelector`
- `ProbabilityMeter`
- `ApplicabilityStatus`
- `ModelMetadataDisclosure`
- `AnalysisPanel` với layout variant `artifact` và `predict-page`

### 6.3. Route compositions

- Session: `SessionShell`, `ConversationCanvas`, `FloatingComposer`,
  `ArtifactWorkspace`.
- Predict: `PredictHero`, `MoleculeConsole`, `SingleResultDashboard`,
  `BatchResultTable`.

Logic fetch/mutation nên ở page/container như hiện tại. Component visual nhận
props và event; không tự gọi API trừ các component vốn đã có domain query riêng
và việc di chuyển query không đem lại lợi ích rõ.

---

## 7. File plan

### 7.1. Foundation

| File | Thay đổi |
|---|---|
| `frontend/src/styles/fonts.css` | Self-host Inter/JetBrains Mono, bỏ import thừa/remote |
| `frontend/src/styles/theme.css` | White/purple semantic tokens, radius, shadow, type scale, reduced motion |
| `frontend/src/styles/index.css` | Base canvas, selection, scrollbar và focus behavior |
| `frontend/src/components/ui/button.tsx` | Thêm glossy primary và icon-circle size |
| `frontend/src/components/ui/input.tsx` | Chuẩn hóa focus/error/surface |
| `frontend/src/components/ui/textarea.tsx` | Đồng bộ input và composer behavior |
| `frontend/src/components/ui/sidebar.tsx` | Width/rail/mobile constants và visual states |

### 7.2. Session route

| File/component | Thay đổi |
|---|---|
| `pages/WorkbenchPage.tsx` | Composition, scroll fade, floating composer slot; giữ data/SSE/navigation logic |
| `shell/WorkspaceLayout.tsx` | White canvas, responsive shell và safe viewport |
| `shell/AppSidebar.tsx` | IA/navigation mới, thêm Quick Predict, search/active states |
| `shell/SessionRow.tsx` | Active/running/hover/loading states |
| `shell/WorkspaceHeader.tsx` | Compact title, metadata popover, scroll state |
| `shell/ConnectionIndicator.tsx` | Dot-first state + tooltip/error label |
| `workbench/EmptyStateHero.tsx` | Greeting, suggestion chips, capability-aware actions |
| `workbench/MessageComposer.tsx` | Floating composer + progressive disclosure |
| `transcript/Transcript.tsx` | Turn spacing và status grouping |
| `transcript/MessageBubble.tsx` | User/assistant/pending variants |
| `transcript/*Card.tsx`, `RunBlock.tsx` | Compact status/callout patterns |
| `answer/*` | Article typography, claim/limitation styling |
| `artifacts/ArtifactsPanel.tsx` | Picker, local toolbar, responsive behavior |
| `artifacts/ArtifactViewer.tsx` và viewers | Unified viewer shell/metadata |
| `inspector/*` | Sticky tabs, semantic status, code viewer |

### 7.3. Quick Predict route

| File/component | Thay đổi |
|---|---|
| `pages/QuickPredictPage.tsx` | Chia container logic và page composition; bỏ form hai cột cũ |
| `workbench/AnalysisPanel.tsx` | Thêm layout variants dùng chung |
| `workbench/EndpointCard.tsx` | Probability hierarchy + metadata disclosure |
| `workbench/Tox21AssayTable.tsx` | Density, responsive, semantic state |
| `workbench/ApplicabilityChip.tsx` | Applicability status component |
| `workbench/ExplainPanel.tsx` | Disclosure, lazy-heavy content |
| `workbench/ImageUploadDialog.tsx` | Đồng bộ dialog/glass surface/states |
| `workbench/StructureEditorDialog.tsx` | Đồng bộ dialog shell/action hierarchy |
| `components/predict/MoleculeConsole.tsx` | Component mới cho input workflow |
| `components/predict/PredictModeSwitch.tsx` | Single/batch segmented control |
| `components/predict/EndpointSelector.tsx` | Endpoint selectable chips |
| `components/predict/SingleResultDashboard.tsx` | Result overview rộng |
| `components/predict/BatchResultList.tsx` | Summary + expandable rows |

Tên component mới là đề xuất; có thể gộp nếu implementation thực tế nhỏ, nhưng
không để `QuickPredictPage.tsx` tiếp tục giữ toàn bộ markup và state presentation.

---

## 8. State matrix bắt buộc

### Session

- Bootstrap loading / error / ready.
- Message history loading / error / empty / populated.
- Pending send / send failed / active run busy.
- SSE connected / reconnecting / offline / recovered.
- No artifact / selected artifact / unseen artifact / artifact unavailable.
- Sidebar expanded / rail / mobile Sheet.
- Composer empty / focused / multiline / SMILES / image / context / disabled.
- OCR capability available / unavailable / recognition failed / success.

### Quick Predict

- Capabilities loading / partial / failed.
- Single / batch.
- Empty / invalid-hint / backend-invalid / recognized image / drawn structure.
- No endpoint / endpoint unavailable / expert override.
- Predict idle / loading / success / partial / error.
- Batch all-success / mixed / all-error / expanded row.
- Result with/without hERG, Tox21, ClinTox, applicability và XAI.

Mỗi state phải có thiết kế, copy, keyboard path và test case; không chỉ thiết kế
happy path trong mockup.

---

## 9. Accessibility và content rules

- Contrast tối thiểu WCAG AA; text nhỏ không đặt trực tiếp trên gradient/glow.
- Focus-visible rõ trên mọi button, row click được, chip và resizer.
- Touch target tối thiểu 40×40px; action icon quan trọng 44×44px trên mobile.
- Mọi icon-only button có accessible name; tooltip không thay thế label cho
  screen reader.
- Probability/status không truyền nghĩa bằng màu duy nhất.
- `aria-live` chỉ dùng cho send/predict/result status cần thiết; tránh transcript
  đọc lại toàn bộ khi SSE cập nhật.
- Dialog focus trap, close bằng Escape; Sheet trả focus đúng trigger.
- Resizable panel có keyboard affordance hoặc giữ primitive behavior có sẵn.
- Tôn trọng `prefers-reduced-motion`, `prefers-contrast` nếu khả thi.
- Nội dung luôn nói đây là dự đoán/screening; không đổi label thành clinical
  verdict và không dùng visual “safe” quá khẳng định.

---

## 10. Performance guardrails

- Không tăng bundle route đầu chỉ để tạo visual effect; dùng CSS cho glow/glass.
- Giữ lazy-load `StructureEditorDialog` và artifact viewers.
- Không render mọi expanded analysis/XAI trong batch.
- Font variable self-host chỉ gồm weight cần dùng; preload đúng file sans chính.
- Tránh backdrop blur trên vùng scroll lớn; chỉ blur header/composer nhỏ.
- Animation chỉ dùng transform/opacity, hạn chế layout animation ở transcript.
- Giữ bundle budget script hiện có; đo lại chunk `/predict` và `/s/:sessionId`.

---

## 11. Kế hoạch triển khai theo phase

### Phase 0 — Baseline và visual contract

- Chụp baseline desktop/tablet/mobile cho empty session, populated session,
  artifact open, predict empty, single result và batch mixed result.
- Ghi lại DOM behavior quan trọng: SSE không remount, draft persistence, focus,
  URL artifact selection, panel width preference.
- Chốt light palette, type scale và component states trong một UI specimen page
  hoặc Storybook-equivalent nội bộ nếu dự án chưa có Storybook.

**Exit:** token/typography được duyệt; không còn quyết định màu/radius lớn để lại
cho từng component tự chọn.

### Phase 1 — Foundation

- Implement font, token, button/input/textarea, surface, badge, status patterns.
- Thay inline color ở các component được đụng tới bằng semantic class/token.
- Kiểm tra contrast, focus, reduced motion.

**Exit:** primitive states pass visual + keyboard review ở light mode; dark mode
không vỡ nghiêm trọng.

### Phase 2 — Session shell và empty state

- Sidebar, header, empty greeting, initial composer placement.
- Responsive rail/Sheet và route navigation.
- Giữ nguyên Workbench query/SSE logic.

**Exit:** empty session đạt bố cục mục tiêu ở ba breakpoint; create/open/search
session và artifact navigation vẫn chạy.

### Phase 3 — Composer và transcript

- Floating composer, attachment menu, chips, advanced settings.
- Message layout, answer typography, run/system/recovery states.
- Scroll behavior, pending send và reconnect states.

**Exit:** gửi text/SMILES/image/draw, draft, pending bubble, active run và jump to
bottom pass functional tests và keyboard test.

### Phase 4 — Artifact workspace

- Picker, viewer shell, Analysis/Answer/Evidence/Observation/Run inspectors.
- Desktop resize, tablet/mobile Sheet, unseen badge.

**Exit:** mọi artifact sub-route mở đúng selection mà không remount workbench;
viewer states nhất quán và readable.

### Phase 5 — Quick Predict input studio

- Page hero, mode switch, console, endpoint selector, recognition preview,
  advanced options và all input/error/loading states.
- Giữ API calls và expert enforcement hiện tại.

**Exit:** single/batch/OCR/draw behavior đạt parity với UI cũ và responsive.

### Phase 6 — Quick Predict results

- Single dashboard, batch summary/list, endpoint details và XAI.
- Partial/unavailable/error states và restrained reveal motion.

**Exit:** mọi response shape hiện có render đúng; batch lớn không eager-render
heavy content.

### Phase 7 — Polish và regression

- Cross-browser Safari/Chrome/Firefox; iOS Safari viewport/keyboard.
- Accessibility pass, performance/bundle check, screenshot regression.
- Xóa token/font/dead style cũ sau khi xác nhận không còn consumer.

**Exit:** đạt toàn bộ Definition of Done bên dưới.

---

## 12. Testing plan

### Component tests

- Composer disclosure, keyboard send, IME guard, chips remove, disabled/loading.
- Endpoint selector và unavailable reason.
- Result cards với boundary probability 0/1 và threshold marker.
- Batch mixed results, expand/collapse và error mapping.
- Connection/run/recovery semantic variants.
- Artifact picker → đúng URL; close → `/s/:sessionId`.

### Integration/E2E

- Tạo session → empty state → gửi SMILES → running → result artifact auto-open.
- Chuyển artifact không remount transcript/composer/SSE.
- Refresh sub-route artifact khôi phục đúng selection.
- Sidebar search/open/create/collapse persistence.
- Quick Predict single success, invalid SMILES, network error.
- Quick Predict batch mixed result.
- OCR available/unavailable và draw-editor lazy loading.
- Mobile: sidebar Sheet, artifact full-screen Sheet, composer + virtual keyboard.

### Visual regression viewport

- 390×844, 768×1024, 1280×800, 1440×900 và 1728×1117.
- Snapshot các state: empty, focus, loading, error, result, panel open và batch.

### Commands cuối mỗi phase

```bash
cd frontend
npm run typecheck
npm run test
npm run lint:policy
npm run build
```

---

## 13. Definition of Done

- Session có chat canvas tập trung, assistant content đọc tự nhiên, user message
  rõ nhưng không lấn át, composer là điểm tương tác chính.
- Sidebar/header/artifact hỗ trợ workflow hiện tại mà không tạo cảm giác admin UI.
- Predict có input studio hấp dẫn và result hierarchy đọc được trong vài giây;
  single/batch không còn là cùng một list card kéo dài.
- Trắng là màu chủ đạo; tím tạo điểm nhấn/glow có kiểm soát; semantic result không
  bị mã hóa sai bằng brand purple.
- Typography chỉ còn một UI family nhất quán và mono cho scientific data.
- Không regress API, SSE, URL state, draft, optimistic send, preferences, OCR,
  drawing hay expert threshold.
- Không remount workbench khi đổi artifact sub-route.
- Hoạt động đầy đủ ở mobile/tablet/desktop; không horizontal overflow.
- Keyboard navigation, focus, contrast, screen-reader label và reduced motion đạt
  yêu cầu trong mục accessibility.
- Typecheck, tests, policy lint, production build và bundle budget đều pass.

---

## 14. Ngoài scope của đợt refine

- Thay đổi model/predictor, threshold mặc định hoặc cách diễn giải khoa học.
- Thay đổi backend API/event schema, persistence hay authorization.
- Thêm collaboration, share session, export report hoặc chuyển Quick Predict
  thành persisted session khi chưa có product/API contract.
- Copy nguyên logo, trade dress hoặc pixel-level UI của ChatGPT/iOS.
- Dark-mode redesign hoàn chỉnh; chỉ yêu cầu không regress nghiêm trọng trong
  quá trình light-first migration.

---

## 15. Thứ tự ưu tiên nếu cần chia release

1. Token + font + composer + transcript.
2. Session sidebar/header/empty state.
3. Quick Predict input console + single result overview.
4. Artifact panel/viewers.
5. Batch result UX.
6. Motion polish, dark-mode pass và optional micro-interactions.

Thứ tự này đưa phần người dùng chạm nhiều nhất lên trước, đồng thời tạo shared
foundation để Quick Predict và artifact viewers không phải style lại hai lần.
