# Kế hoạch hoàn thiện Report, Explainer và UI session

Ngày lập: 2026-09-09  
Session tham chiếu: `ses_75af97e7c77144d6a103ee95209989c5`  
Run tham chiếu: `run_f65e29981c4040cfa023913f402cb145`  
Report tham chiếu: `rpt_a1793c856baf4b448490eeac0b93e008`

## 1. Mục tiêu bàn giao

Hoàn thiện report thành một tài liệu tự chứa, có thể kiểm chứng và có thể tải xuống, bao gồm:

- kết quả predictor và provenance của model;
- explanation theo từng endpoint/task, có hình và diễn giải đúng giới hạn;
- dữ liệu nghiên cứu ngoài hệ thống, có citation bấm được;
- phần tổng hợp chỉ rõ điểm đồng thuận/mâu thuẫn giữa model và tài liệu;
- UI session không bị thanh nhập chat che nội dung cuối;
- explainer plot dùng màu đỏ/xanh lá theo chiều đóng góp, có legend rõ ràng.

Tài liệu này khởi đầu là backlog/solution design. Phần **1a** dưới đây ghi lại
trạng thái triển khai thực tế; các mục 2–9 giữ nguyên như bản phân tích ban đầu
để còn đọc được lý do của từng quyết định.

## 1a. Trạng thái triển khai (cập nhật 2026-09-09)

Toàn bộ P0 đã triển khai và có test. P1/P2 chưa làm.

| Hạng mục | Trạng thái | Ghi chú |
|---|---|---|
| XAI-01 hợp nhất pipeline explainer | Xong | `application/explanation_identity.py` là nơi duy nhất định nghĩa cache key và schema version. Cả `create_analysis` và `GetOrCreateExplanation` ghi `toxpred-explanation-v3` qua `explanation_observation()`; reader nhận cả v1/v2. Checkpoint của analysis trở thành cache dùng chung. Payload đã có nhưng thiếu figure thì chỉ vẽ figure, không chạy lại backward pass. Reason code: `ExplanationUnavailable`. Readiness thật trong `/health/ready` → `explainer`. |
| XAI-02 plot đỏ/xanh theo dấu | Xong | `toxpred/application/depiction.py`: palette phân kỳ theo `signed_contribution`, scale đối xứng, neutral epsilon, legend trong ảnh, note `+`/`-` trên atom mạnh nhất. `SVG_PALETTE_VERSION`/`SVG_RENDERER_VERSION` đã bump. |
| REP-01 citation bấm được | Xong | `ReportReference` là snapshot immutable, nằm trong content hash; artifact bump lên `toxagent-report-v2`. Đánh số theo lần xuất hiện đầu tiên trong thứ tự đọc. Token `[@evd_...]` được validate như citation của claim. Chỉ HTTPS mới thành link. |
| REP-02 render đầy đủ trong app | Xong | `ReportBlock.tsx` render substance profile, table, figure, explanation + legend + bảng contributor, evidence synthesis, conclusions, recommendations, limitations, references, provenance, gap tại đúng section. Endpoint mới `GET /v1/sessions/{s}/reports/{r}/figures/{f}`. Structure figure trung tính riêng qua `POST /v1/depictions` của predictor. |
| REP-03 export HTML/PDF/Markdown | Xong | `_figure_svgs()` resolve bytes một lần cho cả ba format; `render_pdf` nhận `figure_svgs` (trước đây bỏ mất). Thêm format `markdown_bundle` (.zip gồm `report.md` + `figures/`), thay cho URI `figure:` không mở được. |
| UI-01 composer không che nội dung | Xong | Composer về normal flex flow, bỏ `absolute bottom-0` và `pb-48`; thêm safe-area padding; `useStickToBottom` dùng `ResizeObserver` cho cả viewport và content. |
| P1 chất lượng khoa học / UX | Chưa làm | Xem mục 5. Một số hạng mục đã có sẵn như hệ quả của P0 (bảng top contributor, method note, evidence quality badge, copy citation, gap có link, print stylesheet); phần còn lại (risk matrix, OOD panel, concordance table, conflict section, next experiments, mục lục sticky) chưa. |
| P2 governance / vận hành | Chưa làm | Version history/diff, export manifest, telemetry, retention policy, regression fixture. |

**Quyết định đã tạm chốt** cho mục 9, cần product xác nhận lại:

1. Report vẫn English-first; UI chrome tiếng Việt.
2. Citation mở evidence detail nội bộ trước (anchor tới References), người đọc tự bấm ra nguồn.
3. Markdown giao dưới dạng cả hai: `.md` thuần và `markdown_bundle` (.zip) tự chứa.
4. Red/green luôn kèm `+`/`-` và bảng số liệu, không cần mode palette riêng.
5. Structure image mặc định không có atom numbering; `atom_numbering` là tham số opt-in.

## 2. Kết quả rà soát code hiện tại

Không truy cập được report qua URL `localhost:8088` trong môi trường rà soát vì không có process lắng nghe cổng 8088. Các kết luận dưới đây dựa trên source code hiện có.

### 2.1 Citation đã được lưu nhưng chưa được render thành nguồn bấm được

Artifact hiện đã có:

- `Claim.citation_ids`;
- `EvidenceSynthesis.evidence_ids`;
- `EvidenceRecord.canonical_url`;
- bảng liên kết report–evidence trong persistence.

Tuy nhiên:

- `ReportBlock.tsx` chỉ render `section.body_markdown`, gap và navigation;
- component này chưa render claims, evidence synthesis, references, tables hoặc figures;
- Markdown/HTML renderer chỉ in `evidence_id` dạng text, không resolve thành tiêu đề và URL;
- `compile_report()` nhận các observation/explanation/figure đã resolve nhưng không nhận hoặc nhúng snapshot metadata của evidence vào artifact.

Kết quả: nguồn đã được research và validate vẫn chưa trở thành citation hữu ích đối với người đọc report.

### 2.2 Figure có metadata nhưng chưa có delivery path hoàn chỉnh

- Report artifact có `figures` và mỗi section có `figure_ids`.
- In-app `ReportBlock.tsx` không render `figure_ids`.
- Markdown renderer sinh URL riêng `figure:<figure_id>` nhưng file Markdown tải xuống không có bước resolve scheme này.
- `_render()` gọi HTML/PDF renderer mà không truyền `figure_svgs`; HTML vì thế hiển thị `[figure unavailable]`.
- Chưa có API public theo scope session/report để tải một figure bằng `figure_id`.
- `SubstanceProfile.structure_figure_id` tồn tại nhưng hiện chưa thấy luồng tạo structure figure độc lập; code chỉ đọc giá trị từ `build.stage_state`.

Kết quả: dữ liệu explanation có thể đã tồn tại nhưng hình vẫn không xuất hiện trong UI hoặc bản export.

### 2.3 Có hai pipeline explanation đang lệch schema/cache

Đây là root cause cần ưu tiên kiểm tra cho lỗi explainer:

- analysis pipeline lưu observation với schema `toxpred-explanation-v2`;
- report pipeline (`GetOrCreateExplanation`) chỉ tìm schema `toxpred-explanation-v1`;
- checkpoint của analysis lưu payload đã tính, nhưng report builder không dùng chung checkpoint này;
- report builder có thể không nhìn thấy explanation đã hoàn tất, gọi predictor lại, hoặc tạo gap dù checkpoint/model artifact đã đầy đủ;
- cache identity đang được triển khai ở hai nơi khác nhau, tăng nguy cơ lệch model id, artifact hash và vòng đời figure.

Checkpoint đầy đủ chỉ chứng minh weights/tokenizer có thể được load. Explainer còn phụ thuộc vào provider có `token_attribution`, model được pin đúng, token-offset alignment, timeout budget, object store và bước tạo/sanitize figure.

### 2.4 Explainer plot hiện mất ý nghĩa dấu âm/dương

`depiction.py` hiện:

- dùng palette tím một chiều (`purple-sequential-v1`);
- tô màu theo `relative_importance` không âm;
- không dùng `signed_contribution` khi chọn màu;
- vì vậy không phân biệt phần cấu trúc làm tăng và làm giảm logit của predicted class.

### 2.5 Thanh chat đang overlay transcript

Trong `WorkbenchPage.tsx`:

- composer dùng `absolute bottom-0`;
- transcript dùng `pb-48` cố định để bù chiều cao;
- chiều cao thực của composer thay đổi theo viewport, draft nhiều dòng, attachment và context chip;
- nút “Tin nhắn mới” cũng dùng `bottom-24` cố định.

Do đó phần cuối session có thể nằm dưới composer, đặc biệt trên mobile hoặc khi composer cao hơn bình thường.

## 3. Kiến trúc đích

Luồng mong muốn:

```text
Predictor observation ─┐
Explanation package ───┼─> validated ReportArtifact ─> in-app report
Evidence records ──────┤                         ├───> Markdown bundle
Structure/plot figures ┘                         ├───> self-contained HTML
                                                 └───> PDF
```

Nguyên tắc:

1. Predictor facts, literature facts và agent synthesis phải tách biệt.
2. Citation và figure là structured data; không dựa vào việc model tự chèn URL tùy ý trong prose.
3. Mọi rendering phải đọc từ cùng một immutable artifact.
4. Explanation luôn gắn với đúng `analysis_id`, endpoint/task, `model_id` và artifact fingerprint.
5. Màu sắc chỉ là một kênh hiển thị; luôn có dấu `+/-`, legend và bảng contributor để hỗ trợ accessibility.

## 4. Backlog đề xuất

### P0 — REP-01: Citation bấm được trong report

#### Giải pháp

Thêm snapshot nguồn đã resolve vào report artifact, ví dụ:

```json
{
  "references": [
    {
      "evidence_id": "evd_...",
      "title": "...",
      "canonical_url": "https://...",
      "authors": ["..."],
      "published_at": "...",
      "provider": "europepmc",
      "identifier": {"doi": "...", "pmid": "..."},
      "retrieved_at": "..."
    }
  ]
}
```

Metadata này phải được lấy từ `evidence_by_id` sau validation và đưa vào content hash của artifact. Không fetch lại web khi người dùng mở report.

Quy tắc render:

- đánh số nguồn ổn định theo lần xuất hiện đầu tiên trong report;
- external-evidence item hiển thị marker `[n]` bấm được;
- section `References` hiển thị title, tác giả/năm, provider/identifier và link HTTPS;
- link mở tab mới với `rel="noopener noreferrer nofollow"`;
- nếu URL không hợp lệ hoặc không phải HTTPS, vẫn hiển thị citation metadata nhưng không tạo link;
- citation chưa resolve phải hiện cảnh báo, không được biến mất âm thầm;
- claim về predictor chỉ trỏ observation/field path, không dùng paper thay thế model fact.

Để citation nằm đúng câu, bổ sung marker có cấu trúc thay vì cho model tự viết URL. Hai lựa chọn:

- ngắn hạn: section renderer suy ra danh sách nguồn từ `section.claim_ids -> claim.citation_ids` và render “Sources for this section”;
- đích nên làm: report prose dùng token được kiểm soát như `[@evd_xxx]`; validator kiểm tra token tồn tại và được claim/evidence synthesis trong section sử dụng, renderer mới đổi token thành `[n]`.

#### Acceptance criteria

- Report có external research thì luôn có ít nhất một citation hoặc một gap `no_relevant_evidence/provider_unavailable`.
- Click citation trong UI mở đúng `canonical_url`.
- HTML/PDF/Markdown có cùng thứ tự và nội dung references.
- Không render `javascript:`, `data:` hoặc URL không được allowlist.
- Citation id không resolve tạo validation error hoặc trạng thái gap rõ ràng.

#### Tests

- unit test numbering/dedup citation;
- renderer golden test cho Markdown và HTML;
- React test link target/rel và unsafe URL;
- contract test artifact chứa immutable reference snapshot;
- e2e: research → build report → click source.

### P0 — XAI-01: Hợp nhất pipeline explainer và sửa lỗi checkpoint

#### Giải pháp

Chỉ duy trì một application service để tạo/reuse explanation package cho cả analysis và report:

- thống nhất schema version mới, ví dụ `toxpred-explanation-v3`;
- cache key bắt buộc gồm canonical SMILES, endpoint, task, `model_id`, weights hash, tokenizer hash, attribution method và version của alignment;
- migration/read adapter cho observation `v1` và `v2` hiện có;
- analysis checkpoint thành cache dùng chung thay vì cache riêng mà report không đọc;
- khi cache payload đã có nhưng figure chưa có, chỉ tạo/sanitize/store figure; không chạy backward pass lại;
- chỉ cache `completed` hoặc `partial` có numeric attribution hợp lệ; không cache failure tạm thời;
- persist explanation observation, figure metadata và attachment theo một transaction logic nhất quán;
- report builder resolve package bằng target identity, không chỉ bằng schema string.

Thêm capability/readiness thực tế:

- model loaded và checksum đúng;
- provider có callable `token_attribution`;
- thử một smoke explanation nhỏ khi startup/admission hoặc trong diagnostics;
- trả reason code cụ thể: `model_unavailable`, `attribution_unsupported`, `alignment_failed`, `budget_exceeded`, `figure_store_failed`;
- phân biệt numeric explanation thành công nhưng figure lỗi với explanation computation lỗi.

#### Trình tự debug đề xuất

1. Gọi `/v1/models` và xác nhận model được load, capability đúng.
2. Gọi predictor `/v1/explanations` trực tiếp với đúng `model_id`, endpoint và task.
3. Kiểm tra response có `status`, atoms/bonds, signed contribution và `depiction_svg`.
4. Kiểm tra observation schema/version và cache key trong control DB.
5. Kiểm tra attachment/object tồn tại, SHA-256 đúng.
6. Kiểm tra report section tham chiếu đúng `explanation_id` và `figure_id`.

#### Acceptance criteria

- Explanation đã hoàn tất trong analysis được report reuse, không gọi backward pass lần hai.
- Đổi weights/tokenizer hash làm cache miss; cùng artifact cho cache hit.
- Không reuse explanation giữa model hoặc Tox21 task khác nhau.
- Figure lỗi không làm mất numeric contributors.
- Report hiển thị gap đúng reason code nếu explanation thật sự không thể tạo.

#### Tests

- compatibility test cho v1/v2 → v3;
- integration test analysis → report và assert predictor explain chỉ được gọi một lần;
- cache invalidation test theo weights/tokenizer/method/alignment version;
- timeout/retry test;
- failure classification test.

### P0 — REP-02: Render đầy đủ predictor, tables và figures trong app

#### Giải pháp

Mở rộng `ReportBlock.tsx` để render từ structured artifact:

- substance profile và structure depiction;
- predictor results dạng table/card;
- explanation figure, caption, legend và contributor table;
- external evidence synthesis và citation;
- integrated interpretation;
- conclusions, recommendations, limitations;
- provenance appendix dạng collapsible details;
- gaps nằm ngay section bị ảnh hưởng.

Không chỉ render `body_markdown`. `table_ids`, `figure_ids`, claims và các typed extras đều phải có component tương ứng.

Thêm endpoint có authorization theo scope:

```text
GET /v1/sessions/{session_id}/reports/{report_id}/figures/{figure_id}
```

Endpoint phải xác nhận figure thuộc report/session, đọc attachment qua object store, kiểm tra MIME/hash và trả cache headers phù hợp. Frontend fetch blob có auth và revoke object URL khi unmount.

Tạo structure figure riêng từ canonical SMILES trong report build. Không tái sử dụng explanation heatmap làm hình cấu trúc trung tính.

#### Acceptance criteria

- Mỗi `figure_id` trong section đều hiện đúng hình hoặc visible gap.
- Figure không thuộc report/session trả 404.
- Caption và alt text luôn có.
- Không có broken image khi refresh hoặc sau khi token hết hạn.
- Report UI vẫn đọc được trên mobile và khi artifact panel hẹp.

### P0 — REP-03: Sửa export HTML/PDF/Markdown

#### Giải pháp

- Trước khi gọi renderer, resolve toàn bộ figure attachments và truyền `figure_id -> sanitized SVG`.
- HTML/PDF inline SVG đã sanitize để file tự chứa và không fetch resource ngoài.
- PDF renderer phải nhận cùng `figure_svgs`; hiện `render_pdf()` đang tự gọi `render_html(artifact)` mà không có map hình.
- Markdown không nên để URI `figure:` không thể mở. Chọn một trong hai cách:
  - export `.zip` gồm `report.md` và thư mục `figures/`; hoặc
  - tạo URL tải figure ổn định có authorization phù hợp.
- References trong cả ba format dùng snapshot đã nhúng trong artifact.

#### Acceptance criteria

- HTML/PDF tải về xem được toàn bộ hình khi offline.
- Markdown bundle không có link figure chết.
- Cùng report version cho cùng claims, citation numbering, captions và limitations ở mọi format.

### P0 — UI-01: Không để composer che nội dung cuối session

#### Giải pháp ưu tiên

Đưa composer về normal flex flow:

```text
chat column (flex-col, h-full)
├── header (shrink-0)
├── transcript scroller (flex-1, min-h-0, overflow-y-auto)
└── composer region (shrink-0, safe-area padding)
```

Cụ thể:

- bỏ `absolute bottom-0` khỏi composer wrapper;
- bỏ `pb-48` cố định, chỉ giữ padding nội dung thông thường;
- composer region có nền/gradient riêng nhưng chiếm layout height thật;
- đặt nút “Tin nhắn mới” tương đối với transcript viewport, không dùng `bottom-24` cố định;
- thêm `padding-bottom: env(safe-area-inset-bottom)` trên mobile;
- giữ logic chỉ auto-scroll nếu người dùng đang gần cuối;
- dùng bottom sentinel + `scrollIntoView()` hoặc `ResizeObserver` để khi composer đổi chiều cao, message cuối vẫn nhìn thấy.

#### Acceptance criteria

- Ở cuối trang, đáy message/report cuối nằm phía trên composer và có khoảng cách tối thiểu 16 px.
- Composer 1–8 dòng, có attachment/context chip vẫn không che content.
- Resize desktop panel, mobile keyboard và orientation change không gây overlay.
- Khi người dùng đang đọc phía trên, message mới không kéo họ xuống; nút jump vẫn xuất hiện.
- Khi đang ở cuối, message streaming tiếp tục bám đáy.

#### Tests

- component test cho stick-to-bottom và composer resize;
- Playwright ở desktop/tablet/mobile;
- visual regression với report dài, composer nhiều dòng và mobile safe area.

### P0 — XAI-02: Plot đỏ/xanh lá theo signed contribution

#### Quy ước semantic

- đỏ: đóng góp dương, làm tăng logit/xác suất của class đang được giải thích;
- xanh lá: đóng góp âm, làm giảm logit/xác suất của class đang được giải thích;
- xám: gần 0 hoặc không đủ tín hiệu;
- luôn ghi rõ class/target, ví dụ “toward hERG blocker” thay vì chỉ ghi “positive”.

#### Cách tính màu

- lấy `signed_contribution`, không lấy `relative_importance` không dấu;
- scale đối xứng theo `max(abs(contribution))` trong molecule;
- độ đậm/alpha biểu diễn magnitude;
- atom và bond dùng cùng normalization và legend;
- có neutral epsilon để tránh tô màu nhiễu rất nhỏ;
- bump `SVG_PALETTE_VERSION` và `SVG_RENDERER_VERSION` để cache figure cũ không bị hiểu là palette mới.

Màu gợi ý:

- red: `#D73027`;
- green: `#1A9850`;
- neutral: `#BDBDBD`.

Vì đỏ–xanh lá khó phân biệt với một số người dùng, bổ sung:

- legend có nhãn `+ increases target` và `− decreases target`;
- bảng top contributors có dấu, atom index, symbol và giá trị;
- có thể dùng viền liền/đứt hoặc ký hiệu `+/-` ở top atoms, không dựa duy nhất vào màu.

#### Acceptance criteria

- Payload có cả contribution dương và âm tạo ra ít nhất một vùng đỏ và một vùng xanh.
- Đổi dấu contribution đổi đúng màu nhưng giữ magnitude tương ứng.
- Legend, caption và alt text nêu đúng target.
- Snapshot/golden SVG test ổn định giữa các lần chạy.

## 5. Các hạng mục bổ sung để report “đầy đủ hơn”

### P1 — Chất lượng khoa học và khả năng đọc

1. **Executive risk matrix theo endpoint**: endpoint/task, probability, threshold, label, applicability, explanation status và evidence relation. Không tạo aggregate “safe/unsafe”.
2. **Applicability/OOD panel**: nêu rõ input có nằm ngoài miền áp dụng hay không và lý do; đặt gần predictor results.
3. **Model vs literature concordance table**: `supports`, `contradicts`, `mixed`, `insufficient`; ghi organism, assay, dose và chất lượng nguồn.
4. **Evidence quality badge**: primary/secondary/database/regulatory, peer-review status nếu provider có dữ liệu, ngày truy xuất.
5. **Conflict section**: không ép một kết luận nếu nguồn mâu thuẫn; trình bày từng phía và khác biệt về assay/dose/species.
6. **Top contributor table**: positive/negative, atom/bond index, signed score, relative magnitude và unmapped mass.
7. **Method note**: attribution method, target logit, model id, artifact hash, alignment version và thời gian chạy.
8. **Threshold context**: threshold source là artifact hay override; nếu override phải nổi bật.
9. **Recommended next experiments**: assay xác nhận, counter-screen, dose-response, replicate; mỗi đề xuất có priority và basis claims.
10. **Human-readable provenance**: thông tin quan trọng hiển thị trước, raw JSON/hash đặt trong details để không làm report khó đọc.

### P1 — UX

1. Mục lục sticky và highlight section đang đọc.
2. Nút “Copy citation”, “Open source”, “Back to claim”.
3. Toggle xem `Summary / Full technical report` nhưng cùng dùng một artifact.
4. Skeleton riêng cho figures/references thay vì block loading chung.
5. Badge `Completed with gaps` có link nhảy tới từng gap.
6. Print stylesheet và page-break rules cho table/figure/caption.

### P2 — Governance và vận hành

1. Version history và diff giữa report rebuilds.
2. Export manifest chứa report hash và checksum từng figure.
3. Telemetry: cache-hit explanation, explain latency, figure-render failure, unresolved citation, export failure.
4. Retention policy rõ ràng cho evidence snapshot, figures và report renderings.
5. Regression fixture cố định cho một molecule có đủ dương/âm, external evidence và gap.

## 6. Thứ tự triển khai khuyến nghị

| Thứ tự | Hạng mục | Phụ thuộc | Kết quả |
|---|---|---|---|
| 1 | XAI-01 | Không | Loại bỏ lỗi/recompute do lệch schema và cache |
| 2 | XAI-02 | XAI-01 | Payload/figure có semantic đỏ–xanh đúng |
| 3 | REP-01 | Không | Artifact có reference snapshot và citation link |
| 4 | REP-02 | XAI-01, REP-01 | Report trong app có đủ data, hình và nguồn |
| 5 | REP-03 | REP-01, REP-02 | HTML/PDF/Markdown export hoàn chỉnh |
| 6 | UI-01 | Không, có thể chạy song song | Không còn content bị composer che |
| 7 | P1 quality/UX | Các P0 tương ứng | Report dễ đọc và hữu ích hơn |

Nếu chia sprint, nên coi XAI-01 + REP-01 + UI-01 là sprint ổn định dữ liệu/UX; XAI-02 + REP-02 + REP-03 là sprint hoàn thiện presentation/export.

## 7. Definition of Done toàn bộ epic

- Một report mới có đủ 11 section bắt buộc hoặc gap rõ ràng ở section tương ứng.
- Predictor values truy ngược được tới observation, field path, model id và artifact hash.
- Mọi fact lấy từ research có citation bấm được tới URL HTTPS đã chuẩn hoá.
- Report in-app, HTML, PDF và Markdown/bundle hiển thị cùng nội dung cốt lõi.
- Có structure image trung tính và explainer image riêng theo endpoint/task.
- Explainer phân biệt đóng góp tăng/giảm bằng đỏ/xanh, có legend và bảng số liệu.
- Explanation đã tính không bị recompute khi identity/cache key không đổi.
- Message/report cuối không bị composer che ở desktop, tablet hoặc mobile.
- Test unit, integration, contract, e2e và visual regression tương ứng đều pass.
- Không đưa ra kết luận tổng hợp “safe/unsafe”; limitations và evidence gaps luôn hiển thị.

## 8. Những file dự kiến bị tác động

Backend/control:

- `src/toxagent/application/explanation.py`
- `src/toxagent/application/create_analysis.py`
- `src/toxagent/application/submit_report_draft.py`
- `src/toxagent/domain/report.py`
- `src/toxagent/report/compiler.py`
- `src/toxagent/report/renderers.py`
- `src/toxagent/api/routes.py`
- migration/schema/repository cho artifact version mới nếu nhúng `references`

Backend/predictor:

- `src/toxpred/application/depiction.py`
- `src/toxpred/application/explain.py`
- capability/diagnostic tests cho provider attribution

Frontend:

- `src/components/transcript/ReportBlock.tsx`
- các component mới: `ReportFigure`, `ReportTable`, `ReportReferences`, `ExplanationLegend`
- `src/pages/WorkbenchPage.tsx`
- `src/hooks/useStickToBottom.ts`
- `src/lib/api/types.ts`
- `src/lib/api/endpoints.ts`

## 9. Quyết định cần product/design xác nhận

Các mục này không chặn việc sửa P0 về correctness, nhưng cần chốt trước khi polish UI:

1. Report mặc định dùng tiếng Việt hay giữ English-first như profile hiện tại?
2. Citation click mở thẳng external URL hay mở evidence detail nội bộ trước rồi mới mở nguồn?
3. Markdown được giao dưới dạng `.zip` tự chứa hay chấp nhận URL figure cần đăng nhập?
4. Red/green có cần thêm mode palette thân thiện với người mù màu hay luôn bật pattern/`+/-`?
5. Structure image có cần atom numbering mặc định hay chỉ bật trong phần explanation?

