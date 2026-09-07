# Kế hoạch tinh gọn workspace và bàn giao ToxAgent

**Ngày đánh giá:** 2026-09-07
**Phạm vi:** tổ chức Git/workspace, clone, cấu hình, khởi động và tài liệu bàn giao
**Ưu tiên số 1:** khách nhận repo có ít lựa chọn nhất nhưng vẫn chạy được, kiểm tra được và bảo trì được

## 1. Kết luận đề xuất

Giữ ToxAgent là **một monorepo**. Không tách frontend, control-plane, predictor và OCR thành nhiều repo ở lần bàn giao này.

Tạo một **bản phân phối dành cho khách hàng** chỉ chứa source và tài sản cần để chạy sản phẩm. Cách sạch nhất là một private repo bàn giao riêng, có một nhánh mặc định và các release tag. Nếu bắt buộc phải dùng remote hiện tại, dùng một orphan branch đã squash, ví dụ `handoff/v1`, và yêu cầu clone shallow đúng nhánh.

Chuẩn hóa đúng một happy path:

```bash
git clone <customer-repo-url> toxagent
cd toxagent
cp .env.example .env
./bin/toxagent setup
./bin/toxagent up
```

Sau đó khách chỉ truy cập một URL do lệnh `up` in ra. `./bin/toxagent doctor`, `status`, `logs`, `smoke`, `down` là toàn bộ bề mặt vận hành thông thường.

Docker Compose là cách chạy được hỗ trợ chính thức cho khách. Cách chạy Python/Conda trực tiếp vẫn có thể tồn tại trong tài liệu developer, nhưng không xuất hiện trong Quick Start.

## 2. Đánh giá hiện trạng

### 2.1 Điểm đã tốt

- Ranh giới runtime đã khá rõ: frontend, control-plane, ToxPred, ToxOCR và PostgreSQL là các process/container riêng.
- Mỗi service có Dockerfile riêng; ToxPred, ToxOCR và control-plane không dùng chung môi trường Python xung đột.
- `compose.yaml` hiện tại parse hợp lệ với `.env.stack.example`.
- CI đã kiểm tra từng service, migration PostgreSQL, container smoke, frontend và SBOM.
- Model manifest có checksum và provenance; đây là nền tảng tốt để bàn giao artifact an toàn.
- Cache, virtualenv, local data, Node modules và phần lớn model thí nghiệm đã được ignore.

### 2.2 Chưa đủ gọn để bàn giao

Hiện trạng **gọn về boundary kỹ thuật nhưng chưa gọn về trải nghiệm người nhận**.

1. README gốc vẫn giới thiệu repo là `ToxPred` predictor-only và nói frontend/agent layer đã bị loại bỏ. Trong khi cây hiện tại đã có lại `frontend/`, `toxagent-control/`, `toxocr/` và full-stack Compose. Khách không thể biết đâu là sản phẩm chính xác chỉ từ entry point của repo.

2. `compose.yaml`, các overlay và `.env.stack.example` đang là file chưa được Git theo dõi tại thời điểm đánh giá. Nếu push đúng HEAD hiện tại, khách clone sẽ không nhận được happy path mới.

3. `.env.example` đang được theo dõi nhưng thuộc kiến trúc cũ (`MODEL_SERVER_URL_AWS/GCP`, Gemini, local LLM). `.env.stack.example` mới lại mang tên phụ. Có hai nguồn cấu hình cạnh tranh và nguồn có tên chuẩn là nguồn sai cho full stack.

4. Có ít nhất ba cách setup đang cạnh tranh:
   - Conda + Uvicorn trong README, chỉ chạy predictor;
   - `scripts/run_local_phase3.sh`, cần hai Python environment, npm và OpenCode cài trên host;
   - Docker Compose cùng các overlay CPU/GPU/OpenCode.

5. Default Compose chưa thật sự là clone-and-run:
   - ToxOCR không được phép download checkpoint theo template mặc định nhưng chưa có bước provisioning đơn giản;
   - OpenCode vẫn là runtime ngoài Compose, cần URL, thư mục host, auth và model/provider đúng phiên bản;
   - nhiều secret/URL kỹ thuật bị đẩy thẳng sang người dùng;
   - frontend cần token nhưng template hiện không làm rõ cách cấp và dùng token.

6. Root hiện có nhiều nhóm nội dung khác mục đích: runtime source, training code, benchmark, test data, model thí nghiệm, ba file slide, audit, deployment cũ và tài liệu kế hoạch. Đây là workspace nghiên cứu/phát triển hợp lý, nhưng không phải cây bàn giao tối giản.

7. Snapshot hiện tại có 686 file được theo dõi, khoảng 43 MB dữ liệu checkout; khoảng 36 MB là checkpoint/hình/presentation/XLSX. Riêng máy phát triển có khoảng 1.8 GB `models/`, 1.1 GB `.cache/`, 181 MB `.data/` và hơn 500 MB `node_modules`, nhưng phần lớn là ignored và **không đi theo fresh clone**. Không nên đánh đồng độ lớn workspace local với kích thước bản bàn giao.

8. Artifact manifest vẫn khai báo ClinTox optional nhưng blocked vì thiếu tokenizer. Một capability chưa chạy được không nên xuất hiện trong bản khách hàng mặc định trừ khi UI/API diễn đạt rõ là unavailable.

9. Chưa có `LICENSE`, release/version policy và tài liệu customer configuration ngắn gọn ở root. Đây là thiếu sót bàn giao, không chỉ là vấn đề thẩm mỹ.

10. Deploy workflow hiện chủ yếu deploy predictor lên một branch cụ thể; nó chưa chứng minh full stack trong đúng hình dạng sẽ giao cho khách.

## 3. Nguyên tắc tinh gọn

Áp dụng các quy tắc sau khi quyết định giữ hay bỏ một file:

- Một tác vụ phổ biến chỉ có một cách chính thức để làm.
- Root chỉ chứa entry point; chi tiết nằm sau `apps/`, `infra/`, `docs/` hoặc `dev/`.
- Default phải chạy được; tính năng optional phải dùng profile/overlay và không chặn default.
- Giá trị không bí mật, ổn định phải có default trong source; `.env` chỉ chứa lựa chọn deployment và secret thực sự.
- Khách không phải hiểu internal topology để khởi động sản phẩm.
- Không đưa lịch sử nghiên cứu vào distribution chỉ vì có thể cần lại; Git nội bộ đã giữ lịch sử đó.
- Không dùng Git sparse-checkout, submodule hoặc script clone từng phần. Chúng giảm dung lượng nhưng tăng kiến thức Git mà khách phải sở hữu.
- Không tạo một “mega Python environment”; xung đột Torch của ToxPred và MolScribe là lý do chính đáng để giữ container riêng.

## 4. Hình dạng repo đích

```text
toxagent/
├── README.md
├── LICENSE
├── CHANGELOG.md
├── VERSION
├── .env.example
├── .gitignore
├── compose.yaml
├── bin/
│   └── toxagent
├── apps/
│   ├── frontend/
│   ├── control/
│   ├── predictor/
│   └── ocr/
├── artifacts/
│   └── predictor-manifest.yaml
├── infra/
│   ├── compose/
│   │   ├── gpu.yaml
│   │   └── external-opencode.yaml
│   └── cloud/                 # chỉ giữ target cloud thực sự bàn giao
├── docs/
│   ├── GETTING_STARTED.md
│   ├── CONFIGURATION.md
│   ├── OPERATIONS.md
│   ├── ARCHITECTURE.md
│   ├── MODEL_CARD.md
│   └── DEVELOPMENT.md
└── tests/
    └── smoke/
```

Đây là hình dạng đích về mặt sản phẩm. Khi triển khai, nên thực hiện theo hai bước để giảm rủi ro:

- **Bước A — tinh gọn bề mặt trước:** giữ nguyên đường dẫn package hiện tại, sửa README/config/Compose/command wrapper và loại nội dung không giao. Bản này đã đủ để bàn giao.
- **Bước B — gom vật lý vào `apps/`:** chỉ thực hiện sau khi Bước A xanh hoàn toàn. Việc move source làm đổi Docker context, import path, CI path filter, scripts và tài liệu; không nên trộn nó với thay đổi onboarding trong cùng một commit.

Nếu deadline ngắn, dừng ở Bước A. “Gọn cho khách” quan trọng hơn một cây thư mục đẹp nhưng vừa trải qua refactor lớn.

## 5. Phân loại nội dung hiện tại

| Nhóm | Xử lý trong bản bàn giao | Ghi chú |
|---|---|---|
| `frontend/` | Giữ | app runtime |
| `toxagent-control/` | Giữ, sau đó đổi thành `apps/control/` | app runtime + migration + profile |
| `toxpred/` | Giữ, sau đó đổi thành `apps/predictor/toxpred/` | predictor API |
| `toxocr/` | Giữ | OCR service |
| `backend/` | Tách phần ToxPred import thực sự vào predictor | hiện request path còn import model factory từ đây |
| `artifacts/predictor-manifest.yaml` | Giữ và đổi tên rõ | nguồn sự thật của runtime model |
| `models/` | Chỉ giữ production artifact cần thiết hoặc provisioning manifest | bỏ checkpoint thí nghiệm khỏi distribution |
| `config/` | Chỉ giữ config mà production predictor tham chiếu | training config chuyển sang dev repo/branch |
| `tests/` và test từng app | Giữ test contract/unit cần bảo trì | không cần ship fixture benchmark lớn nếu khách chỉ vận hành |
| `benchmarks/`, `test_data/` | Chuyển sang nhánh/repo nội bộ hoặc `dev/` tùy hợp đồng | không nằm trong default handoff nếu khách không retrain/evaluate |
| `scripts/` | Chỉ giữ script vận hành/release | script train/eval chuyển sang `dev/` hoặc repo research |
| `environment.yml`, root `requirements.txt` | Không dùng trong customer Quick Start | giữ trong developer distribution nếu khách cần phát triển model |
| `docs/spec/`, `docs/archive/`, `audit_5_9.md` | Không giao trong default branch | lưu repo nội bộ hoặc release evidence bundle |
| PPTX, slide assets/build scripts | Không giao trong source distribution | phát hành riêng trong handover documents |
| `cloudbuild.tox-agent.yaml`, Firebase/GCP cũ | Chỉ giữ nếu đó là deployment target đã ký với khách | tránh nhiều cloud path cùng được coi là chuẩn |
| local ignored: `.cache/`, `.data/`, `.venv/`, `node_modules/`, `dist/`, logs | Không commit, không copy | có lệnh `clean` riêng nếu cần; không tự xóa dữ liệu user |

## 6. Chiến lược Git và remote

### 6.1 Phương án khuyến nghị: delivery repo riêng

Tạo private repo như `toxagent-delivery`, chỉ có:

- một default branch `main`;
- lịch sử bắt đầu từ snapshot đã sanitize;
- tag bất biến `v1.0.0`, `v1.0.1`, ...;
- branch protection, required CI và CODEOWNERS;
- không có experimental branches, secret đã xóa, binary lịch sử và tài liệu nội bộ.

Ưu điểm quyết định: khách chỉ cần `git clone <url>`, không cần nhớ branch hay `--depth`; repo mà họ có quyền xem cũng chính là phạm vi đã bàn giao.

Không dùng `git filter-repo` trực tiếp lên remote phát triển đang dùng. Xuất snapshot đã kiểm tra sang repo mới để không rewrite lịch sử của team.

### 6.2 Fallback nếu phải dùng cùng remote

Tạo orphan branch `handoff/v1`, squash thành snapshot sạch, bảo vệ branch và gắn release tag. Lệnh bàn giao:

```bash
git clone --branch handoff/v1 --single-branch --depth 1 <repo-url> toxagent
```

Đây chỉ là fallback. Người có quyền trên remote vẫn có thể fetch branch/lịch sử khác; vì vậy nó không tạo ranh giới bảo mật hay IP thật sự.

### 6.3 Trước khi publish

- Chốt hoặc cất an toàn 28 file modified và các file untracked hiện tại; tuyệt đối không refactor trên dirty worktree chưa phân loại.
- Chạy secret scan trên toàn snapshot và, nếu dùng cùng remote, trên toàn lịch sử reachable.
- Kiểm tra license của source, model weights, MolScribe, base images và OpenCode; thêm `THIRD_PARTY_NOTICES.md` nếu cần.
- Chốt quyền của khách: vận hành binary/image, đọc source, sửa source, retrain model hay phân phối lại. Phạm vi này quyết định có giao `dev/`, weights và training data hay không.
- Không giao `.env`, provider auth, service-account JSON, database dump hoặc `.data/opencode-home`.

## 7. Setup và config đích

### 7.1 Một file cấu hình duy nhất

Đổi `.env.stack.example` thành `.env.example`; xóa hoặc archive `.env.example` kiến trúc cũ. Template mới chia đúng ba phần:

1. **Required:** provider/model hoặc external runtime endpoint, một admin/bootstrap token, secret ký capability.
2. **Common optional:** frontend port, CPU/GPU, artifact location.
3. **Advanced:** không liệt kê trong template chính; tài liệu hóa trong `docs/CONFIGURATION.md` và để code/Compose dùng default.

Không bắt khách điền các internal URL như `http://toxpred:8080` hoặc `http://toxocr:8090`; các giá trị đó cố định trong Compose. `TOXAGENT_ENV`, runtime version, default timeout và tool budget cũng là versioned application defaults, không phải input onboarding.

`setup` phải sinh secret ngẫu nhiên nếu chúng chưa có, thay vì yêu cầu khách tự nghĩ chuỗi; lệnh không bao giờ ghi đè giá trị đã tồn tại.

### 7.2 Một topology mặc định

`compose.yaml` mặc định phải là full CPU evaluation stack và tự đủ để health check thành công. Hai overlay duy nhất:

- `infra/compose/gpu.yaml`: chỉ override accelerator/image/volume cần cho GPU;
- `infra/compose/external-opencode.yaml`: chỉ dùng khi khách đã có runtime OpenCode riêng.

Tránh `compose.opencode.yaml` mang ý nghĩa local bridge là một phần của Quick Start. Chi tiết host networking này thuộc developer/advanced operations.

### 7.3 Giải quyết hai blocker hiện tại

**ToxOCR checkpoint**

Chọn một trong hai cách và chỉ hỗ trợ một cách mặc định:

- tốt nhất cho onboarding: `setup` tải checkpoint đã pin vào `.artifacts/`, verify SHA-256, rồi Compose mount read-only;
- nếu hợp đồng cho phép: phát hành checkpoint trong một image/versioned artifact bundle riêng để `docker compose pull` là đủ.

Không để service tự download không kiểm soát ở request time. Production vẫn có thể cấm network hoàn toàn.

**OpenCode/provider**

Trước khi refactor cần một decision gate về quyền phân phối OpenCode 1.17.11:

- nếu được phép containerize: thêm runtime service đã pin digest/version vào default Compose; provider key được truyền bằng Docker secret hoặc `.env`;
- nếu không được phép: `setup`/`doctor` kiểm tra binary đúng version, auth và model, sau đó wrapper tự start bridge. Khách vẫn dùng một command, dù runtime nằm trên host;
- nếu khách cung cấp runtime endpoint: dùng overlay external và chỉ yêu cầu URL + model/provider.

Không ship một default Compose phụ thuộc vào runtime bên ngoài mà README không tự phát hiện và kiểm tra.

### 7.4 Command wrapper tối thiểu

`bin/toxagent` là shell wrapper mỏng quanh Docker Compose, không chứa business logic. Bề mặt đề xuất:

```text
setup    tạo/validate .env, provision artifact, không ghi đè secret
doctor   kiểm tra Docker/Compose, port, dung lượng, GPU tùy chọn, checksum, runtime auth
up       pull/build rồi start; chờ readiness; in URL và trạng thái
down     stop nhưng giữ database
status   hiển thị health của từng service
logs     gom log theo service
smoke    chạy một request tối thiểu end-to-end
backup   backup database/config cần bàn giao vận hành
restore  restore có xác nhận target
```

Không thêm Makefile/Taskfile như một lớp command thứ hai trừ khi chỉ alias về wrapper; thêm tool mới sẽ làm setup kém gọn.

## 8. Tài liệu bàn giao tối thiểu

Root `README.md` không vượt quá mức cần để người mới đạt lần chạy đầu tiên:

- sản phẩm là gì;
- prerequisites;
- năm lệnh Quick Start;
- URL và login/token ban đầu;
- link đến config/operations/development;
- limitations quan trọng (không phải medical decision system, model calibration, capability unavailable nếu có).

Các tài liệu còn lại có ranh giới rõ:

- `GETTING_STARTED.md`: clean-machine walkthrough và expected output;
- `CONFIGURATION.md`: bảng biến `name / required / default / secret / restart required / example`;
- `OPERATIONS.md`: logs, health, backup/restore, update/rollback, GPU và artifact rotation;
- `ARCHITECTURE.md`: service boundaries và data flow;
- `DEVELOPMENT.md`: chạy test/build và local source mode;
- `MODEL_CARD.md`: performance, provenance, intended use, limitations;
- `CHANGELOG.md`: thay đổi có ảnh hưởng khách theo version.

Spec, brainstorm, progress log và slide không link từ customer README và không nằm trong delivery repo mặc định.

## 9. Kế hoạch thực hiện theo commit

### Phase 0 — đóng băng phạm vi bàn giao

1. Commit/stash có chủ đích dirty worktree hiện tại.
2. Chốt ba decision: có giao training/evaluation không, artifact/model được phân phối bằng Git hay registry, OpenCode có được containerize không.
3. Lập allowlist file được giao; dùng allowlist thay vì cố duy trì denylist ngày càng dài.
4. Chụp baseline test, image digest và artifact checksum.

**Exit:** có một manifest nội dung bàn giao được product owner ký duyệt.

### Phase 1 — tạo một happy path, chưa move source

1. Commit Compose/overlay và template env mới.
2. Viết lại root README cho full stack.
3. Thêm `bin/toxagent` với `setup`, `doctor`, `up`, `down`, `status`, `logs`, `smoke`.
4. Provision ToxOCR/model artifact có checksum.
5. Giải quyết OpenCode theo decision Phase 0.
6. Thêm customer smoke test từ frontend URL qua control-plane đến predictor.

**Exit:** fresh clone trên máy sạch chạy bằng đúng Quick Start, không cần đọc source hay sửa YAML.

### Phase 2 — cắt distribution sạch

1. Loại slide, archive/spec nội bộ, audit, model thí nghiệm, training output và cloud path không dùng khỏi snapshot giao khách.
2. Chỉ giữ production artifact được manifest tham chiếu. Nếu giữ weights trong Git, giữ đúng bundle cần chạy; nếu dùng registry, bootstrap tải version + checksum cố định.
3. Thêm LICENSE, third-party notices, VERSION và CHANGELOG.
4. Tạo delivery repo hoặc orphan branch; gắn tag release.

**Exit:** normal clone chỉ nhận nội dung trong hợp đồng, không nhận lịch sử/dev debris.

### Phase 3 — gom cây thư mục vật lý

1. Move từng service vào `apps/`, mỗi commit một service.
2. Chuyển code model trong `backend/` mà predictor runtime thực sự import vào predictor; phần training sang `dev/` hoặc repo research.
3. Cập nhật Docker contexts, imports, manifest paths, CI filters và docs sau từng move.
4. Không đổi behavior/API trong phase này.

**Exit:** root còn tối đa khoảng 10–12 entry có ý nghĩa và mọi test baseline vẫn tương đương.

### Phase 4 — chứng minh khả năng bàn giao

1. CI checkout đúng delivery snapshot, không dùng cache ẩn.
2. Chạy `./bin/toxagent doctor`, `setup`, `up`, `smoke`, `down` trên clean Linux runner.
3. Test hai trường hợp: first run có network và restart không network với artifact đã cache.
4. Test migration từ release trước, backup/restore và rollback image/tag.
5. Một người chưa tham gia dự án thực hiện README mà không nhận hướng dẫn miệng; ghi lại mọi chỗ phải hỏi.

**Exit:** ký biên bản release/handover kèm tag, checksum, SBOM, known limitations và recovery procedure.

## 10. Tiêu chí nghiệm thu

Bản bàn giao chỉ được coi là gọn khi đạt tất cả điều kiện sau:

- Clone command không cần sparse-checkout/submodule và không cần biết lịch sử repo.
- Quick Start có tối đa năm command, chỉ một luồng chính thức.
- Một `.env.example`; không có template cũ cạnh tranh.
- Người dùng chỉ phải quyết định provider credential/model và thông tin deployment thực sự cần thiết.
- `doctor` báo rõ tất cả prerequisite trước khi build/start lâu.
- `up` hoặc thành công và in URL, hoặc thất bại với tên service + hành động sửa cụ thể.
- Default CPU stack đạt readiness; optional GPU/external runtime không ảnh hưởng default.
- Không có secret, local cache, database, log, auth OpenCode hoặc artifact thí nghiệm trong clone.
- README mô tả đúng full stack hiện có.
- Mọi image/artifact/dependency quan trọng được pin; model/checkpoint được verify checksum.
- CI chạy trên đúng shape bàn giao và có end-to-end smoke.
- Có backup/restore, upgrade/rollback và known limitations đủ để khách tự vận hành.

Chỉ số mục tiêu bổ sung:

- root ≤ 12 entry có ý nghĩa;
- một command facade;
- một default Compose + tối đa hai overlay;
- zero required manual edit trong YAML/source;
- time-to-first-healthy được đo và ghi trong release note;
- clean `git status` ngay sau `setup` nhờ mọi generated state nằm trong ignored directories.

## 11. Những việc không nên làm

- Không tách thành bốn repo service ở thời điểm bàn giao.
- Không đưa toàn bộ 1.8 GB model workspace local vào Git/LFS.
- Không giữ cả README predictor-only và README full-stack ở cấp root.
- Không coi Conda, host Python, Docker và cloud deploy là bốn Quick Start tương đương.
- Không move toàn bộ cây source và đổi onboarding trong cùng một commit lớn.
- Không xóa file ignored/local trước khi owner xác nhận; chúng không ảnh hưởng fresh clone và có thể chứa kết quả chưa backup.
- Không dùng default giả an toàn nhưng làm stack không thể ready, ví dụ tắt checkpoint download mà không có provisioning path.
- Không hardcode credential mẫu có vẻ dùng được; `setup` phải sinh secret hoặc yêu cầu credential thật một cách rõ ràng.

## 12. Quyết định cuối cùng

Setup hiện tại **chưa đủ gọn để giao trực tiếp**, dù nền tảng container/service boundary đã đúng hướng. Việc có giá trị cao nhất không phải đổi tên mọi thư mục ngay mà là:

1. tạo distribution sạch;
2. thống nhất Docker Compose thành một happy path;
3. hợp nhất cấu hình thành một `.env.example`;
4. tự động hóa artifact/OpenCode preflight qua một wrapper;
5. viết lại README đúng với full stack;
6. chứng minh mọi thứ từ fresh clone trong CI.

Sau khi sáu điểm trên hoàn tất, việc gom source vào `apps/` là bước làm đẹp và tăng khả năng bảo trì, không còn là blocker bàn giao. Đây là thứ tự vừa gọn cho khách, vừa tránh tạo rủi ro refactor không cần thiết ngay trước release.
