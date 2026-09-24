# ToxAgent database migration runbook

Runbook này áp dụng cho control plane `toxagent-control` chạy PostgreSQL. Nó
quy định cách đưa một Alembic revision vào production; không phải hướng dẫn
khởi tạo database local, và không được thay thế bằng `metadata.create_all()`.

## Nguyên tắc không thương lượng

- **Forward-only ở production.** Không chạy `alembic downgrade` trên dữ liệu
  production. Nếu revision vừa deploy có lỗi, dùng hotfix forward-compatible,
  hoặc restore backup vào một database mới rồi chuyển traffic theo incident
  plan. Migration phải có `downgrade()` cho môi trường disposable/dev, không
  có nghĩa downgrade production được phép.
- **Một migration writer.** Migration job là một workload riêng, chỉ có một
  execution cho một database. App replica không chạy Alembic lúc startup.
- **Backup trước DDL/data change.** Không có backup đã kiểm tra khả năng
  restore thì migration chưa được phép bắt đầu.
- **Compatibility trước tối ưu.** Khi rolling deploy, schema phải đọc/ghi được
  bởi cả binary cũ lẫn binary mới trong toàn bộ compatibility window. Tách
  rename/drop/NOT NULL mới thành ít nhất hai release (expand → migrate traffic
  → contract).
- **Không log URL/credential.** `TOXAGENT_DATABASE_URL` chỉ có trong secret
  store/migration job; output chỉ được ghi revision, timestamp và deployment
  id.

## Điều kiện trước khi chạy

1. Pull request đã xanh job `postgres-migrations`: PostgreSQL trống được
   `alembic upgrade head`, sau đó integration/E2E repository path chạy trên
   schema đã migrate.
2. Review migration bằng tay: `upgrade()` là deterministic, có lock/timeout
   phù hợp, có ước lượng thời lượng và ảnh hưởng write lock. Data backfill lớn
   chạy bằng job resumable, có checkpoint, **không** nhét vào transaction DDL
   dài.
3. Revision đã chạy thành công trên staging clone đủ đại diện về kích thước và
   schema history. Ghi lại duration, lock/wait và query plan nếu có index mới.
4. Xác nhận backup PostgreSQL mới nhất nằm trong retention policy và một
   restore drill gần đây thành công. Nếu attachment/raw evidence bị migration
   tham chiếu, xác nhận snapshot object store tương ứng; database backup một
   mình không khôi phục blob.
5. Có owner on-call cho app và database, một cửa sổ thay đổi, ngưỡng abort,
   cùng deployment revision cũ để giữ/rollback app binary trong compatibility
   window.

## Thiết kế migration tương thích rolling deploy

| Pha | Được làm | Không được làm |
|---|---|---|
| Expand | thêm nullable column/table/index, code mới đọc fallback và dual-write khi cần | rename/drop column, thêm NOT NULL không backfill, đổi semantic payload đang được binary cũ ghi |
| Backfill | job checkpointed, rate limited, idempotent; quan sát lag/lock | một `UPDATE` toàn bảng trong migration deploy |
| Switch | bật read path mới sau khi tất cả replica đủ version và backfill hoàn tất | bỏ fallback khi replica cũ hoặc worker retry còn hoạt động |
| Contract | revision forward mới xoá path/column sau compatibility window và backup checkpoint mới | `downgrade` production hoặc contract trong cùng release expand |

`alembic upgrade head` phải được pin theo image/release đã review; không để
migration job lấy source branch thay đổi trong lúc chạy.

## Quy trình pre-deploy và migrate

Chạy từ working directory `toxagent-control` trong migration job có extra
`postgres` (bao gồm `psycopg` sync cho Alembic; app vẫn dùng `asyncpg`).

1. Scale/hold rollout app sao cho không có binary không tương thích được đưa
   vào trước migration. Không cần tắt replica cũ nếu revision là expand
   compatible.
2. Tạo backup theo công cụ managed PostgreSQL của môi trường; đợi trạng thái
   backup/snapshot là completed và ghi identifier vào change record.
3. Đặt `TOXAGENT_DATABASE_URL` từ secret runtime. Xác minh revision hiện tại:

   ```bash
   python -m alembic -c alembic.ini current
   python -m alembic -c alembic.ini history --verbose
   ```

4. Chạy duy nhất một job:

   ```bash
   python -m alembic -c alembic.ini upgrade head
   python -m alembic -c alembic.ini current
   ```

   `current` phải đúng revision head mà release manifest ghi. Lưu output đã
   redact, duration, backup id và image digest trong change record.
5. Roll out app binary compatible, chờ readiness của tất cả replica. Kiểm smoke
   read-only/API bằng identity vận hành đã được cấp; kiểm outbox, DB error rate
   và migration lock/wait metrics. Không đưa test payload khoa học vào dữ liệu
   người dùng production chỉ để smoke.
6. Sau compatibility window và khi không còn worker/retry binary cũ, mới mở
   change contract trong release forward kế tiếp.

## Sự cố và quyết định rollback

| Tình huống | Hành động |
|---|---|
| CI/staging fail trước production | Dừng. Sửa revision/source; không sửa tay production schema để “cho qua”. |
| Migration job fail trước commit | Giữ app ở binary tương thích, thu thập error/lock state đã redact, xác nhận DB revision rồi sửa forward. |
| Migration fail một phần hoặc data backfill sai | Dừng rollout, bảo toàn evidence/incident timeline. Chọn hotfix forward idempotent nếu integrity còn tốt; nếu không, restore backup vào database mới và làm restore drill/validation trước cutover. |
| App binary mới fail nhưng schema expand compatible | Roll back **app binary** về version cũ compatible; không downgrade schema. |
| Schema đã contract sai hoặc corruption | Kích hoạt incident/restore plan; không chạy Alembic downgrade trên database đang là source of truth. |

Sau mọi incident, reconcile `alembic current`, outbox sequence, run state và
object-store references trước khi mở lại write traffic. Postmortem phải ghi
backup id, revision, duration, lock symptom, dữ liệu ảnh hưởng và preventive
test/runbook update.

## Bằng chứng cần giữ cho mỗi migration

- PR/review link, migration revision và image digest.
- Log CI PostgreSQL migration/integration đã redact.
- Staging clone duration/lock observation.
- Backup identifier và ngày restore drill gần nhất.
- Production `current` trước/sau, thời gian job, người thực hiện và smoke
  result.
- Thời điểm compatibility window kết thúc trước khi contract schema.

Phần rộng hơn (deploy topology, secret rotation, backup/restore drill, SLO và
canary) thuộc W6/W9; runbook này chỉ chốt policy migration W4-05.
