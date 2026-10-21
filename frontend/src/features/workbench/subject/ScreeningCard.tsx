import type { ScreeningDetail } from '../../../shared/api/endpoints';

/**
 * Thẻ artifact của một activity screening (AG-UI, §10.2).
 *
 * Ba vùng tách nhau vì đó là ba loại dữ kiện khác nhau: giá trị mô hình dự
 * đoán, quyết định của policy, và trạng thái workflow. Không gộp để người đọc
 * không nhầm một quyết định policy thành một phép đo.
 *
 * `not_run` và `failed` luôn có nhãn chữ kèm câu nói rõ "không phải kết quả
 * âm tính" — màu sắc một mình không đủ, và một stage chưa chạy không bao giờ
 * được trình bày như đã kiểm tra và an toàn.
 *
 * Các trường lồng nhau của screening là JSON mở trong OpenAPI, nên đọc qua
 * các accessor hẹp dưới đây: thiếu trường thì nói "không rõ", không đoán.
 */
export function ScreeningCard({ screening }: { screening: ScreeningDetail }) {
  const subject = dict(screening.subject);
  const target = dict(subject.target);
  const estimate = screening.activity_estimate == null ? null : dict(screening.activity_estimate);
  const value = dict(estimate?.estimate);
  const interval = numbers(value.prediction_interval);
  const model = dict(estimate?.model);
  const applicability = dict(estimate?.applicability);
  const decision = screening.prioritisation_decision == null
    ? null
    : dict(screening.prioritisation_decision);
  const policy = dict(decision?.policy);
  const stages = dict(screening.stages);
  const activity = dict(stages.activity);
  const prioritisation = dict(stages.prioritisation);
  const toxicity = dict(stages.toxicity);

  return (
    <section className="space-y-3 text-sm" aria-label="Thẻ screening" style={{ color: 'var(--text)' }}>
      <header className="space-y-1">
        <h3 className="text-sm font-semibold">
          {text(target.target_id) ?? 'Không rõ target'}
          {text(target.variant) ? ` (${text(target.variant)})` : ''} ·{' '}
          {text(subject.assay_context) ?? 'không rõ bối cảnh assay'}
        </h3>
        <p className="font-mono text-xs" style={{ color: 'var(--text-muted)' }}>
          {text(subject.canonical_smiles)}
        </p>
        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
          {screening.mode === 'replay'
            ? 'Đọc lại từ bản ghi đã đóng băng (không chạy mô hình).'
            : 'Đang phục vụ trực tiếp.'}
        </p>
      </header>

      <div className="space-y-1">
        <h4 className="text-xs font-semibold uppercase" style={{ color: 'var(--text-muted)' }}>
          Ước lượng hoạt tính
        </h4>
        {estimate == null ? (
          <p>
            Không có ước lượng:{' '}
            {text(activity.status) === 'failed'
              ? `stage hoạt tính thất bại (${text(activity.failure_code) ?? 'không rõ mã'}).`
              : 'stage hoạt tính chưa tạo ra ước lượng nào.'}{' '}
            <strong>Đây không phải kết quả âm tính hay bằng chứng không hoạt tính.</strong>
          </p>
        ) : (
          <>
            <p>
              pIC50 dự đoán <strong>{text(value.predicted_pic50) ?? String(value.predicted_pic50)}</strong>
              {interval.length === 2
                ? ` · khoảng ${Math.round((number(value.interval_level) ?? 0) * 100)}%: ${interval[0]}–${interval[1]}`
                : ''}
            </p>
            <p style={{ color: 'var(--text-muted)' }}>
              Mô hình {text(model.model_id) ?? 'không rõ'}{' '}
              {screening.model_admitted
                ? '(đã được admit)'
                : '(chưa được admit — chưa đủ thẩm định để ra quyết định)'}
              {'. '}
              Applicability: ligand {text(applicability.ligand) ?? 'không rõ'}, target{' '}
              {text(applicability.target) ?? 'không rõ'}, chung {text(applicability.joint) ?? 'không rõ'}.
            </p>
            <p style={{ color: 'var(--text-muted)' }}>
              pIC50 là tiềm lực in-vitro trên target và bối cảnh này, không phải hiệu lực điều trị. Khoảng dự đoán
              là độ phủ trung bình trên phần dư thẩm định, không phải bảo đảm cho riêng chất này.
            </p>
          </>
        )}
      </div>

      <div className="space-y-1">
        <h4 className="text-xs font-semibold uppercase" style={{ color: 'var(--text-muted)' }}>
          Quyết định ưu tiên (policy)
        </h4>
        {decision == null ? (
          <p>Chưa có quyết định policy nào cho screening này.</p>
        ) : (
          <>
            <p>
              <strong>{(text(decision.status) ?? 'không rõ').toUpperCase()}</strong>
              {' · '}
              {strings(decision.reason_codes).join(', ') || 'không có mã lý do'}
            </p>
            <p style={{ color: 'var(--text-muted)' }}>
              Ngưỡng pIC50 {number(policy.threshold_pic50) ?? 'không rõ'} (
              {text(policy.threshold_source) ?? 'không rõ nguồn'}). Đây là đầu ra của một quy tắc cố định,
              không phải một phép đo và không phải kết luận của người điều tra.
            </p>
          </>
        )}
      </div>

      <div className="space-y-1">
        <h4 className="text-xs font-semibold uppercase" style={{ color: 'var(--text-muted)' }}>
          Trạng thái các stage
        </h4>
        <ul className="space-y-1">
          <li>
            Hoạt tính: <strong>{stageLabel(text(activity.status))}</strong>
            {text(activity.failure_code) ? ` (${text(activity.failure_code)})` : ''}
          </li>
          <li>
            Ưu tiên: <strong>{stageLabel(text(prioritisation.status))}</strong>
            {text(prioritisation.decision) ? ` → ${text(prioritisation.decision)}` : ''}
          </li>
          <li>
            Độc tính: <strong>{stageLabel(text(toxicity.status))}</strong>
            {text(toxicity.not_run_reason) ? ` (${text(toxicity.not_run_reason)})` : ''}
            {text(toxicity.failure_code) ? ` (${text(toxicity.failure_code)})` : ''}
            {text(toxicity.status) !== 'completed' && (
              <>
                {' '}
                <strong>Chưa chạy hoặc thất bại không phải kết quả âm tính.</strong>
              </>
            )}
            {toxicity.frozen_analysis_available === true
              ? ' Có phân tích độc tính đã đóng băng kèm theo.'
              : ''}
          </li>
        </ul>
      </div>
    </section>
  );
}

type Dict = Record<string, unknown>;

function dict(value: unknown): Dict {
  return value !== null && typeof value === 'object' ? (value as Dict) : {};
}

function text(value: unknown): string | undefined {
  if (typeof value === 'string') return value;
  return typeof value === 'number' ? String(value) : undefined;
}

function number(value: unknown): number | undefined {
  return typeof value === 'number' ? value : undefined;
}

function numbers(value: unknown): number[] {
  return Array.isArray(value) ? value.filter((item): item is number => typeof item === 'number') : [];
}

function strings(value: unknown): string[] {
  return Array.isArray(value) ? value.map(String) : [];
}

function stageLabel(status: string | undefined): string {
  switch (status) {
    case 'completed':
      return 'đã chạy xong';
    case 'failed':
      return 'thất bại';
    case 'not_run':
      return 'chưa chạy';
    default:
      return 'không rõ';
  }
}
