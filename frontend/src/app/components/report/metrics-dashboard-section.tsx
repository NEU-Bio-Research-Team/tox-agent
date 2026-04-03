import type { FinalReport } from '../../../lib/api';

interface MetricsDashboardSectionProps {
  finalReport: FinalReport;
  language: 'vi' | 'en';
}

function formatMaybeNumber(value: number | undefined): string {
  if (typeof value !== 'number' || Number.isNaN(value)) return 'N/A';
  return value.toFixed(4);
}

function formatMaybePercent(value: number | undefined): string {
  if (typeof value !== 'number' || Number.isNaN(value)) return 'N/A';
  if (value > 1.0) return `${value.toFixed(1)}%`;
  return `${(value * 100).toFixed(1)}%`;
}

function firstDefinedNumber(values: Array<number | undefined>): number | undefined {
  for (const value of values) {
    if (typeof value === 'number' && Number.isFinite(value)) {
      return value;
    }
  }
  return undefined;
}

export function MetricsDashboardSection({ finalReport, language }: MetricsDashboardSectionProps) {
  const clinical = finalReport.sections.clinical_toxicity;
  const ood = finalReport.sections.ood_assessment;
  const inference = finalReport.sections.inference_context;
  const metrics = inference?.clinical_reference_metrics || {};

  const auc = metrics.test_auc_roc;
  const prAuc = metrics.test_pr_auc;
  const recallAt05 = firstDefinedNumber([
    metrics.toxic_recall_t_0_50,
    metrics.test_toxic_recall_t_0_50,
    metrics.recall_t_0_50,
  ]);
  const recallAt035 = firstDefinedNumber([
    metrics.toxic_recall_t_0_35,
    metrics.test_toxic_recall_t_0_35,
    metrics.recall_t_0_35,
  ]);
  const falseNegativesKnown = firstDefinedNumber([
    metrics.false_negatives_known_toxic,
    metrics.test_false_negatives_known_toxic,
  ]);
  const oodFlaggedMolecules = firstDefinedNumber([
    metrics.ood_flagged_molecules,
    metrics.test_ood_flagged_molecules,
  ]);
  const threshold = Number(clinical?.threshold_used ?? 0.35);
  const pToxic = Number(clinical?.probability ?? 0);
  const thresholdPolicy = String(inference?.threshold_policy || 'N/A').toUpperCase();
  const failureRegistrySize = Number(finalReport.sections.failure_registry?.registry_size ?? 0);
  const failureRegistryMatched = Boolean(finalReport.sections.failure_registry?.matched);
  const oodFlag = Boolean(ood?.flag);

  return (
    <section id="metrics">
      <h2 className="text-2xl font-bold mb-6" style={{ color: 'var(--text)' }}>
        {language === 'vi' ? '§0 Bảng chỉ số quyết định' : '§0 Decision Metrics Dashboard'}
      </h2>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="rounded-xl p-5" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
          <p className="text-xs uppercase mb-1" style={{ color: 'var(--text-muted)' }}>AUC-ROC</p>
          <p className="font-mono text-2xl font-semibold" style={{ color: 'var(--text)' }}>{formatMaybeNumber(auc)}</p>
          <p className="text-xs mt-1" style={{ color: 'var(--text-faint)' }}>
            {language === 'vi' ? 'Độ tốt ranking tổng quát của model tham chiếu.' : 'Reference model ranking quality.'}
          </p>
        </div>

        <div className="rounded-xl p-5" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
          <p className="text-xs uppercase mb-1" style={{ color: 'var(--text-muted)' }}>PR-AUC</p>
          <p className="font-mono text-2xl font-semibold" style={{ color: 'var(--text)' }}>{formatMaybeNumber(prAuc)}</p>
          <p className="text-xs mt-1" style={{ color: 'var(--text-faint)' }}>
            {language === 'vi' ? 'Hữu ích khi dữ liệu mất cân bằng toxic/non-toxic.' : 'Useful under toxic/non-toxic class imbalance.'}
          </p>
        </div>

        <div className="rounded-xl p-5" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
          <p className="text-xs uppercase mb-1" style={{ color: 'var(--text-muted)' }}>Toxic Recall @ 0.50</p>
          <p className="font-mono text-2xl font-semibold" style={{ color: 'var(--text)' }}>{formatMaybePercent(recallAt05)}</p>
          <p className="text-xs mt-1" style={{ color: 'var(--text-faint)' }}>
            {language === 'vi' ? 'Độ nhạy toxic ở ngưỡng mặc định bảo thủ.' : 'Toxic sensitivity at conservative default threshold.'}
          </p>
        </div>

        <div className="rounded-xl p-5" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
          <p className="text-xs uppercase mb-1" style={{ color: 'var(--text-muted)' }}>Toxic Recall @ 0.35</p>
          <p className="font-mono text-2xl font-semibold" style={{ color: 'var(--text)' }}>{formatMaybePercent(recallAt035)}</p>
          <p className="text-xs mt-1" style={{ color: 'var(--text-faint)' }}>
            {language === 'vi' ? 'Độ nhạy toxic sau tối ưu ngưỡng safety-first.' : 'Toxic sensitivity after safety-first threshold tuning.'}
          </p>
        </div>

        <div className="rounded-xl p-5" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
          <p className="text-xs uppercase mb-1" style={{ color: 'var(--text-muted)' }}>
            {language === 'vi' ? 'Ngưỡng lâm sàng đang dùng' : 'Clinical threshold in use'}
          </p>
          <p className="font-mono text-2xl font-semibold" style={{ color: 'var(--text)' }}>{threshold.toFixed(2)}</p>
          <p className="text-xs mt-1" style={{ color: 'var(--text-faint)' }}>
            {language === 'vi' ? `Policy: ${thresholdPolicy}` : `Policy: ${thresholdPolicy}`}
          </p>
        </div>

        <div className="rounded-xl p-5" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
          <p className="text-xs uppercase mb-1" style={{ color: 'var(--text-muted)' }}>p_toxic</p>
          <p className="font-mono text-2xl font-semibold" style={{ color: 'var(--text)' }}>{pToxic.toFixed(4)}</p>
          <p className="text-xs mt-1" style={{ color: 'var(--text-faint)' }}>
            {language === 'vi' ? 'Xác suất từ mô hình lâm sàng cho molecule hiện tại.' : 'Clinical model probability for the current molecule.'}
          </p>
        </div>

        <div className="rounded-xl p-5" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
          <p className="text-xs uppercase mb-1" style={{ color: 'var(--text-muted)' }}>
            {language === 'vi' ? 'False Negatives (known toxic)' : 'False Negatives (known toxic)'}
          </p>
          <p className="font-mono text-2xl font-semibold" style={{ color: 'var(--text)' }}>
            {typeof falseNegativesKnown === 'number' ? String(falseNegativesKnown) : String(failureRegistrySize || 'N/A')}
          </p>
          <p className="text-xs mt-1" style={{ color: 'var(--text-faint)' }}>
            {language === 'vi'
              ? (failureRegistryMatched ? 'Molecule hiện tại khớp failure registry.' : 'Số mẫu known toxic đang được registry theo dõi.')
              : (failureRegistryMatched ? 'Current molecule matches failure registry.' : 'Number of known-toxic samples tracked in registry.')}
          </p>
        </div>

        <div className="rounded-xl p-5" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
          <p className="text-xs uppercase mb-1" style={{ color: 'var(--text-muted)' }}>
            {language === 'vi' ? 'OOD-flagged molecules' : 'OOD-flagged molecules'}
          </p>
          <p
            className="font-mono text-2xl font-semibold"
            style={{ color: oodFlag ? 'var(--accent-red)' : 'var(--accent-green)' }}
          >
            {typeof oodFlaggedMolecules === 'number' ? String(oodFlaggedMolecules) : (ood?.ood_risk || 'LOW')}
          </p>
          <p className="text-xs mt-1" style={{ color: 'var(--text-faint)' }}>
            {language === 'vi'
              ? (oodFlag ? 'Molecule hiện tại nằm ngoài phân phối huấn luyện.' : 'Nếu thiếu số lượng tổng, đây là trạng thái OOD của mẫu hiện tại.')
              : (oodFlag ? 'Current molecule is out-of-distribution.' : 'If aggregate count is unavailable, this is current molecule OOD status.')}
          </p>
        </div>
      </div>
    </section>
  );
}
