import { Copy, RefreshCw } from 'lucide-react';
import { Button } from '../ui/button';
import { normalizeRiskLevel } from '../../risk-level';
import type {
  FailureRegistrySection,
  OodAssessmentSection,
  RecommendationPayload,
  RiskLevel,
  RiskLevelCode,
  StructuredRecommendation,
} from '../../../lib/api';

interface AIRecommendationsSectionProps {
  summary: string;
  recommendations?: RecommendationPayload | null;
  riskLevel: RiskLevel;
  language: 'vi' | 'en';
  reliabilityWarning?: string | null;
  oodAssessment?: OodAssessmentSection;
  recommendationSource?: string;
  recommendationSourceDetail?: string;
  failureRegistry?: FailureRegistrySection;
  runtimeMode?: string;
  runtimeNote?: string | null;
}

interface NormalizedRecommendation {
  action: string;
  rationale?: string;
  priority?: 'HIGH' | 'MEDIUM' | 'LOW' | string;
  actionType?: 'experimental' | 'structural' | 'regulatory' | 'monitoring' | string;
}

const PRIORITY_STYLE: Record<string, { bg: string; text: string }> = {
  HIGH: { bg: 'rgba(239,68,68,0.12)', text: 'var(--accent-red)' },
  MEDIUM: { bg: 'rgba(245,158,11,0.14)', text: 'var(--accent-yellow)' },
  LOW: { bg: 'rgba(34,197,94,0.14)', text: 'var(--accent-green)' },
};

const ACTION_TYPE_STYLE: Record<string, string> = {
  experimental: 'var(--accent-blue)',
  structural: '#f59e0b',
  regulatory: '#ef4444',
  monitoring: '#22c55e',
};

function getRiskLabel(riskLevel: RiskLevelCode, language: 'vi' | 'en') {
  if (riskLevel === 'CRITICAL') return language === 'vi' ? 'Cảnh báo khẩn cấp' : 'Critical alert';
  if (riskLevel === 'HIGH') return language === 'vi' ? 'Rủi ro cao' : 'High risk';
  if (riskLevel === 'MODERATE') return language === 'vi' ? 'Rủi ro trung bình' : 'Moderate risk';
  if (riskLevel === 'LOW') return language === 'vi' ? 'Rủi ro thấp' : 'Low risk';
  return language === 'vi' ? 'Không xác định' : 'Unknown';
}

function normalizeRecommendations(
  recommendations: AIRecommendationsSectionProps['recommendations'],
): NormalizedRecommendation[] {
  if (Array.isArray(recommendations)) {
    const normalized: NormalizedRecommendation[] = [];
    for (const item of recommendations) {
      if (typeof item === 'string' && item.trim().length > 0) {
        normalized.push({ action: item.trim() });
        continue;
      }

      if (item && typeof item === 'object') {
        const structured = item as StructuredRecommendation;
        const action = typeof structured.action === 'string' ? structured.action.trim() : '';
        if (!action) {
          continue;
        }
        normalized.push({
          action,
          rationale: typeof structured.rationale === 'string' ? structured.rationale.trim() : undefined,
          priority: typeof structured.priority === 'string' ? structured.priority.toUpperCase() : undefined,
          actionType: typeof structured.action_type === 'string' ? structured.action_type.toLowerCase() : undefined,
        });
      }
    }
    return normalized;
  }

  if (recommendations && typeof recommendations === 'object') {
    const content = (recommendations as { content?: string }).content;
    if (typeof content === 'string' && content.trim().length > 0) {
      return [{ action: content.trim() }];
    }
  }

  return [];
}

export function AIRecommendationsSection({
  summary,
  recommendations,
  riskLevel,
  language,
  reliabilityWarning,
  oodAssessment,
  recommendationSource,
  recommendationSourceDetail,
  failureRegistry,
  runtimeMode,
  runtimeNote,
}: AIRecommendationsSectionProps) {
  const normalizedRisk = normalizeRiskLevel(riskLevel);
  const recommendationItems = normalizeRecommendations(recommendations);

  const handleCopy = async () => {
    const content = [
    summary,
    '',
    ...recommendationItems.map((item, index) => {
      const meta: string[] = [];
      if (item.priority) meta.push(`priority=${item.priority}`);
      if (item.actionType) meta.push(`type=${item.actionType}`);
      const metaText = meta.length > 0 ? ` [${meta.join(', ')}]` : '';
      const rationaleText = item.rationale ? `\n   rationale: ${item.rationale}` : '';
      return `${index + 1}. ${item.action}${metaText}${rationaleText}`;
    }),
  ].join('\n');
    try {
      await navigator.clipboard.writeText(content);
    } catch {
      // Ignore clipboard errors in unsupported environments.
    }
  };

  return (
    <section id="recommendations" className="scroll-mt-24 lg:scroll-mt-20">
      <h2 className="text-2xl font-bold mb-6" style={{ color: 'var(--text)' }}>
        {language === 'vi' ? '§6 Khuyến nghị AI' : '§6 AI Recommendations'}
      </h2>

      {failureRegistry?.matched && (
        <div
          className="rounded-xl p-4 mb-4"
          style={{
            backgroundColor: 'rgba(245,158,11,0.08)',
            border: '1px solid rgba(245,158,11,0.35)',
            color: 'var(--accent-yellow)',
          }}
        >
          <p className="font-semibold mb-1">
            {language === 'vi' ? 'Khớp Failure Registry' : 'Failure Registry Match'}
            {failureRegistry.entry?.id ? `: ${failureRegistry.entry.id}` : ''}
          </p>
          {failureRegistry.entry?.recommended_action && (
            <p className="text-sm">{failureRegistry.entry.recommended_action}</p>
          )}
        </div>
      )}

      {oodAssessment?.flag && (
        <div
          className="rounded-xl p-4 mb-4"
          style={{
            backgroundColor: 'rgba(239,68,68,0.08)',
            border: '1px solid rgba(239,68,68,0.35)',
            color: 'var(--accent-red)',
          }}
        >
          <p className="font-semibold mb-1">{language === 'vi' ? 'Cảnh báo OOD' : 'OOD Warning'}: {oodAssessment.ood_risk}</p>
          <p className="text-sm">{oodAssessment.reason}</p>
          {oodAssessment.recommendation && <p className="text-sm mt-1">{oodAssessment.recommendation}</p>}
        </div>
      )}

      {reliabilityWarning && !oodAssessment?.flag && (
        <div
          className="rounded-xl p-4 mb-4"
          style={{
            backgroundColor: 'rgba(245,158,11,0.08)',
            border: '1px solid rgba(245,158,11,0.35)',
            color: 'var(--accent-yellow)',
          }}
        >
          <p className="text-sm">{reliabilityWarning}</p>
        </div>
      )}

      {runtimeMode === 'deterministic_fallback' && (
        <div
          className="rounded-xl p-4 mb-4"
          style={{
            backgroundColor: 'rgba(59,130,246,0.08)',
            border: '1px solid rgba(59,130,246,0.35)',
            color: 'var(--accent-blue)',
          }}
        >
          <p className="font-semibold mb-1">
            {language === 'vi' ? 'Agent Runtime: Deterministic Fallback' : 'Agent Runtime: Deterministic fallback'}
          </p>
          <p className="text-sm">
            {language === 'vi'
              ? 'ADK runtime không khả dụng ở request này, hệ thống dùng pipeline deterministic để bảo toàn kết quả phân tích.'
              : 'ADK runtime was not available for this request; deterministic pipeline was used to preserve analysis output.'}
          </p>
          {runtimeNote && <p className="text-sm mt-1">{runtimeNote}</p>}
        </div>
      )}

      <div
        className="rounded-r-xl p-8"
        style={{
          background: 'linear-gradient(135deg, var(--surface) 0%, var(--bg) 100%)',
          borderLeft: '3px solid var(--accent-blue)',
        }}
      >
        <div className="mb-6 flex flex-wrap items-center gap-2">
          <svg className="w-3 h-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <rect x="3" y="3" width="18" height="18" rx="2" />
            <path d="M9 9h.01M15 9h.01M9 15h6" />
          </svg>
          <span className="text-xs uppercase tracking-wider" style={{ color: 'var(--text-faint)' }}>
            {language === 'vi' ? 'AI · WriterAgent · Tổng hợp báo cáo' : 'AI · WriterAgent · Report synthesis'}
          </span>
          <span className="text-xs px-2 py-1 rounded" style={{ backgroundColor: 'var(--surface-alt)', color: 'var(--text-muted)' }}>
            {language === 'vi' ? 'Nguồn' : 'Source'}: {recommendationSource || 'unknown'}
          </span>
          {recommendationSourceDetail && (
            <span className="text-xs px-2 py-1 rounded" style={{ backgroundColor: 'var(--surface-alt)', color: 'var(--text-muted)' }}>
              {recommendationSourceDetail}
            </span>
          )}
        </div>

        <div className="space-y-6">
          <div>
            <h3 className="text-xs font-bold uppercase mb-2" style={{ color: 'var(--text-muted)', letterSpacing: '0.08em' }}>
              {language === 'vi' ? 'TÓM TẮT ĐIỀU HÀNH' : 'EXECUTIVE SUMMARY'}
            </h3>
            <p className="text-base leading-relaxed" style={{ color: 'var(--text)', lineHeight: '1.7' }}>
              {summary || (language === 'vi' ? 'Không có executive_summary từ backend.' : 'No executive summary from backend.')}
            </p>
          </div>

          <div>
            <h3 className="text-xs font-bold uppercase mb-2" style={{ color: 'var(--text-muted)', letterSpacing: '0.08em' }}>
              {language === 'vi' ? 'MỨC RỦI RO' : 'RISK LEVEL'}
            </h3>
            <p className="text-base font-semibold" style={{ color: 'var(--text)' }}>
              {normalizedRisk.code} · {getRiskLabel(normalizedRisk.code, language)}
            </p>
            {normalizedRisk.description && (
              <p className="text-sm mt-1" style={{ color: 'var(--text-muted)' }}>
                {normalizedRisk.description}
              </p>
            )}
          </div>

          <div>
            <h3 className="text-xs font-bold uppercase mb-3" style={{ color: 'var(--text-muted)', letterSpacing: '0.08em' }}>
              {language === 'vi' ? 'KHUYẾN NGHỊ' : 'RECOMMENDATIONS'}
            </h3>
            <div className="space-y-2">
              {recommendationItems.length === 0 && (
                <p className="text-base" style={{ color: 'var(--text-muted)' }}>
                  {language === 'vi' ? 'Không có khuyến nghị bổ sung.' : 'No additional recommendations.'}
                </p>
              )}
              {recommendationItems.map((item, index) => {
                const priority = String(item.priority || '').toUpperCase();
                const actionType = String(item.actionType || '').toLowerCase();
                const priorityStyle = PRIORITY_STYLE[priority];
                const actionTypeColor = ACTION_TYPE_STYLE[actionType] || 'var(--text-muted)';

                return (
                  <div
                    key={`${index}-${item.action.slice(0, 20)}`}
                    className="rounded-xl border p-4"
                    style={{
                      backgroundColor: 'var(--surface)',
                      borderColor: 'var(--border)',
                    }}
                  >
                    <div className="mb-2 flex flex-wrap items-center gap-2">
                      <span className="text-xs font-mono" style={{ color: 'var(--text-faint)' }}>
                        #{index + 1}
                      </span>

                      {priorityStyle && (
                        <span
                          className="rounded-full px-2 py-0.5 text-xs font-semibold"
                          style={{
                            backgroundColor: priorityStyle.bg,
                            color: priorityStyle.text,
                          }}
                        >
                          {priority}
                        </span>
                      )}

                      {actionType && (
                        <span
                          className="rounded-full border px-2 py-0.5 text-xs font-mono"
                          style={{
                            borderColor: actionTypeColor,
                            color: actionTypeColor,
                          }}
                        >
                          {actionType}
                        </span>
                      )}
                    </div>

                    <p className="text-base font-semibold" style={{ color: 'var(--text)' }}>
                      {item.action}
                    </p>

                    {item.rationale && (
                      <p className="mt-2 text-sm" style={{ color: 'var(--text-muted)' }}>
                        {item.rationale}
                      </p>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        <div className="mt-8 flex flex-col gap-3 pt-6 sm:flex-row sm:items-center" style={{ borderTop: '1px solid var(--border)' }}>
          <Button variant="ghost" size="sm" className="text-sm" style={{ color: 'var(--text-muted)' }} disabled>
            <RefreshCw className="w-4 h-4 mr-2" />
            {language === 'vi' ? 'Tạo lại báo cáo' : 'Regenerate report'}
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleCopy}
            className="text-sm"
            style={{ color: 'var(--text-muted)' }}
          >
            <Copy className="w-4 h-4 mr-2" />
            {language === 'vi' ? 'Sao chép văn bản' : 'Copy text'}
          </Button>
        </div>
      </div>
    </section>
  );
}
