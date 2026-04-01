import { Copy, RefreshCw } from 'lucide-react';
import { Button } from '../ui/button';
import type { RiskLevel } from '../../../lib/api';

interface AIRecommendationsSectionProps {
  summary: string;
  recommendations: string[];
  riskLevel: RiskLevel;
}

function getRiskLabel(riskLevel: RiskLevel) {
  if (riskLevel === 'CRITICAL') return 'Canh bao khan cap';
  if (riskLevel === 'HIGH') return 'Rui ro cao';
  if (riskLevel === 'MODERATE') return 'Rui ro trung binh';
  if (riskLevel === 'LOW') return 'Rui ro thap';
  return 'Khong xac dinh';
}

export function AIRecommendationsSection({ summary, recommendations, riskLevel }: AIRecommendationsSectionProps) {
  const handleCopy = async () => {
    const content = [summary, '', ...recommendations.map((item, index) => `${index + 1}. ${item}`)].join('\n');
    try {
      await navigator.clipboard.writeText(content);
    } catch {
      // Ignore clipboard errors in unsupported environments.
    }
  };

  return (
    <section id="recommendations">
      <h2 className="text-2xl font-bold mb-6" style={{ color: 'var(--text)' }}>
        §5 AI Recommendations
      </h2>

      <div
        className="rounded-r-xl p-8"
        style={{
          background: 'linear-gradient(135deg, var(--surface) 0%, var(--bg) 100%)',
          borderLeft: '3px solid var(--accent-blue)',
        }}
      >
        <div className="flex items-center gap-2 mb-6">
          <svg className="w-3 h-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <rect x="3" y="3" width="18" height="18" rx="2" />
            <path d="M9 9h.01M15 9h.01M9 15h6" />
          </svg>
          <span className="text-xs uppercase tracking-wider" style={{ color: 'var(--text-faint)' }}>
            AI · WriterAgent · Tong hop bao cao
          </span>
        </div>

        <div className="space-y-6">
          <div>
            <h3 className="text-xs font-bold uppercase mb-2" style={{ color: 'var(--text-muted)', letterSpacing: '0.08em' }}>
              TOM TAT DIEU HANH
            </h3>
            <p className="text-base leading-relaxed" style={{ color: 'var(--text)', lineHeight: '1.7' }}>
              {summary || 'Khong co executive_summary tu backend.'}
            </p>
          </div>

          <div>
            <h3 className="text-xs font-bold uppercase mb-2" style={{ color: 'var(--text-muted)', letterSpacing: '0.08em' }}>
              MUC RUI RO
            </h3>
            <p className="text-base font-semibold" style={{ color: 'var(--text)' }}>
              {riskLevel} · {getRiskLabel(riskLevel)}
            </p>
          </div>

          <div>
            <h3 className="text-xs font-bold uppercase mb-3" style={{ color: 'var(--text-muted)', letterSpacing: '0.08em' }}>
              KHUYEN NGHI
            </h3>
            <div className="space-y-2">
              {recommendations.length === 0 && (
                <p className="text-base" style={{ color: 'var(--text-muted)' }}>
                  Khong co khuyen nghi bo sung.
                </p>
              )}
              {recommendations.map((item, index) => (
                <div key={`${index}-${item.slice(0, 20)}`} className="flex items-start gap-3">
                  <span style={{ color: 'var(--accent-green)', fontSize: '18px' }}>•</span>
                  <p className="text-base flex-1" style={{ color: 'var(--text)' }}>
                    {item}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="flex items-center gap-3 mt-8 pt-6" style={{ borderTop: '1px solid var(--border)' }}>
          <Button variant="ghost" size="sm" className="text-sm" style={{ color: 'var(--text-muted)' }} disabled>
            <RefreshCw className="w-4 h-4 mr-2" />
            Tao lai bao cao
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleCopy}
            className="text-sm"
            style={{ color: 'var(--text-muted)' }}
          >
            <Copy className="w-4 h-4 mr-2" />
            Sao chep van ban
          </Button>
        </div>
      </div>
    </section>
  );
}
