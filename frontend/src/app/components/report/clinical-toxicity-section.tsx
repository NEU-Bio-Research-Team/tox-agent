import type { ClinicalSection } from '../../../lib/api';

interface ClinicalToxicitySectionProps {
  data: ClinicalSection;
  language: 'vi' | 'en';
}

function getClinicalColor(probability: number) {
  if (probability >= 0.7) return 'var(--accent-red)';
  if (probability >= 0.3) return 'var(--accent-yellow)';
  return 'var(--accent-green)';
}

export function ClinicalToxicitySection({ data, language }: ClinicalToxicitySectionProps) {
  const pToxic = Number(data?.probability ?? 0);
  const confidence = Number(data?.confidence ?? 0);
  const threshold = Number(data?.threshold_used ?? 0.35);
  const verdict = data?.verdict || 'UNKNOWN';
  const interpretation = data?.interpretation || (
    language === 'vi'
      ? 'Không có diễn giải chi tiết từ backend.'
      : 'No detailed interpretation from backend.'
  );

  const accentColor = getClinicalColor(pToxic);

  return (
    <section id="clinical">
      <h2 className="text-2xl font-bold mb-6" style={{ color: 'var(--text)' }}>
        {language === 'vi' ? '§1 Độc tính lâm sàng' : '§1 Clinical Toxicity'}
      </h2>

      <div className="grid grid-cols-1 md:grid-cols-[1fr_2fr] gap-6">
        <div className="rounded-xl p-6" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
          <div className="relative mb-4">
            <svg viewBox="0 0 200 200" className="w-full max-w-[200px] mx-auto">
              <circle cx="100" cy="100" r="80" fill="none" stroke="var(--border)" strokeWidth="16" />
              <circle
                cx="100"
                cy="100"
                r="80"
                fill="none"
                stroke={accentColor}
                strokeWidth="16"
                strokeDasharray={`${Math.min(Math.max(pToxic, 0), 1) * 502} 502`}
                strokeLinecap="round"
                transform="rotate(-90 100 100)"
                style={{ transition: 'stroke-dasharray 800ms ease-out' }}
              />
              <text
                x="100"
                y="100"
                textAnchor="middle"
                dominantBaseline="middle"
                className="font-mono font-bold"
                style={{ fontSize: '36px', fill: accentColor }}
              >
                {pToxic.toFixed(2)}
              </text>
              <text
                x="100"
                y="130"
                textAnchor="middle"
                className="text-xs uppercase"
                style={{ fill: 'var(--text-faint)', letterSpacing: '0.1em' }}
              >
                p_toxic
              </text>
            </svg>
          </div>

          <div className="space-y-3">
            <div>
              <div className="flex items-center gap-2 mb-1">
                <div className="w-3 h-3 rounded-full" style={{ backgroundColor: 'var(--accent-green)' }} />
                <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                  {language === 'vi' ? 'Ít độc (0-0.3)' : 'Non-toxic (0-0.3)'}
                </span>
              </div>
              <div className="flex items-center gap-2 mb-1">
                <div className="w-3 h-3 rounded-full" style={{ backgroundColor: 'var(--accent-yellow)' }} />
                <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                  {language === 'vi' ? 'Cảnh báo (0.3-0.7)' : 'Warning (0.3-0.7)'}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full" style={{ backgroundColor: 'var(--accent-red)' }} />
                <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                  {language === 'vi' ? 'Độc cao (0.7-1.0)' : 'Toxic (0.7-1.0)'}
                </span>
              </div>
            </div>
          </div>
        </div>

        <div className="space-y-4">
          <div className="rounded-xl p-5" style={{ backgroundColor: `${accentColor}22`, border: `1px solid ${accentColor}` }}>
            <div className="text-xl font-bold mb-1 uppercase" style={{ color: accentColor }}>
              {verdict}
            </div>
            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
              {language === 'vi' ? 'Nhãn độc tính' : 'Toxicity Label'}
            </p>
          </div>

          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm" style={{ color: 'var(--text-muted)' }}>
                {language === 'vi' ? 'Độ tin cậy' : 'Confidence'}
              </span>
              <span className="font-semibold" style={{ color: 'var(--text)' }}>{(confidence * 100).toFixed(0)}%</span>
            </div>
            <div className="w-full h-2 rounded-full" style={{ backgroundColor: 'var(--border)' }}>
              <div
                className="h-full rounded-full"
                style={{
                  width: `${Math.min(Math.max(confidence, 0), 1) * 100}%`,
                  backgroundColor: 'var(--accent-blue)',
                  transition: 'width 600ms cubic-bezier(0.34, 1.56, 0.64, 1)',
                }}
              />
            </div>
          </div>

          <div className="rounded-lg p-4" style={{ backgroundColor: 'var(--surface-alt)' }}>
            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
              {language === 'vi' ? 'Xác suất mô hình' : 'Model probability'}
            </p>
            <p className="font-mono text-lg font-semibold" style={{ color: 'var(--text)' }}>
              {pToxic.toFixed(4)}
            </p>
            <p className="text-xs mt-1" style={{ color: 'var(--text-faint)' }}>
              {language === 'vi' ? 'Ngưỡng sử dụng' : 'Threshold used'}: {threshold.toFixed(2)}
            </p>
          </div>

          <div className="rounded-lg p-4 italic" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
            <p className="text-sm" style={{ color: 'var(--text-muted)', lineHeight: '1.7' }}>
              {interpretation}
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
