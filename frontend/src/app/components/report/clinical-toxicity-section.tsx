export function ClinicalToxicitySection() {
  const pToxic = 0.23;
  const confidence = 0.89;
  const threshold = 0.50;

  return (
    <section id="clinical">
      <h2 className="text-2xl font-bold mb-6" style={{ color: 'var(--text)' }}>
        §1 Clinical Toxicity
      </h2>

      <div className="grid grid-cols-1 md:grid-cols-[1fr_2fr] gap-6">
        {/* Gauge */}
        <div className="rounded-xl p-6" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
          <div className="relative mb-4">
            <svg viewBox="0 0 200 200" className="w-full max-w-[200px] mx-auto">
              {/* Background circle */}
              <circle
                cx="100"
                cy="100"
                r="80"
                fill="none"
                stroke="var(--border)"
                strokeWidth="16"
              />
              {/* Progress arc */}
              <circle
                cx="100"
                cy="100"
                r="80"
                fill="none"
                stroke="var(--accent-green)"
                strokeWidth="16"
                strokeDasharray={`${pToxic * 502} 502`}
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
                style={{ fontSize: '36px', fill: 'var(--accent-green)' }}
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
                <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Non-toxic (0-0.3)</span>
              </div>
              <div className="flex items-center gap-2 mb-1">
                <div className="w-3 h-3 rounded-full" style={{ backgroundColor: 'var(--accent-yellow)' }} />
                <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Warning (0.3-0.7)</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full" style={{ backgroundColor: 'var(--accent-red)' }} />
                <span className="text-xs" style={{ color: 'var(--text-muted)' }}>Toxic (0.7-1.0)</span>
              </div>
            </div>
          </div>
        </div>

        {/* Details */}
        <div className="space-y-4">
          <div className="rounded-xl p-5" style={{ backgroundColor: 'rgba(34,197,94,0.08)', border: '1px solid var(--accent-green)' }}>
            <div className="text-xl font-bold mb-1" style={{ color: 'var(--accent-green)' }}>
              NON-TOXIC
            </div>
            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>Toxicity Label</p>
          </div>

          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm" style={{ color: 'var(--text-muted)' }}>Confidence</span>
              <span className="font-semibold" style={{ color: 'var(--text)' }}>{(confidence * 100).toFixed(0)}%</span>
            </div>
            <div className="w-full h-2 rounded-full" style={{ backgroundColor: 'var(--border)' }}>
              <div
                className="h-full rounded-full"
                style={{ 
                  width: `${confidence * 100}%`, 
                  backgroundColor: 'var(--accent-blue)',
                  transition: 'width 600ms cubic-bezier(0.34, 1.56, 0.64, 1)'
                }}
              />
            </div>
          </div>

          <div className="rounded-lg p-4" style={{ backgroundColor: 'var(--surface-alt)' }}>
            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>Threshold used</p>
            <p className="font-mono text-lg font-semibold" style={{ color: 'var(--text)' }}>{threshold.toFixed(2)}</p>
          </div>

          <div className="rounded-lg p-4 italic" style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}>
            <p className="text-sm" style={{ color: 'var(--text-muted)', lineHeight: '1.7' }}>
              Phân tử này có xác suất độc tính thấp ({(pToxic * 100).toFixed(0)}%). 
              Phân loại: <strong style={{ color: 'var(--accent-green)' }}>NON-TOXIC</strong> với độ tin cậy {(confidence * 100).toFixed(0)}%.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
