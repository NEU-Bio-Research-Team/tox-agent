import { motion } from 'motion/react';
import { ArrowRight, CheckCircle, AlertTriangle } from 'lucide-react';
import { Button } from './ui/button';

interface QuickVerdictCardProps {
  onViewReport: () => void;
}

export function QuickVerdictCard({ onViewReport }: QuickVerdictCardProps) {
  const pToxic = 0.23;
  const verdict = pToxic >= 0.7 ? 'TOXIC' : pToxic < 0.3 ? 'NON-TOXIC' : 'UNCERTAIN';
  const confidence = 0.89;
  const topRisk = { name: 'SR-HSE', score: 0.32, description: 'Stress Response - Heat Shock Element' };

  const verdictColor = verdict === 'TOXIC' ? 'var(--accent-red)' : verdict === 'NON-TOXIC' ? 'var(--accent-green)' : 'var(--accent-yellow)';
  const verdictBg = verdict === 'TOXIC' ? 'rgba(239,68,68,0.12)' : verdict === 'NON-TOXIC' ? 'rgba(34,197,94,0.12)' : 'rgba(245,158,11,0.12)';

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.4 }}
      className="mb-8 rounded-2xl p-7 shadow-2xl"
      style={{
        backgroundColor: 'var(--surface)',
        border: `1.5px solid ${verdictColor}`,
        boxShadow: `0 0 20px ${verdict === 'TOXIC' ? 'rgba(239,68,68,0.25)' : 'rgba(34,197,94,0.25)'}`
      }}
    >
      <h3 className="text-lg font-semibold mb-6" style={{ color: 'var(--text)' }}>
        KẾT QUẢ PHÂN TÍCH NHANH
      </h3>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-5 mb-6">
        {/* Verdict Badge */}
        <div className="rounded-xl p-5" style={{ backgroundColor: verdictBg, border: `1px solid ${verdictColor}` }}>
          <div className="flex items-center gap-2 mb-2">
            {verdict === 'NON-TOXIC' ? (
              <CheckCircle className="w-6 h-6" style={{ color: verdictColor }} />
            ) : (
              <AlertTriangle className="w-6 h-6" style={{ color: verdictColor }} />
            )}
            <span className="text-xl font-bold uppercase" style={{ color: verdictColor }}>
              {verdict}
            </span>
          </div>
          <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
            Độ tin cậy: {(confidence * 100).toFixed(0)}%
          </p>
        </div>

        {/* p_toxic Gauge */}
        <div className="rounded-xl p-5" style={{ backgroundColor: 'var(--surface-alt)', border: '1px solid var(--border)' }}>
          <div className="relative">
            <svg viewBox="0 0 120 60" className="w-full">
              {/* Track */}
              <path
                d="M 10 50 A 50 50 0 0 1 110 50"
                fill="none"
                stroke="var(--border)"
                strokeWidth="8"
                strokeLinecap="round"
              />
              {/* Fill arc */}
              <path
                d="M 10 50 A 50 50 0 0 1 110 50"
                fill="none"
                stroke={verdictColor}
                strokeWidth="8"
                strokeLinecap="round"
                strokeDasharray={`${pToxic * 157} 157`}
                style={{ transition: 'stroke-dasharray 800ms ease-out' }}
              />
            </svg>
            <div className="text-center -mt-2">
              <div className="font-mono text-2xl font-bold" style={{ color: verdictColor }}>
                {pToxic.toFixed(2)}
              </div>
              <div className="text-xs uppercase tracking-wide" style={{ color: 'var(--text-faint)' }}>
                p_toxic
              </div>
            </div>
          </div>
        </div>

        {/* Top Risk */}
        <div className="rounded-xl p-5" style={{ backgroundColor: 'var(--surface-alt)', border: '1px solid var(--border)' }}>
          <p className="text-sm mb-2" style={{ color: 'var(--text-muted)' }}>
            Rủi ro cao nhất
          </p>
          <div className="font-semibold text-base mb-2" style={{ color: 'var(--text)' }}>
            {topRisk.name}
          </div>
          <div className="w-full h-2 rounded-full mb-1" style={{ backgroundColor: 'var(--border)' }}>
            <div
              className="h-full rounded-full"
              style={{ width: `${topRisk.score * 100}%`, backgroundColor: verdictColor }}
            />
          </div>
          <div className="flex items-center justify-between">
            <span className="font-mono text-xs" style={{ color: 'var(--text-muted)' }}>
              {topRisk.score.toFixed(2)}
            </span>
          </div>
          <p className="text-xs mt-2" style={{ color: 'var(--text-faint)' }}>
            {topRisk.description}
          </p>
        </div>
      </div>

      {/* View Full Report CTA */}
      <Button
        onClick={onViewReport}
        className="w-full h-14 text-base font-semibold rounded-lg group"
        style={{
          background: 'linear-gradient(135deg, var(--accent-blue) 0%, #2563eb 100%)',
          color: '#ffffff'
        }}
      >
        Xem Báo Cáo Đầy Đủ
        <ArrowRight className="w-5 h-5 ml-2 transition-transform group-hover:translate-x-1" />
      </Button>
    </motion.div>
  );
}
