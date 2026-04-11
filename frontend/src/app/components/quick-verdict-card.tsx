import { motion } from 'motion/react';
import { ArrowRight, CheckCircle, AlertTriangle } from 'lucide-react';
import { Button } from './ui/button';
import type { FinalReport } from '../../lib/api';
import { normalizeRiskLevel } from '../risk-level';

interface QuickVerdictCardProps {
  finalReport: FinalReport;
  onViewReport: () => void;
}

function getRiskPalette(riskLevel: string) {
  if (riskLevel === 'CRITICAL' || riskLevel === 'HIGH') {
    return {
      color: 'var(--accent-red)',
      bg: 'rgba(239,68,68,0.12)',
    };
  }
  if (riskLevel === 'MODERATE') {
    return {
      color: 'var(--accent-yellow)',
      bg: 'rgba(245,158,11,0.12)',
    };
  }
  return {
    color: 'var(--accent-green)',
    bg: 'rgba(34,197,94,0.12)',
  };
}

export function QuickVerdictCard({ finalReport, onViewReport }: QuickVerdictCardProps) {
  const pToxic = Number(finalReport.sections.clinical_toxicity?.probability ?? 0);
  const confidence = Number(finalReport.sections.clinical_toxicity?.confidence ?? 0);
  const clinicalVerdict = finalReport.sections.clinical_toxicity?.verdict ?? 'UNKNOWN';
  const normalizedRisk = normalizeRiskLevel(finalReport.risk_level);
  const riskLevel = normalizedRisk.code;

  const taskScores = finalReport.sections.mechanism_toxicity?.task_scores ?? {};
  const sortedTasks = Object.entries(taskScores).sort((a, b) => (b[1] ?? 0) - (a[1] ?? 0));
  const highestRiskName =
    finalReport.sections.mechanism_toxicity?.highest_risk || sortedTasks[0]?.[0] || 'N/A';
  const highestRiskScore =
    taskScores[highestRiskName] ?? sortedTasks[0]?.[1] ?? 0;

  const palette = getRiskPalette(riskLevel);

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.4 }}
      className="mb-8 rounded-2xl p-7 shadow-2xl"
      style={{
        backgroundColor: 'var(--surface)',
        border: `1.5px solid ${palette.color}`,
        boxShadow: `0 0 20px ${palette.color}33`
      }}
    >
      <h3 className="text-lg font-semibold mb-6" style={{ color: 'var(--text)' }}>
        PREDICTION SUMMARIZE
      </h3>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-5 mb-6">
        {/* Verdict Badge */}
        <div className="rounded-xl p-5" style={{ backgroundColor: palette.bg, border: `1px solid ${palette.color}` }}>
          <div className="flex items-center gap-2 mb-2">
            {riskLevel === 'LOW' ? (
              <CheckCircle className="w-6 h-6" style={{ color: palette.color }} />
            ) : (
              <AlertTriangle className="w-6 h-6" style={{ color: palette.color }} />
            )}
            <span className="text-xl font-bold uppercase" style={{ color: palette.color }}>
              {riskLevel}
            </span>
          </div>
          <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
            Confidence: {(confidence * 100).toFixed(0)}%
          </p>
          {normalizedRisk.description && (
            <p className="text-xs mt-2" style={{ color: 'var(--text-faint)' }}>
              {normalizedRisk.description}
            </p>
          )}
          <p className="text-xs mt-2" style={{ color: 'var(--text-faint)' }}>
            Clinical verdict: {clinicalVerdict}
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
                stroke={palette.color}
                strokeWidth="8"
                strokeLinecap="round"
                strokeDasharray={`${pToxic * 157} 157`}
                style={{ transition: 'stroke-dasharray 800ms ease-out' }}
              />
            </svg>
            <div className="text-center -mt-2">
              <div className="font-mono text-2xl font-bold" style={{ color: palette.color }}>
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
            Top Risk Factor
          </p>
          <div className="font-semibold text-base mb-2" style={{ color: 'var(--text)' }}>
            {highestRiskName}
          </div>
          <div className="w-full h-2 rounded-full mb-1" style={{ backgroundColor: 'var(--border)' }}>
            <div
              className="h-full rounded-full"
              style={{ width: `${Number(highestRiskScore) * 100}%`, backgroundColor: palette.color }}
            />
          </div>
          <div className="flex items-center justify-between">
            <span className="font-mono text-xs" style={{ color: 'var(--text-muted)' }}>
              {Number(highestRiskScore).toFixed(2)}
            </span>
          </div>
          <p className="text-xs mt-2" style={{ color: 'var(--text-faint)' }}>
            Highest mechanism signal từ task_scores.
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
        View Full Report
        <ArrowRight className="w-5 h-5 ml-2 transition-transform group-hover:translate-x-1" />
      </Button>
    </motion.div>
  );
}
