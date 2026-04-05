import { ArrowLeft, Download, AlertTriangle, CheckCircle } from 'lucide-react';
import { Button } from './ui/button';
import type { FinalReport } from '../../lib/api';

interface ReportHeaderProps {
  finalReport: FinalReport;
  language: 'vi' | 'en';
  onNewAnalysis: () => void;
}

function getRiskStyle(riskLevel: string) {
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

function formatTimestamp(value: string | undefined, language: 'vi' | 'en') {
  if (!value) return 'N/A';
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString(language === 'vi' ? 'vi-VN' : 'en-GB', { hour12: false });
}

export function ReportHeader({ finalReport, language, onNewAnalysis }: ReportHeaderProps) {
  const metadata = finalReport.report_metadata;
  const clinical = finalReport.sections.clinical_toxicity;

  const verdict = finalReport.risk_level || 'UNKNOWN';
  const confidence = Number(clinical?.confidence ?? 0);
  const verdictStyle = getRiskStyle(verdict);

  const compoundName = metadata.compound_name || 'Unknown compound';
  const shownSmiles = metadata.canonical_smiles || metadata.smiles || 'N/A';

  return (
    <div className="border-b" style={{ borderColor: 'var(--border)' }}>
      <div className="max-w-[1400px] mx-auto px-10 py-8">
        {/* Action Buttons */}
        <div className="flex items-center justify-between mb-6">
          <Button
            variant="ghost"
            onClick={onNewAnalysis}
            className="text-sm"
            style={{ color: 'var(--text-muted)' }}
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            {language === 'vi' ? 'Phân tích mới' : 'New analysis'}
          </Button>
          <Button
            variant="outline"
            onClick={() => window.print()}
            className="text-sm"
            style={{ borderColor: 'var(--accent-blue)', color: 'var(--accent-blue)' }}
          >
            <Download className="w-4 h-4 mr-2" />
            {language === 'vi' ? 'Tải PDF' : 'Download PDF'}
          </Button>
        </div>

        {/* Molecule Info */}
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <h1 className="text-4xl font-bold mb-2" style={{ color: 'var(--text)' }}>
              {compoundName}
            </h1>
            <p className="text-base italic mb-3" style={{ color: 'var(--text-muted)' }}>
              {metadata.compound_name || (language === 'vi' ? 'Không có tên thường dùng' : 'No common name available')}
            </p>
            <p className="font-mono text-sm mb-2" style={{ color: 'var(--text-faint)', wordBreak: 'break-all' }}>
              {shownSmiles}
            </p>
            <p className="text-xs mt-3" style={{ color: 'var(--text-faint)' }}>
              {formatTimestamp(metadata.analysis_timestamp, language)}
            </p>
          </div>

          {/* Verdict Badge */}
          <div className="rounded-xl px-6 py-4" style={{ backgroundColor: verdictStyle.bg, border: `1px solid ${verdictStyle.color}` }}>
            <div className="flex items-center gap-2 mb-1">
              {verdict === 'LOW' ? (
                <CheckCircle className="w-6 h-6" style={{ color: verdictStyle.color }} />
              ) : (
                <AlertTriangle className="w-6 h-6" style={{ color: verdictStyle.color }} />
              )}
              <span className="text-2xl font-bold uppercase" style={{ color: verdictStyle.color }}>
                {verdict}
              </span>
            </div>
            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
              {language === 'vi' ? 'Độ tin cậy' : 'Confidence'}: {(confidence * 100).toFixed(0)}%
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
