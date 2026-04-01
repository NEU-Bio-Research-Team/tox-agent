import { ArrowLeft, Download, AlertTriangle } from 'lucide-react';
import { Button } from './ui/button';

interface ReportHeaderProps {
  smiles: string;
  onNewAnalysis: () => void;
}

export function ReportHeader({ smiles, onNewAnalysis }: ReportHeaderProps) {
  const verdict = 'NON-TOXIC';
  const confidence = 89;
  const verdictColor = 'var(--accent-green)';
  const verdictBg = 'rgba(34,197,94,0.12)';

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
            Phân tích mới
          </Button>
          <Button
            variant="outline"
            className="text-sm"
            style={{ borderColor: 'var(--accent-blue)', color: 'var(--accent-blue)' }}
          >
            <Download className="w-4 h-4 mr-2" />
            Tải PDF
          </Button>
        </div>

        {/* Molecule Info */}
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <h1 className="text-4xl font-bold mb-2" style={{ color: 'var(--text)' }}>
              Aspirin
            </h1>
            <p className="text-base italic mb-3" style={{ color: 'var(--text-muted)' }}>
              Acetylsalicylic Acid
            </p>
            <p className="font-mono text-sm mb-2" style={{ color: 'var(--text-faint)', wordBreak: 'break-all' }}>
              {smiles}
            </p>
            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
              CID: 2244 · MW: 180.16 · C9H8O4
            </p>
            <p className="text-xs mt-3" style={{ color: 'var(--text-faint)' }}>
              31/03/2026, 14:23:07 ICT
            </p>
          </div>

          {/* Verdict Badge */}
          <div className="rounded-xl px-6 py-4" style={{ backgroundColor: verdictBg, border: `1px solid ${verdictColor}` }}>
            <div className="flex items-center gap-2 mb-1">
              <AlertTriangle className="w-6 h-6" style={{ color: verdictColor }} />
              <span className="text-2xl font-bold uppercase" style={{ color: verdictColor }}>
                {verdict}
              </span>
            </div>
            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
              Confidence: {confidence}%
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
