import { useEffect, useMemo, useState } from 'react';
import type { FinalReport } from '../../lib/api';
import { normalizeRiskLevel } from '../risk-level';

const sections = [
  { id: 'metrics', label: 'Metrics' },
  { id: 'clinical', label: 'Clinical Toxicity' },
  { id: 'mechanism', label: 'Mechanism' },
  { id: 'structural', label: 'Structural' },
  { id: 'molrag', label: 'MolRAG Evidence' },
  { id: 'literature', label: 'Literature' },
  { id: 'recommendations', label: 'AI Recommendations' },
];

interface ReportSidebarProps {
  finalReport: FinalReport;
  language: 'vi' | 'en';
}

function riskColor(riskLevel: string) {
  if (riskLevel === 'CRITICAL' || riskLevel === 'HIGH') return 'var(--accent-red)';
  if (riskLevel === 'MODERATE') return 'var(--accent-yellow)';
  return 'var(--accent-green)';
}

export function ReportSidebar({ finalReport, language }: ReportSidebarProps) {
  const [activeSection, setActiveSection] = useState('metrics');
  const pToxic = Number(finalReport.sections.clinical_toxicity?.probability ?? 0);
  const threshold = Number(finalReport.sections.clinical_toxicity?.threshold_used ?? 0.35);
  const normalizedRisk = normalizeRiskLevel(finalReport.risk_level);
  const riskLevel = normalizedRisk.code;
  const cid = finalReport.sections.literature_context?.compound_id?.cid;
  const assayHits = Number(finalReport.sections.mechanism_toxicity?.assay_hits ?? 0);
  const oodRisk = finalReport.sections.ood_assessment?.ood_risk || 'LOW';
  const compoundName =
    finalReport.report_metadata.compound_name ||
    finalReport.report_metadata.common_name ||
    finalReport.report_metadata.iupac_name ||
    finalReport.sections.literature_context?.query_name_used ||
    'N/A';
  const currentRiskColor = riskColor(riskLevel);

  const sectionLabels: Record<string, string> = {
    metrics: language === 'vi' ? 'B\u1ea3ng ch\u1ec9 s\u1ed1' : 'Metrics',
    clinical: language === 'vi' ? '\u0110\u1ed9c t\u00ednh l\u00e2m s\u00e0ng' : 'Clinical Toxicity',
    mechanism: language === 'vi' ? 'C\u01a1 ch\u1ebf \u0111\u1ed9c t\u00ednh' : 'Mechanism',
    structural: language === 'vi' ? 'Gi\u1ea3i th\u00edch c\u1ea5u tr\u00fac' : 'Structural',
    molrag: language === 'vi' ? 'MolRAG Evidence' : 'MolRAG Evidence',
    literature: language === 'vi' ? 'T\u00e0i li\u1ec7u tham kh\u1ea3o' : 'Literature',
    recommendations: language === 'vi' ? 'Khuy\u1ebfn ngh\u1ecb AI' : 'AI Recommendations',
  };

  const quickStats = useMemo(
    () => [
      {
        label: 'p_toxic',
        value: (
          <>
            <div className="mb-1 h-1.5 w-full rounded-full" style={{ backgroundColor: 'var(--border)' }}>
              <div
                className="h-full rounded-full"
                style={{ width: `${Math.min(Math.max(pToxic, 0), 1) * 100}%`, backgroundColor: currentRiskColor }}
              />
            </div>
            <p className="font-mono text-xs font-semibold" style={{ color: 'var(--text)' }}>
              {pToxic.toFixed(2)}
            </p>
          </>
        ),
      },
      {
        label: 'Label',
        value: (
          <>
            <p className="text-sm font-semibold" style={{ color: currentRiskColor }}>
              {riskLevel}
            </p>
            {normalizedRisk.description && (
              <p className="mt-1 text-xs" style={{ color: 'var(--text-faint)' }}>
                {normalizedRisk.description}
              </p>
            )}
          </>
        ),
      },
      {
        label: 'Clinical threshold',
        value: (
          <p className="font-mono text-sm font-semibold" style={{ color: 'var(--text)' }}>
            {threshold.toFixed(2)}
          </p>
        ),
      },
      {
        label: 'Assay hits',
        value: (
          <p className="text-sm font-semibold" style={{ color: 'var(--text)' }}>
            {assayHits}
          </p>
        ),
      },
      {
        label: 'OOD',
        value: (
          <p
            className="text-sm font-semibold"
            style={{
              color:
                oodRisk === 'HIGH'
                  ? 'var(--accent-red)'
                  : oodRisk === 'MEDIUM'
                    ? 'var(--accent-yellow)'
                    : 'var(--accent-green)',
            }}
          >
            {oodRisk}
          </p>
        ),
      },
      {
        label: 'CID',
        value: (
          <p className="font-mono text-sm font-semibold" style={{ color: 'var(--text)' }}>
            {cid ?? 'N/A'}
          </p>
        ),
      },
      {
        label: 'Compound',
        value: (
          <p className="text-sm font-semibold" style={{ color: 'var(--text)' }}>
            {compoundName}
          </p>
        ),
      },
    ],
    [assayHits, cid, compoundName, currentRiskColor, normalizedRisk.description, oodRisk, pToxic, riskLevel, threshold],
  );

  useEffect(() => {
    if (typeof window === 'undefined' || typeof IntersectionObserver === 'undefined') {
      return;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        const visibleEntries = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio);

        if (visibleEntries.length > 0) {
          setActiveSection(visibleEntries[0].target.id);
        }
      },
      {
        rootMargin: '-120px 0px -45% 0px',
        threshold: [0.15, 0.35, 0.6],
      },
    );

    const elements = sections
      .map((section) => document.getElementById(section.id))
      .filter((element): element is HTMLElement => element !== null);

    elements.forEach((element) => observer.observe(element));

    return () => {
      observer.disconnect();
    };
  }, []);

  return (
    <aside
      className="border-b px-4 py-4 md:px-6 lg:sticky lg:top-16 lg:h-[calc(100vh-4rem)] lg:border-r lg:border-b-0 lg:p-6 lg:overflow-y-auto"
      style={{ borderColor: 'var(--border)' }}
    >
      <div className="mb-0 lg:mb-6">
        <h3 className="mb-4 text-sm font-semibold uppercase" style={{ color: 'var(--text-muted)', letterSpacing: '0.05em' }}>
          {language === 'vi' ? 'N\u1ed9i dung b\u00e1o c\u00e1o' : 'Report Sections'}
        </h3>
        <nav className="flex gap-2 overflow-x-auto pb-1 lg:block lg:space-y-2 lg:overflow-visible lg:pb-0">
          {sections.map((section) => (
            <button
              key={section.id}
              onClick={() => {
                setActiveSection(section.id);
                if (typeof window !== 'undefined') {
                  document.getElementById(section.id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
              }}
              className="shrink-0 whitespace-nowrap rounded-lg px-3 py-2 text-left text-sm transition-colors lg:w-full"
              style={{
                backgroundColor: activeSection === section.id ? 'var(--accent-blue-muted)' : 'transparent',
                color: activeSection === section.id ? 'var(--accent-blue)' : 'var(--text-muted)',
              }}
            >
              {activeSection === section.id ? '●' : '○'} {sectionLabels[section.id] || section.label}
            </button>
          ))}
        </nav>
      </div>

      <div className="border-t pt-4 lg:pt-6" style={{ borderColor: 'var(--border)' }}>
        <h3 className="mb-3 text-xs font-semibold uppercase" style={{ color: 'var(--text-faint)', letterSpacing: '0.05em' }}>
          {language === 'vi' ? 'Th\u00f4ng tin nhanh' : 'Quick Stats'}
        </h3>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-1">
          {quickStats.map((stat) => (
            <div
              key={stat.label}
              className="rounded-lg p-3"
              style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}
            >
              <p className="mb-1 text-xs" style={{ color: 'var(--text-muted)' }}>
                {stat.label}
              </p>
              {stat.value}
            </div>
          ))}
        </div>
      </div>
    </aside>
  );
}
