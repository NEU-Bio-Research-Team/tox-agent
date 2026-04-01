import { useNavigate } from 'react-router';
import { Navbar } from '../components/navbar';
import { ReportHeader } from '../components/report-header';
import { ReportSidebar } from '../components/report-sidebar';
import { ClinicalToxicitySection } from '../components/report/clinical-toxicity-section';
import { MechanismProfilingSection } from '../components/report/mechanism-profiling-section';
import { StructuralExplanationSection } from '../components/report/structural-explanation-section';
import { LiteratureContextSection } from '../components/report/literature-context-section';
import { AIRecommendationsSection } from '../components/report/ai-recommendations-section';
import { useReport } from '../../lib/ReportContext';

export function ReportPage() {
  const navigate = useNavigate();
  const { report, setReport, error } = useReport();

  if (!report?.final_report) {
    return (
      <div style={{ minHeight: '100vh', backgroundColor: 'var(--bg)', fontFamily: 'Inter, sans-serif' }}>
        <Navbar />
        <main className="max-w-3xl mx-auto px-6 py-16">
          <div
            className="rounded-xl p-6"
            style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}
          >
            <h1 className="text-2xl font-bold mb-3" style={{ color: 'var(--text)' }}>
              Chua co du lieu bao cao
            </h1>
            <p className="mb-6" style={{ color: 'var(--text-muted)' }}>
              Hay quay lai trang chinh va chay phan tich truoc khi mo trang report.
            </p>
            {error && (
              <p className="mb-6" style={{ color: 'var(--accent-red)' }}>
                Loi gan day: {error}
              </p>
            )}
            <button
              onClick={() => navigate('/')}
              className="px-4 py-2 rounded-lg"
              style={{ backgroundColor: 'var(--accent-blue)', color: '#fff' }}
            >
              Ve trang phan tich
            </button>
          </div>
        </main>
      </div>
    );
  }

  const finalReport = report.final_report;

  return (
    <div style={{ minHeight: '100vh', backgroundColor: 'var(--bg)', fontFamily: 'Inter, sans-serif' }}>
      <Navbar />
      
      <ReportHeader
        finalReport={finalReport}
        onNewAnalysis={() => {
          setReport(null);
          navigate('/');
        }}
      />

      <div className="grid grid-cols-1 lg:grid-cols-[280px_1fr] max-w-[1400px] mx-auto">
        {/* Sidebar */}
        <ReportSidebar finalReport={finalReport} />

        {/* Main Content */}
        <main className="p-10 space-y-12 max-w-[860px]">
          <ClinicalToxicitySection data={finalReport.sections.clinical_toxicity} />
          <MechanismProfilingSection data={finalReport.sections.mechanism_toxicity} />
          <StructuralExplanationSection data={finalReport.sections.structural_explanation} />
          <LiteratureContextSection data={finalReport.sections.literature_context} />
          <AIRecommendationsSection
            summary={finalReport.executive_summary}
            recommendations={finalReport.sections.recommendations || []}
            riskLevel={finalReport.risk_level}
          />
        </main>
      </div>
    </div>
  );
}
