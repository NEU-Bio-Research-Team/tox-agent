import { useNavigate } from 'react-router';
import { Navbar } from '../components/navbar';
import { AIChatbot } from '../components/ai-chatbot';
import { Footer } from '../components/footer';
import { ReportHeader } from '../components/report-header';
import { ReportSidebar } from '../components/report-sidebar';
import { ClinicalToxicitySection } from '../components/report/clinical-toxicity-section';
import { MechanismProfilingSection } from '../components/report/mechanism-profiling-section';
import { MetricsDashboardSection } from '../components/report/metrics-dashboard-section';
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
              There's no report to display
            </h1>
            <p className="mb-6" style={{ color: 'var(--text-muted)' }}>
              Please return to the analysis page and submit a compound to generate a toxicity report.
            </p>
            {error && (
              <p className="mb-6" style={{ color: 'var(--accent-red)' }}>
                Error: {error}
              </p>
            )}
            <button
              onClick={() => navigate('/')}
              className="px-4 py-2 rounded-lg"
              style={{ backgroundColor: 'var(--accent-blue)', color: '#fff' }}
            >
              Return to Analysis Page
            </button>
          </div>
        </main>
      </div>
    );
  }

  const finalReport = report.final_report;
  const reportLanguage = finalReport.report_metadata?.language === 'vi' ? 'vi' : 'en';

  return (
    <div style={{ minHeight: '100vh', backgroundColor: 'var(--bg)', fontFamily: 'Inter, sans-serif' }}>
      <Navbar />
      
      <ReportHeader
        finalReport={finalReport}
        language={reportLanguage}
        onNewAnalysis={() => {
          setReport(null);
          navigate('/');
        }}
      />

      <div className="grid grid-cols-1 lg:grid-cols-[280px_1fr] max-w-[1400px] mx-auto">
        {/* Sidebar */}
        <ReportSidebar finalReport={finalReport} language={reportLanguage} />

        {/* Main Content */}
        <main className="p-10 space-y-12 max-w-[860px]">
          <MetricsDashboardSection finalReport={finalReport} language={reportLanguage} />
          <ClinicalToxicitySection data={finalReport.sections.clinical_toxicity} language={reportLanguage} />
          <MechanismProfilingSection data={finalReport.sections.mechanism_toxicity} language={reportLanguage} />
          <StructuralExplanationSection data={finalReport.sections.structural_explanation} language={reportLanguage} />
          <LiteratureContextSection data={finalReport.sections.literature_context} language={reportLanguage} />
          <AIRecommendationsSection
            summary={finalReport.executive_summary}
            recommendations={finalReport.sections.recommendations || []}
            riskLevel={finalReport.risk_level}
            language={reportLanguage}
            reliabilityWarning={finalReport.sections.reliability_warning}
            oodAssessment={finalReport.sections.ood_assessment}
            recommendationSource={finalReport.sections.recommendation_source}
            recommendationSourceDetail={finalReport.sections.recommendation_source_detail}
            failureRegistry={finalReport.sections.failure_registry}
            runtimeMode={report.runtime_mode}
            runtimeNote={report.runtime_note}
          />
        </main>
      </div>
      
      {/* AI Chatbot - only on report page */}
      <AIChatbot />


      <Footer />
    </div>
  );
}
