import { useState } from 'react';
import { PanelLeft, PanelLeftClose } from 'lucide-react';
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
import { MolragEvidenceSection } from '../components/report/molrag-evidence-section';
import { LiteratureContextSection } from '../components/report/literature-context-section';
import { AIRecommendationsSection } from '../components/report/ai-recommendations-section';
import { Button } from '../components/ui/button';
import { useReport } from '../../lib/ReportContext';

export function ReportPage() {
  const navigate = useNavigate();
  const { report, setReport, error } = useReport();
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

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
              onClick={() => navigate('/analyze')}
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
  const reportLanguage = 'en';

  return (
    <div style={{ minHeight: '100vh', backgroundColor: 'var(--bg)', fontFamily: 'Inter, sans-serif' }}>
      <Navbar />
      
      <ReportHeader
        finalReport={finalReport}
        language={reportLanguage}
        onNewAnalysis={() => {
          setReport(null);
          navigate('/analyze');
        }}
      />

      <div className="mx-auto flex max-w-[1400px] justify-end px-4 pt-4 md:px-6 lg:px-10">
        <Button
          variant="outline"
          size="sm"
          onClick={() => setIsSidebarOpen((current) => !current)}
          className="gap-2"
          style={{ borderColor: 'var(--border)', color: 'var(--text)' }}
        >
          {isSidebarOpen ? <PanelLeftClose className="h-4 w-4" /> : <PanelLeft className="h-4 w-4" />}
          <span className="lg:hidden">{isSidebarOpen ? 'Hide sections' : 'Sections'}</span>
          <span className="hidden lg:inline">{isSidebarOpen ? 'Hide report sections' : 'Show report sections'}</span>
        </Button>
      </div>

      <div
        className="max-w-[1400px] mx-auto lg:grid lg:items-start"
        style={{ gridTemplateColumns: isSidebarOpen ? '280px minmax(0, 1fr)' : '0 minmax(0, 1fr)' }}
      >
        {/* Sidebar */}
        <ReportSidebar
          finalReport={finalReport}
          language={reportLanguage}
          isOpen={isSidebarOpen}
          onToggle={() => setIsSidebarOpen((current) => !current)}
        />

        {/* Main Content */}
        <main
          className={`min-w-0 w-full max-w-[860px] p-4 md:p-6 lg:p-10 space-y-10 lg:space-y-12 ${
            isSidebarOpen ? 'hidden lg:block' : 'block'
          }`}
        >
          <MetricsDashboardSection finalReport={finalReport} language={reportLanguage} />
          <ClinicalToxicitySection data={finalReport.sections.clinical_toxicity} language={reportLanguage} />
          <MechanismProfilingSection data={finalReport.sections.mechanism_toxicity} language={reportLanguage} />
          <StructuralExplanationSection data={finalReport.sections.structural_explanation} language={reportLanguage} />
          <MolragEvidenceSection
            data={finalReport.sections.molrag_evidence}
            fusionResult={finalReport.sections.fusion_result}
            language={reportLanguage}
          />
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
      <AIChatbot
        chatSessionId={report.chat_session_id ?? null}
        analysisSessionId={report.session_id}
        reportState={{
          smiles_input: finalReport.report_metadata.smiles,
          final_report: finalReport,
          evidence_qa_result: report.evidence_qa_result,
        }}
      />


      <Footer />
    </div>
  );
}
