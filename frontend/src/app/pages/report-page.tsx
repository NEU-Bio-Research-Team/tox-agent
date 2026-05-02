import { useEffect, useState } from 'react';
import { MessageSquare, PanelLeft, PanelLeftClose, Plus } from 'lucide-react';
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
import { useAuth } from '../components/contexts/auth-context';
import { listChatSessionsByAnalysisId, type ChatSessionRecord } from '../../lib/chat-history';

export function ReportPage() {
  const navigate = useNavigate();
  const { report, setReport, error } = useReport();
  const { user } = useAuth();
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [chatSessions, setChatSessions] = useState<ChatSessionRecord[]>([]);

  const analysisSessionId = report?.session_id ?? null;

  useEffect(() => {
    if (!user?.id || !analysisSessionId) return;
    let cancelled = false;
    listChatSessionsByAnalysisId(user.id, analysisSessionId)
      .then((sessions) => {
        if (!cancelled) setChatSessions(sessions);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [user?.id, analysisSessionId]);

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

      {/* Chat History Panel */}
      {chatSessions.length > 0 && (
        <div className="mx-auto max-w-[1400px] px-4 py-6 md:px-6 lg:px-10">
          <div
            className="rounded-2xl border p-5"
            style={{ backgroundColor: 'var(--surface)', borderColor: 'var(--border)' }}
          >
            <div className="mb-4 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <MessageSquare className="h-4 w-4" style={{ color: 'var(--accent-blue)' }} />
                <h2 className="text-sm font-semibold" style={{ color: 'var(--text)' }}>
                  Previous Chat Sessions for This Report
                </h2>
              </div>
              <Button
                size="sm"
                variant="outline"
                className="gap-1"
                style={{ borderColor: 'var(--border)', color: 'var(--text)' }}
                onClick={() =>
                  navigate('/chat', {
                    state: {
                      analysisSessionId: report.session_id,
                      chatSessionId: null,
                      reportState: {
                        smiles_input: finalReport.report_metadata.smiles,
                        final_report: finalReport,
                        evidence_qa_result: report.evidence_qa_result,
                      },
                    },
                  })
                }
              >
                <Plus className="h-3 w-3" />
                New Chat
              </Button>
            </div>
            <div className="space-y-2">
              {chatSessions.map((session) => (
                <button
                  key={session.sessionId}
                  type="button"
                  className="w-full rounded-xl border px-4 py-3 text-left transition-colors hover:bg-[var(--surface-alt)]"
                  style={{ borderColor: 'var(--border)' }}
                  onClick={() =>
                    navigate('/chat', {
                      state: {
                        chatSessionId: session.sessionId,
                        analysisSessionId: report.session_id,
                        reportState: {
                          smiles_input: finalReport.report_metadata.smiles,
                          final_report: finalReport,
                          evidence_qa_result: report.evidence_qa_result,
                        },
                      },
                    })
                  }
                >
                  <p className="truncate text-sm font-medium" style={{ color: 'var(--text)' }}>
                    {session.title ?? `Session ${session.sessionId.slice(0, 8)}`}
                  </p>
                  {session.lastMessagePreview && (
                    <p className="mt-0.5 truncate text-xs" style={{ color: 'var(--text-muted)' }}>
                      {session.lastMessagePreview}
                    </p>
                  )}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      <Footer />
    </div>
  );
}
