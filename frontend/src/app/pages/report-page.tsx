import { useLocation, useNavigate } from 'react-router';
import { Navbar } from '../components/navbar';
import { ReportHeader } from '../components/report-header';
import { ReportSidebar } from '../components/report-sidebar';
import { ClinicalToxicitySection } from '../components/report/clinical-toxicity-section';
import { MechanismProfilingSection } from '../components/report/mechanism-profiling-section';
import { StructuralExplanationSection } from '../components/report/structural-explanation-section';
import { LiteratureContextSection } from '../components/report/literature-context-section';
import { AIRecommendationsSection } from '../components/report/ai-recommendations-section';
import { AIChatbot } from '../components/ai-chatbot';

export function ReportPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const smiles = location.state?.smiles || 'CC(=O)Oc1ccccc1C(=O)O';

  return (
    <div style={{ minHeight: '100vh', backgroundColor: 'var(--bg)', fontFamily: 'Inter, sans-serif' }}>
      <Navbar />
      
      <ReportHeader smiles={smiles} onNewAnalysis={() => navigate('/')} />

      <div className="grid grid-cols-1 lg:grid-cols-[280px_1fr] max-w-[1400px] mx-auto">
        {/* Sidebar */}
        <ReportSidebar />

        {/* Main Content */}
        <main className="p-10 space-y-12 max-w-[860px]">
          <ClinicalToxicitySection />
          <MechanismProfilingSection />
          <StructuralExplanationSection />
          <LiteratureContextSection />
          <AIRecommendationsSection />
        </main>
      </div>
      
      {/* AI Chatbot - only on report page */}
      <AIChatbot />
    </div>
  );
}