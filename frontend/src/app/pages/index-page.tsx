import { useState } from 'react';
import { useNavigate } from 'react-router';
import { Navbar } from '../components/navbar';
import { HeroSection } from '../components/hero-section';
import { AgentProgressPanel } from '../components/agent-progress-panel';
import { QuickVerdictCard } from '../components/quick-verdict-card';
import { agentAnalyze } from '../../lib/api';
import { useReport } from '../../lib/ReportContext';

export function IndexPage() {
  const navigate = useNavigate();
  const [analysisComplete, setAnalysisComplete] = useState(false);
  const [smilesInput, setSmilesInput] = useState('CC(=O)Oc1ccccc1C(=O)O');
  const { report, setReport, isLoading, setIsLoading, error, setError } = useReport();

  const handleAnalyze = async () => {
    const smiles = smilesInput.trim();
    if (!smiles) {
      setError('Vui long nhap SMILES truoc khi phan tich.');
      return;
    }

    setIsLoading(true);
    setError(null);
    setReport(null);
    setAnalysisComplete(false);

    try {
      const result = await agentAnalyze(smiles);
      if (result.validation_status && result.validation_status !== 'VALID') {
        throw new Error(`Validation failed: ${result.validation_status}`);
      }
      if (!result.final_report) {
        throw new Error('API response missing final_report');
      }

      setReport(result);
      setAnalysisComplete(true);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unexpected error during analysis';
      setError(message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleViewReport = () => {
    navigate('/report');
  };

  const handleSmilesChange = (value: string) => {
    setSmilesInput(value);
    if (analysisComplete) {
      setAnalysisComplete(false);
    }
  };

  return (
    <div style={{ 
      minHeight: '100vh', 
      backgroundColor: 'var(--bg)',
      fontFamily: 'Inter, sans-serif'
    }}>
      <Navbar />
      
      <main className="max-w-5xl mx-auto px-6">
        {/* Hero / Input Section */}
        <HeroSection 
          value={smilesInput}
          onChange={handleSmilesChange}
          onAnalyze={handleAnalyze}
          isAnalyzing={isLoading}
        />

        {error && (
          <div
            className="mb-6 rounded-lg px-4 py-3"
            style={{
              backgroundColor: 'rgba(239,68,68,0.08)',
              border: '1px solid rgba(239,68,68,0.35)',
              color: 'var(--accent-red)',
            }}
          >
            {error}
          </div>
        )}

        {/* Agent Progress Panel */}
        {(isLoading || analysisComplete) && (
          <AgentProgressPanel
            isAnalyzing={isLoading}
            events={report?.agent_events ?? []}
          />
        )}

        {/* Quick Verdict Card */}
        {analysisComplete && report?.final_report && (
          <QuickVerdictCard
            finalReport={report.final_report}
            onViewReport={handleViewReport}
          />
        )}
      </main>
    </div>
  );
}
