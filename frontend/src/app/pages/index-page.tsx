import { useState } from 'react';
import { useNavigate } from 'react-router';
import { Navbar } from '../components/navbar';
import { HeroSection } from '../components/hero-section';
import { AgentProgressPanel } from '../components/agent-progress-panel';
import { QuickVerdictCard } from '../components/quick-verdict-card';
import { agentAnalyzeStream, type AgentAnalyzeResponse, type AgentEventRecord } from '../../lib/api';
import { useReport } from '../../lib/ReportContext';
import { Footer } from '../components/footer';
import { SmilesHistory, addToHistory } from '../components/smiles-history';
import { useAuth } from '../components/contexts/auth-context';
import { normalizeRiskLevel } from '../risk-level';

function toHistoryVerdict(riskCode: string): 'toxic' | 'warning' | 'non-toxic' {
  switch (riskCode) {
    case 'CRITICAL':
    case 'HIGH':
      return 'toxic';
    case 'MODERATE':
      return 'warning';
    default:
      return 'non-toxic';
  }
}

function toBoundedScore(value: number | null | undefined): number {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return 0;
  }
  if (value < 0) {
    return 0;
  }
  if (value > 1) {
    return 1;
  }
  return value;
}

export function IndexPage() {
  const navigate = useNavigate();
  const { user } = useAuth();
  const [smilesInput, setSmilesInput] = useState('CC(=O)Oc1ccccc1C(=O)O');
  const [streamEvents, setStreamEvents] = useState<AgentEventRecord[]>([]);
  const {
    report,
    setReport,
    isLoading,
    setIsLoading,
    error,
    setError,
    preferences,
  } = useReport();
  const hasReport = Boolean(report && Object.keys(report.final_report ?? {}).length > 0);
  const progressEvents = report?.agent_events?.length ? report.agent_events : streamEvents;
  const showProgressPanel = isLoading || hasReport || streamEvents.length > 0;

  const handleAnalyze = async (opts: { binaryModel: string; toxTypeModel: string }) => {
    const smiles = smilesInput.trim();
    if (!smiles) {
      setError('Please enter a SMILES string before analysis.');
      return;
    }

    setIsLoading(true);
    setError(null);
    setReport(null);
    setStreamEvents([]);

    try {
      const stream = agentAnalyzeStream(smiles, {
        language: 'en',
        clinicalThreshold: preferences.clinicalThreshold,
        mechanismThreshold: preferences.mechanismThreshold,
        inferenceBackend: preferences.inferenceBackend,
        binaryToxModel: opts.binaryModel,     
        toxTypeModel: opts.toxTypeModel,
      });
      let result: AgentAnalyzeResponse | null = null;

      for await (const event of stream) {
        if (event.type === 'agent_event' && event.event) {
          setStreamEvents((prev) => [...prev, event.event as AgentEventRecord]);
          continue;
        }

        if (event.type === 'done' && event.result) {
          result = event.result as AgentAnalyzeResponse;
          setStreamEvents(result.agent_events ?? []);
          continue;
        }

        if (event.type === 'error') {
          throw new Error(event.message || event.error || 'Unexpected streaming analysis error');
        }
      }

      if (!result) {
        throw new Error('No final response received from analysis stream');
      }

      if (result.validation_status && result.validation_status !== 'VALID') {
        throw new Error(`Validation failed: ${result.validation_status}`);
      }
      if (!result.final_report || Object.keys(result.final_report).length === 0) {
        throw new Error('API response missing final_report');
      }

      setReport(result);

      // Persist analysis history without blocking the main UI flow.
      const normalizedRisk = normalizeRiskLevel(result.final_report.risk_level).code;
      const probability = result.final_report.sections.clinical_toxicity?.probability;

      void addToHistory(
        smiles,
        toHistoryVerdict(normalizedRisk),
        toBoundedScore(probability),
        user?.id,
        {
          language: 'en',
          inferenceBackend: preferences.inferenceBackend,
          binaryModel: opts.binaryModel,
          toxTypeModel: opts.toxTypeModel,
          sessionId: result.session_id,
          riskLevel: normalizedRisk,
        },
      ).catch((historyError) => {
        console.warn('Failed to persist analysis history:', historyError);
      });
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
    if (!report) {
      setStreamEvents([]);
    }
  };

  const handleSelectFromHistory = (smiles: string) => {
    setSmilesInput(smiles);
    window.scrollTo({ top: 0, behavior: 'smooth' });
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
        {showProgressPanel && (
          <AgentProgressPanel
            isAnalyzing={isLoading}
            events={progressEvents}
          />
        )}

        {/* Quick Verdict Card */}
        {hasReport && report?.final_report && (
          <QuickVerdictCard
            finalReport={report.final_report}
            onViewReport={handleViewReport}
          />
        )}

        {/* SMILES History */}
        {!isLoading && !hasReport && streamEvents.length === 0 && (
          <div className="py-12">
            <SmilesHistory onSelectSmiles={handleSelectFromHistory} uid={user?.id ?? null} />
          </div>
        )}
      </main>
      <Footer />
    </div>
  );
}
