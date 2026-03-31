import { useState } from 'react';
import { useNavigate } from 'react-router';
import { Navbar } from '../components/navbar';
import { HeroSection } from '../components/hero-section';
import { AgentProgressPanel } from '../components/agent-progress-panel';
import { QuickVerdictCard } from '../components/quick-verdict-card';

export function IndexPage() {
  const navigate = useNavigate();
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisComplete, setAnalysisComplete] = useState(false);
  const [smilesInput, setSmilesInput] = useState('CC(=O)Oc1ccccc1C(=O)O');

  const handleAnalyze = async () => {
    setIsAnalyzing(true);
    setAnalysisComplete(false);
    
    // Simulate analysis
    await new Promise(resolve => setTimeout(resolve, 5000));
    
    setIsAnalyzing(false);
    setAnalysisComplete(true);
  };

  const handleViewReport = () => {
    navigate('/report', { state: { smiles: smilesInput } });
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
          onChange={setSmilesInput}
          onAnalyze={handleAnalyze}
          isAnalyzing={isAnalyzing}
        />

        {/* Agent Progress Panel */}
        {(isAnalyzing || analysisComplete) && (
          <AgentProgressPanel isAnalyzing={isAnalyzing} />
        )}

        {/* Quick Verdict Card */}
        {analysisComplete && (
          <QuickVerdictCard onViewReport={handleViewReport} />
        )}
      </main>
    </div>
  );
}
