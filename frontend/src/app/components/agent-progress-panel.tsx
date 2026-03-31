import { motion } from 'motion/react';
import { CheckCircle, Loader2, Circle } from 'lucide-react';
import { useEffect, useState } from 'react';

interface AgentProgressPanelProps {
  isAnalyzing: boolean;
}

interface AgentStatus {
  name: string;
  status: 'pending' | 'running' | 'done' | 'error';
  progress: number;
  message: string;
  time?: string;
  result?: {
    label: string;
    value: string;
  };
}

export function AgentProgressPanel({ isAnalyzing }: AgentProgressPanelProps) {
  const [agents, setAgents] = useState<AgentStatus[]>([
    { name: 'InputValidator', status: 'pending', progress: 0, message: '' },
    { name: 'ScreeningAgent', status: 'pending', progress: 0, message: '' },
    { name: 'ResearcherAgent', status: 'pending', progress: 0, message: '' },
    { name: 'WriterAgent', status: 'pending', progress: 0, message: '' },
  ]);

  const [logs, setLogs] = useState<Array<{ time: string; agent: string; message: string }>>([]);

  useEffect(() => {
    if (!isAnalyzing) return;

    const addLog = (agent: string, message: string) => {
      const time = new Date().toLocaleTimeString('en-GB', { hour12: false });
      setLogs(prev => [...prev, { time, agent, message }]);
    };

    // Simulate agent execution
    const sequence = async () => {
      // InputValidator
      setAgents(prev => prev.map((a, i) => i === 0 ? { ...a, status: 'running', message: 'Validating SMILES...' } : a));
      addLog('InputValidator', 'Starting SMILES validation');
      await new Promise(r => setTimeout(r, 500));
      setAgents(prev => prev.map((a, i) => i === 0 ? { ...a, status: 'done', progress: 100, time: '0.3s' } : a));
      addLog('InputValidator', '✓ SMILES validated');

      // Parallel: ScreeningAgent + ResearcherAgent
      setAgents(prev => prev.map((a, i) => 
        i === 1 ? { ...a, status: 'running', message: 'GNN forward pass...' } : 
        i === 2 ? { ...a, status: 'running', message: 'Querying PubChem...' } : a
      ));
      addLog('ScreeningAgent', '⟳ Loading GNN model...');
      addLog('ResearcherAgent', '⟳ Querying PubChem CID...');

      // Progress simulation
      for (let i = 0; i < 100; i += 20) {
        await new Promise(r => setTimeout(r, 400));
        setAgents(prev => prev.map((a, idx) => 
          idx === 1 || idx === 2 ? { ...a, progress: Math.min(i + 20, 100) } : a
        ));
      }

      await new Promise(r => setTimeout(r, 500));
      setAgents(prev => prev.map((a, i) => 
        i === 1 ? { 
          ...a, 
          status: 'done', 
          progress: 100, 
          time: '3.2s',
          result: { label: 'p_toxic:', value: '0.23' }
        } : 
        i === 2 ? { 
          ...a, 
          status: 'done', 
          progress: 100, 
          time: '5.1s',
          result: { label: '5 papers found', value: 'CID: 2244' }
        } : a
      ));
      addLog('ScreeningAgent', '✓ GNN forward pass done');
      addLog('ResearcherAgent', '✓ Fetching papers complete');

      // WriterAgent
      await new Promise(r => setTimeout(r, 300));
      setAgents(prev => prev.map((a, i) => i === 3 ? { ...a, status: 'running', message: 'Generating report...' } : a));
      addLog('WriterAgent', '⟳ Generating comprehensive report...');
      
      for (let i = 0; i < 100; i += 25) {
        await new Promise(r => setTimeout(r, 300));
        setAgents(prev => prev.map((a, idx) => 
          idx === 3 ? { ...a, progress: Math.min(i + 25, 100) } : a
        ));
      }

      await new Promise(r => setTimeout(r, 300));
      setAgents(prev => prev.map((a, i) => i === 3 ? { ...a, status: 'done', progress: 100, time: '2.1s' } : a));
      addLog('WriterAgent', '✓ Report generation complete');
    };

    sequence();
  }, [isAnalyzing]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="mb-6 rounded-2xl p-6 shadow-lg"
      style={{ backgroundColor: 'var(--surface)', border: '1px solid var(--border)' }}
    >
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold" style={{ color: 'var(--text)' }}>
          Pipeline Phân Tích
        </h3>
      </div>

      {/* Pipeline Visualization */}
      <div className="space-y-4 mb-6">
        {/* InputValidator */}
        <AgentNode agent={agents[0]} />

        {/* Parallel indicator */}
        <div className="flex items-center justify-center">
          <div className="text-xs uppercase tracking-widest" style={{ color: 'var(--text-faint)', letterSpacing: '0.1em' }}>
            PARALLEL
          </div>
        </div>

        {/* Parallel agents */}
        <div className="grid grid-cols-2 gap-4">
          <AgentNode agent={agents[1]} />
          <AgentNode agent={agents[2]} />
        </div>

        {/* WriterAgent */}
        <AgentNode agent={agents[3]} />
      </div>

      {/* Streaming Log */}
      <div 
        className="rounded-lg p-4 font-mono text-xs max-h-32 overflow-y-auto"
        style={{ backgroundColor: 'var(--bg)', color: 'var(--text-muted)' }}
      >
        {logs.map((log, idx) => (
          <motion.div
            key={idx}
            initial={{ opacity: 0, y: 5 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-1"
          >
            <span style={{ color: 'var(--text-faint)' }}>{log.time}</span>
            {' '}
            <span style={{ color: 'var(--accent-blue)' }}>{log.agent}</span>
            {' '}
            <span>{log.message}</span>
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
}

function AgentNode({ agent }: { agent: AgentStatus }) {
  const getStatusIcon = () => {
    switch (agent.status) {
      case 'done':
        return <CheckCircle className="w-5 h-5" style={{ color: 'var(--accent-green)' }} />;
      case 'running':
        return <Loader2 className="w-5 h-5 animate-spin" style={{ color: 'var(--accent-blue)' }} />;
      case 'error':
        return <span style={{ color: 'var(--accent-red)' }}>✕</span>;
      default:
        return <Circle className="w-5 h-5" style={{ color: 'var(--border)' }} />;
    }
  };

  return (
    <div
      className="rounded-lg p-4 transition-all"
      style={{
        backgroundColor: agent.status === 'running' ? 'var(--accent-blue-muted)' : agent.status === 'error' ? 'rgba(239,68,68,0.08)' : 'transparent',
        border: '1px solid var(--border)'
      }}
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          {getStatusIcon()}
          <span className="font-medium text-sm" style={{ color: 'var(--text)' }}>
            {agent.name}
          </span>
        </div>
        {agent.time && (
          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
            {agent.time}
          </span>
        )}
      </div>

      {agent.status === 'running' && (
        <>
          <div className="w-full h-1 rounded-full mb-2" style={{ backgroundColor: 'var(--border)' }}>
            <motion.div
              className="h-full rounded-full"
              style={{ backgroundColor: 'var(--accent-blue)' }}
              initial={{ width: 0 }}
              animate={{ width: `${agent.progress}%` }}
              transition={{ duration: 0.3 }}
            />
          </div>
          <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
            {agent.message}
          </p>
        </>
      )}

      {agent.result && agent.status === 'done' && (
        <div className="mt-2 px-3 py-2 rounded-md text-xs" style={{ backgroundColor: 'var(--surface-alt)' }}>
          <span style={{ color: 'var(--text-muted)' }}>{agent.result.label}</span>{' '}
          <span className="font-mono font-semibold" style={{ color: 'var(--text)' }}>
            {agent.result.value}
          </span>
        </div>
      )}
    </div>
  );
}
