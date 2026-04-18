import { motion } from 'motion/react';
import { CheckCircle, Loader2, Circle } from 'lucide-react';
import type { AgentEventRecord } from '../../lib/api';

interface AgentProgressPanelProps {
  isAnalyzing: boolean;
  events: AgentEventRecord[];
}

interface AgentStatus {
  name: string;
  status: 'pending' | 'running' | 'done' | 'error';
  progress: number;
  message: string;
}

interface LogLine {
  time: string;
  agent: string;
  message: string;
}

const AGENT_ORDER = ['InputValidator', 'ScreeningAgent', 'ResearcherAgent', 'WriterAgent'];

function getCurrentTimeLabel() {
  return new Date().toLocaleTimeString('en-GB', { hour12: false });
}

function getCallName(call: Record<string, unknown>): string {
  const directName = call.name;
  if (typeof directName === 'string' && directName) {
    return directName;
  }

  const nestedName = (call.functionCall as { name?: unknown } | undefined)?.name;
  if (typeof nestedName === 'string' && nestedName) {
    return nestedName;
  }

  return 'tool_call';
}

function getResponseName(response: Record<string, unknown>): string {
  const directName = response.name;
  if (typeof directName === 'string' && directName) {
    return directName;
  }

  const nestedName = (response.response as { name?: unknown } | undefined)?.name;
  if (typeof nestedName === 'string' && nestedName) {
    return nestedName;
  }

  return 'tool_result';
}

function buildFallbackState(isAnalyzing: boolean): { agents: AgentStatus[]; logs: LogLine[] } {
  if (isAnalyzing) {
    return {
      agents: [
        {
          name: 'InputValidator',
          status: 'running',
          progress: 40,
          message: 'Checking SMILES and Health Endpoints...',
        },
        {
          name: 'ScreeningAgent',
          status: 'running',
          progress: 55,
          message: 'Running toxicity analysis model...',
        },
        {
          name: 'ResearcherAgent',
          status: 'running',
          progress: 50,
          message: 'Querying PubChem/PubMed...',
        },
        {
          name: 'WriterAgent',
          status: 'pending',
          progress: 0,
          message: 'Waiting for report generation...',
        },
      ],
      logs: [
        {
          time: getCurrentTimeLabel(),
          agent: 'System',
          message: 'Waiting for detailed events...',
        },
      ],
    };
  }

  return {
    agents: AGENT_ORDER.map((name) => ({
      name,
      status: 'done',
      progress: 100,
      message: 'Completed (fallback)',
    })),
    logs: [
      {
        time: getCurrentTimeLabel(),
        agent: 'System',
        message: 'No agent_events available, using fallback mode',
      },
    ],
  };
}

function buildEventDrivenState(events: AgentEventRecord[], isAnalyzing: boolean): { agents: AgentStatus[]; logs: LogLine[] } {
  const agentMap = new Map<string, AgentStatus>(
    AGENT_ORDER.map((name) => [
      name,
      { name, status: 'pending', progress: 0, message: 'Waiting...' },
    ]),
  );

  const logs: LogLine[] = [];

  events.forEach((event, index) => {
    const author = event.author || 'System';
    const time = getCurrentTimeLabel();

    if (agentMap.has(author)) {
      const current = agentMap.get(author)!;
      const callName = event.function_calls?.[0] ? getCallName(event.function_calls[0]) : null;
      const responseName = event.function_responses?.[0]
        ? getResponseName(event.function_responses[0])
        : null;

      current.status = event.is_final ? 'done' : 'running';
      current.progress = event.is_final ? 100 : Math.max(current.progress, 65);
      current.message =
        (callName && `Calling ${callName}...`) ||
        (responseName && `Received ${responseName} result`) ||
        event.text_preview ||
        (event.is_final ? 'Completed' : 'Processing...');

      logs.push({
        time,
        agent: author,
        message: event.is_final ? 'Done' : current.message,
      });
    }

    if (event.function_calls?.length) {
      event.function_calls.forEach((call) => {
        logs.push({
          time,
          agent: author,
          message: `Tool call: ${getCallName(call)}`,
        });
      });
    }

    if (event.text_preview && !event.function_calls?.length) {
      logs.push({
        time,
        agent: author,
        message: event.text_preview,
      });
    }

    if (event.function_responses?.length) {
      event.function_responses.forEach((response) => {
        logs.push({
          time,
          agent: author,
          message: `Tool result: ${getResponseName(response)}`,
        });
      });
    }

    if (index === events.length - 1 && event.is_final && agentMap.has('WriterAgent')) {
      const writer = agentMap.get('WriterAgent')!;
      writer.status = 'done';
      writer.progress = 100;
      writer.message = writer.message || 'Completed report generation';
    }
  });

  if (!isAnalyzing) {
    agentMap.forEach((agent) => {
      if (agent.status === 'running') {
        agent.status = 'done';
        agent.progress = 100;
      }
    });
  }

  return {
    agents: AGENT_ORDER.map((name) => agentMap.get(name)!),
    logs: logs.slice(-24),
  };
}

export function AgentProgressPanel({ isAnalyzing, events }: AgentProgressPanelProps) {
  const hasEvents = events.length > 0;
  const state = hasEvents
    ? buildEventDrivenState(events, isAnalyzing)
    : buildFallbackState(isAnalyzing);

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
          Analysis Pipeline
        </h3>
      </div>

      <div className="space-y-4 mb-6">
        <AgentNode agent={state.agents[0]} />

        <div className="flex items-center justify-center">
          <div
            className="text-xs uppercase tracking-widest"
            style={{ color: 'var(--text-faint)', letterSpacing: '0.1em' }}
          >
            PARALLEL
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <AgentNode agent={state.agents[1]} />
          <AgentNode agent={state.agents[2]} />
        </div>

        <AgentNode agent={state.agents[3]} />
      </div>

      <div
        className="rounded-lg p-4 font-mono text-xs max-h-32 overflow-y-auto"
        style={{ backgroundColor: 'var(--bg)', color: 'var(--text-muted)' }}
      >
        {state.logs.map((log, idx) => (
          <motion.div
            key={`${log.time}-${log.agent}-${idx}`}
            initial={{ opacity: 0, y: 5 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-1"
          >
            <span style={{ color: 'var(--text-faint)' }}>{log.time}</span>{' '}
            <span style={{ color: 'var(--accent-blue)' }}>{log.agent}</span>{' '}
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
        return <span style={{ color: 'var(--accent-red)' }}>x</span>;
      default:
        return <Circle className="w-5 h-5" style={{ color: 'var(--border)' }} />;
    }
  };

  return (
    <div
      className="rounded-lg p-4 transition-all"
      style={{
        backgroundColor:
          agent.status === 'running'
            ? 'var(--accent-blue-muted)'
            : agent.status === 'error'
              ? 'rgba(239,68,68,0.08)'
              : 'transparent',
        border: '1px solid var(--border)',
      }}
    >
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          {getStatusIcon()}
          <span className="font-medium text-sm" style={{ color: 'var(--text)' }}>
            {agent.name}
          </span>
        </div>
      </div>

      {(agent.status === 'running' || agent.status === 'done') && (
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
    </div>
  );
}
