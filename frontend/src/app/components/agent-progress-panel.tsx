import { useEffect, useMemo, useState } from 'react';
import { motion } from 'motion/react';
import { Activity, Atom, CheckCircle, Circle, Loader2, Sparkles, Zap } from 'lucide-react';
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

const SCIENCE_TIPS = [
  'LD50 estimates the dose required to cause lethality in 50% of a tested population.',
  'Lipinski\'s Rule of 5 is a fast heuristic for screening drug-like oral candidates.',
  'hERG inhibition is a common early safety flag because it can correlate with QT prolongation.',
  'Toxicity workflows gain reliability when model predictions are paired with literature evidence.',
];

const MOLECULE_NODES = [
  { id: 'n1', x: 44, y: 84, delay: 0 },
  { id: 'n2', x: 108, y: 42, delay: 0.22 },
  { id: 'n3', x: 162, y: 90, delay: 0.4 },
  { id: 'n4', x: 232, y: 54, delay: 0.6 },
  { id: 'n5', x: 278, y: 116, delay: 0.82 },
  { id: 'n6', x: 138, y: 146, delay: 1.04 },
];

const MOLECULE_EDGES = [
  ['n1', 'n2'],
  ['n2', 'n3'],
  ['n3', 'n4'],
  ['n4', 'n5'],
  ['n3', 'n6'],
  ['n1', 'n6'],
  ['n2', 'n6'],
  ['n3', 'n5'],
] as const;

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

function buildFallbackState(isAnalyzing: boolean): { agents: AgentStatus[]; logs: LogLine[] } {
  if (isAnalyzing) {
    return {
      agents: [
        {
          name: 'InputValidator',
          status: 'running',
          progress: 52,
          message: 'Checking SMILES integrity and service readiness...',
        },
        {
          name: 'ScreeningAgent',
          status: 'pending',
          progress: 0,
          message: 'Queued for toxicity model evaluation...',
        },
        {
          name: 'ResearcherAgent',
          status: 'pending',
          progress: 0,
          message: 'Queued for PubChem and PubMed evidence retrieval...',
        },
        {
          name: 'WriterAgent',
          status: 'pending',
          progress: 0,
          message: 'Queued for report synthesis...',
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

      current.status = event.is_final ? 'done' : 'running';
      current.progress = event.is_final ? 100 : Math.max(current.progress, 65);
      current.message =
        (callName && `Calling ${callName}...`) ||
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
  const state = useMemo(
    () => (hasEvents ? buildEventDrivenState(events, isAnalyzing) : buildFallbackState(isAnalyzing)),
    [events, hasEvents, isAnalyzing],
  );
  const [tipIndex, setTipIndex] = useState(0);

  useEffect(() => {
    if (!isAnalyzing) {
      setTipIndex(0);
      return;
    }

    const intervalId = window.setInterval(() => {
      setTipIndex((current) => (current + 1) % SCIENCE_TIPS.length);
    }, 3000);

    return () => {
      window.clearInterval(intervalId);
    };
  }, [isAnalyzing]);

  if (isAnalyzing) {
    return <AnalyzingProgressExperience state={state} tip={SCIENCE_TIPS[tipIndex]} />;
  }

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
          Pipeline Summary
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

function AnalyzingProgressExperience({ state, tip }: { state: { agents: AgentStatus[]; logs: LogLine[] }; tip: string }) {
  const overallProgress = Math.max(
    12,
    Math.round(state.agents.reduce((total, agent) => total + agent.progress, 0) / state.agents.length),
  );
  const activeAgentIndex = getActiveAgentIndex(state.agents);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45 }}
      className="relative mb-6 overflow-hidden rounded-[28px] border p-6 shadow-xl md:p-8"
      style={{
        background:
          'linear-gradient(135deg, rgba(5, 24, 37, 0.98) 0%, rgba(9, 55, 59, 0.96) 46%, rgba(234, 248, 244, 0.98) 100%)',
        border: '1px solid rgba(15, 23, 42, 0.08)',
      }}
    >
      <div className="pointer-events-none absolute inset-0">
        <motion.div
          className="absolute -left-20 top-10 h-48 w-48 rounded-full blur-3xl"
          style={{ backgroundColor: 'rgba(45, 212, 191, 0.18)' }}
          animate={{ scale: [1, 1.18, 1], opacity: [0.3, 0.55, 0.3] }}
          transition={{ duration: 4.8, repeat: Infinity, ease: 'easeInOut' }}
        />
        <motion.div
          className="absolute right-0 top-0 h-56 w-56 rounded-full blur-3xl"
          style={{ backgroundColor: 'rgba(59, 130, 246, 0.14)' }}
          animate={{ scale: [1.12, 0.92, 1.12], opacity: [0.18, 0.32, 0.18] }}
          transition={{ duration: 5.6, repeat: Infinity, ease: 'easeInOut' }}
        />
      </div>

      <div className="relative grid gap-6 xl:grid-cols-[1.2fr_0.88fr]">
        <div className="space-y-6">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <div
                className="mb-3 flex items-center gap-2 text-xs uppercase tracking-[0.28em]"
                style={{ color: 'rgba(229, 245, 242, 0.76)' }}
              >
                <Atom className="h-4 w-4" />
                Live Molecular Scan
              </div>
              <h3 className="text-2xl font-semibold" style={{ color: '#f6fffd' }}>
                Analyzing molecular toxicity
              </h3>
              <p className="mt-2 max-w-xl text-sm leading-6" style={{ color: 'rgba(224, 240, 237, 0.84)' }}>
                Validation, toxicity screening, evidence lookup, and report synthesis are progressing through the agent pipeline.
              </p>
            </div>

            <div
              className="rounded-full border px-4 py-2 text-sm font-medium"
              style={{
                color: '#f6fffd',
                borderColor: 'rgba(255, 255, 255, 0.18)',
                backgroundColor: 'rgba(255, 255, 255, 0.08)',
              }}
            >
              {overallProgress}% complete
            </div>
          </div>

          <div
            className="rounded-[24px] border p-5"
            style={{
              background: 'linear-gradient(135deg, rgba(9, 20, 31, 0.84), rgba(12, 52, 61, 0.72))',
              borderColor: 'rgba(255, 255, 255, 0.12)',
            }}
          >
            <MoleculePulseAnimation />

            <div className="mt-5 space-y-2">
              <div
                className="flex items-center justify-between text-xs uppercase tracking-[0.22em]"
                style={{ color: 'rgba(215, 233, 230, 0.74)' }}
              >
                <span>Scanning structure</span>
                <span>{overallProgress}%</span>
              </div>

              <div className="relative h-2 overflow-hidden rounded-full" style={{ backgroundColor: 'rgba(255, 255, 255, 0.12)' }}>
                <motion.div
                  className="h-full rounded-full"
                  style={{ background: 'linear-gradient(90deg, #2dd4bf 0%, #22c55e 100%)' }}
                  initial={{ width: 0 }}
                  animate={{ width: `${overallProgress}%` }}
                  transition={{ duration: 0.6, ease: 'easeOut' }}
                />
                <motion.div
                  className="absolute inset-y-0 w-24"
                  style={{ background: 'linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.72), transparent)' }}
                  animate={{ x: ['-140%', '430%'] }}
                  transition={{ duration: 1.8, repeat: Infinity, ease: 'linear' }}
                />
              </div>
            </div>
          </div>

          <div
            className="rounded-2xl border p-4"
            style={{ backgroundColor: 'rgba(7, 18, 28, 0.48)', borderColor: 'rgba(255, 255, 255, 0.12)' }}
          >
            <div
              className="mb-3 flex items-center gap-2 text-xs uppercase tracking-[0.22em]"
              style={{ color: 'rgba(215, 233, 230, 0.74)' }}
            >
              <Sparkles className="h-4 w-4" />
              Rotating science tip
            </div>

            <motion.p
              key={tip}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              className="text-sm leading-6"
              style={{ color: '#f2fffc' }}
            >
              {tip}
            </motion.p>
          </div>
        </div>

        <div className="space-y-4">
          <div
            className="rounded-[24px] border p-4"
            style={{ backgroundColor: 'rgba(248, 252, 251, 0.92)', borderColor: 'rgba(8, 47, 73, 0.08)' }}
          >
            <div
              className="mb-4 flex items-center gap-2 text-xs uppercase tracking-[0.22em]"
              style={{ color: 'rgba(10, 37, 46, 0.62)' }}
            >
              <Zap className="h-4 w-4" />
              Agent pipeline
            </div>

            <div className="space-y-3">
              {state.agents.map((agent, index) => (
                <AgentSequenceStep
                  key={agent.name}
                  agent={agent}
                  index={index}
                  isActive={index === activeAgentIndex}
                />
              ))}
            </div>
          </div>

          <div
            className="rounded-[24px] border p-4"
            style={{ backgroundColor: 'rgba(7, 18, 28, 0.82)', borderColor: 'rgba(148, 163, 184, 0.18)' }}
          >
            <div
              className="mb-3 flex items-center gap-2 text-xs uppercase tracking-[0.22em]"
              style={{ color: 'rgba(162, 255, 223, 0.8)' }}
            >
              <Activity className="h-4 w-4" />
              Live log stream
            </div>

            <div className="max-h-56 space-y-2 overflow-y-auto font-mono text-xs leading-5" style={{ color: '#d6fff3' }}>
              {state.logs.map((log, idx) => (
                <motion.div
                  key={`${log.time}-${log.agent}-${idx}`}
                  initial={{ opacity: 0, y: 4 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="rounded-lg px-3 py-2"
                  style={{ backgroundColor: 'rgba(15, 45, 43, 0.32)' }}
                >
                  <span style={{ color: 'rgba(153, 246, 228, 0.66)' }}>{log.time}</span>{' '}
                  <span style={{ color: '#5eead4' }}>{log.agent}</span>{' '}
                  <span style={{ color: '#eafff8' }}>{log.message}</span>
                </motion.div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
}

function MoleculePulseAnimation() {
  const nodeMap = new Map(MOLECULE_NODES.map((node) => [node.id, node]));

  return (
    <div className="relative overflow-hidden rounded-[22px] border p-3" style={{ borderColor: 'rgba(255, 255, 255, 0.08)' }}>
      <svg viewBox="0 0 320 180" className="h-[220px] w-full" role="img" aria-label="Animated molecule scan">
        <defs>
          <linearGradient id="molecule-edge-gradient" x1="0%" x2="100%" y1="0%" y2="0%">
            <stop offset="0%" stopColor="rgba(45, 212, 191, 0.18)" />
            <stop offset="50%" stopColor="rgba(186, 230, 253, 0.92)" />
            <stop offset="100%" stopColor="rgba(34, 197, 94, 0.22)" />
          </linearGradient>
          <linearGradient id="molecule-scan-gradient" x1="0%" x2="100%" y1="0%" y2="0%">
            <stop offset="0%" stopColor="rgba(255, 255, 255, 0)" />
            <stop offset="50%" stopColor="rgba(255, 255, 255, 0.2)" />
            <stop offset="100%" stopColor="rgba(255, 255, 255, 0)" />
          </linearGradient>
        </defs>

        <rect x="0" y="0" width="320" height="180" rx="22" fill="rgba(255, 255, 255, 0.02)" />

        {MOLECULE_EDGES.map(([fromId, toId], index) => {
          const from = nodeMap.get(fromId)!;
          const to = nodeMap.get(toId)!;

          return (
            <motion.line
              key={`${fromId}-${toId}`}
              x1={from.x}
              y1={from.y}
              x2={to.x}
              y2={to.y}
              stroke="url(#molecule-edge-gradient)"
              strokeWidth="2.2"
              strokeLinecap="round"
              initial={{ opacity: 0.22 }}
              animate={{ opacity: [0.22, 0.84, 0.22] }}
              transition={{ duration: 2.6, delay: index * 0.08, repeat: Infinity, ease: 'easeInOut' }}
            />
          );
        })}

        <motion.rect
          x="-80"
          y="0"
          width="96"
          height="180"
          fill="url(#molecule-scan-gradient)"
          animate={{ x: [-96, 336] }}
          transition={{ duration: 2.8, repeat: Infinity, ease: 'linear' }}
        />

        {MOLECULE_NODES.map((node) => (
          <g key={node.id}>
            <motion.circle
              cx={node.x}
              cy={node.y}
              r="14"
              fill="rgba(45, 212, 191, 0.12)"
              animate={{ r: [14, 22, 14], opacity: [0.14, 0.42, 0.14] }}
              transition={{ duration: 2.4, delay: node.delay, repeat: Infinity, ease: 'easeInOut' }}
            />
            <motion.circle
              cx={node.x}
              cy={node.y}
              r="8"
              fill="rgba(224, 255, 250, 0.92)"
              animate={{ opacity: [0.72, 1, 0.72] }}
              transition={{ duration: 2.1, delay: node.delay, repeat: Infinity, ease: 'easeInOut' }}
            />
          </g>
        ))}
      </svg>
    </div>
  );
}

function AgentSequenceStep({
  agent,
  index,
  isActive,
}: {
  agent: AgentStatus;
  index: number;
  isActive: boolean;
}) {
  const statusLabel =
    agent.status === 'done' ? 'Completed' : agent.status === 'running' ? 'Running' : agent.status === 'error' ? 'Issue' : 'Queued';
  const barWidth = agent.status === 'running' ? Math.max(agent.progress, 18) : agent.progress;
  const isDone = agent.status === 'done';
  const isRunning = agent.status === 'running';
  const isError = agent.status === 'error';

  return (
    <motion.div
      animate={{
        scale: isActive ? 1.015 : 1,
        boxShadow: isActive ? '0 16px 40px rgba(13, 148, 136, 0.16)' : '0 0 0 rgba(0, 0, 0, 0)',
      }}
      transition={{ duration: 0.25, ease: 'easeOut' }}
      className="rounded-2xl border p-4"
      style={{
        background: isActive
          ? 'linear-gradient(135deg, rgba(240, 253, 250, 0.98), rgba(220, 252, 231, 0.9))'
          : isDone
            ? 'rgba(240, 253, 244, 0.82)'
            : 'rgba(255, 255, 255, 0.9)',
        borderColor: isActive
          ? 'rgba(13, 148, 136, 0.34)'
          : isDone
            ? 'rgba(34, 197, 94, 0.24)'
            : 'rgba(15, 23, 42, 0.08)',
      }}
    >
      <div className="flex items-start gap-3">
        <div
          className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full border"
          style={{
            borderColor: isError ? 'rgba(239, 68, 68, 0.3)' : isActive ? 'rgba(13, 148, 136, 0.28)' : 'rgba(148, 163, 184, 0.22)',
            backgroundColor: isError ? 'rgba(239, 68, 68, 0.12)' : isActive ? 'rgba(20, 184, 166, 0.12)' : 'rgba(255, 255, 255, 0.84)',
            color: '#0f172a',
          }}
        >
          {isDone ? (
            <CheckCircle className="h-5 w-5" style={{ color: '#16a34a' }} />
          ) : isRunning ? (
            <Loader2 className="h-5 w-5 animate-spin" style={{ color: '#0f766e' }} />
          ) : isError ? (
            <span className="text-sm font-semibold" style={{ color: '#dc2626' }}>!</span>
          ) : (
            <span className="text-sm font-semibold">{index + 1}</span>
          )}
        </div>

        <div className="min-w-0 flex-1">
          <div className="flex items-center justify-between gap-2">
            <p className="text-sm font-semibold" style={{ color: '#0f172a' }}>
              {agent.name}
            </p>
            <span
              className="rounded-full px-2.5 py-1 text-[11px] uppercase tracking-[0.18em]"
              style={{
                color: isDone ? '#166534' : isRunning ? '#115e59' : isError ? '#991b1b' : '#475569',
                backgroundColor: isDone
                  ? 'rgba(34, 197, 94, 0.12)'
                  : isRunning
                    ? 'rgba(20, 184, 166, 0.12)'
                    : isError
                      ? 'rgba(239, 68, 68, 0.1)'
                      : 'rgba(148, 163, 184, 0.14)',
              }}
            >
              {statusLabel}
            </span>
          </div>

          <p className="mt-1 text-xs leading-5" style={{ color: 'rgba(15, 23, 42, 0.68)' }}>
            {agent.message}
          </p>

          <div className="mt-3 h-1.5 overflow-hidden rounded-full" style={{ backgroundColor: 'rgba(148, 163, 184, 0.2)' }}>
            <motion.div
              className="h-full rounded-full"
              style={{
                background: isDone
                  ? 'linear-gradient(90deg, #22c55e 0%, #16a34a 100%)'
                  : isError
                    ? 'linear-gradient(90deg, #f87171 0%, #dc2626 100%)'
                    : 'linear-gradient(90deg, #14b8a6 0%, #06b6d4 100%)',
              }}
              initial={{ width: 0 }}
              animate={{ width: `${barWidth}%` }}
              transition={{ duration: 0.4, ease: 'easeOut' }}
            />
          </div>
        </div>
      </div>
    </motion.div>
  );
}

function getActiveAgentIndex(agents: AgentStatus[]) {
  const errorIndex = agents.findIndex((agent) => agent.status === 'error');
  if (errorIndex >= 0) {
    return errorIndex;
  }

  const runningIndex = agents.findIndex((agent) => agent.status === 'running');
  if (runningIndex >= 0) {
    return runningIndex;
  }

  const completedCount = agents.filter((agent) => agent.status === 'done').length;
  return Math.min(completedCount, agents.length - 1);
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
