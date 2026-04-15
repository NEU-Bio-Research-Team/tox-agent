import { useEffect, useRef, useState } from 'react';
import { useLocation, useNavigate } from 'react-router';
import { ArrowRight, Bot, FlaskConical, Send, Sparkles, User } from 'lucide-react';
import { Navbar } from '../components/navbar';
import { Footer } from '../components/footer';
import { Button } from '../components/ui/button';
import { useAuth } from '../components/contexts/auth-context';
import { agentChat } from '../../lib/api';
import { appendChatTurnToFirestore, loadChatSessionFromFirestore, type PersistedChatMessage } from '../../lib/chat-history';
import { useReport } from '../../lib/ReportContext';

interface ChatRouteState {
  question?: string;
  chatSessionId?: string | null;
  analysisSessionId?: string | null;
  reportState?: {
    smiles_input?: string;
    final_report?: Record<string, unknown>;
    evidence_qa_result?: Record<string, unknown>;
  } | null;
}

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
}

const DEFAULT_PROMPTS = [
  'Explain how ToxAgent evaluates toxicity from a SMILES string.',
  'What parts of the report should I review first?',
  'How should I interpret toxicity probability and confidence?',
];

const REPORT_PROMPTS = [
  'Summarize the key toxicity risk from this report.',
  'What mechanism evidence supports the highest risk?',
  'What should I investigate next before making a decision?',
];

function buildInitialMessage(hasReport: boolean, compoundLabel: string) {
  return hasReport
    ? `I can answer follow-up questions about the current analysis for ${compoundLabel}. Ask about risk, mechanisms, evidence, or recommendations.`
    : 'I can help explain the ToxAgent workflow and UI. Run an analysis first if you want answers grounded in a specific report.';
}

function createAssistantMessage(hasReport: boolean, compoundLabel: string): ChatMessage {
  return {
    id: crypto.randomUUID(),
    role: 'assistant',
    content: buildInitialMessage(hasReport, compoundLabel),
    timestamp: Date.now(),
  };
}

function toPersistedMessage(message: ChatMessage): PersistedChatMessage {
  return {
    role: message.role,
    content: message.content,
    timestamp: message.timestamp,
  };
}

function fromPersistedMessage(message: PersistedChatMessage): ChatMessage {
  return {
    id: crypto.randomUUID(),
    role: message.role,
    content: message.content,
    timestamp: message.timestamp,
  };
}

export function ChatbotPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const { user } = useAuth();
  const { report } = useReport();
  const routeState: ChatRouteState | null =
    typeof location.state === 'object' && location.state !== null
      ? (location.state as ChatRouteState)
      : null;
  const initialQuestion = typeof routeState?.question === 'string' ? routeState.question : '';
  const routeChatSessionId = typeof routeState?.chatSessionId === 'string' ? routeState.chatSessionId : null;
  const routeAnalysisSessionId = typeof routeState?.analysisSessionId === 'string' ? routeState.analysisSessionId : null;
  const routeReportState = routeState?.reportState ?? null;

  const finalReport = report?.final_report ?? null;
  const groundedSmiles =
    finalReport?.report_metadata.smiles ??
    (typeof routeReportState?.smiles_input === 'string' ? routeReportState.smiles_input : null);
  const compoundLabel =
    finalReport?.report_metadata.compound_name ||
    finalReport?.report_metadata.common_name ||
    finalReport?.report_metadata.smiles ||
    'the current compound';
  const hasGroundedReport = Boolean(finalReport);
  const promptList = hasGroundedReport ? REPORT_PROMPTS : DEFAULT_PROMPTS;

  const [messages, setMessages] = useState<ChatMessage[]>([createAssistantMessage(hasGroundedReport, compoundLabel)]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [isHydratingSession, setIsHydratingSession] = useState(false);
  const [activeChatSessionId, setActiveChatSessionId] = useState<string | null>(
    routeChatSessionId ?? report?.chat_session_id ?? null,
  );
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const processedInitialQuestion = useRef(false);

  useEffect(() => {
    setMessages([createAssistantMessage(hasGroundedReport, compoundLabel)]);
    setActiveChatSessionId(routeChatSessionId ?? report?.chat_session_id ?? null);
    processedInitialQuestion.current = false;
  }, [compoundLabel, hasGroundedReport, report?.chat_session_id, report?.session_id, routeChatSessionId]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);

  useEffect(() => {
    let cancelled = false;

    const hydrateChatSession = async () => {
      if (!user?.id || !activeChatSessionId) {
        setIsHydratingSession(false);
        return;
      }

      setIsHydratingSession(true);

      try {
        const storedSession = await loadChatSessionFromFirestore(user.id, activeChatSessionId);
        if (cancelled || !storedSession || storedSession.messages.length === 0) {
          return;
        }

        setMessages(storedSession.messages.map((message) => fromPersistedMessage(message)));
      } catch (error) {
        console.warn('Failed to hydrate report chat session from Firestore:', error);
      } finally {
        if (!cancelled) {
          setIsHydratingSession(false);
        }
      }
    };

    void hydrateChatSession();

    return () => {
      cancelled = true;
    };
  }, [activeChatSessionId, user?.id]);

  const sendMessage = async (messageText?: string) => {
    const userInput = (messageText ?? input).trim();
    if (!userInput || isTyping) {
      return;
    }

    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: userInput,
      timestamp: Date.now(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsTyping(true);

    try {
      const result = await agentChat(userInput, {
        chatSessionId: activeChatSessionId,
        analysisSessionId: routeAnalysisSessionId ?? report?.session_id ?? null,
        reportState: finalReport
          ? {
              smiles_input: finalReport.report_metadata.smiles,
              final_report: finalReport,
              evidence_qa_result: report?.evidence_qa_result,
            }
          : routeReportState,
      });

      const resolvedChatSessionId = result.chat_session_id ?? activeChatSessionId;
      if (resolvedChatSessionId) {
        setActiveChatSessionId(resolvedChatSessionId);
      }

      const assistantMessage: ChatMessage = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: result.response,
        timestamp: Date.now(),
      };

      setMessages((prev) => [...prev, assistantMessage]);

      if (user?.id && resolvedChatSessionId) {
        try {
          await appendChatTurnToFirestore(user.id, {
            sessionId: resolvedChatSessionId,
            analysisSessionId: routeAnalysisSessionId ?? report?.session_id ?? null,
            smiles: groundedSmiles,
            messages: [toPersistedMessage(userMessage), toPersistedMessage(assistantMessage)],
          });
        } catch (persistError) {
          console.warn('Failed to persist report chat session turn:', persistError);
        }
      }
    } catch (error) {
      const fallback = error instanceof Error ? error.message : 'Chat runtime error.';
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: hasGroundedReport
            ? `I couldn't fetch a grounded report answer right now: ${fallback}`
            : `I couldn't reach the assistant right now: ${fallback}`,
          timestamp: Date.now(),
        },
      ]);
    } finally {
      setIsTyping(false);
    }
  };

  useEffect(() => {
    if (!initialQuestion || processedInitialQuestion.current || isHydratingSession) {
      return;
    }

    processedInitialQuestion.current = true;
    void sendMessage(initialQuestion);
  }, [initialQuestion, isHydratingSession]);

  return (
    <div style={{ minHeight: '100vh', backgroundColor: 'var(--bg)', fontFamily: 'Inter, sans-serif' }}>
      <Navbar />

      <main className="mx-auto flex w-full max-w-7xl flex-col gap-6 px-4 py-6 sm:px-6 lg:grid lg:grid-cols-[340px_1fr] lg:px-8">
        <section
          className="rounded-3xl border p-6"
          style={{ backgroundColor: 'var(--surface)', borderColor: 'var(--border)' }}
        >
          <div className="mb-6 flex items-start gap-3">
            <div
              className="flex h-12 w-12 items-center justify-center rounded-2xl"
              style={{ backgroundColor: 'var(--accent-blue-muted)', color: 'var(--accent-blue)' }}
            >
              <Bot className="h-6 w-6" />
            </div>
            <div>
              <p className="text-xs font-semibold uppercase tracking-[0.18em]" style={{ color: 'var(--text-faint)' }}>
                Chat Workspace
              </p>
              <h1 className="text-2xl font-semibold" style={{ color: 'var(--text)' }}>
                ToxAgent Assistant
              </h1>
            </div>
          </div>

          <p className="mb-6 text-sm leading-7" style={{ color: 'var(--text-muted)' }}>
            {hasGroundedReport
              ? `This chat is grounded on the latest report for ${compoundLabel}. Use it to clarify evidence, interpret scores, and plan next steps.`
              : 'This chat page is ready, but there is no active report in memory yet. You can still ask product questions, or run an analysis first for report-grounded answers.'}
          </p>

          <div
            className="mb-6 rounded-2xl border p-4"
            style={{ backgroundColor: 'var(--surface-alt)', borderColor: 'var(--border)' }}
          >
            <div className="mb-3 flex items-center gap-2">
              <FlaskConical className="h-4 w-4" style={{ color: 'var(--accent-blue)' }} />
              <p className="text-sm font-medium" style={{ color: 'var(--text)' }}>
                Session Context
              </p>
            </div>
            <div className="space-y-2 text-sm" style={{ color: 'var(--text-muted)' }}>
              <p>Grounded report: {hasGroundedReport ? 'available' : 'not loaded'}</p>
              <p>Analysis session: {report?.session_id ?? 'none'}</p>
              <p>Chat session: {activeChatSessionId ?? 'new session will be created on first message'}</p>
            </div>
          </div>

          <div className="mb-6">
            <div className="mb-3 flex items-center gap-2">
              <Sparkles className="h-4 w-4" style={{ color: 'var(--accent-blue)' }} />
              <p className="text-sm font-medium" style={{ color: 'var(--text)' }}>
                Suggested Prompts
              </p>
            </div>
            <div className="flex flex-wrap gap-2">
              {promptList.map((prompt) => (
                <button
                  key={prompt}
                  type="button"
                  onClick={() => setInput(prompt)}
                  className="rounded-full border px-3 py-2 text-left text-xs transition-colors hover:bg-[var(--surface)]"
                  style={{
                    borderColor: 'var(--border)',
                    backgroundColor: 'var(--surface-alt)',
                    color: 'var(--text-muted)',
                  }}
                >
                  {prompt}
                </button>
              ))}
            </div>
          </div>

          <Button
            type="button"
            onClick={() => navigate('/')}
            className="w-full justify-between rounded-2xl px-4 py-6"
            style={{ backgroundColor: 'var(--accent-blue)', color: '#ffffff' }}
          >
            Run or restart an analysis
            <ArrowRight className="h-4 w-4" />
          </Button>
        </section>

        <section
          className="flex min-h-[70vh] flex-col overflow-hidden rounded-3xl border"
          style={{ backgroundColor: 'var(--surface)', borderColor: 'var(--border)' }}
        >
          <div
            className="border-b px-5 py-4"
            style={{ backgroundColor: 'var(--surface)', borderColor: 'var(--border)' }}
          >
            <p className="text-sm font-semibold" style={{ color: 'var(--text)' }}>
              Conversation
            </p>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
              {hasGroundedReport
                ? 'Answers should stay aligned with the current report context.'
                : 'Answers may be generic until a report is available.'}
            </p>
          </div>

          <div className="flex-1 space-y-4 overflow-y-auto px-4 py-5 sm:px-5" style={{ backgroundColor: 'var(--bg)' }}>
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex gap-3 ${message.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}
              >
                <div
                  className="flex h-9 w-9 shrink-0 items-center justify-center rounded-2xl"
                  style={{
                    backgroundColor: message.role === 'user' ? 'var(--accent-blue)' : 'var(--surface-alt)',
                    color: message.role === 'user' ? '#ffffff' : 'var(--accent-blue)',
                  }}
                >
                  {message.role === 'user' ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
                </div>
                <div
                  className={`max-w-[85%] rounded-3xl px-4 py-3 ${message.role === 'user' ? 'rounded-tr-md' : 'rounded-tl-md'}`}
                  style={{
                    backgroundColor: message.role === 'user' ? 'var(--accent-blue)' : 'var(--surface)',
                    color: message.role === 'user' ? '#ffffff' : 'var(--text)',
                    border: message.role === 'assistant' ? '1px solid var(--border)' : 'none',
                  }}
                >
                  <p className="text-sm leading-7 whitespace-pre-wrap">{message.content}</p>
                </div>
              </div>
            ))}

            {isTyping && (
              <div className="flex gap-3">
                <div
                  className="flex h-9 w-9 items-center justify-center rounded-2xl"
                  style={{ backgroundColor: 'var(--surface-alt)', color: 'var(--accent-blue)' }}
                >
                  <Bot className="h-4 w-4" />
                </div>
                <div
                  className="rounded-3xl rounded-tl-md border px-4 py-3"
                  style={{ backgroundColor: 'var(--surface)', borderColor: 'var(--border)' }}
                >
                  <div className="flex gap-1.5">
                    <div className="h-2 w-2 animate-bounce rounded-full" style={{ backgroundColor: 'var(--text-muted)', animationDelay: '0ms' }} />
                    <div className="h-2 w-2 animate-bounce rounded-full" style={{ backgroundColor: 'var(--text-muted)', animationDelay: '150ms' }} />
                    <div className="h-2 w-2 animate-bounce rounded-full" style={{ backgroundColor: 'var(--text-muted)', animationDelay: '300ms' }} />
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          <div className="border-t p-4 sm:p-5" style={{ borderColor: 'var(--border)' }}>
            <div className="flex flex-col gap-3 sm:flex-row">
              <textarea
                value={input}
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' && !event.shiftKey) {
                    event.preventDefault();
                    void sendMessage();
                  }
                }}
                placeholder={
                  hasGroundedReport
                    ? 'Ask about risk, evidence, mechanisms, or recommendations...'
                    : 'Ask about the product or run an analysis for grounded report chat...'
                }
                className="min-h-[64px] flex-1 resize-none rounded-2xl border px-4 py-3 text-sm outline-none"
                style={{
                  backgroundColor: 'var(--surface-alt)',
                  borderColor: 'var(--border)',
                  color: 'var(--text)',
                }}
                disabled={isTyping}
              />
              <Button
                type="button"
                onClick={() => void sendMessage()}
                disabled={!input.trim() || isTyping}
                className="h-auto rounded-2xl px-5 py-3 sm:w-auto"
                style={{
                  backgroundColor: input.trim() && !isTyping ? 'var(--accent-blue)' : 'var(--border)',
                  color: '#ffffff',
                }}
              >
                <Send className="mr-2 h-4 w-4" />
                Send
              </Button>
            </div>
          </div>
        </section>
      </main>

      <Footer />
    </div>
  );
}
