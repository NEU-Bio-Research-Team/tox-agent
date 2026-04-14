import { useState, useRef, useEffect } from 'react';
import { MessageCircle, X, Send, Bot, User } from 'lucide-react';
import { Button } from './ui/button';
import { agentChat, type FinalReport } from '../../lib/api';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
}

const AI_PROMPTS = [
  "Summarize the key toxicity risk from this report.",
  "What mechanism evidence supports the highest risk?",
  "Which recommendations should be prioritized first?",
  "What are the key limitations of this report?",
];

interface AIChatbotProps {
  chatSessionId?: string | null;
  analysisSessionId?: string | null;
  reportState?: {
    smiles_input?: string;
    final_report?: FinalReport;
    evidence_qa_result?: Record<string, unknown>;
  } | null;
}

export function AIChatbot({ chatSessionId, analysisSessionId, reportState }: AIChatbotProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      role: 'assistant',
      content:
        "Hi! I can answer follow-up questions grounded on the current toxicity report. Ask me about risk, mechanisms, literature evidence, or recommendations.",
      timestamp: Date.now(),
    }
  ]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [activeChatSessionId, setActiveChatSessionId] = useState<string | null>(chatSessionId ?? null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    setActiveChatSessionId(chatSessionId ?? null);
  }, [chatSessionId]);

  const handleSend = async () => {
    if (!input.trim()) return;
    const userInput = input.trim();

    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      content: userInput,
      timestamp: Date.now(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsTyping(true);

    try {
      const result = await agentChat(userInput, {
        chatSessionId: activeChatSessionId,
        analysisSessionId,
        reportState,
      });

      if (result.chat_session_id) {
        setActiveChatSessionId(result.chat_session_id);
      }

      setMessages(prev => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: result.response,
          timestamp: Date.now(),
        }
      ]);
    } catch (error) {
      const fallback = error instanceof Error
        ? error.message
        : 'Chat runtime error. Please rerun analysis and try again.';
      setMessages(prev => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: `I couldn't fetch a grounded report answer right now: ${fallback}`,
          timestamp: Date.now(),
        }
      ]);
    } finally {
      setIsTyping(false);
    }
  };

  const handlePromptClick = (prompt: string) => {
    setInput(prompt);
  };

  return (
    <>
      {/* Chatbot Toggle Button */}
      {!isOpen && (
        <button
          onClick={() => setIsOpen(true)}
          className="fixed bottom-6 right-6 z-50 w-14 h-14 rounded-full shadow-lg flex items-center justify-center transition-all hover:scale-110"
          style={{ backgroundColor: 'var(--accent-blue)' }}
        >
          <MessageCircle className="w-6 h-6 text-white" />
        </button>
      )}

      {/* Chatbot Window */}
      {isOpen && (
        <div 
          className="fixed bottom-6 right-6 z-50 w-96 rounded-xl shadow-2xl flex flex-col overflow-hidden"
          style={{ 
            backgroundColor: 'var(--surface)', 
            border: '1px solid var(--border)',
            height: '600px',
            maxHeight: 'calc(100vh - 100px)'
          }}
        >
          {/* Header */}
          <div 
            className="p-4 border-b flex items-center justify-between"
            style={{ 
              backgroundColor: 'var(--accent-blue)', 
              borderColor: 'var(--border)'
            }}
          >
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-full bg-white/20 flex items-center justify-center">
                <Bot className="w-5 h-5 text-white" />
              </div>
              <div>
                <h3 className="font-semibold text-white">ToxAgent AI</h3>
                <p className="text-xs text-white/80">Always here to help</p>
              </div>
            </div>
            <button
              onClick={() => setIsOpen(false)}
              className="text-white/80 hover:text-white transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* Messages */}
          <div 
            className="flex-1 overflow-y-auto p-4 space-y-4"
            style={{ backgroundColor: 'var(--bg)' }}
          >
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex gap-3 ${message.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}
              >
                <div 
                  className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0"
                  style={{ 
                    backgroundColor: message.role === 'user' 
                      ? 'var(--accent-blue)' 
                      : 'var(--surface-alt)' 
                  }}
                >
                  {message.role === 'user' ? (
                    <User className="w-4 h-4" style={{ color: '#ffffff' }} />
                  ) : (
                    <Bot className="w-4 h-4" style={{ color: 'var(--accent-blue)' }} />
                  )}
                </div>
                <div
                  className={`rounded-2xl px-4 py-2.5 max-w-[75%] ${
                    message.role === 'user' ? 'rounded-tr-sm' : 'rounded-tl-sm'
                  }`}
                  style={{
                    backgroundColor: message.role === 'user' 
                      ? 'var(--accent-blue)' 
                      : 'var(--surface)',
                    color: message.role === 'user' ? '#ffffff' : 'var(--text)',
                    border: message.role === 'assistant' ? '1px solid var(--border)' : 'none'
                  }}
                >
                  <p className="text-sm leading-relaxed">{message.content}</p>
                </div>
              </div>
            ))}

            {isTyping && (
              <div className="flex gap-3">
                <div 
                  className="w-8 h-8 rounded-full flex items-center justify-center"
                  style={{ backgroundColor: 'var(--surface-alt)' }}
                >
                  <Bot className="w-4 h-4" style={{ color: 'var(--accent-blue)' }} />
                </div>
                <div
                  className="rounded-2xl rounded-tl-sm px-4 py-3"
                  style={{
                    backgroundColor: 'var(--surface)',
                    border: '1px solid var(--border)'
                  }}
                >
                  <div className="flex gap-1">
                    <div className="w-2 h-2 rounded-full animate-bounce" style={{ backgroundColor: 'var(--text-muted)', animationDelay: '0ms' }} />
                    <div className="w-2 h-2 rounded-full animate-bounce" style={{ backgroundColor: 'var(--text-muted)', animationDelay: '150ms' }} />
                    <div className="w-2 h-2 rounded-full animate-bounce" style={{ backgroundColor: 'var(--text-muted)', animationDelay: '300ms' }} />
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Quick Prompts */}
          {messages.length <= 2 && (
            <div className="px-4 py-2 border-t" style={{ borderColor: 'var(--border)' }}>
              <p className="text-xs mb-2" style={{ color: 'var(--text-muted)' }}>Quick questions:</p>
              <div className="flex flex-wrap gap-2">
                {AI_PROMPTS.map((prompt, idx) => (
                  <button
                    key={idx}
                    onClick={() => handlePromptClick(prompt)}
                    className="text-xs px-3 py-1.5 rounded-full hover:bg-[var(--surface-alt)] transition-colors"
                    style={{ 
                      backgroundColor: 'var(--surface)',
                      border: '1px solid var(--border)',
                      color: 'var(--text-muted)'
                    }}
                  >
                    {prompt}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Input */}
          <div className="p-4 border-t" style={{ borderColor: 'var(--border)' }}>
            <div className="flex gap-2">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSend()}
                placeholder="Ask me anything..."
                className="flex-1 px-4 py-2 rounded-lg border outline-none"
                style={{
                  backgroundColor: 'var(--surface-alt)',
                  borderColor: 'var(--border)',
                  color: 'var(--text)'
                }}
              />
              <Button
                onClick={handleSend}
                disabled={!input.trim() || isTyping}
                className="w-10 h-10 rounded-lg flex items-center justify-center"
                style={{ 
                  backgroundColor: input.trim() ? 'var(--accent-blue)' : 'var(--border)',
                  color: '#ffffff'
                }}
              >
                <Send className="w-4 h-4" />
              </Button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
