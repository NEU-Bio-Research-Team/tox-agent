import { useState } from 'react';
import { MessageCircle, X, Send, Bot } from 'lucide-react';
import { useNavigate } from 'react-router';
import { Button } from './ui/button';
import { type FinalReport } from '../../lib/api';

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
  const navigate = useNavigate();
  const [isOpen, setIsOpen] = useState(false);
  const [input, setInput] = useState('');

  const handleSend = () => {
    if (!input.trim()) return;
    const userInput = input.trim();
    setInput('');
    setIsOpen(false);
    navigate('/chat', {
      state: {
        question: userInput,
        chatSessionId: chatSessionId ?? null,
        analysisSessionId: analysisSessionId ?? null,
        reportState: reportState ?? null,
      },
    });
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

          {/* Intro */}
          <div 
            className="flex-1 overflow-y-auto p-4 space-y-4"
            style={{ backgroundColor: 'var(--bg)' }}
          >
            <div
              className="rounded-2xl px-4 py-3"
              style={{
                backgroundColor: 'var(--surface)',
                border: '1px solid var(--border)',
              }}
            >
              <p className="text-sm leading-relaxed" style={{ color: 'var(--text)' }}>
                Type your question and press Enter. We will move to the full chat page and start processing immediately.
              </p>
            </div>
          </div>

          {/* Quick Prompts */}
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

          {/* Input */}
          <div className="p-4 border-t" style={{ borderColor: 'var(--border)' }}>
            <div className="flex gap-2">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSend()}
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
                disabled={!input.trim()}
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
