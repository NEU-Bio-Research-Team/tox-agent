import type { ReactNode } from 'react';

export function MessageBubble({ role, children }: { role: 'user' | 'assistant'; children: ReactNode }) {
  const isUser = role === 'user';
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className="max-w-[85%] rounded-2xl px-4 py-3 text-sm leading-relaxed"
        style={
          isUser
            ? { backgroundColor: 'var(--accent-blue)', color: '#ffffff' }
            : { backgroundColor: 'var(--surface)', border: '1px solid var(--border)', color: 'var(--text)' }
        }
      >
        {children}
      </div>
    </div>
  );
}
