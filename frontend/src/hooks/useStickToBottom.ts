import { useEffect, useRef, useState } from 'react';

const NEAR_BOTTOM_PX = 80;

/**
 * Plan section 8.2: "chỉ tự cuộn nếu người dùng đang ở cuối. Nếu đang đọc
 * phía trên, hiện 'Tin nhắn mới'." `tick` is any value that changes once per
 * new message/answer arriving (e.g. `messages.length`) — this hook doesn't
 * read message content itself so it stays agnostic of the transcript shape.
 */
export function useStickToBottom(tick: number) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [showJump, setShowJump] = useState(false);
  const wasAtBottomRef = useRef(true);
  const lastTickRef = useRef(tick);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const onScroll = () => {
      const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < NEAR_BOTTOM_PX;
      wasAtBottomRef.current = atBottom;
      if (atBottom) setShowJump(false);
    };
    el.addEventListener('scroll', onScroll, { passive: true });
    return () => el.removeEventListener('scroll', onScroll);
  }, []);

  useEffect(() => {
    if (tick === lastTickRef.current) return;
    lastTickRef.current = tick;
    const el = containerRef.current;
    if (!el) return;
    if (wasAtBottomRef.current) {
      el.scrollTop = el.scrollHeight;
    } else {
      setShowJump(true);
    }
  }, [tick]);

  const jumpToBottom = () => {
    const el = containerRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
    wasAtBottomRef.current = true;
    setShowJump(false);
  };

  return { containerRef, showJump, jumpToBottom };
}
