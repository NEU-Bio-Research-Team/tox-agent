import { useCallback, useEffect, useRef, useState } from 'react';

const NEAR_BOTTOM_PX = 80;

/**
 * Plan section 8.2: "chỉ tự cuộn nếu người dùng đang ở cuối. Nếu đang đọc
 * phía trên, hiện 'Tin nhắn mới'." `tick` is any value that changes once per
 * new message/answer arriving (e.g. `messages.length`) — this hook doesn't
 * read message content itself so it stays agnostic of the transcript shape.
 *
 * UI-01 adds two observers, because `tick` is not the only thing that can move
 * the bottom of the transcript out from under the reader:
 *
 * - **The content grows without a new message.** A streaming answer, a report
 *   block finishing its fetch, or an image loading all change the scroll height
 *   while `messages.length` stays put. Keyed only on `tick`, the view stayed
 *   where it was and the newest text grew away below the fold.
 * - **The scroller itself shrinks.** The composer now occupies real layout
 *   height (it used to be absolutely positioned over the transcript), so a draft
 *   growing from one line to eight takes that height *from* the scroller. A
 *   reader sitting at the bottom would otherwise find themselves scrolled up by
 *   however much the composer grew.
 *
 * Both are handled the same way: if the reader was at the bottom, keep them
 * there; if they were reading further up, leave them alone and let the jump
 * button say something arrived.
 */
export function useStickToBottom(tick: number) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [showJump, setShowJump] = useState(false);
  const wasAtBottomRef = useRef(true);
  const lastTickRef = useRef(tick);

  const pin = useCallback(() => {
    const el = containerRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, []);

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
    const el = containerRef.current;
    // `ResizeObserver` is in every browser this app supports, but jsdom in some
    // test setups has no implementation and a missing global would take the
    // whole transcript down rather than degrade the scrolling.
    if (!el || typeof ResizeObserver === 'undefined') return;
    const observer = new ResizeObserver(() => {
      if (wasAtBottomRef.current) pin();
    });
    // The viewport, so a composer that grew or a rotated phone re-pins; and the
    // content, so a streaming answer or a late-loading figure does too.
    observer.observe(el);
    const content = el.firstElementChild;
    if (content) observer.observe(content);
    return () => observer.disconnect();
  }, [pin]);

  useEffect(() => {
    if (tick === lastTickRef.current) return;
    lastTickRef.current = tick;
    if (!containerRef.current) return;
    if (wasAtBottomRef.current) {
      pin();
    } else {
      setShowJump(true);
    }
  }, [tick, pin]);

  const jumpToBottom = () => {
    if (!containerRef.current) return;
    pin();
    wasAtBottomRef.current = true;
    setShowJump(false);
  };

  return { containerRef, showJump, jumpToBottom };
}
