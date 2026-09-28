import { act, renderHook } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { useStickToBottom } from './useStickToBottom';

/**
 * UI-01. The composer used to be absolutely positioned over the transcript with
 * a fixed `pb-48` on the scroller to compensate. The composer's real height
 * depends on the viewport, the draft's line count, attachments and the context
 * chip, so the compensation was wrong in exactly the cases that mattered and the
 * last message ended up underneath it.
 *
 * The fix puts the composer in normal flex flow, which means the scroller now
 * *shrinks* when the composer grows. These tests cover what that requires of the
 * hook: a reader at the bottom stays at the bottom when the viewport resizes or
 * the content grows, and a reader further up is never yanked.
 */

type ResizeCallback = (entries: unknown[], observer: unknown) => void;

let callbacks: ResizeCallback[] = [];

class FakeResizeObserver {
  constructor(callback: ResizeCallback) {
    callbacks.push(callback);
  }
  observe() {}
  disconnect() {}
}

/** A scroller whose geometry the test drives directly. jsdom lays nothing out,
 * so `scrollHeight`/`clientHeight` have to be supplied. */
function scroller({ scrollHeight = 1000, clientHeight = 400, scrollTop = 600 } = {}) {
  const element = document.createElement('div');
  const content = document.createElement('div');
  element.append(content);
  Object.defineProperty(element, 'scrollHeight', {
    get: () => scrollHeight,
    configurable: true,
  });
  Object.defineProperty(element, 'clientHeight', {
    get: () => clientHeight,
    configurable: true,
  });
  let currentTop = scrollTop;
  Object.defineProperty(element, 'scrollTop', {
    get: () => currentTop,
    set: (value: number) => {
      currentTop = value;
    },
    configurable: true,
  });
  return element;
}

function mount(element: HTMLElement, tick = 0) {
  // The ref is populated *during* render, before effects run — the order React
  // itself uses when it attaches a ref to a real DOM node. Setting it after
  // `renderHook` returns would leave the hook's effects having already run
  // against `null`, so no scroll listener and no observer would ever attach and
  // every assertion below would be testing nothing.
  return renderHook(
    (props: { tick: number }) => {
      const hook = useStickToBottom(props.tick);
      (hook.containerRef as { current: HTMLElement | null }).current = element;
      return hook;
    },
    { initialProps: { tick } },
  );
}

describe('useStickToBottom', () => {
  beforeEach(() => {
    callbacks = [];
    vi.stubGlobal('ResizeObserver', FakeResizeObserver);
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('pins to the bottom when a new message arrives and the reader was there', () => {
    // scrollHeight 1000 - scrollTop 600 - clientHeight 400 = 0, i.e. at bottom.
    const element = scroller({ scrollTop: 600 });
    const hook = mount(element);

    hook.rerender({ tick: 1 });

    expect(element.scrollTop).toBe(1000);
    expect(hook.result.current.showJump).toBe(false);
  });

  it('does not yank a reader who has scrolled up, and offers the jump instead', () => {
    const element = scroller({ scrollTop: 0 });
    const hook = mount(element);
    // The scroll listener is what records "not at the bottom".
    act(() => element.dispatchEvent(new Event('scroll')));

    hook.rerender({ tick: 1 });

    expect(element.scrollTop).toBe(0);
    expect(hook.result.current.showJump).toBe(true);
  });

  it('re-pins when the scroller shrinks because the composer grew', () => {
    // The case the fixed `pb-48` could not handle: an eight-line draft takes
    // height from the transcript, and without this the reader is scrolled up by
    // however much the composer grew.
    const element = scroller({ scrollTop: 600 });
    mount(element);
    element.scrollTop = 400;

    act(() => callbacks.forEach((callback) => callback([], null)));

    expect(element.scrollTop).toBe(1000);
  });

  it('re-pins when the content grows without a new message', () => {
    // A streaming answer or a late-loading report block changes scrollHeight
    // while `messages.length` stays put.
    const element = scroller({ scrollTop: 600 });
    mount(element);
    element.scrollTop = 100;
    act(() => element.dispatchEvent(new Event('scroll')));
    // Back to the bottom, so the observer should follow the growth.
    element.scrollTop = 600;
    act(() => element.dispatchEvent(new Event('scroll')));

    act(() => callbacks.forEach((callback) => callback([], null)));

    expect(element.scrollTop).toBe(1000);
  });

  it('leaves a scrolled-up reader alone when the composer resizes', () => {
    const element = scroller({ scrollTop: 0 });
    mount(element);
    act(() => element.dispatchEvent(new Event('scroll')));

    act(() => callbacks.forEach((callback) => callback([], null)));

    expect(element.scrollTop).toBe(0);
  });

  it('jumpToBottom pins and clears the badge', () => {
    const element = scroller({ scrollTop: 0 });
    const hook = mount(element);
    act(() => element.dispatchEvent(new Event('scroll')));
    hook.rerender({ tick: 1 });
    expect(hook.result.current.showJump).toBe(true);

    act(() => hook.result.current.jumpToBottom());

    expect(element.scrollTop).toBe(1000);
    expect(hook.result.current.showJump).toBe(false);
  });

  it('survives an environment with no ResizeObserver', () => {
    // Some jsdom setups have none. A missing global must degrade the scrolling,
    // not take the whole transcript down.
    vi.stubGlobal('ResizeObserver', undefined);
    const element = scroller({ scrollTop: 600 });
    const hook = mount(element);

    hook.rerender({ tick: 1 });

    expect(element.scrollTop).toBe(1000);
  });
});
