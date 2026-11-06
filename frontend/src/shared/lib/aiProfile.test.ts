/**
 * I06 — "Runtime mặc định" was shown for a null profile without checking that
 * a usable default existed.
 *
 * That wording reads as "a default is in force". It was shown in a
 * predictor-only deployment, where there is no runtime at all, and in an agent
 * deployment with no profiles configured — two states where nothing would run.
 *
 * The wording is the fix, so it is what is tested. It now labels the default
 * connection state in the chat's model picker, and choiceForSession below
 * decides what a session runs on when it is opened.
 */
import { describe, expect, it } from 'vitest';

import { choiceForSession, describeAgentProfileChoice, sameChoice } from './aiProfile';

describe('predictor-only deployment', () => {
  const choice = describeAgentProfileChoice({ agentEnabled: false, readyProfileCount: 0 });

  it('does not name a default runtime', () => {
    expect(choice.defaultOptionLabel).toBe('Không có agent runtime');
    expect(choice.defaultOptionLabel).not.toMatch(/mặc định/i);
  });

  it('says which capabilities are unavailable here', () => {
    expect(choice.notice).toMatch(/chỉ chạy dự đoán/i);
  });

  it('disables a choice that cannot take effect', () => {
    expect(choice.disabled).toBe(true);
  });

  it('says the same when profiles exist, because nothing can run them', () => {
    const withProfiles = describeAgentProfileChoice({
      agentEnabled: false,
      readyProfileCount: 3,
    });
    expect(withProfiles.defaultOptionLabel).toBe('Không có agent runtime');
    expect(withProfiles.disabled).toBe(true);
  });
});

describe('agent deployment', () => {
  it('reports an unconfigured AI rather than a default when no profile is ready', () => {
    const choice = describeAgentProfileChoice({ agentEnabled: true, readyProfileCount: 0 });
    expect(choice.defaultOptionLabel).toBe('Chưa cấu hình AI');
    expect(choice.notice).toMatch(/chưa có profile/i);
    // Still selectable: the user can configure one and come back.
    expect(choice.disabled).toBe(false);
  });

  it('offers the server default only once a usable profile exists', () => {
    const choice = describeAgentProfileChoice({ agentEnabled: true, readyProfileCount: 1 });
    expect(choice.defaultOptionLabel).toBe('Runtime mặc định của server');
    expect(choice.notice).toBeNull();
    expect(choice.disabled).toBe(false);
  });
});

describe('before /health/ready answers', () => {
  it('claims nothing either way', () => {
    const choice = describeAgentProfileChoice({ agentEnabled: null, readyProfileCount: 0 });
    // Not "no runtime" — that would be a claim we cannot make yet — and not
    // "default runtime" either.
    expect(choice.defaultOptionLabel).toBe('Chưa cấu hình AI');
    expect(choice.disabled).toBe(false);
  });
});

describe('choiceForSession', () => {
  const ready = ['conn_a', 'conn_b'];
  const a = { connectionId: 'conn_a', modelId: 'gpt-5.1' };
  const b = { connectionId: 'conn_b', modelId: null };

  it('keeps the model a session picked while its connection is ready', () => {
    expect(choiceForSession({ saved: a, lastChoice: b, readyConnectionIds: ready })).toEqual(a);
  });

  it('starts a session with no pick on the last model picked anywhere', () => {
    expect(choiceForSession({ saved: null, lastChoice: b, readyConnectionIds: ready })).toEqual(b);
  });

  it('unpins a connection that is no longer ready, whichever way it was chosen', () => {
    // A deleted or failed connection left pinned would fail every run.
    const gone = { connectionId: 'conn_gone', modelId: 'x' };
    expect(choiceForSession({ saved: gone, lastChoice: gone, readyConnectionIds: ready })).toBeNull();
    expect(choiceForSession({ saved: gone, lastChoice: a, readyConnectionIds: ready })).toEqual(a);
  });
});

describe('sameChoice', () => {
  it("treats a missing model as the connection's own", () => {
    expect(sameChoice({ connectionId: 'c', modelId: null }, { connectionId: 'c', modelId: null })).toBe(true);
    expect(sameChoice({ connectionId: 'c', modelId: 'm' }, { connectionId: 'c', modelId: null })).toBe(false);
    expect(sameChoice(null, null)).toBe(true);
  });
});

describe('a server with no AI of its own', () => {
  it('asks the user to connect one instead of naming a server default', () => {
    const choice = describeAgentProfileChoice({ agentEnabled: true, readyProfileCount: 0, userAiRequired: true });
    expect(choice.defaultOptionLabel).not.toMatch(/server/i);
    expect(choice.notice).toMatch(/không có AI riêng/);
    expect(choice.disabled).toBe(false);
  });

  it('pins a ready connection rather than the runtime that is not there', () => {
    expect(choiceForSession({ saved: null, lastChoice: null, readyConnectionIds: ['con_a', 'con_b'], userAiRequired: true }))
      .toEqual({ connectionId: 'con_a', modelId: null });
    expect(choiceForSession({ saved: null, lastChoice: { connectionId: 'con_b', modelId: 'm' }, readyConnectionIds: ['con_a', 'con_b'], userAiRequired: true }))
      .toEqual({ connectionId: 'con_b', modelId: 'm' });
    expect(choiceForSession({ saved: null, lastChoice: null, readyConnectionIds: [], userAiRequired: true })).toBeNull();
  });

  it('keeps the server runtime where the server has one', () => {
    expect(choiceForSession({ saved: null, lastChoice: null, readyConnectionIds: ['con_a'] })).toBeNull();
  });
});
