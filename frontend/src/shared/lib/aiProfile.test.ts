/**
 * I06 — "Runtime mặc định" was shown for a null profile without checking that
 * a usable default existed.
 *
 * That wording reads as "a default is in force". It was shown in a
 * predictor-only deployment, where there is no runtime at all, and in an agent
 * deployment with no profiles configured — two states where nothing would run.
 *
 * The wording is the fix, so it is what is tested. It now labels the default
 * profile choice in Settings; the per-session popover it used to live in is
 * gone, and profileForSession below decides what a session is pinned to.
 */
import { describe, expect, it } from 'vitest';

import { describeAgentProfileChoice, profileForSession } from './aiProfile';

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

describe('profileForSession', () => {
  const ready = ['conn_a', 'conn_b'];

  it('pins the Settings default while it is ready', () => {
    expect(profileForSession({ defaultProfileId: 'conn_a', currentProfileId: 'conn_b', readyProfileIds: ready })).toBe('conn_a');
  });

  it('honours an explicit choice of the server runtime', () => {
    expect(profileForSession({ defaultProfileId: null, currentProfileId: 'conn_b', readyProfileIds: ready })).toBeNull();
  });

  it('keeps what the session had when no default was ever chosen', () => {
    // A profile pinned through the old per-session popover must survive the
    // popover's removal.
    expect(profileForSession({ defaultProfileId: undefined, currentProfileId: 'conn_b', readyProfileIds: ready })).toBe('conn_b');
  });

  it('unpins a profile that is no longer ready, whichever way it was chosen', () => {
    // A deleted or failed profile left pinned would fail every run, and there
    // is no per-session control left to unpin it.
    expect(profileForSession({ defaultProfileId: 'conn_gone', currentProfileId: null, readyProfileIds: ready })).toBeNull();
    expect(profileForSession({ defaultProfileId: undefined, currentProfileId: 'conn_gone', readyProfileIds: ready })).toBeNull();
  });
});
