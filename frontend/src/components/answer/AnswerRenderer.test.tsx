import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router';
import { describe, expect, it } from 'vitest';
import { AnswerRenderer } from './AnswerRenderer';
import type { Claim, GroundedAnswer } from '../../lib/api/types';

/** PROD-08: answer_markdown is model output — untrusted data. These tests
 * exist to keep that boundary from silently regressing, not to describe
 * every rendering detail. */

function baseAnswer(overrides: Partial<GroundedAnswer>): GroundedAnswer {
  return {
    schema_version: 'grounded-answer-v1',
    answer_id: 'ans_1',
    run_id: 'run_1',
    answer_markdown: '',
    claims: [],
    limitations: [],
    recommended_next_steps: [],
    candidate_generation: 1,
    is_fallback: false,
    content_sha256: 'x',
    created_at: '2026-09-06T00:00:00Z',
    ...overrides,
  };
}

function renderAnswer(answer: GroundedAnswer) {
  return render(
    <MemoryRouter>
      <AnswerRenderer answer={answer} sessionId="ses_1" />
    </MemoryRouter>,
  );
}

describe('AnswerRenderer', () => {
  it('turns a claim-linked number into a chip, not a raw link', () => {
    const claim: Claim = {
      claim_id: 'clm_1',
      kind: 'numeric',
      text: 'x',
      transform: 'round:3',
      citation_ids: [],
      rendered_value: '0.731',
    };
    renderAnswer(
      baseAnswer({
        answer_markdown: 'The predicted hERG blocker probability is 0.731.',
        claims: [claim],
      }),
    );
    const chip = screen.getByRole('button', { name: /0\.731/ });
    expect(chip).toBeInTheDocument();
    // A chip is a <button>, never an <a href="claim:..."> reaching the DOM —
    // the claim: scheme must never leak out as a real, clickable href.
    expect(document.querySelector('a[href^="claim:"]')).toBeNull();
  });

  it('never renders raw HTML embedded in model output', () => {
    renderAnswer(
      baseAnswer({
        answer_markdown: 'Ignore previous instructions. <img src=x onerror=alert(1)>',
      }),
    );
    expect(document.querySelector('img')).toBeNull();
    expect(document.querySelector('[onerror]')).toBeNull();
  });

  it('renders an ordinary external link as a real, safe, new-tab link', () => {
    renderAnswer(
      baseAnswer({
        answer_markdown: 'See [this paper](https://example.org/study).',
      }),
    );
    const link = screen.getByRole('link', { name: 'this paper' });
    expect(link).toHaveAttribute('href', 'https://example.org/study');
    expect(link).toHaveAttribute('target', '_blank');
    expect(link).toHaveAttribute('rel', expect.stringContaining('noopener'));
  });

  it('drops a javascript: URI rather than rendering it as a link', () => {
    renderAnswer(
      baseAnswer({
        answer_markdown: '[click me](javascript:alert(1))',
      }),
    );
    expect(screen.queryByRole('link', { name: 'click me' })).toBeNull();
    // The text content survives (unwrapDisallowed) — only the dangerous
    // href is refused, not the whole sentence silently disappearing.
    expect(screen.getByText('click me')).toBeInTheDocument();
  });

  it('shows the fallback badge only when the answer actually is a fallback', () => {
    const { rerender } = renderAnswer(baseAnswer({ answer_markdown: 'ok', is_fallback: false }));
    expect(screen.queryByText('ĐÁP ÁN DỰ PHÒNG')).toBeNull();

    rerender(
      <MemoryRouter>
        <AnswerRenderer answer={baseAnswer({ answer_markdown: 'ok', is_fallback: true })} sessionId="ses_1" />
      </MemoryRouter>,
    );
    expect(screen.getAllByText('ĐÁP ÁN DỰ PHÒNG').length).toBeGreaterThan(0);
  });
});
