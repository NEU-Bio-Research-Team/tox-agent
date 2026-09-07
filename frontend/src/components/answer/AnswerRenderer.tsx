import { useMemo } from 'react';
import { Link } from 'react-router';
import ReactMarkdown, { defaultUrlTransform } from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { Components } from 'react-markdown';
import type { GroundedAnswer } from '../../lib/api/types';
import { linkifyClaims, CLAIM_LINK_SCHEME } from '../../lib/answerMarkdown';
import { ClaimChip } from './ClaimChip';
import { LimitationBlock } from './LimitationBlock';
import { FallbackBadge } from './FallbackBadge';

/** PROD-08: `answer_markdown` is model output — untrusted data, never an
 * instruction. react-markdown (without rehype-raw) already refuses to render
 * embedded HTML; this allowlist further restricts which block/inline
 * elements can appear at all, and `a`/`img` get their own guards below. */
const ALLOWED_TAGS: string[] = [
  'p', 'strong', 'em', 'ul', 'ol', 'li', 'code', 'pre', 'h3', 'h4', 'blockquote', 'table', 'thead', 'tbody', 'tr', 'th', 'td', 'a',
];

function buildComponents(sessionId: string, answer: GroundedAnswer): Components {
  const claimById = new Map(answer.claims.map((claim) => [claim.claim_id, claim]));
  const orderById = new Map(answer.claims.map((claim, index) => [claim.claim_id, index]));

  const components: Components = {
    a: ({ href, children }) => {
      if (href?.startsWith(CLAIM_LINK_SCHEME)) {
        const claimId = href.slice(CLAIM_LINK_SCHEME.length);
        const claim = claimById.get(claimId);
        if (claim) {
          return <ClaimChip claim={claim} sessionId={sessionId} index={orderById.get(claimId) ?? 0} />;
        }
      }
      // Any other link in model output is untrusted; only allow it to be an
      // ordinary external link, never same-tab navigation inside the app.
      if (href && /^https?:\/\//i.test(href)) {
        return (
          <a href={href} target="_blank" rel="noopener noreferrer nofollow" style={{ color: 'var(--accent-blue)' }}>
            {children}
          </a>
        );
      }
      return <>{children}</>;
    },
    img: () => null, // PROD-08 — no auto-loaded remote content from model output
  };
  return components;
}

export function AnswerRenderer({ answer, sessionId }: { answer: GroundedAnswer; sessionId: string }) {
  const linkedMarkdown = useMemo(() => linkifyClaims(answer.answer_markdown, answer.claims), [answer]);
  const components = useMemo(() => buildComponents(sessionId, answer), [sessionId, answer]);

  return (
    <div className="space-y-4">
      {answer.is_fallback && (
        <div className="flex items-center gap-2">
          <FallbackBadge />
        </div>
      )}

      <div className="prose-sm max-w-none text-sm leading-relaxed" style={{ color: 'var(--text)' }}>
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          allowedElements={ALLOWED_TAGS}
          unwrapDisallowed
          components={components}
          // react-markdown's default urlTransform strips any scheme it
          // doesn't allowlist (http/https/mailto/relative) *before* the `a`
          // component ever runs — silently turning every `claim:<id>` link
          // linkifyClaims produced into `href=""`. Pass claim: links through
          // untouched; every other URL still gets the default sanitizer.
          urlTransform={(url) => (url.startsWith(CLAIM_LINK_SCHEME) ? url : defaultUrlTransform(url))}
        >
          {linkedMarkdown}
        </ReactMarkdown>
      </div>

      <LimitationBlock limitations={answer.limitations} />

      {answer.recommended_next_steps.length > 0 && (
        <div className="rounded-lg p-3" style={{ backgroundColor: 'var(--accent-blue-muted)' }}>
          <p className="mb-1.5 text-xs font-semibold" style={{ color: 'var(--text)' }}>
            Bước tiếp theo đề xuất
          </p>
          <ul className="space-y-1">
            {answer.recommended_next_steps.map((step, index) => {
              const indices = step.basis_claim_ids
                .map((id) => answer.claims.findIndex((c) => c.claim_id === id))
                .filter((idx) => idx >= 0);
              return (
                <li key={index} className="text-xs" style={{ color: 'var(--text-muted)' }}>
                  • {step.text}
                  {indices.length > 0 ? (
                    <span style={{ color: 'var(--accent-blue)' }}>
                      {' '}
                      ({indices.map((idx) => `#${idx + 1}`).join(', ')})
                    </span>
                  ) : (
                    <span className="ml-1 italic" style={{ color: 'var(--text-faint)' }}>
                      (không neo claim)
                    </span>
                  )}
                </li>
              );
            })}
          </ul>
        </div>
      )}

      <div className="space-y-1.5 border-t pt-3 text-xs" style={{ borderColor: 'var(--border)' }}>
        {/* candidate_generation/content_sha256/full IDs belong in the audit
            artifact (AnswerAuditViewer), not this footer — plan section 8.4:
            "không chiếm phần chân mọi message". */}
        <div className="flex flex-wrap items-center gap-2" style={{ color: 'var(--text-faint)' }}>
          <span>
            {answer.claims.length} claim · {answer.limitations.length} giới hạn
          </span>
          <Link to={`/s/${sessionId}/answers/${answer.answer_id}`} className="font-medium underline" style={{ color: 'var(--accent-blue)' }}>
            Nguồn và kiểm chứng →
          </Link>
        </div>
        {answer.is_fallback && (
          <div className="flex flex-wrap items-center gap-2">
            <FallbackBadge />
            <Link
              to={`/s/${sessionId}/runs/${answer.run_id}?tab=validation`}
              className="font-medium underline"
              style={{ color: 'var(--accent-blue)' }}
            >
              kiểm chứng đáp án →
            </Link>
          </div>
        )}
      </div>
    </div>
  );
}
