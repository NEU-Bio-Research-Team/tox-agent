import type { Claim } from './api/types';

export const CLAIM_LINK_SCHEME = 'claim:';

/**
 * N-3 / plan section 8.4, rule 1: match `rendered_value` into `answer_markdown`
 * by exact string, first occurrence, no fuzzy regex. A value that can't be
 * matched is left as plain text — the content of the answer is never edited
 * to force a chip in.
 *
 * Implementation: scan the markdown once; at each position, try every
 * not-yet-placed claim's `rendered_value` (longest first, so "3.15%" is not
 * shadowed by a shorter claim whose value happens to be "3.15"), and splice in
 * a markdown link with a private `claim:` scheme. `components.a` in
 * AnswerRenderer turns that scheme back into a <ClaimChip>; any other href is
 * still ordinary, safe markdown.
 */
export function linkifyClaims(markdown: string, claims: Claim[]): string {
  const candidates = claims
    .filter((claim): claim is Claim & { rendered_value: string } => Boolean(claim.rendered_value) && claim.rendered_value!.length >= 2)
    .map((claim, order) => ({ claim, order }))
    .sort((a, b) => b.claim.rendered_value.length - a.claim.rendered_value.length);

  const placed = new Set<string>();
  let result = '';
  let i = 0;

  outer: while (i < markdown.length) {
    for (const { claim } of candidates) {
      if (placed.has(claim.claim_id)) continue;
      const value = claim.rendered_value;
      // A model frequently writes the raw value inside inline code, e.g.
      // "**`non_blocker`**" — CommonMark never parses markdown syntax
      // *inside* a code span, so inserting a link there (`` `[x](claim:id)` ``)
      // would render as literal bracket text instead of a chip. Check the
      // backtick-wrapped form first and consume the backticks along with it,
      // replacing the whole span with the link — ClaimChip already renders
      // in a monospace style, so this doesn't change how the value looks.
      const backtickWrapped = `\`${value}\``;
      if (markdown.startsWith(backtickWrapped, i)) {
        result += `[${value}](${CLAIM_LINK_SCHEME}${claim.claim_id})`;
        placed.add(claim.claim_id);
        i += backtickWrapped.length;
        continue outer;
      }
      if (markdown.startsWith(value, i)) {
        result += `[${value}](${CLAIM_LINK_SCHEME}${claim.claim_id})`;
        placed.add(claim.claim_id);
        i += value.length;
        continue outer;
      }
    }
    result += markdown[i];
    i += 1;
  }

  return result;
}
