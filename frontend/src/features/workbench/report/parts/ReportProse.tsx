import { Children, Fragment, isValidElement, type ReactNode } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { ReportReference } from '../../../../shared/api/types';
import { CitationMarker } from './ReportReferences';

/** The same pattern the server validates and numbers:
 * `validation/citations.py`'s `CITATION_TOKEN`. */
export const CITATION_TOKEN = /\[@(evd_[0-9a-f]{32})\]/g;

/** Evidence ids cited inline in one section's prose, in reading order. */
export function citedInProse(body: string | null | undefined): string[] {
  const found: string[] = [];
  for (const match of (body ?? '').matchAll(CITATION_TOKEN)) {
    if (!found.includes(match[1])) found.push(match[1]);
  }
  return found;
}

/**
 * Section prose with its `[@evd_...]` tokens rendered as citation markers.
 *
 * The substitution runs on the *parsed* tree, not on the Markdown source. Doing
 * it first — rewriting the string to contain an anchor and then parsing that —
 * would mean feeding generated markup back through a parser that also handles
 * model-authored text, which is the shape of problem the server's content-safety
 * gate exists to prevent. Here the model's text only ever becomes React text
 * nodes, and the only element a paragraph gains is the marker this component
 * built from the artifact's own reference snapshot.
 */
export function ReportProse({
  reportId,
  body,
  references,
}: {
  reportId: string;
  body: string;
  references: Map<string, ReportReference>;
}) {
  /** Replace tokens inside one text node. */
  function inText(text: string, keyPrefix: string): ReactNode {
    const matches = [...text.matchAll(CITATION_TOKEN)];
    if (matches.length === 0) return text;
    const nodes: ReactNode[] = [];
    let cursor = 0;
    matches.forEach((match, index) => {
      const at = match.index ?? 0;
      if (at > cursor) nodes.push(text.slice(cursor, at));
      nodes.push(
        <CitationMarker
          key={`${keyPrefix}-${index}`}
          reportId={reportId}
          reference={references.get(match[1])}
          evidenceId={match[1]}
        />,
      );
      cursor = at + match[0].length;
    });
    if (cursor < text.length) nodes.push(text.slice(cursor));
    return <Fragment>{nodes}</Fragment>;
  }

  /** Walk a rendered subtree, rewriting only its string leaves. */
  function withCitations(children: ReactNode, keyPrefix = 'c'): ReactNode {
    return Children.map(children, (child, index) => {
      if (typeof child === 'string') return inText(child, `${keyPrefix}-${index}`);
      if (isValidElement(child)) {
        const props = child.props as { children?: ReactNode };
        if (props.children !== undefined) {
          return {
            ...child,
            props: {
              ...props,
              children: withCitations(props.children, `${keyPrefix}-${index}`),
            },
          };
        }
      }
      return child;
    });
  }

  return (
    <div className="prose prose-sm max-w-none">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          p: ({ children }) => <p>{withCitations(children)}</p>,
          li: ({ children }) => <li>{withCitations(children)}</li>,
          td: ({ children }) => <td>{withCitations(children)}</td>,
          th: ({ children }) => <th>{withCitations(children)}</th>,
          blockquote: ({ children }) => <blockquote>{withCitations(children)}</blockquote>,
          // Report prose never links out on its own. A citation is a structured
          // reference, and an arbitrary URL in a section body is precisely what
          // the `[@evd_...]` token replaced — so the link text stays and the
          // anchor does not.
          a: ({ children }) => <span>{withCitations(children)}</span>,
          // Nor does it embed remote images, which would be a tracking beacon
          // fired by every reader of the report.
          img: () => null,
        }}
      >
        {body}
      </ReactMarkdown>
    </div>
  );
}
