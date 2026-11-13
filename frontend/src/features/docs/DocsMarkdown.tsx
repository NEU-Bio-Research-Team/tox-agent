import { Children, isValidElement, type ReactElement, type ReactNode } from 'react';
import ReactMarkdown, { type Components } from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Link } from 'react-router';
import { Callout, type Tone } from '../../shared/ui/callout';
import { CardAnatomy } from './CardAnatomy';
import { slugify } from './articles';

function textOf(node: ReactNode): string {
  if (typeof node === 'string' || typeof node === 'number') return String(node);
  if (Array.isArray(node)) return node.map(textOf).join('');
  if (isValidElement(node)) return textOf((node as ReactElement<{ children?: ReactNode }>).props.children);
  return '';
}

const CALLOUT_TONE: Record<string, Tone> = { NOTE: 'brand', WARNING: 'warning', INFO: 'info', TIP: 'neutral' };

/** `> [!NOTE] Title` followed by a blank quoted line and the body becomes a
 * Callout; any other blockquote stays a quotation. */
function Blockquote({ children }: { children?: ReactNode }) {
  const blocks = Children.toArray(children).filter((child) => !(typeof child === 'string' && child.trim() === ''));
  const first = blocks[0];
  const head = isValidElement(first) ? textOf(first) : '';
  const marker = /^\[!(NOTE|WARNING|INFO|TIP)\]\s*(.*)$/.exec(head.trim());
  if (!marker) {
    return <blockquote className="my-4 border-l-2 border-line-strong pl-4 text-ink-secondary">{children}</blockquote>;
  }
  return (
    <Callout tone={CALLOUT_TONE[marker[1]]} title={marker[2] || undefined} className="my-5 text-sm">
      <div className="[&_p]:my-1 [&_p]:text-sm [&_p]:leading-relaxed">{blocks.slice(1)}</div>
    </Callout>
  );
}

const components: Components = {
  h2: ({ children }) => (
    <h2 id={slugify(textOf(children))} className="mt-10 scroll-mt-24 text-2xl font-semibold tracking-tight text-ink first:mt-0">
      {children}
    </h2>
  ),
  h3: ({ children }) => <h3 className="mt-6 text-lg font-semibold text-ink">{children}</h3>,
  p: ({ children }) => <p className="my-3 text-15 leading-[1.75] text-ink">{children}</p>,
  ul: ({ children }) => <ul className="my-3 list-disc space-y-2 pl-6 text-15 leading-[1.7] text-ink marker:text-ink-tertiary">{children}</ul>,
  ol: ({ children }) => <ol className="my-3 list-decimal space-y-2 pl-6 text-15 leading-[1.7] text-ink">{children}</ol>,
  strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
  a: ({ href = '', children }) =>
    href.startsWith('/') ? (
      <Link to={href} className="font-medium text-link underline underline-offset-2">{children}</Link>
    ) : (
      <a href={href} target="_blank" rel="noopener noreferrer" className="font-medium text-link underline underline-offset-2">{children}</a>
    ),
  blockquote: Blockquote,
  table: ({ children }) => (
    <div className="my-4 overflow-x-auto rounded-xl border border-line">
      <table className="w-full border-collapse text-sm">{children}</table>
    </div>
  ),
  th: ({ children }) => <th className="bg-canvas-subtle px-4 py-2.5 text-left text-xs font-semibold uppercase tracking-wider text-ink-tertiary">{children}</th>,
  td: ({ children }) => <td className="border-t border-line px-4 py-2.5 text-ink">{children}</td>,
  pre: ({ children }) => <>{children}</>,
  code: ({ className, children }) => {
    if (/language-card-anatomy/.test(className ?? '')) return <CardAnatomy />;
    if (className) {
      return (
        <pre className="my-4 overflow-x-auto rounded-xl border border-line bg-surface-muted p-4 font-mono text-13 leading-relaxed text-ink">
          <code>{children}</code>
        </pre>
      );
    }
    return <code className="rounded-md bg-brand-soft-strong px-1.5 py-0.5 font-mono text-13 text-brand-ink">{children}</code>;
  },
};

export function DocsMarkdown({ source }: { source: string }) {
  return (
    <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
      {source}
    </ReactMarkdown>
  );
}
