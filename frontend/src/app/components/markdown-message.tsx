import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface MarkdownMessageProps {
	content: string;
	isUser?: boolean;
}

export function MarkdownMessage({ content, isUser = false }: MarkdownMessageProps) {
	if (isUser) {
		return <p className="text-sm leading-7 whitespace-pre-wrap">{content}</p>;
	}

	return (
		<ReactMarkdown
			remarkPlugins={[remarkGfm]}
			components={{
				strong: ({ children }) => (
					<strong className="font-semibold" style={{ color: 'var(--text)' }}>
						{children}
					</strong>
				),
				em: ({ children }) => (
					<em className="italic" style={{ color: 'var(--text-muted)' }}>
						{children}
					</em>
				),
				code: ({ children, className }) => {
					const text = String(children ?? '');
					const isBlock = Boolean(className && className.trim().length > 0);
					if (isBlock) {
						return (
							<pre
								className="my-2 overflow-x-auto rounded-lg p-3 text-xs"
								style={{
									backgroundColor: 'var(--surface-alt)',
									border: '1px solid var(--border)',
									color: 'var(--text)',
								}}
							>
								<code>{text}</code>
							</pre>
						);
					}
					return (
						<code
							className="rounded px-1.5 py-0.5 text-xs font-mono"
							style={{
								backgroundColor: 'var(--surface-alt)',
								color: 'var(--accent-blue)',
								border: '1px solid var(--border)',
							}}
						>
							{text}
						</code>
					);
				},
				ul: ({ children }) => (
					<ul className="my-2 ml-4 list-disc space-y-1 text-sm" style={{ color: 'var(--text-muted)' }}>
						{children}
					</ul>
				),
				ol: ({ children }) => (
					<ol className="my-2 ml-4 list-decimal space-y-1 text-sm" style={{ color: 'var(--text-muted)' }}>
						{children}
					</ol>
				),
				p: ({ children }) => (
					<p className="mb-2 text-sm leading-7 last:mb-0" style={{ color: 'var(--text)' }}>
						{children}
					</p>
				),
				hr: () => <hr className="my-3" style={{ borderColor: 'var(--border)' }} />,
			}}
		>
			{content}
		</ReactMarkdown>
	);
}
