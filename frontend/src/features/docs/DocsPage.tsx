import { useEffect, useMemo, useState } from 'react';
import { Link, Navigate, useLocation, useNavigate, useParams } from 'react-router';
import { ChevronLeft, ChevronRight, ExternalLink, Search } from 'lucide-react';
import { ThemeToggle } from '../../app/shell/ThemeToggle';
import { FEEDBACK_URL, REPO_URL } from '../../app/links';
import { Button } from '../../shared/ui/button';
import { CommandDialog, type CommandItem } from '../../shared/ui/command-dialog';
import { Logo } from '../../shared/ui/logo';
import { NativeSelect } from '../../shared/ui/native-select';
import { cn } from '../../shared/ui/utils';
import { scrollToSection, useActiveSection } from '../../shared/hooks/useActiveSection';
import { ARTICLES, DOC_GROUPS, findArticle, headingsOf } from './articles';
import { DocsMarkdown } from './DocsMarkdown';

/** Ctrl/Cmd+K and the header button open a finder over every article and heading. */
function useDocsSearch() {
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
        event.preventDefault();
        setOpen((current) => !current);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const items = useMemo<CommandItem[]>(
    () =>
      ARTICLES.flatMap((article) => [
        { id: article.slug, group: article.group, label: article.title, hint: 'Bài viết', onSelect: () => navigate(`/docs/${article.slug}`) },
        ...headingsOf(article.body).map((heading) => ({
          id: `${article.slug}#${heading.id}`,
          group: article.group,
          label: heading.text,
          hint: article.title,
          onSelect: () => navigate(`/docs/${article.slug}#${heading.id}`),
        })),
      ]),
    [navigate],
  );

  return { open, setOpen, items };
}

/**
 * The docs frame (plan section 6.3): a top bar with search and the way back into
 * the app, a grouped list of articles, the article, and "Trên trang này". It is
 * public, so it has no rail and needs no sign-in.
 */
export function DocsPage() {
  const { slug } = useParams<{ slug: string }>();
  const { hash } = useLocation();
  const navigate = useNavigate();
  const search = useDocsSearch();
  const article = findArticle(slug);
  const headings = useMemo(() => (article ? headingsOf(article.body) : []), [article]);
  const [active, setActive] = useActiveSection(headings.map((heading) => heading.id));

  useEffect(() => {
    if (article) document.title = `${article.title} · Tài liệu ToxAgent`;
    return () => {
      document.title = 'ToxAgent';
    };
  }, [article]);

  // Arriving on `/docs/x#heading` (from search or a link) scrolls to it; an
  // article opened without a hash starts at the top.
  useEffect(() => {
    if (!article) return;
    const id = hash.replace(/^#/, '');
    if (id) window.setTimeout(() => document.getElementById(id)?.scrollIntoView({ block: 'start' }), 0);
    else window.scrollTo({ top: 0 });
  }, [article, hash]);

  if (!article) return <Navigate to="/docs/intro" replace />;

  const index = ARTICLES.findIndex((item) => item.slug === article.slug);
  const previous = ARTICLES[index - 1];
  const next = ARTICLES[index + 1];

  return (
    <div className="min-h-screen bg-background text-foreground">
      <header className="sticky top-0 z-30 flex flex-wrap items-center gap-x-5 gap-y-2 border-b border-line bg-background/90 px-4 py-2 backdrop-blur md:px-6">
        <Link to="/docs/intro" className="flex items-center gap-2.5">
          <Logo withWordmark />
          <span className="border-l border-line-strong pl-2.5 text-sm text-ink-secondary">Tài liệu</span>
        </Link>
        <button
          type="button"
          onClick={() => search.setOpen(true)}
          className="order-last flex h-9 w-full items-center gap-2.5 rounded-lg border border-line-strong bg-canvas-subtle px-3.5 text-left text-13 text-ink-secondary transition-colors hover:bg-surface-muted sm:order-none sm:mx-auto sm:max-w-md sm:flex-1"
        >
          <Search className="size-4" aria-hidden="true" />
          <span className="flex-1">Tìm trong tài liệu</span>
          <kbd className="rounded-md border border-line-strong bg-background px-1.5 py-px text-xs">Ctrl K</kbd>
        </button>
        <div className="ml-auto flex items-center gap-1 sm:ml-0">
          <Button asChild variant="ghost" size="sm" className="hidden gap-1 text-ink-secondary md:inline-flex">
            <a href={REPO_URL} target="_blank" rel="noopener noreferrer">GitHub <ExternalLink /></a>
          </Button>
          <Button asChild variant="ghost" size="sm" className="hidden gap-1 text-ink-secondary md:inline-flex">
            <a href={FEEDBACK_URL} target="_blank" rel="noopener noreferrer">Góp ý <ExternalLink /></a>
          </Button>
          <ThemeToggle className="text-ink-secondary" />
          <Button asChild className="ml-1 gap-1.5">
            <Link to="/analyze">Mở ứng dụng <ChevronRight /></Link>
          </Button>
        </div>
      </header>

      <div className="mx-auto flex w-full max-w-[1360px]">
        <nav aria-label="Mục lục tài liệu" className="sticky top-[57px] hidden h-[calc(100vh-57px)] w-[272px] shrink-0 space-y-6 overflow-y-auto border-r border-line px-5 py-7 lg:block">
          {DOC_GROUPS.map((group) => (
            <div key={group} className="space-y-0.5">
              <p className="px-2.5 pb-1.5 text-2xs font-semibold uppercase tracking-wider text-ink-tertiary">{group}</p>
              {ARTICLES.filter((item) => item.group === group).map((item) => (
                <Link
                  key={item.slug}
                  to={`/docs/${item.slug}`}
                  aria-current={item.slug === article.slug ? 'page' : undefined}
                  className={cn(
                    'flex h-[34px] items-center rounded-[9px] px-2.5 text-sm transition-colors',
                    item.slug === article.slug ? 'bg-brand-soft-strong font-semibold text-brand-ink' : 'text-ink hover:bg-surface-muted',
                  )}
                >
                  {item.title}
                </Link>
              ))}
            </div>
          ))}
        </nav>

        <main className="min-w-0 flex-1 px-5 py-10 md:px-14">
          <article className="max-w-[720px] space-y-7">
            <div className="lg:hidden">
              <NativeSelect aria-label="Chọn bài" value={article.slug} onChange={(event) => navigate(`/docs/${event.target.value}`)}>
                {ARTICLES.map((item) => (
                  <option key={item.slug} value={item.slug}>{item.group} · {item.title}</option>
                ))}
              </NativeSelect>
            </div>

            <header className="space-y-3.5">
              <p className="text-13 text-ink-secondary">
                {article.group} <span aria-hidden="true">›</span> <span className="text-ink">{article.title}</span>
              </p>
              <h1 className="text-4xl font-bold leading-[1.15] tracking-tight text-ink">{article.title}</h1>
              <p className="text-lg leading-relaxed text-ink-secondary">{article.lede}</p>
            </header>

            <div>
              <DocsMarkdown source={article.body} />
            </div>

            <nav aria-label="Bài trước và bài sau" className="flex flex-wrap gap-3.5 pt-4">
              {previous ? (
                <Link to={`/docs/${previous.slug}`} className="flex min-w-[220px] flex-1 flex-col gap-1 rounded-[14px] border border-line px-[18px] py-3.5 transition-colors hover:bg-surface-muted">
                  <span className="flex items-center gap-1 text-xs text-ink-tertiary"><ChevronLeft className="size-3.5" />Trước</span>
                  <span className="text-sm font-semibold text-ink">{previous.title}</span>
                </Link>
              ) : null}
              {next ? (
                <Link to={`/docs/${next.slug}`} className="flex min-w-[220px] flex-1 flex-col items-end gap-1 rounded-[14px] border border-line px-[18px] py-3.5 text-right transition-colors hover:bg-surface-muted">
                  <span className="flex items-center gap-1 text-xs text-ink-tertiary">Tiếp<ChevronRight className="size-3.5" /></span>
                  <span className="text-sm font-semibold text-ink">{next.title}</span>
                </Link>
              ) : null}
            </nav>

            <p className="border-t border-line pt-5 text-13 text-ink-tertiary">
              ToxAgent — evidence-and-decision-support control plane. Ba phép đo độc lập; không có điểm tổng về độc tính.
            </p>
          </article>
        </main>

        {headings.length > 0 ? (
          <aside aria-label="Trên trang này" className="sticky top-[57px] hidden h-fit w-[232px] shrink-0 space-y-2.5 px-6 py-11 xl:block">
            <p className="text-2xs font-semibold uppercase tracking-wider text-ink-tertiary">Trên trang này</p>
            {headings.map((heading) => (
              <a
                key={heading.id}
                href={`#${heading.id}`}
                aria-current={active === heading.id ? 'location' : undefined}
                onClick={(event) => {
                  event.preventDefault();
                  scrollToSection(heading.id, setActive);
                }}
                className={cn(
                  'block border-l-2 py-1 pl-3 text-13 transition-colors',
                  active === heading.id ? 'border-brand font-semibold text-brand-ink' : 'border-line text-ink-secondary hover:text-ink',
                )}
              >
                {heading.text}
              </a>
            ))}
          </aside>
        ) : null}
      </div>

      <CommandDialog
        open={search.open}
        onOpenChange={search.setOpen}
        items={search.items}
        label="Tìm trong tài liệu"
        placeholder="Tìm bài hoặc mục…"
        emptyText="Không có bài hay mục nào khớp."
      />
    </div>
  );
}
