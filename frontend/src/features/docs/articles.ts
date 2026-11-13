import intro from './content/intro.md?raw';
import quickAnalysis from './content/quick-analysis.md?raw';
import readingResults from './content/reading-results.md?raw';
import limitations from './content/limitations.md?raw';
import sessionsAgent from './content/sessions-agent.md?raw';
import aiConnections from './content/ai-connections.md?raw';
import principles from './content/principles.md?raw';

export const DOC_GROUPS = ['Bắt đầu', 'Hiểu kết quả', 'Làm việc với agent', 'Nguyên tắc'] as const;
export type DocGroup = (typeof DOC_GROUPS)[number];

export interface DocArticle {
  slug: string;
  group: DocGroup;
  title: string;
  /** The article's first paragraph, shown large under the title. */
  lede: string;
  /** The rest of the markdown, after the lede. */
  body: string;
}

/** The documentation, in reading order. Each article is a markdown file under
 * `content/`, so it is edited as text. */
type ArticleSeed = { slug: string; group: DocGroup; title: string; source: string };

/** An article opens with one paragraph that is its lede; the rest is the body. */
function withLede(seed: ArticleSeed): DocArticle {
  const [lede, ...rest] = seed.source.trim().split(/\n\s*\n/);
  return { slug: seed.slug, group: seed.group, title: seed.title, lede, body: rest.join('\n\n') };
}

const SEEDS: ReadonlyArray<ArticleSeed> = [
  { slug: 'intro', group: 'Bắt đầu', title: 'Giới thiệu', source: intro },
  { slug: 'quick-analysis', group: 'Bắt đầu', title: 'Phân tích nhanh', source: quickAnalysis },
  { slug: 'reading-results', group: 'Hiểu kết quả', title: 'Đọc kết quả', source: readingResults },
  { slug: 'limitations', group: 'Hiểu kết quả', title: 'Giới hạn của hệ thống', source: limitations },
  { slug: 'sessions-agent', group: 'Làm việc với agent', title: 'Session và agent', source: sessionsAgent },
  { slug: 'ai-connections', group: 'Làm việc với agent', title: 'Kết nối AI', source: aiConnections },
  { slug: 'principles', group: 'Nguyên tắc', title: 'Bốn ràng buộc của ToxAgent', source: principles },
];

export const ARTICLES: ReadonlyArray<DocArticle> = SEEDS.map(withLede);

export function findArticle(slug: string | undefined): DocArticle | undefined {
  return ARTICLES.find((article) => article.slug === slug);
}

/** Lower-case, strip diacritics, hyphenate: the id a heading gets. */
export function slugify(text: string): string {
  return text
    .normalize('NFD')
    .replace(/[̀-ͯ]/g, '')
    .replace(/đ/gi, 'd')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
}

export interface DocHeading {
  id: string;
  text: string;
}

/** The `##` headings of an article, skipping fenced code. Used for the table of
 * contents and for the search index; the renderer derives ids the same way. */
export function headingsOf(markdown: string): DocHeading[] {
  const headings: DocHeading[] = [];
  let fenced = false;
  for (const line of markdown.split('\n')) {
    if (line.startsWith('```')) fenced = !fenced;
    if (fenced) continue;
    const match = /^## (.+)$/.exec(line);
    if (match) headings.push({ id: slugify(match[1]), text: match[1].trim() });
  }
  return headings;
}
