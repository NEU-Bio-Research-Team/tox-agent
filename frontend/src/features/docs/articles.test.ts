import { describe, expect, it } from 'vitest';
import { ARTICLES, DOC_GROUPS, findArticle, headingsOf, slugify } from './articles';

describe('docs articles', () => {
  it('has unique slugs and every article belongs to a declared group', () => {
    const slugs = ARTICLES.map((article) => article.slug);
    expect(new Set(slugs).size).toBe(slugs.length);
    for (const article of ARTICLES) expect(DOC_GROUPS).toContain(article.group);
  });

  it('gives every article a lede and a body', () => {
    for (const article of ARTICLES) {
      expect(article.lede.length, article.slug).toBeGreaterThan(20);
      expect(article.body.length, article.slug).toBeGreaterThan(50);
    }
  });

  it('gives every heading in an article a distinct id', () => {
    for (const article of ARTICLES) {
      const ids = headingsOf(article.body).map((heading) => heading.id);
      expect(new Set(ids).size, article.slug).toBe(ids.length);
    }
  });

  it('only links to docs articles that exist', () => {
    const link = /\]\(\/docs\/([a-z0-9-]+)(?:#[a-z0-9-]+)?\)/g;
    for (const article of ARTICLES) {
      for (const match of `${article.lede}\n${article.body}`.matchAll(link)) {
        expect(findArticle(match[1]), `${article.slug} links to /docs/${match[1]}`).toBeDefined();
      }
    }
  });

  it('keeps the no-aggregate constraint stated in the principles', () => {
    const principles = findArticle('principles');
    expect(principles?.body).toMatch(/không có điểm tổng/i);
  });
});

describe('slugify', () => {
  it('strips diacritics and hyphenates', () => {
    expect(slugify('Một thẻ cho mỗi phép đo')).toBe('mot-the-cho-moi-phep-do');
    expect(slugify('Đọc kết quả')).toBe('doc-ket-qua');
  });
});

describe('headingsOf', () => {
  it('reads second-level headings and ignores fenced code', () => {
    const markdown = ['## Một', '', '```', '## không phải heading', '```', '', '### con', '', '## Hai'].join('\n');
    expect(headingsOf(markdown).map((heading) => heading.text)).toEqual(['Một', 'Hai']);
  });
});
