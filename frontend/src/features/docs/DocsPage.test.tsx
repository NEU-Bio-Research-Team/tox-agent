import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router';
import { afterEach, describe, expect, it } from 'vitest';
import { DocsPage } from './DocsPage';

afterEach(cleanup);

function renderAt(path: string) {
  window.scrollTo = () => undefined;
  return render(
    <MemoryRouter initialEntries={[path]}>
      <Routes>
        <Route path="/docs/:slug" element={<DocsPage />} />
      </Routes>
    </MemoryRouter>,
  );
}

describe('DocsPage', () => {
  it('shows the article, its place in the list and the headings of the page', () => {
    renderAt('/docs/reading-results');
    expect(screen.getByRole('heading', { level: 1, name: 'Đọc kết quả' })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Đọc kết quả' })).toHaveAttribute('aria-current', 'page');
    const toc = screen.getByRole('complementary', { name: 'Trên trang này' });
    expect(toc).toHaveTextContent('Applicability');
  });

  it('draws a Callout for a [!NOTE] quotation and the annotated card figure', () => {
    renderAt('/docs/reading-results');
    expect(screen.getByText('Không có kết luận gộp')).toBeInTheDocument();
    expect(screen.getByLabelText('Ví dụ một thẻ kết quả')).toBeInTheDocument();
  });

  it('links to the previous and next article', () => {
    renderAt('/docs/reading-results');
    const pager = screen.getByRole('navigation', { name: 'Bài trước và bài sau' });
    expect(pager).toHaveTextContent('Phân tích nhanh');
    expect(pager).toHaveTextContent('Giới hạn của hệ thống');
  });

  it('finds an article from the search dialog', () => {
    renderAt('/docs/intro');
    fireEvent.keyDown(window, { key: 'k', ctrlKey: true });
    const box = screen.getByRole('combobox', { name: 'Tìm trong tài liệu' });
    fireEvent.change(box, { target: { value: 'ket noi ai' } });
    expect(screen.getAllByRole('option').some((option) => option.textContent?.includes('Kết nối AI'))).toBe(true);
  });

  it('sends an unknown article back to the introduction', () => {
    renderAt('/docs/does-not-exist');
    expect(screen.getByRole('heading', { level: 1, name: 'Giới thiệu' })).toBeInTheDocument();
  });
});
