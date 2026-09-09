import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router';
import { describe, expect, it } from 'vitest';
import { ActivityPresence } from './ActivityPresence';

describe('ActivityPresence', () => {
  it('keeps repeated semantic work behind one compact progress line', () => {
    render(
      <MemoryRouter>
        <ActivityPresence
          status="running"
          tools={[]}
          activities={[
            { activity_id: 'a1', phase: 'retrieval', kind: 'literature_search', label_key: 'activity.searching_literature', status: 'completed' },
            { activity_id: 'a2', phase: 'retrieval', kind: 'literature_search', label_key: 'activity.searching_literature', status: 'started' },
          ]}
        />
      </MemoryRouter>,
    );

    expect(screen.getByText('Đang tìm các nghiên cứu liên quan…')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '2 hoạt động' })).toBeInTheDocument();
    expect(screen.queryByText('search_toxicology_evidence')).toBeNull();
  });
});
