import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import { ApiError } from '../../../shared/api/types';
import { ArtifactUnavailable } from './ArtifactUnavailable';

describe('ArtifactUnavailable', () => {
  afterEach(cleanup);

  it('renders an explicit retention state only for a typed expiry outcome', () => {
    const expired = new ApiError(410, {
      error: { code: 'artifact_expired', message: 'expired', retryable: false, details: {} },
    });
    render(<ArtifactUnavailable artifact="evidence" error={expired} />);
    expect(screen.getByText('Nội dung này đã hết hạn lưu')).toBeInTheDocument();
  });

  it('does not infer expiry from a generic not-found response', () => {
    const missing = new ApiError(404, {
      error: { code: 'not_found', message: 'not found', retryable: false, details: {} },
    });
    render(<ArtifactUnavailable artifact="evidence" error={missing} />);
    expect(screen.queryByText('Nội dung này đã hết hạn lưu')).toBeNull();
    expect(screen.getByText(/có thể nó không còn hoặc bạn không có quyền xem/i)).toBeInTheDocument();
  });
});
