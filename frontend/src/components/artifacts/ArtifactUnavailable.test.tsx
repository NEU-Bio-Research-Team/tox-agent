import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import { ApiError } from '../../lib/api/types';
import { ArtifactUnavailable } from './ArtifactUnavailable';

describe('ArtifactUnavailable', () => {
  afterEach(cleanup);

  it('renders an explicit retention state only for a typed expiry outcome', () => {
    const expired = new ApiError(410, {
      error: { code: 'artifact_expired', message: 'expired', retryable: false, details: {} },
    });
    render(<ArtifactUnavailable artifact="evidence" error={expired} />);
    expect(screen.getByText('Artifact đã hết hạn')).toBeInTheDocument();
  });

  it('does not infer expiry from a generic not-found response', () => {
    const missing = new ApiError(404, {
      error: { code: 'not_found', message: 'not found', retryable: false, details: {} },
    });
    render(<ArtifactUnavailable artifact="evidence" error={missing} />);
    expect(screen.queryByText('Artifact đã hết hạn')).toBeNull();
    expect(screen.getByText(/có thể không tồn tại hoặc bạn không có quyền xem/i)).toBeInTheDocument();
  });
});
