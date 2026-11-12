import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

afterEach(cleanup);

const google = vi.hoisted(() => ({ signInWithGoogle: vi.fn() }));
vi.mock('./googleSignIn', async () => {
  const real = await vi.importActual<typeof import('./googleSignIn')>('./googleSignIn');
  return { ...real, signInWithGoogle: google.signInWithGoogle };
});

import { FirebaseSignIn } from './FirebaseSignIn';
import { GoogleSignInError } from './googleSignIn';

describe('FirebaseSignIn', () => {
  afterEach(() => google.signInWithGoogle.mockReset());

  it('offers Google only when the build names the Firebase project', () => {
    render(<FirebaseSignIn config={{ apiKey: 'k' }} />);
    expect(screen.queryByRole('button', { name: /Tiếp tục với Google/ })).toBeNull();
    cleanup();
    render(<FirebaseSignIn config={{ apiKey: 'k', projectId: 'tox-agent' }} />);
    expect(screen.getByRole('button', { name: /Tiếp tục với Google/ })).toBeTruthy();
  });

  it('starts the Google sign-in with the build\'s config', async () => {
    google.signInWithGoogle.mockResolvedValue(undefined);
    const config = { apiKey: 'k', projectId: 'tox-agent' };
    render(<FirebaseSignIn config={config} />);
    fireEvent.click(screen.getByRole('button', { name: /Tiếp tục với Google/ }));
    await waitFor(() => expect(google.signInWithGoogle).toHaveBeenCalledWith(config));
  });

  it('says why Google failed, and says nothing when the window was only closed', async () => {
    const config = { apiKey: 'k', projectId: 'tox-agent' };
    render(<FirebaseSignIn config={config} />);

    google.signInWithGoogle.mockRejectedValueOnce(new GoogleSignInError('auth/operation-not-allowed', 'Đăng nhập bằng Google chưa được bật cho dự án này.'));
    fireEvent.click(screen.getByRole('button', { name: /Tiếp tục với Google/ }));
    expect(await screen.findByRole('alert')).toHaveTextContent('chưa được bật');

    google.signInWithGoogle.mockRejectedValueOnce(new GoogleSignInError('auth/popup-closed-by-user', 'Đã đóng cửa sổ đăng nhập.', true));
    fireEvent.click(screen.getByRole('button', { name: /Tiếp tục với Google/ }));
    await waitFor(() => expect(screen.queryByRole('alert')).toBeNull());
    // The button is usable again, so a person can try once more.
    expect(screen.getByRole('button', { name: /Tiếp tục với Google/ })).not.toBeDisabled();
  });
});
