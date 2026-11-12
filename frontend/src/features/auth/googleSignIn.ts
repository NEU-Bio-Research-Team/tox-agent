/**
 * "Continue with Google", through the Firebase project's own Google provider.
 *
 * The Firebase SDK is used here as a broker and nothing more: it opens the
 * project's sign-in handler in a popup, which sends the person to Google and
 * back, and hands over the resulting ID token and refresh token. Those are
 * then adopted into the same session the email-and-password path keeps
 * (`firebaseAuth.ts`), so renewal, sign-out and every request read one place.
 * The SDK keeps nothing: it runs with in-memory persistence, so there is no
 * second signed-in state in IndexedDB to fall out of step with the first.
 *
 * It is loaded only when the button is pressed. A person who types a password
 * never downloads it.
 */
import { adoptSession, type FirebaseAuthConfig } from './firebaseAuth';

export class GoogleSignInError extends Error {
  constructor(
    readonly code: string,
    message: string,
    /** True when the person simply closed the window: nothing went wrong. */
    readonly cancelled = false,
  ) {
    super(message);
  }
}

const MESSAGES: Record<string, string> = {
  'auth/popup-blocked': 'Trình duyệt đã chặn cửa sổ đăng nhập Google. Hãy cho phép cửa sổ bật lên cho trang này rồi thử lại.',
  'auth/operation-not-allowed': 'Đăng nhập bằng Google chưa được bật cho dự án này.',
  'auth/unauthorized-domain': 'Địa chỉ trang này chưa được Firebase cho phép đăng nhập Google.',
  'auth/account-exists-with-different-credential': 'Email này đã có tài khoản đăng nhập bằng mật khẩu. Hãy đăng nhập bằng email và mật khẩu.',
  'auth/user-disabled': 'Tài khoản này đã bị vô hiệu hoá.',
  'auth/network-request-failed': 'Không kết nối được dịch vụ đăng nhập. Kiểm tra mạng rồi thử lại.',
  'auth/too-many-requests': 'Quá nhiều lần thử. Hãy thử lại sau.',
};

/** What the sign-in popup needs beyond the API key: the project it belongs to. */
export function googleSignInAvailable(config: FirebaseAuthConfig): boolean {
  return Boolean(config.projectId);
}

export async function signInWithGoogle(config: FirebaseAuthConfig): Promise<void> {
  if (!config.projectId) {
    throw new GoogleSignInError('not-configured', 'Đăng nhập bằng Google chưa được cấu hình cho bản này.');
  }
  const [{ getApps, initializeApp }, auth] = await Promise.all([import('firebase/app'), import('firebase/auth')]);
  const app =
    getApps().find((candidate) => candidate.name === 'toxagent-google') ??
    initializeApp(
      {
        apiKey: config.apiKey,
        authDomain: config.authDomain ?? `${config.projectId}.firebaseapp.com`,
        projectId: config.projectId,
      },
      'toxagent-google',
    );
  // `initializeAuth` may run once per app. A second press — after a closed
  // window, a blocked popup — reuses the instance the first one made.
  let instance;
  try {
    instance = auth.initializeAuth(app, {
      persistence: auth.inMemoryPersistence,
      popupRedirectResolver: auth.browserPopupRedirectResolver,
    });
  } catch (cause) {
    if ((cause as { code?: unknown })?.code !== 'auth/already-initialized') throw cause;
    instance = auth.getAuth(app);
  }

  try {
    const result = await auth.signInWithPopup(instance, new auth.GoogleAuthProvider());
    const token = await result.user.getIdTokenResult();
    const lifetimeS = Math.max(60, Math.round((Date.parse(token.expirationTime) - Date.now()) / 1000));
    adoptSession(token.token, result.user.refreshToken, lifetimeS);
  } catch (cause) {
    const code = typeof (cause as { code?: unknown })?.code === 'string' ? (cause as { code: string }).code : 'unknown';
    if (code === 'auth/popup-closed-by-user' || code === 'auth/cancelled-popup-request') {
      throw new GoogleSignInError(code, 'Đã đóng cửa sổ đăng nhập.', true);
    }
    throw new GoogleSignInError(code, MESSAGES[code] ?? 'Đăng nhập bằng Google thất bại.');
  } finally {
    // The session now lives in firebaseAuth.ts. Clear what the SDK holds.
    await auth.signOut(instance).catch(() => undefined);
  }
}
