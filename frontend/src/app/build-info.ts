/** Injected by vite.config.ts at build time. Vitest uses its own config with
 * no `define`, hence the guards. */
declare const __APP_VERSION__: string | undefined;
declare const __APP_BUILD_TIME__: string | undefined;
declare const __APP_COMMIT__: string | undefined;

export const APP_VERSION = typeof __APP_VERSION__ === 'string' ? __APP_VERSION__ : '0.0.0-dev';
export const APP_BUILD_TIME = typeof __APP_BUILD_TIME__ === 'string' ? __APP_BUILD_TIME__ : new Date(0).toISOString();
export const APP_COMMIT = typeof __APP_COMMIT__ === 'string' ? __APP_COMMIT__ : 'unknown';

export const BUILD_TIME_LABEL = `${new Date(APP_BUILD_TIME).toLocaleString('en-GB', {
  year: 'numeric',
  month: '2-digit',
  day: '2-digit',
  hour: '2-digit',
  minute: '2-digit',
  hour12: false,
  timeZone: 'UTC',
})} UTC`;
