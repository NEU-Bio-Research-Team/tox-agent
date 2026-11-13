import { expect, test, type Page } from '@playwright/test';
import { installStoryApi, messages, type Routes } from './fixtures/api';
import { SESSION_ID, answerTable, publishedReply, result, RUN_1, ANSWER_1, session, userMessage, lineagePassed } from './fixtures/stories';

/**
 * The design-system safety net (plan section 7, P0). Every route is opened in the
 * light and the dark theme, at a desktop and at a phone width, and must
 *
 *   - not scroll sideways,
 *   - carry exactly one level-1 heading, and
 *   - paint a screenshot, attached to the report so a person can look at it.
 *
 * Screenshots are attached, not compared: a pixel baseline would pin the browser
 * and the fonts of one machine. Set `VISUAL_DIR` to also write them to disk.
 */

const S = `/v1/sessions/${SESSION_ID}`;
const WORKBENCH: Routes = {
  [`GET ${S}`]: session(),
  [`GET ${S}/screenings`]: [404, { error: { code: 'not_found', message: 'off', retryable: false, details: {} } }],
  [`GET ${S}/messages`]: messages([
    userMessage(1, 'Is there a measured EGFR IC50 that settles the REVIEW?'),
    publishedReply(2, answerTable.answer_markdown, ANSWER_1, RUN_1, 'passed'),
  ]),
  [`GET ${S}/answers/${ANSWER_1}`]: answerTable,
  [`GET ${S}/runs/${RUN_1}/result`]: result(RUN_1, { public_answer_id: ANSWER_1, candidate_answer_id: ANSWER_1 }),
  [`GET ${S}/runs/${RUN_1}/lineage`]: lineagePassed,
};

type Setup = (page: Page) => Promise<void>;
const signedIn: Setup = async (page) => { await installStoryApi(page, SESSION_ID, {}); };
const workbench: Setup = async (page) => { await installStoryApi(page, SESSION_ID, WORKBENCH); };
const anonymous: Setup = async (page) => {
  await page.route(/\/(v1|health)\//, (route) => route.fulfill({ status: 404, body: '{}' }));
};

const ROUTES: Array<{ name: string; path: string; setup: Setup; phone: boolean; h1: boolean }> = [
  { name: 'landing', path: '/', setup: anonymous, phone: true, h1: false },
  { name: 'docs-intro', path: '/docs/intro', setup: anonymous, phone: true, h1: true },
  { name: 'docs-reading-results', path: '/docs/reading-results', setup: anonymous, phone: true, h1: true },
  { name: 'docs-principles', path: '/docs/principles', setup: anonymous, phone: false, h1: true },
  { name: 'sign-in-gate', path: '/sessions', setup: anonymous, phone: false, h1: false },
  { name: 'sessions', path: '/sessions', setup: signedIn, phone: true, h1: true },
  { name: 'analyze', path: '/analyze', setup: signedIn, phone: true, h1: true },
  { name: 'settings', path: '/settings', setup: signedIn, phone: true, h1: true },
  { name: 'workbench', path: `/s/${SESSION_ID}`, setup: workbench, phone: true, h1: true },
];

const VIEWPORTS = [
  { label: 'desktop', size: { width: 1440, height: 900 } },
  { label: 'phone', size: { width: 390, height: 844 } },
];

for (const route of ROUTES) {
  for (const theme of ['light', 'dark'] as const) {
    for (const viewport of VIEWPORTS) {
      if (viewport.label === 'phone' && !route.phone) continue;
      test(`${route.name} · ${theme} · ${viewport.label}`, async ({ page }, testInfo) => {
        await page.setViewportSize(viewport.size);
        await page.addInitScript((value) => { try { localStorage.setItem('theme', value); } catch { /* storage blocked */ } }, theme);
        await route.setup(page);
        await page.goto(route.path);
        await page.waitForLoadState('networkidle').catch(() => undefined);
        await page.waitForTimeout(500);

        expect(await page.locator('html').evaluate((el) => el.classList.contains('dark'))).toBe(theme === 'dark');

        const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
        expect(overflow, 'the page must not scroll sideways').toBeLessThanOrEqual(0);

        if (route.h1) await expect(page.getByRole('heading', { level: 1 })).toHaveCount(1);

        const image = await page.screenshot();
        await testInfo.attach(`${route.name}-${theme}-${viewport.label}`, { body: image, contentType: 'image/png' });
        if (process.env.VISUAL_DIR) {
          const { writeFile, mkdir } = await import('node:fs/promises');
          await mkdir(process.env.VISUAL_DIR, { recursive: true });
          await writeFile(`${process.env.VISUAL_DIR}/${route.name}.${theme}.${viewport.label}.png`, image);
        }
      });
    }
  }
}
