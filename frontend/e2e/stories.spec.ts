import { expect, test, type Page } from '@playwright/test';
import { installStoryApi, messages, type Routes } from './fixtures/api';
import {
  ANSWER_1, ANSWER_2, ANSWER_LEGACY, CASE_ABL1, CASE_EGFR, EVIDENCE_REFUSED, EVIDENCE_TABLE, RUN_1, RUN_2, RUN_3,
  RUN_4, SCREENING_ABL1, SCREENING_EGFR, SESSION_ID, TABLE_CELL, answerLegacy, answerRepaired, answerTable, caseAbl1,
  caseEgfr, cases, changesAbl1, changesFirstTurn, changesSecondTurn, evidenceRefused, evidenceTable, legacyReply,
  legacyResult, lineagePassed, lineageRepaired, lineageWithheld, passagesCell, passagesOutline, passagesRefused,
  publishedReply, queuedRun, result, runWithheld, screeningAbl1, screeningEgfr, screenings, session, userMessage,
  withheldReply,
} from './fixtures/stories';

/**
 * The four user stories of master plan §12, through the browser (§10.5): from
 * choosing a subject to the audit. The API is the control plane's, served from
 * typed fixtures; what is asserted is what a researcher sees and what the
 * browser sends -- never a policy the client computed.
 */

const S = `/v1/sessions/${SESSION_ID}`;

const OVERREACH = 'The retrieved study shows that this exact compound blocks hERG.';

async function open(page: Page, routes: Routes, path = `/s/${SESSION_ID}`) {
  const api = await installStoryApi(page, SESSION_ID, routes);
  await page.goto(path);
  await expect(page.getByRole('heading', { name: 'Story workspace' })).toBeVisible();
  return api;
}

const screeningRoutes: Routes = {
  [`GET ${S}/screenings`]: screenings,
  [`GET ${S}/screenings/${SCREENING_EGFR}`]: screeningEgfr,
  [`GET ${S}/screenings/${SCREENING_ABL1}`]: screeningAbl1,
};

test('US1: one ligand on two targets is two subjects, the card states the screening, and a run keeps its subject', async ({ page }) => {
  let sent = false;
  const api = await open(page, {
    ...screeningRoutes,
    [`GET ${S}`]: () => session(sent ? { active_run: queuedRun(RUN_1) } : {}),
    [`GET ${S}/messages`]: messages([]),
    [`POST ${S}/messages`]: () => {
      sent = true;
      return [202, { message_id: 'msg_1', run_id: RUN_1, run_status: 'queued', selected_intent: 'decision_support',
        lane: 'agentic', events_url: `${S}/events` }];
    },
  });

  const egfr = page.getByRole('button', { name: /^EGFR/ });
  const abl1 = page.getByRole('button', { name: /^ABL1/ });
  await expect(egfr).toHaveAttribute('aria-pressed', 'false');
  await expect(abl1).toHaveAttribute('aria-pressed', 'false');
  await expect(page.getByText(/hệ thống không tự chọn hộ/)).toBeVisible();

  await egfr.click();
  await expect(egfr).toHaveAttribute('aria-pressed', 'true');
  // The numbers are on screen before any model turn, in three separate zones.
  await expect(page.getByText('Ước lượng hoạt tính')).toBeVisible();
  await expect(page.getByText('7.1', { exact: true })).toBeVisible();
  await expect(page.getByText(/6\.8–7\.4/)).toBeVisible();
  await expect(page.getByText('REVIEW', { exact: true })).toBeVisible();
  await expect(page.getByText(/Chưa chạy hoặc thất bại không phải kết quả âm tính/)).toBeVisible();

  await page.getByPlaceholder(/Nhập SMILES hoặc mô tả yêu cầu/i).fill('Why is this screening flagged for review?');
  await page.getByRole('button', { name: 'Gửi' }).click();
  await expect.poll(() => api.posts.length).toBe(1);
  expect(api.posts[0].body.subject_ref).toEqual({ kind: 'activity_screening', screening_id: SCREENING_EGFR });

  // Switching while the run waits changes the next question, not the run.
  await expect(page.getByText(/Run đang chạy vẫn trả lời về chủ thể đã chọn lúc gửi/)).toBeVisible();
  await abl1.click();
  await expect(abl1).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByText(/Hoạt tính:\s*thất bại/)).toBeVisible();
  await expect(page.getByText(/Hoạt tính:\s*thất bại \(descriptor_generation_failed\)/)).toBeVisible();
  expect(api.posts).toHaveLength(1);
  expect(api.posts[0].body.subject_ref).toEqual({ kind: 'activity_screening', screening_id: SCREENING_EGFR });
  expect(api.unmocked).toEqual([]);
});

const us2Routes: Routes = {
  ...screeningRoutes,
  [`GET ${S}`]: session(),
  [`GET ${S}/messages`]: messages([
    userMessage(1, 'Is there a measured EGFR IC50 that settles the REVIEW?'),
    publishedReply(2, answerTable.answer_markdown, ANSWER_1, RUN_1, 'passed'),
  ]),
  [`GET ${S}/answers/${ANSWER_1}`]: answerTable,
  [`GET ${S}/runs/${RUN_1}/result`]: result(RUN_1, { public_answer_id: ANSWER_1, candidate_answer_id: ANSWER_1 }),
  [`GET ${S}/runs/${RUN_1}/lineage`]: lineagePassed,
  [`GET ${S}/evidence`]: { evidence: [evidenceTable, evidenceRefused], count: 2 },
  [`GET ${S}/evidence/${EVIDENCE_TABLE}`]: evidenceTable,
  [`GET ${S}/evidence/${EVIDENCE_TABLE}/passages`]: (_r: unknown, url: URL) =>
    (url.searchParams.get('locator') === TABLE_CELL ? passagesCell : passagesOutline),
  [`GET ${S}/evidence/${EVIDENCE_REFUSED}`]: evidenceRefused,
  [`GET ${S}/evidence/${EVIDENCE_REFUSED}/passages`]: passagesRefused,
};

test('US2: a claim opens on the table row it quotes, the deep link survives a reload, and a refused text says why', async ({ page }) => {
  const api = await open(page, us2Routes);

  await expect(page.getByRole('status', { name: 'Trạng thái phát hành' })).toContainText('Đã kiểm tra hỗ trợ của nguồn');
  // It says the claims were checked against their sources -- and, in words, that this is not scientific correctness.
  await expect(page.getByRole('status', { name: 'Trạng thái phát hành' })).toContainText('không có nghĩa kết luận đúng về khoa học');
  await page.getByRole('link', { name: 'Nguồn và kiểm chứng →' }).click();
  await expect(page).toHaveURL(new RegExp(`/answers/${ANSWER_1}$`));
  await expect(page.getByText('Review vòng 1 (lần đầu)')).toBeVisible();

  await page.getByRole('link', { name: `đoạn ${TABLE_CELL} →` }).click();
  await expect(page).toHaveURL(/locator=table-wrap%3A1%2Frow%3A3%2Fcell%3A2/);
  const table = page.getByRole('table', { name: 'Table 1' });
  await expect(table.getByRole('columnheader', { name: 'EGFR IC50 (nM)' })).toBeVisible();
  await expect(table.locator('td[aria-current="true"]')).toHaveText(/^85/);
  // A comparator stays part of the value, as written.
  await expect(table.getByText('> 10000')).toBeVisible();
  await expect(page.getByText(/Phát lại từ bản chụp đóng băng của europepmc/)).toBeVisible();
  await expect(page.getByText(/trực tiếp về subject/)).toBeVisible();

  await page.reload();
  await expect(page.getByRole('table', { name: 'Table 1' }).locator('td[aria-current="true"]')).toHaveText(/^85/);

  await page.getByRole('button', { name: /TX-301 kinase panel abstract/ }).click();
  await expect(page.getByText('Không đọc được toàn văn')).toBeVisible();
  await expect(page.getByText(/the provider holds no full text for PMC0000002/)).toBeVisible();
  await expect(page.getByText(/bối cảnh \(cùng ligand, target khác\)/)).toBeVisible();
  expect(api.unmocked).toEqual([]);
});

const us3Routes: Routes = {
  [`GET ${S}`]: session(),
  // A toxicity-only deployment: activity replay off, so no screenings (404 is
  // that deployment's answer, and the selector shows nothing).
  [`GET ${S}/screenings`]: [404, { error: { code: 'not_found', message: 'activity replay is off', retryable: false, details: {} } }],
  [`GET ${S}/messages`]: messages([
    userMessage(1, 'Does the literature support the hERG flag?'),
    publishedReply(2, answerRepaired.answer_markdown, ANSWER_2, RUN_2, 'repaired_and_passed'),
    userMessage(3, 'Is the exposure margin safe?'),
    withheldReply(4, 'The answer could not be released. The server-confirmed facts are listed; the rest is in the audit view.', RUN_3),
    userMessage(5, 'And the older answer?'),
    legacyReply(6, ANSWER_LEGACY),
  ]),
  [`GET ${S}/answers/${ANSWER_2}`]: answerRepaired,
  [`GET ${S}/answers/${ANSWER_LEGACY}`]: answerLegacy,
  [`GET ${S}/runs/${RUN_2}/result`]: result(RUN_2, { review_status: 'repaired_and_passed', public_answer_id: ANSWER_2,
    candidate_answer_id: ANSWER_2 }),
  [`GET ${S}/runs/${RUN_3}/result`]: result(RUN_3, { disposition: 'partial_facts', review_status: 'blocked',
    knowledge_outcome: 'evidence_insufficient', detail: { confirmed_facts: [{}, {}], unresolved_claim_ids: ['clm_x'],
      needed_inputs: ['a free Cmax at the intended dose'] } }),
  [`GET ${S}/runs/${RUN_4}/result`]: legacyResult,
  [`GET ${S}/runs/${RUN_2}/lineage`]: lineageRepaired,
  [`GET ${S}/runs/${RUN_3}/lineage`]: lineageWithheld,
  [`GET ${S}/runs/${RUN_3}`]: runWithheld,
};

test('US3: a repaired answer, a withheld result and a legacy answer each say what was checked, and the audit shows why', async ({ page }) => {
  const api = await open(page, us3Routes);

  const banners = page.getByRole('status', { name: 'Trạng thái phát hành' });
  await expect(banners).toHaveCount(3);
  await expect(banners.nth(0)).toContainText('Đã sửa một lượt rồi qua kiểm tra hỗ trợ');
  await expect(banners.nth(1)).toContainText('Chỉ công bố dữ kiện xác nhận được');
  await expect(banners.nth(1)).toContainText('Cần tiếp: a free Cmax at the intended dose');
  await expect(banners.nth(2)).toContainText('Chưa có thông tin kiểm chứng');
  // The candidate the review blocked is never in the transcript.
  await expect(page.getByText(OVERREACH)).toHaveCount(0);

  await page.getByRole('link', { name: 'xem các giai đoạn và lý do trong audit →' }).click();
  await expect(page).toHaveURL(new RegExp(`/runs/${RUN_3}\\?tab=release$`));
  await expect(page.getByRole('tab', { name: 'Phát hành' })).toHaveAttribute('aria-selected', 'true');
  await expect(page.getByText('Người đọc không nhận được gì, và vì sao')).toBeVisible();
  await expect(page.getByText('Cần thêm đầu vào: a free Cmax at the intended dose')).toBeVisible();
  await expect(page.getByText('bản đang phát hành')).toHaveCount(0);

  await page.goto(`/s/${SESSION_ID}/answers/${ANSWER_2}`);
  const lineage = page.getByRole('list', { name: 'Lineage của claim' });
  await expect(lineage).toContainText('overreach · bị xóa · vòng 1: không được hỗ trợ');
  await expect(page.getByText(/Bản sửa chỉ xóa claim/)).toBeVisible();
  const stages = page.getByRole('list', { name: 'Bản nháp theo thứ tự' }).getByRole('listitem');
  await stages.first().getByRole('button', { name: 'Xem văn bản của giai đoạn này' }).click();
  await expect(stages.first()).toContainText('Bản nháp, chỉ để audit — không phải câu trả lời đã phát hành.');
  await expect(stages.first()).toContainText(OVERREACH);

  // Reload keeps every release state; the candidate still never reaches the chat.
  await page.goto(`/s/${SESSION_ID}`);
  await expect(page.getByRole('status', { name: 'Trạng thái phát hành' })).toHaveCount(3);
  await expect(page.getByText(OVERREACH)).toHaveCount(0);
  expect(api.unmocked).toEqual([]);
});

const us4Routes: Routes = {
  ...screeningRoutes,
  [`GET ${S}`]: session(),
  [`GET ${S}/messages`]: messages([]),
  [`GET ${S}/evidence`]: { evidence: [evidenceTable], count: 1 },
  [`GET ${S}/evidence/${EVIDENCE_TABLE}`]: evidenceTable,
  [`GET ${S}/evidence/${EVIDENCE_TABLE}/passages`]: passagesOutline,
  [`GET ${S}/cases`]: cases,
  [`GET ${S}/cases/${CASE_EGFR}`]: caseEgfr,
  [`GET ${S}/cases/${CASE_ABL1}`]: caseAbl1,
  [`GET ${S}/cases/${CASE_EGFR}/runs/${RUN_2}/changes`]: changesSecondTurn,
  [`GET ${S}/cases/${CASE_EGFR}/runs/${RUN_1}/changes`]: changesFirstTurn,
  [`GET ${S}/cases/${CASE_ABL1}/runs/${RUN_3}/changes`]: changesAbl1,
  [`POST ${S}/cases/${CASE_EGFR}/context`]: caseEgfr,
};

test('US4: the case board says what a turn changed and kept, and another target is another case', async ({ page }) => {
  const api = await open(page, us4Routes, `/s/${SESSION_ID}/evidence/${EVIDENCE_TABLE}`);

  await page.getByText(/Điều tra khoa học \(2\)/).click();
  await expect(page.getByLabel('Subject của hồ sơ')).toContainText('target EGFR');
  const changed = page.getByRole('list', { name: 'Đã đổi' });
  await expect(changed).toContainText('nhà nghiên cứu thêm bối cảnh: measured_pIC50 = 6.2 pIC50 (nguồn: researcher)');
  await expect(changed).toContainText('h1: đang mở → bị yếu đi');
  const kept = page.getByRole('list', { name: 'Giữ nguyên' });
  await expect(kept).toContainText(`screening ${SCREENING_EGFR}: cùng hash cccccccccccc`);
  await expect(kept).toContainText('quyết định policy và trạng thái stage không đổi');
  await expect(kept).toContainText('Đổi target hoặc ligand là một hồ sơ khác');

  await page.getByLabel('Chọn lượt').selectOption({ label: 'Lượt 1' });
  await expect(page.getByText(/Lượt đầu tiên của hồ sơ/)).toBeVisible();

  // Context the next turn will see, typed by the researcher.
  await page.getByLabel('Loại dữ liệu').fill('free_cmax');
  await page.getByLabel('Giá trị', { exact: true }).fill('0.05 µM');
  await page.getByRole('button', { name: 'Thêm vào hồ sơ' }).click();
  await expect.poll(() => api.posts.length).toBe(1);
  expect(api.posts[0]).toEqual({ path: `${S}/cases/${CASE_EGFR}/context`, body: { key: 'free_cmax', value: '0.05 µM' } });

  await page.getByLabel('Chọn hồ sơ').selectOption(CASE_ABL1);
  await expect(page.getByLabel('Subject của hồ sơ')).toContainText('target ABL1');
  await expect(page.getByText(/Lượt đầu tiên của hồ sơ/)).toBeVisible();
  expect(api.unmocked).toEqual([]);
});

test('a phone-width layout and the keyboard reach the subject and the release state', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await open(page, us2Routes);

  await expect(page.getByRole('status', { name: 'Trạng thái phát hành' })).toBeVisible();
  const egfr = page.getByRole('button', { name: /^EGFR/ });
  await egfr.focus();
  await page.keyboard.press('Space');
  await expect(egfr).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByText('Ước lượng hoạt tính')).toBeVisible();
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  expect(overflow).toBeLessThanOrEqual(0);
});
