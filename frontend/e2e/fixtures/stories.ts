/**
 * The four user stories (master plan §12) as the API states them.
 *
 * Every payload is typed by the control plane's generated OpenAPI types, and
 * `tsconfig.e2e.json` puts this file under `npm run typecheck`: a field the
 * server renames is a compile error here, the same as in the app. The shapes
 * follow responses captured from a local deployment of the same commit (a
 * protocol-v2 arm on openai/gpt-6-luna, 2026-10-22); the values are synthetic
 * engineering fixtures, never claims about a real molecule.
 */
import type { components } from '../../src/shared/api/openapi';

type S = components['schemas'];

export const SESSION_ID = 'ses_story';
const NOW = '2026-10-22T00:30:00+00:00';
const SMILES = 'CCOc1ccc2nc(S(N)(=O)=O)sc2c1';

export const SCREENING_EGFR = 'scr_egfr_wt';
export const SCREENING_ABL1 = 'scr_abl1_wt';
export const EVIDENCE_TABLE = 'evd_' + 'a'.repeat(32);
export const EVIDENCE_REFUSED = 'evd_' + 'b'.repeat(32);
export const RUN_1 = 'run_' + '1'.repeat(32);
export const RUN_2 = 'run_' + '2'.repeat(32);
export const RUN_3 = 'run_' + '3'.repeat(32);
export const RUN_4 = 'run_' + '4'.repeat(32);
export const ANSWER_1 = 'ans_' + '1'.repeat(32);
export const ANSWER_2 = 'ans_' + '2'.repeat(32);
export const ANSWER_LEGACY = 'ans_' + '9'.repeat(32);
export const CASE_EGFR = 'scase_' + 'e'.repeat(32);
export const CASE_ABL1 = 'scase_' + 'f'.repeat(32);
export const TABLE_CELL = 'table-wrap:1/row:3/cell:2';

// --------------------------------------------------------------- session

export function session(overrides: Partial<S['SessionProjection']> = {}): S['SessionProjection'] {
  return {
    session_id: SESSION_ID, status: 'active', preferred_language: 'vi', title: 'Story workspace',
    title_source: 'manual', title_status: 'ready', version: 2, created_at: NOW, updated_at: NOW,
    latest_event_sequence: 0, active_run: null, recent_runs: [], active_analysis: null,
    ...overrides,
  };
}

export function queuedRun(runId: string): S['RunProjection'] {
  return {
    run_id: runId, status: 'queued', lane: 'agentic', intent: 'decision_support',
    trigger_message_id: 'msg_queued', runtime_binding_id: null, recovery_of_run_id: null,
    failure_code: null, potentially_billed: false, deadline_at: NOW, created_at: NOW,
    started_at: null, ended_at: null,
  };
}

// --------------------------------------------------------------- screenings

function summary(id: string, targetId: string, status: string, overrides: Partial<S['ScreeningSummary']> = {}): S['ScreeningSummary'] {
  return {
    screening_id: id, status, canonical_smiles: SMILES,
    target: { target_id: targetId }, assay_context: 'IC50|wild_type', intended_target_role: 'intended_target',
    toxicity_run: false, has_activity_estimate: true, has_prioritisation_decision: true,
    has_toxicity_snapshot: false, ligand_names: ['TX-301'], model_admitted: false,
    record_sha256: 'c'.repeat(64), source: 'replay', attached_at: NOW, ...overrides,
  };
}

export const screenings: S['ScreeningList'] = {
  mode: 'replay',
  screenings: [
    summary(SCREENING_EGFR, 'EGFR', 'completed_with_toxicity_not_run'),
    summary(SCREENING_ABL1, 'ABL1', 'activity_failed', { has_activity_estimate: false, has_prioritisation_decision: false }),
  ],
};

export const screeningEgfr: S['ScreeningDetail'] = {
  screening_id: SCREENING_EGFR,
  subject: { canonical_smiles: SMILES, target: { target_id: 'EGFR', uniprot_id: 'P00533' }, assay_context: 'IC50|wild_type' },
  subject_key: 'activity:' + 'e'.repeat(40), intended_target_role: 'intended_target',
  status: 'completed_with_toxicity_not_run',
  stages: {
    activity: { status: 'completed', failure_code: null },
    prioritisation: { status: 'completed', decision: 'review' },
    toxicity: { status: 'not_run', not_run_reason: 'review_required_awaiting_researcher', failure_code: null,
      authorised_by: null, frozen_analysis_available: false },
  },
  activity_estimate: {
    estimate: { predicted_pic50: 7.1, prediction_interval: [6.8, 7.4], interval_level: 0.9 },
    applicability: { ligand: 'ok', target: 'ok', joint: 'ok' },
    model: { model_id: 'synthetic-activity-fixture-v1', admitted: false },
  },
  prioritisation_decision: {
    status: 'review', reason_codes: ['prediction_interval_crosses_target_threshold'],
    policy: { threshold_pic50: 7.0, threshold_source: 'programme_default' },
  },
  artifact_refs: [{ kind: 'screening', source_id: SCREENING_EGFR, content_sha256: 'c'.repeat(64) }],
  required_limitations: ['activity_model_not_admitted', 'toxicity_not_run_not_negative'],
  mode: 'replay', serving: false, model_admitted: false, record_sha256: 'c'.repeat(64), attached_at: NOW,
};

export const screeningAbl1: S['ScreeningDetail'] = {
  ...screeningEgfr,
  screening_id: SCREENING_ABL1,
  subject: { canonical_smiles: SMILES, target: { target_id: 'ABL1', uniprot_id: 'P00519' }, assay_context: 'IC50|wild_type' },
  subject_key: 'activity:' + 'f'.repeat(40),
  status: 'activity_failed',
  stages: {
    activity: { status: 'failed', failure_code: 'descriptor_generation_failed' },
    prioritisation: { status: 'not_run' },
    toxicity: { status: 'not_run', not_run_reason: 'activity_stage_failed', failure_code: null,
      authorised_by: null, frozen_analysis_available: false },
  },
  activity_estimate: null,
  prioritisation_decision: null,
  artifact_refs: [{ kind: 'screening', source_id: SCREENING_ABL1, content_sha256: 'd'.repeat(64) }],
  record_sha256: 'd'.repeat(64),
};

// --------------------------------------------------------------- messages

function part(index: number, type: string, content: Record<string, unknown>, owner: string): S['MessagePart'] {
  return { part_id: `prt_${owner}_${index}`, index, type, content };
}

export function userMessage(sequence: number, text: string): S['Message'] {
  return { message_id: `msg_u${sequence}`, role: 'user', sequence, created_at: NOW, client_message_id: null,
    parts: [part(0, 'text', { text }, `u${sequence}`)] };
}

export function publishedReply(sequence: number, text: string, answerId: string, runId: string, review: string): S['Message'] {
  const owner = `a${sequence}`;
  return {
    message_id: `msg_${owner}`, role: 'assistant', sequence, created_at: NOW, client_message_id: null,
    parts: [
      part(0, 'text', { text }, owner),
      part(1, 'answer_ref', { answer_id: answerId }, owner),
      part(2, 'result_ref', { run_id: runId, disposition: 'publish', review_status: review }, owner),
    ],
  };
}

export function withheldReply(sequence: number, text: string, runId: string): S['Message'] {
  const owner = `a${sequence}`;
  return {
    message_id: `msg_${owner}`, role: 'assistant', sequence, created_at: NOW, client_message_id: null,
    parts: [
      part(0, 'text', { text }, owner),
      part(1, 'result_ref', { run_id: runId, disposition: 'partial_facts', review_status: 'blocked' }, owner),
    ],
  };
}

export function legacyReply(sequence: number, answerId: string): S['Message'] {
  const owner = `a${sequence}`;
  return {
    message_id: `msg_${owner}`, role: 'assistant', sequence, created_at: NOW, client_message_id: null,
    parts: [part(0, 'text', { text: 'An older answer.' }, owner), part(1, 'answer_ref', { answer_id: answerId }, owner)],
  };
}

// --------------------------------------------------------------- answers

const EGFR_TEXT = 'Table 1 of the retrieved study reports an EGFR IC50 of 85 nM for TX-301 in a biochemical assay.';

export const answerTable: S['GroundedAnswer'] = {
  schema_version: 'grounded-answer-v2', answer_id: ANSWER_1, run_id: RUN_1,
  answer_markdown: `${EGFR_TEXT} The screening's REVIEW is a policy decision; the toxicity stage did not run.`,
  candidate_generation: 1, content_sha256: '1'.repeat(64), created_at: NOW, is_fallback: false,
  claims: [{
    claim_id: 'clm_' + '1'.repeat(32), kind: 'scientific', text: EGFR_TEXT, transform: 'identity',
    citation_ids: [EVIDENCE_TABLE], evidence_spans: [{ evidence_id: EVIDENCE_TABLE, locator: TABLE_CELL }],
    local_ref: 'egfr_ic50',
  }],
  limitations: [{ code: 'toxicity_not_run_not_negative', text: 'The toxicity stage did not run; that is not a negative result.' }],
  recommended_next_steps: [],
  release: null,
};

const SERIES = 'A retrieved study reports hERG channel blockade in a screened series.';
const OVERREACH = 'The retrieved study shows that this exact compound blocks hERG.';

export const answerRepaired: S['GroundedAnswer'] = {
  ...answerTable, answer_id: ANSWER_2, run_id: RUN_2, answer_markdown: SERIES, candidate_generation: 2,
  claims: [{ claim_id: 'clm_' + '2'.repeat(32), kind: 'scientific', text: SERIES, transform: 'identity',
    citation_ids: [EVIDENCE_TABLE], local_ref: 'series' }],
  limitations: [{ code: 'evidence_scope_limited', text: 'One retrieved study.' }],
};

export const answerLegacy: S['GroundedAnswer'] = {
  ...answerTable, answer_id: ANSWER_LEGACY, run_id: RUN_4, answer_markdown: 'An older answer.', claims: [], limitations: [],
};

// --------------------------------------------------------------- run results

export function result(runId: string, overrides: Partial<S['RunResult']> = {}): S['RunResult'] {
  return {
    run_id: runId, recorded: true, release_policy: 'strict-reviewed-v1', schema_version: 'run-result-v1',
    disposition: 'publish', review_status: 'passed', execution_outcome: 'completed',
    knowledge_outcome: 'supported_assessment', release_policy_sha256: '0'.repeat(64),
    public_answer_id: null, candidate_answer_id: null, detail: {}, created_at: NOW, note: null, ...overrides,
  };
}

export const legacyResult: S['RunResult'] = {
  run_id: RUN_4, recorded: false, release_policy: 'legacy',
  note: 'this run has no recorded result; whether its answer was reviewed is not known',
};

// --------------------------------------------------------------- lineage

function stage(index: number, phase: string, overrides: Partial<S['LineageStage']>): S['LineageStage'] {
  return {
    index, phase, generation: index + 1, outcome: 'accepted', answer_id: null, hard_valid: true, is_public: false,
    violation_codes: [], claim_count: 1, limitation_codes: [], answer_markdown: '', created_at: NOW, ...overrides,
  };
}

const NO_MISSING: S['LineageMissing'] = {
  needed_inputs: [], unresolved_claim_ids: [], lost_limitation_codes: [], confirmed_fact_count: 0,
  reason: null, failure_code: null,
};

export const lineagePassed: S['RunStageLineage'] = {
  schema_version: 'stage-lineage-v1', run_id: RUN_1, recorded: true, release_policy: 'strict-reviewed-v1',
  disposition: 'publish', review_status: 'passed', execution_outcome: 'completed',
  public_answer_id: ANSWER_1, candidate_answer_id: ANSWER_1,
  stages: [stage(0, 'initial', { answer_id: ANSWER_1, is_public: true, answer_markdown: answerTable.answer_markdown })],
  reviews: [{ round: 1, phase: 'review_initial', status: 'completed', reason: null, answer_id: ANSWER_1,
    counts: { supported: 1 }, blocking_claim_ids: [],
    verdicts: [{ claim_id: 'clm_' + '1'.repeat(32), key: 'egfr_ic50', text: EGFR_TEXT, verdict: 'supported', reason: 'Table 1 row 3 states 85 nM' }] }],
  claims: [{ key: 'egfr_ic50', kind: 'scientific', candidate_text: EGFR_TEXT, final_text: EGFR_TEXT, change: 'kept',
    verdicts: [{ round: 1, verdict: 'supported', reason: 'Table 1 row 3 states 85 nM' }] }],
  repair: null, missing: NO_MISSING,
};

export const lineageRepaired: S['RunStageLineage'] = {
  ...lineagePassed, run_id: RUN_2, review_status: 'repaired_and_passed', public_answer_id: ANSWER_2, candidate_answer_id: ANSWER_2,
  stages: [
    stage(0, 'initial', { answer_id: 'ans_' + '7'.repeat(32), claim_count: 2, answer_markdown: `${SERIES} ${OVERREACH}` }),
    stage(1, 'semantic_repair', { answer_id: ANSWER_2, is_public: true, answer_markdown: SERIES }),
  ],
  reviews: [
    { round: 1, phase: 'review_initial', status: 'completed', reason: null, answer_id: 'ans_' + '7'.repeat(32),
      counts: { supported: 1, not_supported: 1 }, blocking_claim_ids: ['clm_x'],
      verdicts: [
        { claim_id: 'clm_s', key: 'series', text: SERIES, verdict: 'supported', reason: 'the abstract says so' },
        { claim_id: 'clm_x', key: 'overreach', text: OVERREACH, verdict: 'not_supported', reason: 'the source tested a series, not this compound' },
      ] },
    { round: 2, phase: 'review_final', status: 'completed', reason: null, answer_id: ANSWER_2, counts: { supported: 1 },
      blocking_claim_ids: [], verdicts: [{ claim_id: 'clm_' + '2'.repeat(32), key: 'series', text: SERIES, verdict: 'supported', reason: 'the abstract says so' }] },
  ],
  claims: [
    { key: 'series', kind: 'scientific', candidate_text: SERIES, final_text: SERIES, change: 'kept',
      verdicts: [{ round: 1, verdict: 'supported' }, { round: 2, verdict: 'supported' }] },
    { key: 'overreach', kind: 'scientific', candidate_text: OVERREACH, final_text: null, change: 'removed',
      verdicts: [{ round: 1, verdict: 'not_supported', reason: 'the source tested a series, not this compound' }] },
  ],
  repair: { removed_claim_ids: ['overreach'], added_claim_ids: [], reworded_claim_ids: [], removal_only: true,
    lost_limitation_codes: [] },
};

export const lineageWithheld: S['RunStageLineage'] = {
  ...lineageRepaired, run_id: RUN_3, disposition: 'partial_facts', review_status: 'blocked',
  public_answer_id: null, candidate_answer_id: 'ans_' + '8'.repeat(32),
  stages: lineageRepaired.stages.map((s) => ({ ...s, is_public: false })),
  missing: { needed_inputs: ['a free Cmax at the intended dose'], unresolved_claim_ids: ['clm_x'],
    lost_limitation_codes: [], confirmed_fact_count: 2, reason: 'a claim was still not supported after the one repair',
    failure_code: null },
};

export const runWithheld: S['RunDetail'] = {
  run_id: RUN_3, status: 'completed', lane: 'agentic', intent: 'decision_support', trigger_message_id: 'msg_u5',
  runtime_binding_id: null, recovery_of_run_id: null, failure_code: null, potentially_billed: true,
  deadline_at: NOW, created_at: NOW, started_at: NOW, ended_at: NOW, runtime: null,
  tool_calls: [], usage: { status: 'unknown', events: [] },
};

// --------------------------------------------------------------- evidence

function evidence(id: string, title: string, overrides: Partial<S['EvidenceRecordView']> = {}): S['EvidenceRecordView'] {
  return {
    evidence_id: id, title, authors: ['Synthetic A'], published_at: '2025-01-01', source_type: 'article',
    source_quality_tier: 'primary', identifier: { doi: null, pmid: null, pmcid: 'PMC0000001', cid: null, other: null },
    canonical_url: null, abstract_or_excerpt: 'Synthetic engineering fixture.', normalized_facts: {},
    status: 'accepted', rejection_reason: null, provider: 'snapshot', retrieved_at: NOW, content_sha256: 'e'.repeat(64),
    untrusted_external_content: true, full_text: null, relevance_assessment: {}, ...overrides,
  };
}

export const evidenceTable = evidence(EVIDENCE_TABLE, 'TX-301 EGFR IC50 by a biochemical assay (synthetic)', {
  relevance_assessment: { relevance: 'direct', reason_codes: ['compound_match', 'target_match'],
    assessed_scope: { ligand: 'TX-301', target_id: 'EGFR' } },
});

export const evidenceRefused = evidence(EVIDENCE_REFUSED, 'TX-301 kinase panel abstract (synthetic)', {
  relevance_assessment: { relevance: 'contextual', reason_codes: ['ligand_on_other_target'],
    assessed_scope: { ligand: 'TX-301', target_id: 'ABL1' } },
});

const READ: S['EvidenceTextAccess'] = {
  state: 'full_text_retrieved', reason: null, full_text_id: 'PMC0000001', passage_count: 6, omitted: {}, truncated: false,
  provenance: { replay: true, served_by: 'snapshot', captured_from: 'europepmc', captured_at: '2026-10-21T00:00:00+00:00',
    parser_version: 'jats-passages-v1' },
};

export const passagesCell: S['EvidencePassagesView'] = {
  schema_version: 'evidence-passages-v1', evidence_id: EVIDENCE_TABLE, access: READ, measurement: null,
  focus: {
    locator: TABLE_CELL, found: true,
    passage: { locator: TABLE_CELL, section: 'Table 1', kind: 'table_cell', text: '85', truncated: false, cells: [] },
    table: {
      locator: 'table-wrap:1', label: 'Table 1', caption: 'Table 1 Kinase inhibition by biochemical assay, 1 mM ATP',
      header_locator: 'table-wrap:1/row:1', header: ['Compound', 'EGFR IC50 (nM)', 'ABL1 IC50 (nM)'],
      row_locator: 'table-wrap:1/row:3', row_kind: 'table_row', cells: ['TX-301', '85', '> 10000'], cited_cell: 2,
      columns: [
        { header: 'Compound', value: 'TX-301', cited: false },
        { header: 'EGFR IC50 (nM)', value: '85', cited: true },
        { header: 'ABL1 IC50 (nM)', value: '> 10000', cited: false },
      ],
    },
  },
  passages: [], untrusted_external_content: true,
};

export const passagesOutline: S['EvidencePassagesView'] = {
  ...passagesCell, focus: null,
  passages: [
    { locator: 'sec:1/p:1', section: 'Results', kind: 'paragraph', preview: 'Inhibition is summarised in Table 1.', truncated: false },
    { locator: 'table-wrap:1/row:3', section: 'Table 1', kind: 'table_row', preview: 'Compound: TX-301 | EGFR IC50 (nM): 85 | ABL1 IC50 (nM): > 10000', truncated: false },
  ],
};

export const passagesRefused: S['EvidencePassagesView'] = {
  schema_version: 'evidence-passages-v1', evidence_id: EVIDENCE_REFUSED,
  access: { state: 'access_failed', reason: 'the provider holds no full text for PMC0000002', full_text_id: 'PMC0000002',
    passage_count: 0, omitted: {}, truncated: false, provenance: {} },
  measurement: null, focus: null, passages: [], untrusted_external_content: true,
};

// --------------------------------------------------------------- cases

const COVERAGE: S['CaseCoverage'] = { hypotheses: 1, with_any_source: 1, with_independent_direct_evidence: 1,
  with_counterevidence_considered: 1, open_uncertainties: 0, blocking_uncertainties: 0 };

function caseBody(id: string, targetId: string, runs: string[], overrides: Partial<S['ScientificCase']> = {}): S['ScientificCase'] {
  return {
    schema_version: 'scientific-case-v1', case_id: id, session_id: SESSION_ID,
    subject_key: `activity:${targetId.toLowerCase()}`, question: `Is TX-301 worth a direct ${targetId} assay?`,
    decision_context: '', subject_refs: [`screening:${targetId}`], requester: 'study-user',
    data_scope: { external_search: true, reason: '', run_id: null }, status: 'open',
    context: [], hypotheses: [], evidence: [], uncertainties: [], actions: [], next_tests: [],
    conclusion: { can_say: [], cannot_say: [], what_would_change: [], approver: '', run_id: null },
    runs: runs.map((runId) => ({ run_id: runId, goal: 'q', stop_reason: 'sufficient', answer_id: null })),
    coverage: COVERAGE, revision: 8, created_at: NOW, updated_at: NOW,
    subject: { kind: 'activity_screening', identity: { canonical_smiles: SMILES, target: { target_id: targetId }, assay_context: 'IC50|wild_type' } },
    ...overrides,
  };
}

export const caseEgfr = caseBody(CASE_EGFR, 'EGFR', [RUN_1, RUN_2], {
  context: [{ id: 'c1', key: 'measured_pIC50', value: '6.2', note: '', actor: 'user', run_id: null, unit: 'pIC50', source: 'researcher' }],
  hypotheses: [{ id: 'h1', statement: 'TX-301 is potent on EGFR', kind: 'model_signal', refutation_condition: 'A measured pIC50 below 7',
    status: 'weakened', status_reason: 'the measured value is below the interval', actor: 'model', run_id: RUN_1 }],
});

export const caseAbl1 = caseBody(CASE_ABL1, 'ABL1', [RUN_3], { revision: 3 });

function caseSummary(body: S['ScientificCase']): S['ScientificCaseSummary'] {
  return {
    case_id: body.case_id, question: body.question, subject_key: body.subject_key, subject_refs: body.subject_refs,
    status: body.status, revision: body.revision, hypotheses: body.hypotheses.length, evidence: body.evidence.length,
    open_uncertainties: 0, runs: body.runs.length, coverage: body.coverage, updated_at: NOW, requester: body.requester,
    external_search: true,
  };
}

export const cases: S['ScientificCaseListResponse'] = { cases: [caseSummary(caseEgfr), caseSummary(caseAbl1)] };

const SECTION: S['CaseSectionChanges'] = { added: [], changed: [], kept: [], removed: [] };
const KEPT_ARTIFACT = { kind: 'screening', source_id: SCREENING_EGFR, content_sha256: 'c'.repeat(64) };

export const changesSecondTurn: S['CaseRunChanges'] = {
  schema_version: 'case-changes-v1', case_id: CASE_EGFR, run_id: RUN_2, previous_run_id: RUN_1,
  from_revision: 5, to_revision: 8, subject: { subject_key: caseEgfr.subject_key, kept: true },
  updates_by_actor: { user: 1, model: 2, server: 2 }, operations: [],
  context: { ...SECTION, added: [{ id: 'c1', key: 'measured_pIC50', value: '6.2', unit: 'pIC50', source: 'researcher', scope: null, actor: 'user' }] },
  hypotheses: { ...SECTION, changed: [{ id: 'h1', statement: 'TX-301 is potent on EGFR', status: 'weakened',
    status_reason: 'the measured value is below the interval', actor: 'model',
    changes: [{ field: 'status', before: 'open', after: 'weakened' }] }] },
  evidence: SECTION, uncertainties: SECTION, next_tests: SECTION,
  conclusion: { changed: false, before: null, after: {}, released_kept: null },
  data_scope: { changed: false, before: null, after: {} }, run: { disposition: 'publish' },
  artifacts: { compared: true, reason: null, added: [], changed: [], removed: [], kept: [KEPT_ARTIFACT] },
};

export const changesFirstTurn: S['CaseRunChanges'] = {
  ...changesSecondTurn, run_id: RUN_1, previous_run_id: null, from_revision: 0, to_revision: 5,
  context: SECTION, hypotheses: { ...SECTION, added: [{ id: 'h1', statement: 'TX-301 is potent on EGFR', status: 'open', actor: 'model' }] },
  artifacts: { compared: false, reason: 'the first run of the case', added: [KEPT_ARTIFACT], changed: [], kept: [], removed: [] },
};

export const changesAbl1: S['CaseRunChanges'] = {
  ...changesFirstTurn, case_id: CASE_ABL1, run_id: RUN_3, subject: { subject_key: caseAbl1.subject_key, kept: true },
};
