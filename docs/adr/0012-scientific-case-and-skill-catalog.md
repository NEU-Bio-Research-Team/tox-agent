# ADR 0012 — A durable scientific case on the live path, and investigation methods as loadable skills

**Status:** accepted (behind default-off flags) · **Date:** 2026-09-25 · **Source:**
`docs/RETHINK_TOXAGENT_AGENTIC_RESEARCH_EVALUATION_VI.md` (§3, §4),
executed through `docs/backlog/SCIENTIFIC_INVESTIGATION_BACKLOG.md`

## Context

`DecisionSupportStateV1` (ADR 0011) made one run's goal, propositions, coverage
and stop reason into data. It ends with the run. A research question does not:
the researcher returns with an assay result, an exposure, a second paper, and
the product then has only a transcript and the previous run's proposition list
to continue from. Coverage is also computed from the relations the accepted
answer itself wrote, so it shows that a ref exists, not that the evidence is
direct, independent, or that counter-evidence was looked for.

The RETHINK review asks for the investigation itself to become the product
object — a case with a decision question, competing hypotheses, a ledger of
evidence for and against, a ledger of what is unknown, the agent's reasons for
acting and stopping, and a conditional conclusion — and for investigation
*methods* to grow as reviewed, versioned skills instead of new workflow code
or an ever-longer static prompt.

The superseded `ScientificAgentKernel` had a `CaseState` of its own. ADR 0011
retired that runtime; reviving it would reintroduce the second source of truth
ADR 0011 removed.

## Decision

1. **`ScientificCaseV1` (`domain/scientific_case.py`) is the cross-turn case on
   the live `AgentRuntimeGateway` path.** One open case per session and subject
   (analysis); a subject switch opens another case. It is the fold of an
   append-only update log (`scientific_case_events`); the snapshot
   (`scientific_cases`) is written only together with the events that produced
   it, under a revision check. Ids are `scase_…`, a prefix distinct from the
   kernel's `case_…`, so neither can resolve as the other.
2. **Evidence in the case is an artifact.** Every ledger entry cites an
   observation, an evidence record, a report or a context item the user
   supplied. There is no `agent_synthesis` source class; an inference is a
   hypothesis. A hypothesis becomes `supported`/`refuted`/`weakened` only with
   a ledger entry of that stance; a conclusion line the case "can say" names the
   entries it rests on. Ref coverage and quality coverage (direct independent
   evidence; counter-evidence considered) are reported side by side and never
   combined.
3. **The run's typed product is `DecisionDossierV1`,** compiled by the server
   from the case when a `decision_support` run ends and stored once per run.
   It keeps the three explanation layers apart: model attribution (with the
   explainer's measured verdict), independent scientific evidence, and the
   agent's recorded decisions. `submit_grounded_answer` remains the only way a
   turn ends; the dossier is compiled from state, not submitted, so there is
   still one exit and one validator.
4. **The model writes to the case through two closed tools**,
   `get_scientific_case` and `update_scientific_case` (typed operations; refs
   checked against artifacts the session really has). The server attaches runs,
   records what the accepted answer cited — as unlinked `contextual` entries,
   for any answer schema — plus, under grounded-answer v2, its relations with
   their stance, and finishes runs. The user adds context through the API.
   (Amended 2026-09-26, W9-04: the ledger no longer depends on
   `answer_draft_v2`; a source already in the ledger is not recorded twice.)
5. **Investigation methods are skills loaded on demand.** The prompt carries
   only an index of names and descriptions (the Agent Skills discovery model);
   bodies and references are read through two closed MCP tools,
   `read_scientific_skill` and `read_skill_reference`, never through the
   runtime's own `skill`/`read` permissions. A separate list tool was dropped:
   the index already is the listing, and it kept the fully-flagged
   `decision_support` surface under its ceiling. A skill is a hash-pinned package (`SKILL.md` +
   `skill.manifest.json` + `references/`); it cannot grant a tool, and the
   catalog does not advertise a skill whose required tools the run lacks. Runs
   record the skills they were shown and the ones they loaded.
6. **Everything is behind default-off flags** (`scientific_case_v1`, and the
   skill catalog's own flag). With them off the tool surface, its schema hash
   and the prompt are byte-identical to before. Turning either on by default is
   a release decision taken on the paired comparison study and TAB-Suite, not
   here.

## Consequences

- Migration 0020 is additive (three new tables). The kernel's tables stay
  superseded and untouched; retiring them remains ADR 0011's separate change.
- `tools.registry.FLAG_GATED_TOOLS` is the one table of flag-gated tools; the
  bootstrap applies it and `docs/CAPABILITY_MATRIX.md` reports it.
- The comparison study (`evals/investigation`) can grade a dossier, not only a
  chat answer, and its logs carry case ids and revisions.
- Not decided here: value-of-information scoring (RETHINK L3), learning from
  lab outcomes (L4), and any multi-agent split (ADR 0011 point 6 still holds).
