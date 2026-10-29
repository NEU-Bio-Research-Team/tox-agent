# Architecture of the agent layer

Written for engineers working on this repository.

## The path of one turn

```
message
  │  admission: intent, subject binding, budget          application/conversation/
  ▼
run ──► turn assembly                                    harness/gateway/context.py
  │       route ─ goal contract ─ case view ─ profile
  ▼
model turn (one dispatch)                                harness/gateway/runtime_gateway.py
  │       reads artifacts, retrieves evidence,
  │       runs procedures, writes the case through
  │       typed operations
  ▼
submit_grounded_answer ──► hard validator                validation/answer/
  │                          every number cites a field
  │                          every required limitation declared
  ▼
reviewer turn (independent)                              application/investigation/claim_review.py
  │       one verdict per claim, against its sources
  ▼
release ──► answer, or the grounded part, or nothing     application/investigation/release_gate.py
  │
  ▼
run result: disposition, review status, stop reason
```

Everything the model did is recorded: each tool call with its phase and
duration, each rejected draft with its violations, each review verdict with its
reason, and the contract the turn actually read.

## Subject binding

A decision-support message is bound to **one** subject when it is accepted: the
analysis or the activity screening it names, or the session's only one. Two
subjects and no selection is a clarification, never the first one.

The binding travels frozen in the run envelope to the gateway, the case, the
decision state and the dossier. It is what makes "another screening of the same
ligand is another subject" enforceable rather than advisory: evidence retrieved
for one subject cannot be cited as direct evidence about another.

Flag: `subject_binding_v1`.

## The four serving routes

A question's *intent* says which pipeline runs. Its *route* says how much
investigation it deserves. The router is deterministic and spends no model
call.

| Route | Taken when | Runs as |
|---|---|---|
| `read_explain` | The subject and its artifact are bound, the question is shaped as a read, and nothing in it asks for a judgement | Profile `decision_support_read`: artifact reads and the answer. No search, no case writes, no skills, no calculations |
| `investigate` | Anything asking for a judgement, and the default | The full loop, profile `decision_support` |
| `update` | A researcher value is in force that the previous run of this subject did not see, or a changed premise left claims stale | The full loop; the contract names each new value and asks what it changes *and* what it leaves unchanged |
| `compare_batch` | Comparison wording **and** more than one candidate | The full loop plus `compare_candidates`; one row per candidate, from that candidate's own artifact |
| `undetermined` | No subject bound, or a comparison asked for with one candidate | Profile `decision_support_read`; the contract asks the clarifying question first |

Two rules the router will not break:

- **A keyword never decides a research question.** Any assessment marker
  ("contradict", "margin", "should we", "why") outranks every read marker, so
  "what is the margin" does not reach the cheap route.
- **A changed premise outranks the wording.** A value read off a stale claim is
  wrong whatever the question looked like.

Routing decisions carry reason codes, and the benchmark reports routes and
their cost separately from task success.

Code: `application/investigation/router.py`. Flag: `investigator_contract_v1`.

## The goal contract

One projection, written once per turn, of what the run was asked to do: the
route, the bound subject, what must be answered, the actions available (the
tool names the registry resolved for this turn's profile — not a wish list),
the admitted budget, and the typed stop conditions.

It has no `gold` parameter and cannot be given one; a unit test asserts the
parameter set. Required claims come from the question, never from a hidden
benchmark answer.

A contract is not editable mid-run: a contract that changed would let a run
that answered a different question look like one that answered this one.

Code: `application/investigation/goal_contract.py`.

## The scientific case

Per session and subject, carried across turns. Four groups of data: context the
researcher supplied, hypotheses, an evidence ledger with scopes, and
uncertainties — plus actions, derived claims and a conclusion.

**It is append-only, through typed operations.** The model may send nine of
them (`set_question`, `add_hypothesis`, `revise_hypothesis`, `record_evidence`,
`record_uncertainty`, `resolve_uncertainty`, `record_action`,
`propose_next_test`, `set_conclusion`); the rest belong to the researcher or
the server. Every operation is logged, and the case snapshot must equal the
replay of its log.

Four things in it are worth knowing about:

**Measurement values** (`domain/measurement_value.py`). A declared value
becomes the set of values it permits, in canonical units, with the conversion
recorded. This separates three questions that were one string comparison
before: are these the same measurement (identity), is this the same number
written differently (representation), and do these two numbers agree
(numeric). A µM record and an nM record are no longer "different"; `> 30 µM`
against a "below 10 µM" rule is a determinate no, while `< 30 µM` is
undetermined. Two disjoint values are *not* a contradiction unless the caller
declares what counts as a real difference, because that is an assay judgement.

**Researcher values** (`domain/scientific_case/values.py`). The context log is
already an ordered set of versions per key, so correcting a value appends a
second item and the first stays citable: `case_value:free_cmax@v2` resolves the
same way in every later run. A value nobody verified is `user_asserted` — the
weakest standing, never the strongest — and carries the limitation that says
so. `sme_verified` cannot be awarded by a run.

**Claim dependencies** (`domain/scientific_case/dependencies.py`). A derived
claim records its inputs as *references*, its rule and its assumptions. A
justification that copied the number would still look valid after the number
changed, which is the bug this exists to prevent. When a premise moves,
`affected()` walks transitively, and a claim is marked `needs_reassessment` only
when *every* justification lost a premise — one standing route means the claim
is not in doubt. Nothing is deleted: the earlier answer was computed from those
inputs, and the next answer has to be able to say what changed.

**The case view** (`application/investigation/case_view.py`). What the turn
actually sees, in six groups: the subject and contract; the observations with
their identifiers; the conclusion in force and what changed; evidence,
counter-evidence and the metadata records did not state; open gaps, prior
searches and budget left; and the skills for this task. It carries
*references and counts*, and says how many it withheld — content stays behind a
tool.

Flag: `scientific_case_v1`.

## Tools and procedures

Tools are registered per capability profile, and a profile is a small closed
set — a unit test enforces a ceiling, because a large roster costs money and
misroutes. There is no shell, no filesystem and no raw web access in any
conversational profile.

`run_scientific_procedure` is one tool for five determinate operations over
records the run already read:

| Procedure | Answers |
|---|---|
| `normalise_measurements` | Each record's value in canonical units, with the conversion and the decisive fields it did not state |
| `deduplicate_by_origin` | One record per experiment — two rows of one measurement are not two independent sources |
| `compare_measurements` | Whether two records' scopes are comparable, then whether their values agree under a declared tolerance |
| `check_dependencies` | Which derived claims a changed premise left stale, plus unresolved references and cycles |
| `compare_candidates` | The session's candidates on one declared criterion, grouped by scale, never a combined score |

The procedure name is an enum, so an unknown one is a schema refusal before any
code runs. A procedure reaches no provider, so a repeat is free and a frozen
benchmark can replay it. A procedure proposes; the investigator commits.

Code: `application/investigation/procedures.py`,
`tools/definitions/procedures.py`. Flag: `scientific_procedures_v1`.

## Gaps and stopping

Seven kinds of unresolved requirement, typed because they are not
interchangeable: `identity_or_scope`, `no_relevant_evidence`, `span_not_read`,
`measurement_unresolved`, `disagreement_unexplained`, `only_researcher_has_it`,
`provider_or_corpus`. A run that reports "insufficient" without saying which of
the seven it hit has told the researcher nothing they can use.

A run ends on one of eight stop reasons. Three distinctions in them matter:

- `provider_failure` is **not** `insufficient_evidence`. One says the product
  could not look; the other says the science is unsettled.
- `answered_from_artifacts` is not `insufficient_evidence` either: the answer
  read what the run already held and there was nothing to search for.
- `needs_user_input` names the input only the researcher has.

An access failure is **recorded**, not inferred from an empty result: an empty
result and a refused request look alike afterwards and mean opposite things.

Code: `domain/investigation_gap.py`, `domain/decision_state.py`.

## Verification and release

Three gates, in order, and each one can only subtract:

1. **The hard validator** (`validation/answer/`). Every number in the prose
   must be a claim; every numeric claim must cite an observation field that
   holds that value; every limitation the cited sources require must be
   declared. A quoted passage must name the locator it came from. This is
   mechanical and cannot be argued with.
2. **The semantic reviewer** (`claim_review.py`). An independent turn, one
   verdict per claim, judged against a bundle the *server* assembles: the
   passages the claim quotes, the measured value the record states, the scope
   this run's relations declared, and — for a computed value — its formula and
   every input with its provenance. For a researcher-supplied input the bundle
   says so, because a cited record does not have to make the researcher's
   statement.
3. **The release gate** (`release_gate.py`). Under `strict-reviewed-v1` an
   accepted answer is a *candidate*; one bounded repair turn is allowed; what
   the reviewer still blocks is withheld. A run can publish the grounded part
   and withhold the rest (`partial_facts`), which is more useful than
   publishing nothing and honest in a way publishing everything is not.

If a model spends both attempts and still fails validation, the server writes a
**deterministic fallback** containing no model text at all — only facts read
from the run's own bound subject, each with its field path. It answers about
*that* subject, never whichever analysis the session touched last.

## Budget

One effective budget per run, named by a preset so it is a recorded fact rather
than whatever the environment happened to say. It counts every logical tool
invocation, reserves slots for the submission, the review and the repair, and
caps invalid calls so a malformed-call loop cannot spend a run. Each call is
recorded under its phase: `investigator`, `hard_correction`, `review_initial`,
`semantic_repair`, `review_final`.

Code: `application/runs/budget.py`, `tools/runner.py`.

## State ownership

| State | Owner | The model's access |
|---|---|---|
| Session, run, budget | Server | None |
| Subject binding | Server, at admission | Read, through the contract and the subject line |
| Analyses, screenings, policy decisions | Workflow | Read, through tools, by id |
| Evidence records and full text | Server, on retrieval | Read, through tools |
| Scientific case | Server, through typed operations | Nine operations, through one tool |
| Observations | Whichever component produced them | Cite by id and field path |
| Answers, drafts, reviews, releases | Server | Submit a draft; never release |

The single rule behind the table: the model proposes, the server records. Every
value in a released answer resolves to a stored field, which is what makes an
answer auditable after the fact.
