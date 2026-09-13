# Explanation policy

## Scope

One `get_or_create_explanation` call covers exactly one `(endpoint, task)`
target. `tox21` requires a `task`; the other endpoints refuse one.

The explanation is computed with the same model that produced this snapshot's
prediction for that endpoint. You do not choose the model and cannot pass one.

## Reuse

A repeat call for the same target on the same analysis returns the stored
explanation with `reused: true` and does not spend the predictor budget again.
`get_explanation_package` re-reads one without any chance of creating one; use
it while drafting.

## What the fields mean

| Field | Meaning |
|---|---|
| `signed_contribution` | Direction and size of an atom's contribution to the score. Positive pushes toward the predicted class. |
| `relative_importance` | The atom's share of total attribution magnitude. |
| `unmapped_importance` | Share of attribution that mapped to no atom. `null` means the explainer did not report it; `0.0` means none. |
| `status` | `completed`, `partial`, or `failed`. |
| `method` | The attribution method and the token-to-atom alignment used. |

`null` and `0.0` are different facts and are reported differently.

## Wording that is refused

- "causes", "is responsible for", "the mechanism is", "proves"
- "the toxicophore is X"
- any statement that the figure shows a chemical interaction

## Wording that is correct

- "the score responded most strongly to ..."
- "these atoms contributed toward the predicted class; these contributed against it"
- "X of the attribution mass could not be mapped to a specific atom"
