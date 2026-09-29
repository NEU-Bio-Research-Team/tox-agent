# Scope comparison checklist

Use the rows that the sources can answer; leave the rest as unknown rather than
guessing. This page is background for your reasoning, not a source: do not cite
it, and do not write its numbers into an answer — an answer states only numbers
a tool handed you.

| Dimension | Ask | Typical reason two sources disagree |
|---|---|---|
| Compound | Same parent, salt, prodrug, metabolite, enantiomer? | An analogue or metabolite result reported as the compound's |
| Endpoint | Binding, ion flux, or current? Reporter activity or cytotoxicity? | Different measurements of related biology |
| System | Species, cell line, primary cells, tissue, whole animal? | Species or expression-system differences |
| Conditions | Temperature, voltage protocol, serum protein, incubation time | Protocol dependence of potency |
| Exposure | Concentrations tested vs concentration reached (free Cmax) | Potent in vitro, never reached in vivo — or the reverse |
| Measure | IC50 / EC50, % effect at one concentration, active/inactive call | Values that are not on the same scale |
| Evidence type | Primary experiment, review, label, prediction | A review restating one primary study counts once |

## hERG assay types and what each measures

| Assay | Measures | Reading |
|---|---|---|
| Manual or automated patch clamp | hERG (IKr) current directly | The usual functional reference for an IC50 |
| Radioligand binding displacement (e.g. dofetilide) | Binding to the channel | Binding without functional block is possible; not a current measurement |
| Thallium-flux (fluorescence) | Ion flux through the channel | Functional but indirect; potency can differ from patch clamp |
| In vivo / clinical QT | Integrated repolarisation effect | Depends on exposure, other channels and physiology |

## Potency and exposure

The usual frame for hERG is the margin between the IC50 and the free
(unbound) plasma Cmax at the intended dose. Redfern et al. (2003) observed that
drugs associated with torsade de pointes mostly had hERG IC50 values close to
free therapeutic concentrations, and proposed a *provisional* 30-fold margin
([Cardiovasc Res 58(1):32–45](https://academic.oup.com/cardiovascres/article/58/1/32/295425)).
The margin a programme accepts is its own decision and has been revisited since
([2020 review](https://www.sciencedirect.com/science/article/abs/pii/S105687192030229X));
never state a margin as a safety verdict.

## Tox21

Tox21 assays are quantitative high-throughput in-vitro screens. An active call
is a concentration-response readout in one assay system; cytotoxicity and
assay interference (for example autofluorescence) can produce actives that do
not reflect the target biology. Assays are independent: a count of active
assays is not a severity.
