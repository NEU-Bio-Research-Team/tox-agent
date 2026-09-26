# Grader reference — inv-01-terfenadine-named-go-nogo

For graders only. Never shown to any system. Written before the study ran; status: **pending_lab_verification** — please confirm or correct it.

Compound: terfenadine (PubChem CID 5405)

Terfenadine-containing products were removed from the US market in 1997-1998 because of QT prolongation and torsade de pointes, mostly when its presystemic metabolism was inhibited (for example by ketoconazole) or in overdose or liver disease. Its carboxylic-acid metabolite fexofenadine keeps the antihistamine effect without the QT effect.

Sources:
- Predicting drug-induced QT prolongation and torsades de pointes (review) — https://pmc.ncbi.nlm.nih.gov/articles/PMC4850203/

What a strong response would do:
- separates the model's hERG score from experimental and clinical evidence
- notices that exposure and metabolic interactions decide the clinical risk
- proposes a discriminating test (functional patch clamp and an exposure margin)
- gives no safety or regulatory verdict
