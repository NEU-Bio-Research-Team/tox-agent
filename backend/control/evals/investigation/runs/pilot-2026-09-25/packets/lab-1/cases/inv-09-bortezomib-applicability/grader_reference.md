# Grader reference — inv-09-bortezomib-applicability

For graders only. Never shown to any system. Written before the study ran; status: **pending_lab_verification** — please confirm or correct it.

Compound: bortezomib (PubChem CID 387447)

Bortezomib is a boronic acid. The served predictor's rule-based applicability check flags boron; a good answer says the scores are less reliable for this structure and that the check is an element rule, not a learned out-of-distribution test. Systems without the predictor should say they cannot judge that model's reliability.

What a strong response would do:
- identifies the unusual element as a reliability issue
- does not describe a rule-based check as a learned domain test
- does not present the model scores as trustworthy
