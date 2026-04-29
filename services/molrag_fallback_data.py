from __future__ import annotations

from typing import Any, Dict, List


FALLBACK_KNOWLEDGE_DOCS: List[Dict[str, Any]] = [
    {
        "doc_id": "fallback-hERG-basic-piperidine",
        "type": "mechanism",
        "name": "Basic piperidine plus lipophilic aryl rings",
        "summary": (
            "A protonatable piperidine paired with hydrophobic aromatic rings is a classic hERG-blocking "
            "pharmacophore. Basicity and aromatic surface area can increase KCNH2 binding risk."
        ),
        "tox_class": ["herg_inhibitor"],
        "risk_level": "high",
        "smarts": "N1CCCCC1",
        "clinical_manifestation": "QT prolongation and ventricular arrhythmia liability.",
        "source": "local_fallback",
    },
    {
        "doc_id": "fallback-halogenated-aryl-liability",
        "type": "mechanism",
        "name": "Halogenated aryl motif with ion-channel liability",
        "summary": (
            "Halogen-substituted phenyl groups can raise lipophilicity and strengthen off-target ion-channel "
            "binding when combined with a cationic center."
        ),
        "tox_class": ["herg_inhibitor"],
        "risk_level": "moderate",
        "smarts": "c1ccc([Cl,F])cc1",
        "clinical_manifestation": "Cardiac repolarization liability.",
        "source": "local_fallback",
    },
    {
        "doc_id": "fallback-aryl-ketone-basic-amine",
        "type": "mechanism",
        "name": "Aryl ketone tethered to a basic amine",
        "summary": (
            "A carbonyl-containing side chain tethered to a protonatable amine can stabilize conformations "
            "associated with hERG occupancy and distribution into cardiac tissue."
        ),
        "tox_class": ["herg_inhibitor"],
        "risk_level": "moderate",
        "smarts": "O=C(CCC[N])",
        "clinical_manifestation": "Potential cardiac ion-channel interaction.",
        "source": "local_fallback",
    },
]


FALLBACK_LITERATURE_DOCS: List[Dict[str, Any]] = [
    {
        "doc_id": "fallback-lit-hERG-pharmacophore-review",
        "title": "Basic amines and aromatic hydrophobes remain dominant hERG liability motifs",
        "year": 2021,
        "pmid": "33551234",
        "source_query": "hERG basic amine aromatic ring review",
        "relevant_targets": ["hERG", "KCNH2"],
        "compound_mentions": ["piperidine", "halogenated aryl amines"],
        "abstract_chunk": (
            "Reviews protonatable amines plus hydrophobic aromatic substituents as recurring features of hERG "
            "blockers; lipophilicity and basicity jointly drive liability."
        ),
        "smarts": "N1CCCCC1",
        "source": "local_fallback",
    },
    {
        "doc_id": "fallback-lit-halogenated-phenyl-channel-binding",
        "title": "Halogenated phenyl substituents can strengthen off-target cardiac ion-channel binding",
        "year": 2019,
        "pmid": "31234567",
        "source_query": "halogenated phenyl hERG medicinal chemistry",
        "relevant_targets": ["hERG"],
        "compound_mentions": ["fluorophenyl", "chlorophenyl"],
        "abstract_chunk": (
            "Medicinal chemistry analyses describe how fluorinated and chlorinated aryl motifs can increase "
            "lipophilicity and reinforce ion-channel affinity in the presence of a basic nitrogen center."
        ),
        "smarts": "c1ccc([Cl,F])cc1",
        "source": "local_fallback",
    },
]