from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from xml.etree import ElementTree as ET

import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
except Exception as exc:  # pragma: no cover - runtime dependency check
    raise RuntimeError(
        "firebase-admin is required. Install dependencies first, for example: "
        "pip install firebase-admin rdkit-pypi sentence-transformers requests pandas"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV_SOURCES: Sequence[Tuple[Path, str]] = (
    (PROJECT_ROOT / "test_data" / "reference_panel.csv", "reference_panel"),
    (PROJECT_ROOT / "test_data" / "screening_library.csv", "screening_library"),
    (PROJECT_ROOT / "test_data" / "full_test_set.csv", "full_test_set"),
    (PROJECT_ROOT / "test_data" / "toxic_compounds.csv", "toxic_compounds"),
    (PROJECT_ROOT / "test_data" / "smiles_only.csv", "smiles_only"),
)

SMARTS_ALERT_MAP: Dict[str, str] = {
    "michael_acceptor": "[CH2]=[CH][C](=O)",
    "nitro_aromatic": "c1ccccc1[N+](=O)[O-]",
    "aniline": "c1ccccc1[NH2]",
    "aldehyde": "[CX3H1](=O)",
    "epoxide": "C1OC1",
    "quinone": "O=C1C=CC(=O)C=C1",
    "herg_pharmacophore": "[$(c1ccccc1)]CC[NH]",
    "acyl_halide": "[CX3](=O)[Cl,Br,I,F]",
}

PUBMED_QUERIES: Sequence[str] = (
    "cardiotoxicity hERG drug prediction machine learning",
    "hepatotoxicity DILI prediction deep learning SMILES",
    "structural alert toxicity SMARTS prediction",
    "molecular fingerprint toxicity classification GNN",
    "drug induced liver injury reactive metabolite",
    "cardiac arrhythmia QT prolongation drug",
    "ADMET toxicity prediction transformer",
)

TARGET_KEYWORDS: Dict[str, Sequence[str]] = {
    "hERG": ("herg", "kv11.1", "qt prolongation"),
    "Nav1.5": ("nav1.5", "sodium channel cardiac"),
    "CYP3A4": ("cyp3a4", "cytochrome p450 3a4"),
    "CYP2D6": ("cyp2d6",),
    "hERG/Nav1.5": ("cardiac ion channel",),
    "ALT/AST": ("liver enzym", "dili", "hepatotoxic"),
    "Mitochondria": ("mitochondri", "atp synthase"),
}

_CURATED_MECHANISMS: Sequence[Dict[str, Any]] = (
    {
        "doc_id": "mechanism:herg_inhibition",
        "type": "mechanism",
        "title": "hERG Channel Inhibition",
        "summary": (
            "hERG (Kv11.1) potassium channel blockade prolongs cardiac QT interval, "
            "with risk of Torsades de Pointes."
        ),
        "risk_level": "high",
        "clinical_manifestation": "QT prolongation, Torsades de Pointes, sudden cardiac death",
        "structural_alerts": ["basic nitrogen", "aromatic ring", "LogP > 3", "herg_pharmacophore"],
        "associated_scaffolds": ["piperidine", "phenothiazine", "quinolone", "terfenadine"],
        "key_refs": ["PMID:12345678", "ChEMBL_assay:CHEMBL829"],
    },
    {
        "doc_id": "mechanism:dili",
        "type": "mechanism",
        "title": "Drug-Induced Liver Injury (DILI)",
        "summary": (
            "Hepatocellular damage can emerge via reactive metabolite formation, "
            "mitochondrial dysfunction, or immune-mediated mechanisms."
        ),
        "risk_level": "high",
        "clinical_manifestation": "Elevated ALT/AST, fulminant hepatic failure",
        "structural_alerts": ["michael_acceptor", "nitro_aromatic", "quinone", "acyl_halide"],
        "associated_scaffolds": ["acetaminophen scaffold", "diclofenac scaffold"],
        "key_refs": ["PMID:24121004", "DILIrank_dataset"],
    },
    {
        "doc_id": "mechanism:reactive_metabolite",
        "type": "mechanism",
        "title": "Reactive Metabolite Formation",
        "summary": (
            "CYP450-mediated bioactivation can generate electrophilic intermediates that "
            "form covalent protein adducts."
        ),
        "risk_level": "high",
        "clinical_manifestation": "Idiosyncratic drug reactions, haptenization, immune activation",
        "structural_alerts": ["michael_acceptor", "epoxide", "quinone", "aldehyde"],
        "associated_scaffolds": ["furan", "thiophene", "aniline"],
        "key_refs": ["PMID:18351689"],
    },
    {
        "doc_id": "mechanism:mitochondrial_toxicity",
        "type": "mechanism",
        "title": "Mitochondrial Toxicity",
        "summary": (
            "Uncoupling oxidative phosphorylation or inhibiting ETC complexes can reduce ATP "
            "synthesis and induce cellular stress."
        ),
        "risk_level": "medium",
        "clinical_manifestation": "Lactic acidosis, myopathy, peripheral neuropathy",
        "structural_alerts": ["nitro_group", "biguanide"],
        "associated_scaffolds": ["rotenone scaffold", "biguanide"],
        "key_refs": ["PMID:15504116"],
    },
)

SMARTS_PATTERNS: Dict[str, Any] = {
    name: Chem.MolFromSmarts(smarts)
    for name, smarts in SMARTS_ALERT_MAP.items()
}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Seed MolRAG Firestore collections from local datasets and public APIs."
    )
    parser.add_argument(
        "--service-account",
        default=os.getenv("FIREBASE_SERVICE_ACCOUNT", "serviceAccountKey.json"),
        help=(
            "Path to Firebase service account JSON. If not found, script falls back to "
            "Application Default Credentials (ADC)."
        ),
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=("compounds", "knowledge", "literature", "all"),
        default=("all",),
        help="Seed one or more phases. Default: all.",
    )
    parser.add_argument(
        "--database-id",
        default=os.getenv("FIRESTORE_DATABASE_ID", "(default)"),
        help="Firestore database id to seed. Example: (default) or tox-agent-db.",
    )
    parser.add_argument(
        "--pubmed-max-results",
        type=int,
        default=15,
        help="Max PubMed articles per query for literature phase.",
    )
    parser.add_argument(
        "--pubchem-sleep",
        type=float,
        default=0.2,
        help="Delay between PubChem requests in seconds.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print progress without writing to Firestore.",
    )
    return parser


def _resolve_phases(raw_phases: Sequence[str]) -> List[str]:
    if any(phase == "all" for phase in raw_phases):
        return ["compounds", "knowledge", "literature"]
    deduped: List[str] = []
    for phase in raw_phases:
        if phase not in deduped:
            deduped.append(phase)
    return deduped


def _init_firestore(service_account_path: str, database_id: str = "(default)"):
    app = firebase_admin.get_app() if firebase_admin._apps else None

    normalized_database_id = _clean_text(database_id) or "(default)"

    def _client_for_database():
        if normalized_database_id and normalized_database_id != "(default)":
            try:
                return firestore.client(database_id=normalized_database_id)
            except TypeError:
                return firestore.client()
        return firestore.client()

    if app is not None:
        return _client_for_database()

    candidate = Path(service_account_path).expanduser()
    if candidate.exists():
        cred = credentials.Certificate(str(candidate))
        firebase_admin.initialize_app(cred)
        print(f"[setup] Firebase initialized with service account: {candidate}")
        return _client_for_database()

    try:
        firebase_admin.initialize_app()
        print("[setup] Firebase initialized with ADC (no service account file found).")
        return _client_for_database()
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize Firebase Admin SDK. Provide --service-account with a valid JSON "
            "or configure Application Default Credentials."
        ) from exc


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def _normalize_label_int(raw_value: Any) -> Optional[int]:
    text = _clean_text(raw_value).lower()
    if text == "":
        return None

    if text in {"toxic", "true", "positive", "high", "1", "1.0"}:
        return 1
    if text in {"non-toxic", "nontoxic", "safe", "false", "negative", "low", "0", "0.0"}:
        return 0

    try:
        return 1 if float(text) >= 0.5 else 0
    except Exception:
        return None


def detect_tox_class(smiles: str, label_int: Optional[int]) -> List[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or label_int == 0:
        return []

    hits: List[str] = []
    for name, pattern in SMARTS_PATTERNS.items():
        if pattern is not None and mol.HasSubstructMatch(pattern):
            hits.append(name)

    if label_int == 1 and not hits:
        return ["general_toxicity"]
    return hits


def pubchem_enrich(smiles: str) -> Dict[str, str]:
    try:
        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/"
            f"{requests.utils.quote(smiles)}/property/IUPACName,MolecularFormula,InChIKey/JSON"
        )
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return {}

        props = response.json().get("PropertyTable", {}).get("Properties", [])
        if not props:
            return {}

        first = props[0]
        return {
            "iupac_name": _clean_text(first.get("IUPACName")),
            "formula": _clean_text(first.get("MolecularFormula")),
            "inchikey": _clean_text(first.get("InChIKey")),
        }
    except Exception:
        return {}


def _infer_tox_class_from_name(name: str) -> List[str]:
    lowered = _clean_text(name).lower()
    mapping: Dict[str, Sequence[str]] = {
        "herg": ("herg_inhibitor",),
        "liver": ("hepatotoxic",),
        "hepat": ("hepatotoxic",),
        "nitro": ("reactive_metabolite", "nitro_reduction"),
        "michael": ("reactive_metabolite", "protein_adduct"),
        "quinone": ("reactive_metabolite", "oxidative_stress"),
        "epoxide": ("reactive_metabolite",),
        "mutagenic": ("genotoxic",),
        "ames": ("genotoxic",),
    }

    classes: List[str] = []
    for keyword, values in mapping.items():
        if keyword in lowered:
            classes.extend(values)

    deduped = sorted(set(classes))
    return deduped if deduped else ["general_alert"]


def _extract_targets(text: str) -> List[str]:
    lowered = _clean_text(text).lower()
    found: List[str] = []
    for target, keywords in TARGET_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            found.append(target)
    return found


def seed_compounds(db, *, dry_run: bool = False, pubchem_sleep: float = 0.2) -> int:
    collection = db.collection("molrag_compounds") if not dry_run else None
    seen_inchikeys: Set[str] = set()
    written = 0

    for csv_path, source in CSV_SOURCES:
        if not csv_path.exists():
            print(f"[compounds] skip missing file: {csv_path}")
            continue

        dataframe = pd.read_csv(csv_path)
        dataframe.columns = [str(column).strip().lower() for column in dataframe.columns]
        smiles_col = next((column for column in dataframe.columns if "smiles" in column), None)
        label_col = next((column for column in dataframe.columns if "label" in column or "toxic" in column), None)
        name_col = next((column for column in dataframe.columns if "name" in column), None)
        notes_col = next((column for column in dataframe.columns if "note" in column), None)

        if smiles_col is None:
            print(f"[compounds] skip {csv_path.name}: no smiles column")
            continue

        for _, row in dataframe.iterrows():
            smiles = _clean_text(row.get(smiles_col))
            if not smiles:
                continue

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            canonical_smiles = Chem.MolToSmiles(mol)
            inchi_value = Chem.MolToInchi(mol)
            inchikey = _clean_text(Chem.InchiToInchiKey(inchi_value) if inchi_value else "")
            if not inchikey:
                continue
            if inchikey in seen_inchikeys:
                continue
            seen_inchikeys.add(inchikey)

            label_int = _normalize_label_int(row.get(label_col) if label_col else None)
            label_text = "Unknown"
            if label_int == 1:
                label_text = "Toxic"
            elif label_int == 0:
                label_text = "Non-toxic"

            enriched = pubchem_enrich(smiles)
            if pubchem_sleep > 0:
                time.sleep(pubchem_sleep)

            common_name = _clean_text(row.get(name_col) if name_col else "") or enriched.get("iupac_name") or canonical_smiles[:40]
            notes = _clean_text(row.get(notes_col) if notes_col else "")
            formula = rdMolDescriptors.CalcMolFormula(mol)
            if not formula:
                formula = enriched.get("formula", "")

            payload = {
                "compound_id": inchikey,
                "common_name": common_name,
                "iupac_name": enriched.get("iupac_name", ""),
                "smiles": smiles,
                "canonical_smiles": canonical_smiles,
                "label": label_text,
                "tox_class": detect_tox_class(smiles, label_int),
                "source_dataset": source,
                "notes": notes,
                "molecular_weight": round(Descriptors.MolWt(mol), 2),
                "formula": formula,
                "created_at": firestore.SERVER_TIMESTAMP if not dry_run else None,
            }

            if not dry_run and collection is not None:
                collection.document(inchikey).set(payload)
            written += 1
            print(f"[compounds] {'(dry-run) ' if dry_run else ''}seeded: {common_name} ({source})")

    print(f"[compounds] total written: {written}")
    return written


def _build_chembl_next_url(next_path: str) -> str:
    if next_path.startswith("http://") or next_path.startswith("https://"):
        return next_path
    return f"https://www.ebi.ac.uk{next_path}"


def _iter_chembl_pages(*, start_url: str, response_key: str) -> Iterable[List[Dict[str, Any]]]:
    url = start_url
    while url:
        response = requests.get(url, timeout=20)
        if response.status_code == 404:
            raise requests.HTTPError("Endpoint not found", response=response)
        response.raise_for_status()
        data = response.json()

        records = data.get(response_key, [])
        if isinstance(records, list):
            yield records

        next_path = data.get("page_meta", {}).get("next")
        if next_path:
            url = _build_chembl_next_url(str(next_path))
            time.sleep(0.25)
        else:
            url = ""


def _seed_structural_alerts_from_chembl(collection, *, dry_run: bool) -> int:
    endpoint_candidates: Sequence[Tuple[str, str]] = (
        ("https://www.ebi.ac.uk/chembl/api/data/structuralalert.json?limit=200&offset=0", "objects"),
        ("https://www.ebi.ac.uk/chembl/api/data/structural_alert.json?limit=200&offset=0", "structural_alerts"),
    )

    for start_url, response_key in endpoint_candidates:
        try:
            written = 0
            for records in _iter_chembl_pages(start_url=start_url, response_key=response_key):
                for alert in records:
                    alert_name = _clean_text(alert.get("alert_name") or alert.get("name"))
                    alertset_name = _clean_text(alert.get("alertset_name") or alert.get("alert_set"))
                    alert_id = _clean_text(alert.get("alert_id") or alert.get("id"))
                    if not alert_id:
                        continue

                    normalized_alertset = alertset_name.lower().replace(" ", "_") or "unknown_alertset"
                    doc_id = f"alert:{normalized_alertset}_{alert_id}"

                    payload = {
                        "doc_id": doc_id,
                        "type": "structural_alert",
                        "name": alert_name,
                        "smarts": _clean_text(alert.get("smarts")),
                        "severity": "medium",
                        "alertset": alertset_name,
                        "tox_class": _infer_tox_class_from_name(alert_name),
                        "mechanism_ref": "",
                        "examples": [],
                    }

                    if not dry_run and collection is not None:
                        collection.document(doc_id).set(payload)
                    written += 1

            if written > 0:
                print(f"[knowledge] structural alerts source: {start_url}")
                return written
        except requests.HTTPError as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code == 404:
                continue
            print(f"[knowledge] structural alert fetch error: {type(exc).__name__}: {str(exc)[:180]}")
            return 0
        except Exception as exc:
            print(f"[knowledge] structural alert parse error: {type(exc).__name__}: {str(exc)[:180]}")
            return 0

    print("[knowledge] structural alert endpoint unavailable; fallback to drug_warning only.")
    return 0


def _warning_severity_from_type(warning_type: str) -> str:
    lowered = _clean_text(warning_type).lower()
    if "black box" in lowered:
        return "high"
    if "withdraw" in lowered or "contra" in lowered:
        return "high"
    if "boxed" in lowered:
        return "high"
    if "warning" in lowered:
        return "medium"
    return "low"


def _seed_drug_warnings_from_chembl(collection, *, dry_run: bool) -> int:
    written = 0
    start_url = "https://www.ebi.ac.uk/chembl/api/data/drug_warning.json?limit=200&offset=0"

    try:
        for records in _iter_chembl_pages(start_url=start_url, response_key="drug_warnings"):
            for warning in records:
                warning_id = _clean_text(warning.get("warning_id"))
                if not warning_id:
                    continue

                warning_class = _clean_text(warning.get("warning_class"))
                warning_type = _clean_text(warning.get("warning_type"))
                warning_desc = _clean_text(warning.get("warning_description"))
                warning_country = _clean_text(warning.get("warning_country"))
                warning_year = warning.get("warning_year")
                molecule_chembl_id = _clean_text(warning.get("molecule_chembl_id"))
                parent_molecule_chembl_id = _clean_text(warning.get("parent_molecule_chembl_id"))

                refs = warning.get("warning_refs", [])
                ref_urls: List[str] = []
                if isinstance(refs, list):
                    for item in refs:
                        if not isinstance(item, dict):
                            continue
                        ref_url = _clean_text(item.get("ref_url"))
                        if ref_url:
                            ref_urls.append(ref_url)
                ref_urls = ref_urls[:3]

                name = warning_class or warning_type or "drug_warning"
                summary_parts = [
                    f"Type: {warning_type}" if warning_type else "",
                    f"Class: {warning_class}" if warning_class else "",
                    f"Country: {warning_country}" if warning_country else "",
                    f"Description: {warning_desc}" if warning_desc else "",
                ]
                summary = " | ".join(part for part in summary_parts if part)
                if not summary:
                    summary = "Post-marketing safety warning reported in ChEMBL."

                tox_infer_text = " ".join(
                    value for value in (warning_class, warning_type, warning_desc) if value
                )

                doc_id = f"warning:{warning_id}"
                payload = {
                    "doc_id": doc_id,
                    "type": "drug_warning",
                    "name": name,
                    "summary": summary,
                    "severity": _warning_severity_from_type(warning_type),
                    "tox_class": _infer_tox_class_from_name(tox_infer_text),
                    "warning_type": warning_type,
                    "warning_class": warning_class,
                    "warning_country": warning_country,
                    "warning_year": warning_year,
                    "molecule_chembl_id": molecule_chembl_id,
                    "parent_molecule_chembl_id": parent_molecule_chembl_id,
                    "references": ref_urls,
                }

                if not dry_run and collection is not None:
                    collection.document(doc_id).set(payload)
                written += 1
    except Exception as exc:
        print(f"[knowledge] drug_warning fetch error: {type(exc).__name__}: {str(exc)[:180]}")
        return written

    print("[knowledge] drug warnings source: https://www.ebi.ac.uk/chembl/api/data/drug_warning.json")
    return written


def seed_knowledge(db, *, dry_run: bool = False) -> int:
    collection = db.collection("molrag_knowledge") if not dry_run else None
    written = 0

    written += _seed_structural_alerts_from_chembl(collection, dry_run=dry_run)
    written += _seed_drug_warnings_from_chembl(collection, dry_run=dry_run)

    for mechanism in _CURATED_MECHANISMS:
        if not dry_run and collection is not None:
            collection.document(str(mechanism["doc_id"])).set(dict(mechanism))
        written += 1
        print(f"[knowledge] {'(dry-run) ' if dry_run else ''}seeded: {mechanism['doc_id']}")

    print(f"[knowledge] total written: {written}")
    return written


def fetch_pubmed_abstracts(query: str, *, max_results: int = 15) -> List[Dict[str, Any]]:
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    search_response = requests.get(
        f"{base_url}/esearch.fcgi",
        params={"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json"},
        timeout=12,
    )
    search_response.raise_for_status()
    pmids = search_response.json().get("esearchresult", {}).get("idlist", [])
    if not pmids:
        return []

    time.sleep(0.4)

    fetch_response = requests.post(
        f"{base_url}/efetch.fcgi",
        data={"db": "pubmed", "id": ",".join(pmids), "rettype": "abstract", "retmode": "xml"},
        timeout=20,
    )
    fetch_response.raise_for_status()

    root = ET.fromstring(fetch_response.text)
    articles: List[Dict[str, Any]] = []

    for article in root.findall(".//PubmedArticle"):
        try:
            pmid = _clean_text(article.findtext(".//PMID"))
            title = _clean_text(article.findtext(".//ArticleTitle"))
            abstract_parts = article.findall(".//AbstractText")
            abstract = " ".join(_clean_text(part.text) for part in abstract_parts if _clean_text(part.text))
            if not pmid or not abstract:
                continue

            year_raw = _clean_text(article.findtext(".//PubDate/Year"))
            year = int(year_raw) if year_raw.isdigit() else 2020
            chemicals = [
                _clean_text(item.findtext("NameOfSubstance"))
                for item in article.findall(".//Chemical")
            ]
            chemicals = [name for name in chemicals if name][:5]

            articles.append(
                {
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract,
                    "year": year,
                    "chemicals": chemicals,
                }
            )
        except Exception:
            continue

    return articles


def _load_embedding_model():
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # pragma: no cover - runtime dependency check
        raise RuntimeError(
            "sentence-transformers is required for the literature phase. "
            "Install with: pip install sentence-transformers"
        ) from exc

    return SentenceTransformer("allenai/scibert_scivocab_uncased")


def seed_literature(db, *, max_results: int = 15, dry_run: bool = False) -> int:
    collection = db.collection("molrag_literature") if not dry_run else None
    embed_model = _load_embedding_model()

    seen_pmids: Set[str] = set()
    written = 0

    for query in PUBMED_QUERIES:
        print(f"[literature] fetching query: {query}")
        try:
            articles = fetch_pubmed_abstracts(query, max_results=max_results)
        except Exception as exc:
            print(f"[literature] skip query due to fetch error: {type(exc).__name__}: {str(exc)[:180]}")
            continue

        for article in articles:
            pmid = article["pmid"]
            if pmid in seen_pmids:
                continue
            seen_pmids.add(pmid)

            text_to_embed = f"{article['title']}. {article['abstract'][:1000]}"
            embedding = embed_model.encode(text_to_embed).tolist()
            combined_text = f"{article['title']} {article['abstract']}"
            targets = _extract_targets(combined_text)

            payload = {
                "doc_id": f"pubmed:{pmid}",
                "title": article["title"],
                "abstract_chunk": article["abstract"][:2000],
                "year": article["year"],
                "embedding": embedding,
                "compound_mentions": article["chemicals"],
                "relevant_targets": targets,
                "source_query": query,
            }

            if not dry_run and collection is not None:
                collection.document(f"pubmed:{pmid}").set(payload)
            written += 1
            print(f"[literature] {'(dry-run) ' if dry_run else ''}seeded PMID:{pmid}")

        time.sleep(0.5)

    print(f"[literature] total written: {written}")
    return written


def _run_selected_phases(db, *, phases: Sequence[str], dry_run: bool, pubchem_sleep: float, pubmed_max_results: int) -> Dict[str, int]:
    results: Dict[str, int] = {}

    if "compounds" in phases:
        print("=== Phase 1: Seeding molrag_compounds ===")
        results["compounds"] = seed_compounds(db, dry_run=dry_run, pubchem_sleep=pubchem_sleep)

    if "knowledge" in phases:
        print("\n=== Phase 2: Seeding molrag_knowledge ===")
        results["knowledge"] = seed_knowledge(db, dry_run=dry_run)

    if "literature" in phases:
        print("\n=== Phase 3: Seeding molrag_literature ===")
        results["literature"] = seed_literature(db, max_results=pubmed_max_results, dry_run=dry_run)

    return results


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    phases = _resolve_phases(args.phases)
    db = _init_firestore(args.service_account, args.database_id)
    print(f"[setup] Firestore database id: {args.database_id}")
    summary = _run_selected_phases(
        db,
        phases=phases,
        dry_run=bool(args.dry_run),
        pubchem_sleep=float(args.pubchem_sleep),
        pubmed_max_results=int(args.pubmed_max_results),
    )

    print("\n=== Seeding Summary ===")
    for phase in phases:
        print(f"- {phase}: {summary.get(phase, 0)} docs")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
