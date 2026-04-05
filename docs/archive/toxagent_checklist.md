# ToxAgent — Checklist & Đặc Tả Implementation
**Google ADK Multi-Agent System | GDGoC Hackathon Vietnam 2026**
**Cập nhật:** 31/03/2026 | **Kiến trúc:** FastAPI model_server → Tool Layer → Agent Layer

---

## Mục lục

1. [Tổng quan kiến trúc](#0-tổng-quan-kiến-trúc)
2. [Phase 2 — Tool Layer Checklist](#phase-2--tool-layer-checklist)
3. [Phase 3+4 — Agent Layer Checklist](#phase-34--agent-layer-checklist)
4. [Sơ đồ Data Flow (ASCII)](#sơ-đồ-data-flow-ascii)
5. [Thứ tự Implementation (Priority Queue)](#thứ-tự-implementation-priority-queue)

---

## 0. Tổng quan kiến trúc

```
model_server (FastAPI) ← đã hoàn thành
    └── /analyze        → trả về AnalyzeResponse (clinical + mechanism + explanation + final_verdict)
    └── /predict        → ClinicalToxicityOutput + MechanismToxicityOutput
    └── /predict/batch  → list[AnalyzeResponse]
    └── /explain        → ToxicityExplanationOutput
    └── /health         → status

Tool Layer (tools/)     ← Phase 2 (2 file)
    └── tools/tox_tools.py       → wrap /analyze, /predict/batch, /health
    └── tools/research_tools.py  → PubChem API + PubMed E-utilities

Agent Layer (agents/)   ← Phase 3+4 (5 agent)
    └── agents/screening_agent.py
    └── agents/researcher_agent.py
    └── agents/writer_agent.py
    └── agents/orchestrator_agent.py
    (ExplainerAgent → gộp vào ScreeningAgent, xem lý do ở §3.1)
```

> **Lý do chỉ cần 2 file tool:**  
> Endpoint `/analyze` đã trả về `ClinicalToxicityOutput + MechanismToxicityOutput + ToxicityExplanationOutput` trong một lần gọi duy nhất. Không cần tool riêng cho từng sub-model. Tool layer chỉ cần: (1) wrap `/analyze` + batch + health, và (2) gọi PubChem/PubMed cho context y văn.

---

## Phase 2 — Tool Layer Checklist

### Cấu trúc file

```
tools/
├── __init__.py
├── tox_tools.py          # Wrap model_server endpoints
└── research_tools.py     # PubChem + PubMed
```

---

### File 1: `tools/tox_tools.py`

**Phụ thuộc:**
```python
import httpx
import os
from rdkit import Chem  # pip install rdkit
from typing import Optional

MODEL_SERVER_URL = os.getenv("MODEL_SERVER_URL", "http://localhost:8000")
```

---

#### Hàm 1: `validate_smiles`

```python
def validate_smiles(smiles: str) -> dict:
    """
    Kiểm tra tính hợp lệ của chuỗi SMILES bằng RDKit cục bộ (không gọi API).

    Dùng khi: Agent cần xác nhận input trước khi gửi lên model_server.
    Không dùng thay thế cho /analyze — chỉ là pre-check nhanh.

    Args:
        smiles (str): Chuỗi SMILES cần kiểm tra, ví dụ "CC(=O)Oc1ccccc1C(=O)O".

    Returns:
        dict với các key:
            - valid (bool): True nếu SMILES hợp lệ về mặt cú pháp.
            - canonical_smiles (str | None): Dạng canonical nếu hợp lệ, None nếu không.
            - error (str | None): Thông báo lỗi nếu không hợp lệ.
            - atom_count (int | None): Số nguyên tử nếu hợp lệ.
    """
```

**Chi tiết implementation:**
```python
def validate_smiles(smiles: str) -> dict:
    # Bước 1: Kiểm tra empty/None
    if not smiles or not smiles.strip():
        return {"valid": False, "canonical_smiles": None,
                "error": "SMILES rỗng hoặc null", "atom_count": None}
    # Bước 2: Parse với RDKit
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return {"valid": False, "canonical_smiles": None,
                "error": f"RDKit không thể parse SMILES: '{smiles}'", "atom_count": None}
    # Bước 3: Trả canonical form
    canonical = Chem.MolToSmiles(mol)
    return {
        "valid": True,
        "canonical_smiles": canonical,
        "error": None,
        "atom_count": mol.GetNumAtoms()
    }
```

| Thuộc tính | Giá trị |
|---|---|
| Endpoint wraps | Không — dùng RDKit local |
| ADK dispatch trigger | "kiểm tra SMILES", "SMILES có hợp lệ không", "validate cấu trúc" |
| Failure mode | Nếu RDKit không cài → `ImportError`; fallback: gọi PubChem `/compound/smiles/{smiles}/JSON` và kiểm tra HTTP 200 |

---

#### Hàm 2: `analyze_molecule`

```python
def analyze_molecule(smiles: str) -> dict:
    """
    Phân tích độc tính toàn diện của một phân tử từ chuỗi SMILES.

    Gọi endpoint /analyze trên model_server, trả về kết quả đầy đủ gồm:
    phân tích độc tính lâm sàng (GATv2 xSmiles), cơ chế độc tính (Tox21 GATv2),
    giải thích cấu trúc (GNNExplainer), và kết luận tổng hợp cuối cùng.

    Dùng khi: Cần phân tích một phân tử đơn lẻ theo yêu cầu người dùng.
    Không dùng cho batch (>1 phân tử) — dùng analyze_molecules_batch thay thế.

    Args:
        smiles (str): Chuỗi SMILES hợp lệ của phân tử cần phân tích.
                      Nên validate trước bằng validate_smiles().

    Returns:
        dict với các key:
            - smiles (str): SMILES gốc từ input.
            - canonical_smiles (str): Dạng canonical do model_server chuẩn hóa.
            - clinical (dict): {p_toxic, label, confidence} từ GATv2 xSmiles.
            - mechanism (dict): {task_scores (12 Tox21 tasks), active_tasks,
                                  highest_risk_task, assay_hits} từ Tox21 GATv2.
            - explanation (dict): {top_atoms (list[10]), top_bonds (list[10]),
                                    heatmap_base64 (str)} từ GNNExplainer.
            - final_verdict (str): Kết luận tổng hợp clinical + mechanism.
            - error (str | None): Thông báo lỗi nếu gọi thất bại.
    """
```

**Chi tiết implementation:**
```python
def analyze_molecule(smiles: str) -> dict:
    try:
        response = httpx.post(
            f"{MODEL_SERVER_URL}/analyze",
            json={"smiles": smiles},
            timeout=30.0
        )
        response.raise_for_status()
        data = response.json()
        data["error"] = None
        return data
    except httpx.TimeoutException:
        return {"error": f"model_server timeout sau 30s cho SMILES: {smiles}",
                "smiles": smiles, "final_verdict": "PHÂN TÍCH THẤT BẠI — timeout"}
    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP {e.response.status_code}: {e.response.text}",
                "smiles": smiles, "final_verdict": "PHÂN TÍCH THẤT BẠI"}
    except Exception as e:
        return {"error": str(e), "smiles": smiles, "final_verdict": "PHÂN TÍCH THẤT BẠI"}
```

| Thuộc tính | Giá trị |
|---|---|
| Endpoint wraps | `POST /analyze` |
| ADK dispatch trigger | "phân tích độc tính", "analyze molecule", "xem kết quả độc tính" |
| Timeout | 30 giây |
| Failure mode | Trả về dict với `error` key; agent đọc và escalate |

---

#### Hàm 3: `analyze_molecules_batch`

```python
def analyze_molecules_batch(smiles_list: list[str]) -> dict:
    """
    Phân tích độc tính song song cho nhiều phân tử (batch mode).

    Gọi endpoint /predict/batch trên model_server. Phù hợp khi người dùng
    cung cấp danh sách SMILES hoặc khi cần so sánh nhiều hợp chất cùng lúc.
    Giới hạn: tối đa 50 phân tử mỗi lần gọi.

    Args:
        smiles_list (list[str]): Danh sách chuỗi SMILES cần phân tích.
                                  Tối đa 50 phần tử.

    Returns:
        dict với các key:
            - results (list[dict]): Mỗi phần tử là AnalyzeResponse cho một SMILES,
                                    có cùng cấu trúc như analyze_molecule().
            - total (int): Số phân tử đã gửi.
            - success_count (int): Số phân tử phân tích thành công.
            - error (str | None): Lỗi cấp batch nếu toàn bộ request thất bại.
    """
```

**Chi tiết implementation:**
```python
def analyze_molecules_batch(smiles_list: list[str]) -> dict:
    if len(smiles_list) > 50:
        return {"error": "Batch tối đa 50 phân tử. Hãy chia nhỏ danh sách.",
                "results": [], "total": len(smiles_list), "success_count": 0}
    try:
        response = httpx.post(
            f"{MODEL_SERVER_URL}/predict/batch",
            json={"smiles_list": smiles_list},
            timeout=120.0  # Batch cần timeout dài hơn
        )
        response.raise_for_status()
        results = response.json()  # list[AnalyzeResponse]
        success_count = sum(1 for r in results if r.get("final_verdict") != "PHÂN TÍCH THẤT BẠI")
        return {"results": results, "total": len(smiles_list),
                "success_count": success_count, "error": None}
    except Exception as e:
        return {"error": str(e), "results": [], "total": len(smiles_list), "success_count": 0}
```

| Thuộc tính | Giá trị |
|---|---|
| Endpoint wraps | `POST /predict/batch` |
| ADK dispatch trigger | "phân tích nhiều phân tử", "batch analysis", "so sánh danh sách" |
| Timeout | 120 giây |

---

#### Hàm 4: `check_model_server_health`

```python
def check_model_server_health() -> dict:
    """
    Kiểm tra trạng thái hoạt động của model_server.

    Gọi endpoint /health để xác nhận server đang chạy trước khi thực hiện
    phân tích. Dùng trong OrchestratorAgent khi khởi động session mới.

    Returns:
        dict với các key:
            - healthy (bool): True nếu server đang hoạt động bình thường.
            - status (str): Thông báo trạng thái từ server.
            - latency_ms (float): Thời gian phản hồi tính bằng millisecond.
            - error (str | None): Mô tả lỗi nếu server không phản hồi.
    """
```

| Thuộc tính | Giá trị |
|---|---|
| Endpoint wraps | `GET /health` |
| ADK dispatch trigger | "server có hoạt động không", "kiểm tra health" |
| Timeout | 5 giây |

---

### File 2: `tools/research_tools.py`

**Phụ thuộc:**
```python
import httpx
import urllib.parse
from typing import Optional

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PUBMED_BASE  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_API_KEY = os.getenv("PUBMED_API_KEY", "")  # Optional — tăng rate limit
```

---

#### Hàm 5: `get_compound_info_pubchem`

```python
def get_compound_info_pubchem(smiles: str) -> dict:
    """
    Lấy thông tin hợp chất từ PubChem dựa trên chuỗi SMILES.

    Truy vấn PubChem PUG REST API để lấy tên IUPAC, tên thông thường,
    molecular formula, molecular weight, CID (PubChem Compound ID),
    và danh sách synonyms. Dùng để làm giàu bối cảnh cho báo cáo cuối.

    Args:
        smiles (str): Chuỗi SMILES hợp lệ của hợp chất cần tra cứu.

    Returns:
        dict với các key:
            - cid (int | None): PubChem Compound ID.
            - iupac_name (str | None): Tên IUPAC chính thức.
            - common_name (str | None): Tên thông thường (synonym đầu tiên).
            - molecular_formula (str | None): Công thức phân tử (vd: "C9H8O4").
            - molecular_weight (float | None): Khối lượng phân tử (g/mol).
            - synonyms (list[str]): Tối đa 5 tên gọi khác.
            - pubchem_url (str | None): URL trang PubChem của hợp chất.
            - error (str | None): Thông báo lỗi nếu không tìm thấy.
    """
```

**Chi tiết implementation:**
```python
def get_compound_info_pubchem(smiles: str) -> dict:
    encoded = urllib.parse.quote(smiles, safe="")
    try:
        # Bước 1: Lấy CID từ SMILES
        cid_resp = httpx.get(
            f"{PUBCHEM_BASE}/compound/smiles/{encoded}/cids/JSON",
            timeout=10.0
        )
        cid_resp.raise_for_status()
        cid = cid_resp.json()["IdentifierList"]["CID"][0]

        # Bước 2: Lấy properties
        props_resp = httpx.get(
            f"{PUBCHEM_BASE}/compound/cid/{cid}/property/"
            "IUPACName,MolecularFormula,MolecularWeight/JSON",
            timeout=10.0
        )
        props_resp.raise_for_status()
        props = props_resp.json()["PropertyTable"]["Properties"][0]

        # Bước 3: Lấy synonyms (tối đa 5)
        syn_resp = httpx.get(
            f"{PUBCHEM_BASE}/compound/cid/{cid}/synonyms/JSON",
            timeout=10.0
        )
        synonyms = []
        if syn_resp.status_code == 200:
            synonyms = syn_resp.json()["InformationList"]["Information"][0].get("Synonym", [])[:5]

        return {
            "cid": cid,
            "iupac_name": props.get("IUPACName"),
            "common_name": synonyms[0] if synonyms else None,
            "molecular_formula": props.get("MolecularFormula"),
            "molecular_weight": props.get("MolecularWeight"),
            "synonyms": synonyms,
            "pubchem_url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
            "error": None
        }
    except Exception as e:
        return {"cid": None, "iupac_name": None, "common_name": None,
                "molecular_formula": None, "molecular_weight": None,
                "synonyms": [], "pubchem_url": None, "error": str(e)}
```

| Thuộc tính | Giá trị |
|---|---|
| API | PubChem PUG REST |
| ADK dispatch trigger | "tên hợp chất là gì", "công thức phân tử", "CID PubChem", "tra cứu tên thuốc" |
| Rate limit | 5 req/s (không cần key); không cần authen |

---

#### Hàm 6: `search_toxicity_literature`

```python
def search_toxicity_literature(compound_name: str, max_results: int = 5) -> dict:
    """
    Tìm kiếm tài liệu khoa học về độc tính của hợp chất trên PubMed.

    Truy vấn PubMed E-utilities (esearch + efetch) để lấy danh sách bài báo
    liên quan đến độc tính của hợp chất. Trả về tiêu đề, tác giả, năm,
    journal, và abstract tóm tắt. Dùng cho ResearcherAgent.

    Ưu tiên query: "{compound_name} toxicity mechanism" cho các bài báo cơ chế,
    lọc thêm "[MeSH Terms]" nếu cần chính xác.

    Args:
        compound_name (str): Tên hợp chất (IUPAC hoặc tên thông thường),
                              ví dụ "aspirin" hoặc "acetylsalicylic acid".
        max_results (int): Số bài báo tối đa cần lấy. Mặc định 5, tối đa 10.

    Returns:
        dict với các key:
            - articles (list[dict]): Mỗi phần tử gồm:
                {pmid, title, authors (list), year, journal, abstract_snippet (150 ký tự)}
            - total_found (int): Tổng số bài báo PubMed tìm thấy cho query này.
            - query_used (str): Query thực tế đã gửi lên PubMed.
            - error (str | None): Thông báo lỗi nếu request thất bại.
    """
```

**Chi tiết implementation:**
```python
def search_toxicity_literature(compound_name: str, max_results: int = 5) -> dict:
    max_results = min(max_results, 10)  # Hard cap
    query = f"{compound_name} toxicity mechanism"
    encoded_query = urllib.parse.quote(query)
    api_key_param = f"&api_key={PUBMED_API_KEY}" if PUBMED_API_KEY else ""

    try:
        # Bước 1: esearch — lấy danh sách PMID
        search_resp = httpx.get(
            f"{PUBMED_BASE}/esearch.fcgi?db=pubmed&term={encoded_query}"
            f"&retmax={max_results}&retmode=json&sort=relevance{api_key_param}",
            timeout=15.0
        )
        search_resp.raise_for_status()
        search_data = search_resp.json()["esearchresult"]
        pmids = search_data.get("idlist", [])
        total_found = int(search_data.get("count", 0))

        if not pmids:
            return {"articles": [], "total_found": 0,
                    "query_used": query, "error": None}

        # Bước 2: efetch — lấy abstract cho từng PMID
        ids_str = ",".join(pmids)
        fetch_resp = httpx.get(
            f"{PUBMED_BASE}/efetch.fcgi?db=pubmed&id={ids_str}"
            f"&rettype=abstract&retmode=json{api_key_param}",
            timeout=20.0
        )
        # Parse XML/JSON efetch — giữ đơn giản: trả pmid list + titles từ esummary
        summary_resp = httpx.get(
            f"{PUBMED_BASE}/esummary.fcgi?db=pubmed&id={ids_str}&retmode=json{api_key_param}",
            timeout=15.0
        )
        summary_data = summary_resp.json().get("result", {})

        articles = []
        for pmid in pmids:
            art = summary_data.get(pmid, {})
            authors = [a.get("name", "") for a in art.get("authors", [])[:3]]
            articles.append({
                "pmid": pmid,
                "title": art.get("title", "N/A"),
                "authors": authors,
                "year": art.get("pubdate", "")[:4],
                "journal": art.get("source", "N/A"),
                "abstract_snippet": art.get("title", "")[:150],  # fallback đơn giản
                "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            })

        return {"articles": articles, "total_found": total_found,
                "query_used": query, "error": None}
    except Exception as e:
        return {"articles": [], "total_found": 0,
                "query_used": query, "error": str(e)}
```

| Thuộc tính | Giá trị |
|---|---|
| API | NCBI PubMed E-utilities (esearch + esummary) |
| ADK dispatch trigger | "tài liệu khoa học", "bài báo về độc tính", "nghiên cứu liên quan", "literature" |
| Rate limit | 3 req/s (không key), 10 req/s (có API key) |

---

#### Hàm 7: `get_pubchem_bioassay_data`

```python
def get_pubchem_bioassay_data(cid: int) -> dict:
    """
    Lấy dữ liệu bioassay độc tính từ PubChem cho một hợp chất.

    Truy vấn PubChem BioAssay API để lấy kết quả thử nghiệm sinh học
    liên quan đến độc tính (active/inactive trong các assay Tox21, AID).
    Bổ sung cho kết quả mechanism từ model_server bằng dữ liệu thực nghiệm.

    Args:
        cid (int): PubChem Compound ID. Lấy từ get_compound_info_pubchem().

    Returns:
        dict với các key:
            - cid (int): PubChem CID đầu vào.
            - active_assays (list[dict]): Các assay mà hợp chất có kết quả ACTIVE,
                                          mỗi phần tử: {aid, assay_name, activity_outcome}.
            - total_assays_tested (int): Tổng số assay đã thử nghiệm.
            - tox21_active_count (int): Số assay Tox21 có kết quả active.
            - error (str | None): Thông báo lỗi.
    """
```

| Thuộc tính | Giá trị |
|---|---|
| API | PubChem BioAssay REST (`/assay/activity`) |
| ADK dispatch trigger | "kết quả bioassay", "thử nghiệm sinh học", "Tox21 data từ PubChem" |
| Ưu tiên | MEDIUM — implement sau validate + analyze + literature |

---

### Checklist file `tools/__init__.py`

```python
# tools/__init__.py
from .tox_tools import (
    validate_smiles,
    analyze_molecule,
    analyze_molecules_batch,
    check_model_server_health,
)
from .research_tools import (
    get_compound_info_pubchem,
    search_toxicity_literature,
    get_pubchem_bioassay_data,
)

__all__ = [
    "validate_smiles",
    "analyze_molecule",
    "analyze_molecules_batch",
    "check_model_server_health",
    "get_compound_info_pubchem",
    "search_toxicity_literature",
    "get_pubchem_bioassay_data",
]
```

### Tóm tắt Tool Layer

| # | Hàm | File | Endpoint / API | Priority |
|---|---|---|---|---|
| 1 | `validate_smiles` | `tox_tools.py` | RDKit local | 🔴 MUST |
| 2 | `analyze_molecule` | `tox_tools.py` | `POST /analyze` | 🔴 MUST |
| 3 | `analyze_molecules_batch` | `tox_tools.py` | `POST /predict/batch` | 🟡 SHOULD |
| 4 | `check_model_server_health` | `tox_tools.py` | `GET /health` | 🟡 SHOULD |
| 5 | `get_compound_info_pubchem` | `research_tools.py` | PubChem PUG REST | 🔴 MUST |
| 6 | `search_toxicity_literature` | `research_tools.py` | PubMed E-utilities | 🔴 MUST |
| 7 | `get_pubchem_bioassay_data` | `research_tools.py` | PubChem BioAssay | 🟢 NICE |

---

## Phase 3+4 — Agent Layer Checklist

### 3.1 Lý do gộp ExplainerAgent vào ScreeningAgent

`/analyze` đã trả về `ToxicityExplanationOutput` (top_atoms, top_bonds, heatmap_base64) trong **cùng một API call** với clinical + mechanism. Tách thành 2 agent nghĩa là:
- Gọi API 2 lần (một lần cho ScreeningAgent, một lần cho ExplainerAgent) → lãng phí 1.5–3 giây
- Thêm overhead state management không cần thiết
- Không có giá trị phân tích tăng thêm (cả hai đều chỉ đọc dữ liệu từ response)

**Quyết định:** `ScreeningAgent` đọc toàn bộ `AnalyzeResponse` gồm cả `explanation`. Field `heatmap_base64` được lưu vào `session.state["heatmap_base64"]` để `WriterAgent` đính kèm vào báo cáo.

---

### Agent 1: ScreeningAgent

**File:** `agents/screening_agent.py`

```python
from google.adk.agents import LlmAgent
from tools import validate_smiles, analyze_molecule

screening_agent = LlmAgent(
    name="ScreeningAgent",
    model="gemini-2.0-flash",
    description="""Phân tích độc tính lâm sàng và cơ chế của một phân tử từ SMILES.
    Dùng khi cần phân loại độc tính, xem điểm Tox21, hoặc giải thích cấu trúc nguyên tử.""",
    instruction="""
    Bạn là chuyên gia phân tích độc tính phân tử. Nhiệm vụ:
    1. Gọi validate_smiles(smiles) để kiểm tra input. Nếu không hợp lệ, dừng và báo lỗi.
    2. Gọi analyze_molecule(smiles) để lấy kết quả đầy đủ.
    3. Lưu kết quả phân tích vào output_key.
    4. Nếu analyze_molecule trả về error, báo cáo lỗi và đề xuất SMILES thay thế.

    SMILES cần phân tích: {smiles_input}
    """,
    tools=[validate_smiles, analyze_molecule],
    output_key="screening_result",
)
```

#### Đặc tả chi tiết

| Thuộc tính | Giá trị |
|---|---|
| **File** | `agents/screening_agent.py` |
| **ADK name** | `"ScreeningAgent"` |
| **Model** | `gemini-2.0-flash` (nhanh, đủ cho tool calling) |
| **Role** | Phân tích độc tính đơn phân tử — validate SMILES → gọi /analyze → trả kết quả có cấu trúc |

**Input (đọc từ session state):**
```python
session.state["smiles_input"]   # str — SMILES do OrchestratorAgent set
```

**Tools available:**
- `validate_smiles(smiles: str) → dict`
- `analyze_molecule(smiles: str) → dict`

**Processing flow:**
```
1. Đọc session.state["smiles_input"]
2. Gọi validate_smiles(smiles_input)
   ├─ valid == False → set state["screening_error"] = error_msg; DỪNG
   └─ valid == True → tiếp tục
3. Gọi analyze_molecule(canonical_smiles)  ← dùng canonical từ bước 2
   ├─ error != None → set state["screening_error"] = error; DỪNG
   └─ error == None → tiếp tục
4. Đọc và diễn giải các trường:
   - clinical.p_toxic, clinical.label, clinical.confidence
   - mechanism.active_tasks, mechanism.highest_risk_task, mechanism.assay_hits
   - explanation.top_atoms, explanation.top_bonds, explanation.heatmap_base64
   - final_verdict
5. Tạo structured summary bằng ngôn ngữ tự nhiên
6. Ghi vào session.state["screening_result"] qua output_key
```

**Output (ghi vào session.state):**
```python
session.state["screening_result"] = """
{
  "summary": "Phân tử X có nguy cơ độc tính lâm sàng CAO (p_toxic=0.87, confidence=0.91)...",
  "clinical": {"p_toxic": 0.87, "label": "TOXIC", "confidence": 0.91},
  "mechanism": {
    "active_tasks": ["NR-AR", "SR-MMP"],
    "highest_risk_task": "NR-AR",
    "assay_hits": 3
  },
  "explanation": {
    "top_atoms": [...],
    "heatmap_base64": "data:image/png;base64,..."
  },
  "final_verdict": "TOXIC",
  "canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O"
}
"""
# Đồng thời WriterAgent có thể đọc qua {screening_result} placeholder
```

**Failure mode + Fallback:**

| Lỗi | Hành động |
|---|---|
| SMILES không hợp lệ | Set `state["screening_error"]`, gợi ý dùng công cụ chuyển đổi tên → SMILES |
| model_server timeout | Retry 1 lần với timeout 60s; nếu vẫn fail → set error, OrchestratorAgent quyết định |
| `/analyze` trả 5xx | Set `state["screening_error"]`, check health bằng `check_model_server_health()` |
| RDKit không cài | Fallback: gọi PubChem `/compound/smiles/{s}/cids/JSON` để validate |

---

### Agent 2: ResearcherAgent

**File:** `agents/researcher_agent.py`

```python
from google.adk.agents import LlmAgent
from tools import get_compound_info_pubchem, search_toxicity_literature, get_pubchem_bioassay_data

researcher_agent = LlmAgent(
    name="ResearcherAgent",
    model="gemini-1.5-pro",
    description="""Tra cứu thông tin khoa học về hợp chất từ PubChem và PubMed.
    Dùng khi cần tên IUPAC, dữ liệu bioassay, hoặc tài liệu về cơ chế độc tính.""",
    instruction="""
    Bạn là chuyên gia tra cứu tài liệu dược học. Nhiệm vụ:
    1. Lấy thông tin hợp chất từ PubChem dựa trên SMILES.
    2. Tìm tài liệu khoa học về độc tính trên PubMed.
    3. Kết hợp thông tin thành context phong phú cho WriterAgent.

    SMILES cần tra cứu: {smiles_input}
    """,
    tools=[get_compound_info_pubchem, search_toxicity_literature, get_pubchem_bioassay_data],
    output_key="research_result",
)
```

#### Đặc tả chi tiết

| Thuộc tính | Giá trị |
|---|---|
| **File** | `agents/researcher_agent.py` |
| **ADK name** | `"ResearcherAgent"` |
| **Model** | `gemini-1.5-pro` (cần reasoning tốt hơn để tổng hợp literature) |
| **Role** | Tra cứu PubChem + PubMed → trả context bối cảnh y văn cho WriterAgent |

**Input (đọc từ session state):**
```python
session.state["smiles_input"]     # str — SMILES từ OrchestratorAgent
# ResearcherAgent chạy SONG SONG với ScreeningAgent → không phụ thuộc nhau
```

**Tools available:**
- `get_compound_info_pubchem(smiles: str) → dict`
- `search_toxicity_literature(compound_name: str, max_results: int) → dict`
- `get_pubchem_bioassay_data(cid: int) → dict`

**Processing flow:**
```
1. Đọc session.state["smiles_input"]
2. Gọi get_compound_info_pubchem(smiles_input)
   ├─ error != None → ghi note "Không tìm thấy trên PubChem", tiếp tục
   └─ Lấy: cid, iupac_name, common_name, molecular_formula, molecular_weight
3. Xác định tên tốt nhất để search:
   ├─ Ưu tiên: common_name (vd: "aspirin")
   └─ Fallback: iupac_name (vd: "2-acetyloxybenzoic acid")
4. Gọi search_toxicity_literature(best_name, max_results=5)
   ├─ error != None → ghi note "Không tìm thấy tài liệu PubMed", tiếp tục
   └─ Lấy: articles (list[5]), total_found
5. Nếu cid không None: gọi get_pubchem_bioassay_data(cid)
   └─ Lấy: active_assays, tox21_active_count
6. Tổng hợp thành research_context có cấu trúc
7. Ghi vào session.state["research_result"] qua output_key
```

**Output (ghi vào session.state):**
```python
session.state["research_result"] = """
{
  "compound_info": {
    "cid": 2244,
    "iupac_name": "2-acetyloxybenzoic acid",
    "common_name": "Aspirin",
    "molecular_formula": "C9H8O4",
    "molecular_weight": 180.16,
    "pubchem_url": "https://pubchem.ncbi.nlm.nih.gov/compound/2244"
  },
  "literature": {
    "total_found": 1284,
    "articles": [
      {"pmid": "12345", "title": "...", "year": "2023", "journal": "Toxicology"},
      ...
    ]
  },
  "bioassay_summary": {
    "tox21_active_count": 2,
    "active_assays": [{"aid": "...", "assay_name": "..."}]
  }
}
"""
```

**Failure mode + Fallback:**

| Lỗi | Hành động |
|---|---|
| PubChem không tìm thấy SMILES | Set compound_info = null, dùng SMILES string làm tên search PubMed |
| PubMed rate limit (429) | Retry sau 1s x2; nếu fail → articles = [], ghi note trong output |
| CID null → không gọi bioassay | Skip `get_pubchem_bioassay_data`, để bioassay_summary = null |
| Toàn bộ research thất bại | Ghi `state["research_result"] = {"error": "...", "compound_info": null, "literature": null}` |
| WriterAgent nhận research = null | WriterAgent fallback: viết báo cáo chỉ dựa trên screening_result |

---

### Agent 3: WriterAgent

**File:** `agents/writer_agent.py`

```python
from google.adk.agents import LlmAgent

writer_agent = LlmAgent(
    name="WriterAgent",
    model="gemini-1.5-pro",
    description="""Tổng hợp kết quả phân tích độc tính và tài liệu khoa học thành báo cáo
    chuyên nghiệp bằng tiếng Việt. Dùng sau khi ScreeningAgent và ResearcherAgent hoàn thành.""",
    instruction="""
    Bạn là chuyên gia viết báo cáo phân tích độc tính dược phẩm.
    
    Dữ liệu đầu vào:
    - Kết quả phân tích mô hình: {screening_result}
    - Thông tin hợp chất & tài liệu: {research_result}
    - SMILES gốc: {smiles_input}
    
    Hãy tạo một báo cáo hoàn chỉnh theo đúng cấu trúc được yêu cầu.
    Nếu research_result không có dữ liệu, vẫn viết báo cáo dựa trên screening_result.
    Trả về JSON có cấu trúc như đã định nghĩa trong output schema.
    """,
    tools=[],  # WriterAgent không cần tool — chỉ tổng hợp dữ liệu từ state
    output_key="final_report",
)
```

#### Đặc tả chi tiết

| Thuộc tính | Giá trị |
|---|---|
| **File** | `agents/writer_agent.py` |
| **ADK name** | `"WriterAgent"` |
| **Model** | `gemini-1.5-pro` (cần chất lượng ngôn ngữ cao, reasoning tổng hợp) |
| **Role** | Đọc screening_result + research_result → sinh báo cáo Markdown/JSON cuối cùng bằng tiếng Việt |

**Input (đọc từ session state):**
```python
session.state["screening_result"]    # dict/str — từ ScreeningAgent
session.state["research_result"]     # dict/str — từ ResearcherAgent
session.state["smiles_input"]        # str — SMILES gốc
```

**Tools available:** Không có (pure synthesis agent)

**Processing flow:**
```
1. Đọc screening_result từ state
   ├─ Nếu có screening_error → báo cáo lỗi, DỪNG với error report
   └─ Parse: clinical, mechanism, explanation, final_verdict
2. Đọc research_result từ state
   ├─ Nếu null/error → đánh dấu "Không có dữ liệu tài liệu"
   └─ Parse: compound_info, literature.articles, bioassay_summary
3. Xác định mức độ rủi ro tổng hợp:
   - CRITICAL: p_toxic > 0.8 AND active_tasks >= 3
   - HIGH:     p_toxic > 0.6 OR active_tasks >= 2
   - MODERATE: p_toxic > 0.4 OR active_tasks >= 1
   - LOW:      p_toxic <= 0.4 AND active_tasks == 0
4. Viết từng section của báo cáo (xem Output schema)
5. Đính kèm heatmap_base64 vào section giải thích cấu trúc
6. Format output thành JSON có cấu trúc
7. Ghi vào session.state["final_report"] qua output_key
```

**Output Schema (ghi vào session.state["final_report"]):**
```json
{
  "report_metadata": {
    "smiles": "CC(=O)Oc1ccccc1C(=O)O",
    "canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O",
    "compound_name": "Aspirin",
    "analysis_timestamp": "2026-03-31T15:32:00+07:00",
    "report_version": "1.0"
  },
  "executive_summary": "Phân tử Aspirin (C9H8O4) được phân loại là...",
  "risk_level": "LOW | MODERATE | HIGH | CRITICAL",
  "sections": {
    "clinical_toxicity": {
      "verdict": "TOXIC | NON-TOXIC",
      "probability": 0.87,
      "confidence": 0.91,
      "interpretation": "Mô hình GATv2 xSmiles phân loại..."
    },
    "mechanism_toxicity": {
      "active_tox21_tasks": ["NR-AR", "SR-MMP"],
      "highest_risk": "NR-AR",
      "assay_hits": 3,
      "task_details": {
        "NR-AR": {"score": 0.72, "meaning": "Thụ thể androgen nhân"}
      }
    },
    "structural_explanation": {
      "key_atoms": "Nguyên tử C3, O7 đóng góp cao nhất vào dự đoán độc tính...",
      "key_bonds": "...",
      "heatmap_base64": "data:image/png;base64,..."
    },
    "literature_context": {
      "compound_id": {"cid": 2244, "pubchem_url": "..."},
      "relevant_papers": [
        {"title": "...", "year": "2023", "pmid": "12345", "url": "..."}
      ],
      "bioassay_evidence": "Tox21 xác nhận 2 assay active..."
    },
    "recommendations": [
      "Tiếp tục thử nghiệm in vitro cho cơ chế NR-AR",
      "Xem xét chỉnh sửa cấu trúc tại vị trí C3"
    ]
  }
}
```

**Failure mode + Fallback:**

| Lỗi | Hành động |
|---|---|
| `screening_result` null | Trả error report: "Không thể phân tích — ScreeningAgent thất bại" |
| `research_result` null | Viết báo cáo không có section literature_context; ghi note "Tra cứu tài liệu không khả dụng" |
| LLM output không valid JSON | OrchestratorAgent gọi lại WriterAgent với explicit JSON formatting instruction |
| heatmap_base64 quá lớn (>1MB) | Strip heatmap, ghi note "Heatmap available via /explain endpoint" |

---

### Agent 4: OrchestratorAgent

**File:** `agents/orchestrator_agent.py`

```python
from google.adk.agents import SequentialAgent, ParallelAgent, LlmAgent
from tools import check_model_server_health, validate_smiles

# Import các agent con
from agents.screening_agent import screening_agent
from agents.researcher_agent import researcher_agent
from agents.writer_agent import writer_agent

# Bước 1: Validate + Health check (LlmAgent với tools)
input_validator = LlmAgent(
    name="InputValidator",
    model="gemini-2.0-flash",
    instruction="""
    Kiểm tra SMILES input và health của model_server.
    SMILES: {smiles_input}
    1. Gọi check_model_server_health(). Nếu unhealthy → DỪNG với lỗi hệ thống.
    2. Gọi validate_smiles(smiles_input). Nếu không hợp lệ → DỪNG với lỗi input.
    3. Nếu cả hai OK → output "VALID" vào state.
    """,
    tools=[check_model_server_health, validate_smiles],
    output_key="validation_status",
)

# Bước 2: Parallel execution
parallel_analysis = ParallelAgent(
    name="ParallelAnalysis",
    sub_agents=[screening_agent, researcher_agent],
    description="Chạy ScreeningAgent và ResearcherAgent song song để giảm latency.",
)

# Bước 3: Synthesis
# writer_agent đã định nghĩa ở trên

# Orchestrator tổng thể
orchestrator = SequentialAgent(
    name="ToxAgentOrchestrator",
    sub_agents=[input_validator, parallel_analysis, writer_agent],
    description="Điều phối toàn bộ pipeline phân tích độc tính ToxAgent.",
)

root_agent = orchestrator
```

#### Đặc tả chi tiết

| Thuộc tính | Giá trị |
|---|---|
| **File** | `agents/orchestrator_agent.py` |
| **ADK name** | `"ToxAgentOrchestrator"` |
| **Loại** | `SequentialAgent` (deterministic orchestration) |
| **Role** | Điều phối 3 bước: validate → [screening ‖ research] → write |

**Input (nhận từ user message):**
```python
# User gửi message chứa SMILES, ví dụ:
# "Phân tích độc tính của CC(=O)Oc1ccccc1C(=O)O"
# OrchestratorAgent extract SMILES và set:
session.state["smiles_input"] = "CC(=O)Oc1ccccc1C(=O)O"
```

**Sub-agents (thực thi theo thứ tự):**
```
Step 1: InputValidator      (LlmAgent, sequential)
         ↓ state["validation_status"] = "VALID"
Step 2: ParallelAnalysis    (ParallelAgent)
         ├── ScreeningAgent → state["screening_result"]
         └── ResearcherAgent → state["research_result"]
Step 3: WriterAgent         (LlmAgent, sequential)
         ↓ state["final_report"]
```

**Processing flow:**
```
1. Nhận user message → extract SMILES (LLM parsing nếu cần)
2. Set session.state["smiles_input"] = extracted_smiles
3. Khởi động SequentialAgent pipeline:
   a. InputValidator: check health + validate SMILES
      - Nếu fail → escalate lên root, trả lỗi về user
   b. ParallelAnalysis: chạy ScreeningAgent + ResearcherAgent đồng thời
      - Race condition prevention: mỗi agent dùng key state riêng
   c. WriterAgent: tổng hợp → final_report
4. Đọc state["final_report"] và trả về user
5. (Optional) Reflection step: đánh giá chất lượng báo cáo
```

**Reflection (Optional, Phase 4):**
```python
# Sau khi WriterAgent hoàn thành, OrchestratorAgent có thể thêm LlmAgent:
quality_checker = LlmAgent(
    name="QualityChecker",
    model="gemini-2.0-flash",
    instruction="""
    Đánh giá báo cáo trong {final_report}:
    - Có đủ 5 sections không?
    - risk_level có nhất quán với p_toxic không?
    - Nếu thiếu → set state["needs_revision"] = True và mô tả thiếu gì
    - Nếu OK → set state["needs_revision"] = False
    """,
    output_key="quality_check",
)
```

**Session State Schema (toàn bộ keys):**

```python
{
    # Input
    "smiles_input": str,                    # Set bởi: OrchestratorAgent
    
    # Validation
    "validation_status": str,               # Set bởi: InputValidator → "VALID" | "INVALID"
    
    # Parallel results
    "screening_result": dict | str,         # Set bởi: ScreeningAgent
    "screening_error": str | None,          # Set bởi: ScreeningAgent (khi lỗi)
    "research_result": dict | str,          # Set bởi: ResearcherAgent
    
    # Final output
    "final_report": dict | str,             # Set bởi: WriterAgent
    
    # Optional reflection
    "quality_check": dict | str,            # Set bởi: QualityChecker
    "needs_revision": bool,                 # Set bởi: QualityChecker
}
```

**Failure mode + Fallback:**

| Lỗi | Hành động |
|---|---|
| model_server unhealthy | Trả ngay "Hệ thống phân tích hiện không khả dụng. Vui lòng thử lại." |
| SMILES không extract được từ message | Hỏi lại user: "Vui lòng cung cấp chuỗi SMILES của phân tử cần phân tích." |
| ParallelAgent: ScreeningAgent fail | WriterAgent nhận screening_result = null → trả partial report với lỗi rõ ràng |
| ParallelAgent: ResearcherAgent fail | WriterAgent viết báo cáo không có literature section |
| WriterAgent fail sau 2 retries | OrchestratorAgent trả raw screening_result dạng JSON thô với note "Báo cáo đầy đủ không khả dụng" |
| Toàn bộ pipeline fail | Log lỗi, trả generic error message, không crash session |

---

### Tóm tắt Agent Layer

| Agent | File | Model | Loại ADK | Tools | output_key |
|---|---|---|---|---|---|
| InputValidator | `orchestrator_agent.py` | gemini-2.0-flash | `LlmAgent` | health, validate | `validation_status` |
| ScreeningAgent | `screening_agent.py` | gemini-2.0-flash | `LlmAgent` | validate, analyze | `screening_result` |
| ResearcherAgent | `researcher_agent.py` | gemini-1.5-pro | `LlmAgent` | pubchem, pubmed, bioassay | `research_result` |
| WriterAgent | `writer_agent.py` | gemini-1.5-pro | `LlmAgent` | (none) | `final_report` |
| ParallelAnalysis | `orchestrator_agent.py` | — | `ParallelAgent` | — | — |
| ToxAgentOrchestrator | `orchestrator_agent.py` | — | `SequentialAgent` | — | — |

---

## Sơ đồ Data Flow (ASCII)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              USER INPUT                                         │
│   "Phân tích độc tính của [SMILES]"                                             │
│   Ví dụ: "CC(=O)Oc1ccccc1C(=O)O"                                              │
└───────────────────────────────┬─────────────────────────────────────────────────┘
                                │ Content(role="user", parts=[Part(text=...)])
                                ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        ToxAgentOrchestrator                                      │
│                         (SequentialAgent)                                        │
│                                                                                  │
│  session.state["smiles_input"] = "CC(=O)Oc1ccccc1C(=O)O"  ◄── extracted SMILES│
└───────────────────────────────┬─────────────────────────────────────────────────┘
                                │
                                ▼ Step 1
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          InputValidator                                          │
│                           (LlmAgent)                                             │
│                                                                                  │
│  Tool calls:                                                                     │
│  ├── check_model_server_health() ──► GET /health ──► {"healthy": true}          │
│  └── validate_smiles(smiles)     ──► RDKit         ──► {"valid": true, ...}     │
│                                                                                  │
│  output_key → state["validation_status"] = "VALID"                              │
└───────────────────────────────┬─────────────────────────────────────────────────┘
                                │ state["validation_status"] == "VALID"
                                ▼ Step 2
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          ParallelAnalysis                                        │
│                           (ParallelAgent)                                        │
│                                                                                  │
│         Đọc: state["smiles_input"]                                               │
│                        │                                                        │
│          ┌─────────────┴──────────────┐                                         │
│          │ (concurrent)               │                                          │
│          ▼                            ▼                                          │
│  ┌──────────────────┐    ┌──────────────────────┐                               │
│  │  ScreeningAgent  │    │   ResearcherAgent     │                               │
│  │  (LlmAgent)      │    │   (LlmAgent)          │                               │
│  │                  │    │                        │                               │
│  │ Tool calls:      │    │ Tool calls:            │                               │
│  │ ├ validate_smiles│    │ ├ get_compound_info()  │                               │
│  │ └ analyze_mol()  │    │ ├ search_literature()  │                               │
│  │   │              │    │ └ get_bioassay_data()  │                               │
│  │   ▼              │    │   │                    │                               │
│  │ POST /analyze    │    │   ▼                    │                               │
│  │ model_server     │    │ PubChem REST API       │                               │
│  │   │              │    │ PubMed E-utilities     │                               │
│  │   ▼              │    │   │                    │                               │
│  │ AnalyzeResponse  │    │   ▼                    │                               │
│  │ {clinical,       │    │ {cid, iupac_name,      │                               │
│  │  mechanism,      │    │  articles[5],          │                               │
│  │  explanation,    │    │  bioassay_summary}     │                               │
│  │  final_verdict}  │    │                        │                               │
│  │   │              │    │   │                    │                               │
│  │   output_key ────►    │   output_key ──────────►                              │
│  │ state["screening_result"]  state["research_result"]                           │
│  └──────────────────┘    └──────────────────────┘                               │
└───────────────────────────────┬─────────────────────────────────────────────────┘
                                │ state["screening_result"] + state["research_result"]
                                ▼ Step 3
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            WriterAgent                                           │
│                             (LlmAgent)                                           │
│                                                                                  │
│  Đọc từ state:                                                                   │
│  ├── {screening_result}   → clinical + mechanism + explanation + final_verdict  │
│  ├── {research_result}    → compound_info + literature + bioassay               │
│  └── {smiles_input}       → SMILES gốc                                          │
│                                                                                  │
│  Tổng hợp → báo cáo hoàn chỉnh                                                  │
│                                                                                  │
│  output_key → state["final_report"] = {                                         │
│    report_metadata, executive_summary, risk_level,                               │
│    sections: {clinical_toxicity, mechanism_toxicity,                             │
│               structural_explanation, literature_context,                        │
│               recommendations}                                                   │
│  }                                                                               │
└───────────────────────────────┬─────────────────────────────────────────────────┘
                                │ state["final_report"]
                                ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              FINAL RESPONSE                                      │
│                                                                                  │
│  Báo cáo phân tích độc tính toàn diện:                                           │
│  ┌─────────────────────────────────────────────────────┐                        │
│  │ # Báo cáo Độc tính: Aspirin (C9H8O4)               │                        │
│  │ **Mức độ rủi ro:** LOW                              │                        │
│  │                                                     │                        │
│  │ ## 1. Độc tính Lâm sàng                             │                        │
│  │ p_toxic = 0.12 → NON-TOXIC (confidence: 0.89)      │                        │
│  │                                                     │                        │
│  │ ## 2. Cơ chế Độc tính (Tox21)                       │                        │
│  │ Active tasks: 0/12 → Không kích hoạt thụ thể nào   │                        │
│  │                                                     │                        │
│  │ ## 3. Giải thích Cấu trúc                           │                        │
│  │ [Heatmap GNNExplainer đính kèm]                     │                        │
│  │                                                     │                        │
│  │ ## 4. Bối cảnh Y văn                                │                        │
│  │ PubMed: 1284 bài báo | PubChem CID: 2244            │                        │
│  │                                                     │                        │
│  │ ## 5. Khuyến nghị                                   │                        │
│  │ - An toàn ở liều điều trị thông thường...           │                        │
│  └─────────────────────────────────────────────────────┘                        │
└─────────────────────────────────────────────────────────────────────────────────┘

Latency estimate (rough):
  InputValidator:    ~1-2s  (health + rdkit local)
  ScreeningAgent:    ~3-5s  (POST /analyze → GATv2 inference)
  ResearcherAgent:   ~3-6s  (PubChem + PubMed REST calls)
  [Parallel]:        ~3-6s  (max của hai agent trên)
  WriterAgent:       ~4-8s  (gemini-1.5-pro synthesis)
  ─────────────────────────
  TOTAL END-TO-END:  ~8-16s per analysis
```

---

## Thứ tự Implementation (Priority Queue)

> **Mục tiêu MVP demo trong 48 giờ:** User nhập SMILES → nhận báo cáo độc tính hoàn chỉnh.

### Sprint 1 — MVP Core (Ưu tiên cao nhất)

| # | Task | File | Ước tính | Ghi chú |
|---|---|---|---|---|
| 1 | Implement `validate_smiles` | `tools/tox_tools.py` | 30 phút | RDKit đã cài? Nếu chưa: `pip install rdkit` |
| 2 | Implement `analyze_molecule` | `tools/tox_tools.py` | 30 phút | Wrap POST /analyze; test với aspirin SMILES |
| 3 | Test tool calls thủ công | terminal | 15 phút | `python -c "from tools.tox_tools import *; print(analyze_molecule('CC(=O)Oc1ccccc1C(=O)O'))"` |
| 4 | Implement `ScreeningAgent` | `agents/screening_agent.py` | 45 phút | LlmAgent + 2 tools + output_key |
| 5 | Implement `get_compound_info_pubchem` | `tools/research_tools.py` | 45 phút | Test với CID 2244 (aspirin) |
| 6 | Implement `search_toxicity_literature` | `tools/research_tools.py` | 45 phút | Test esearch + esummary |
| 7 | Implement `ResearcherAgent` | `agents/researcher_agent.py` | 45 phút | LlmAgent + 3 tools + output_key |
| 8 | Implement `WriterAgent` | `agents/writer_agent.py` | 1 giờ | Không cần tools; test template injection |
| 9 | Implement `OrchestratorAgent` (basic) | `agents/orchestrator_agent.py` | 1 giờ | SequentialAgent wrapper trước; thêm Parallel sau |
| 10 | End-to-end test MVP | `main.py` hoặc `adk web` | 30 phút | Chạy pipeline với 3 SMILES test cases |

**Subtotal Sprint 1: ~7 giờ**

---

### Sprint 2 — Parallel + Robustness

| # | Task | File | Ước tính | Ghi chú |
|---|---|---|---|---|
| 11 | Upgrade Orchestrator: SequentialAgent → ParallelAgent | `agents/orchestrator_agent.py` | 30 phút | Wrap ScreeningAgent + ResearcherAgent trong ParallelAgent |
| 12 | Implement `check_model_server_health` | `tools/tox_tools.py` | 20 phút | GET /health + latency measure |
| 13 | Implement `InputValidator` sub-agent | `agents/orchestrator_agent.py` | 30 phút | Thêm vào đầu SequentialAgent pipeline |
| 14 | Error handling đầy đủ cho tất cả tools | `tools/` | 1 giờ | Retry logic, timeout handling, fallback messages |
| 15 | Implement `analyze_molecules_batch` | `tools/tox_tools.py` | 30 phút | POST /predict/batch |
| 16 | Implement `get_pubchem_bioassay_data` | `tools/research_tools.py` | 45 phút | Bổ sung BioAssay context |
| 17 | Viết unit tests cho Tool Layer | `tests/test_tools.py` | 1.5 giờ | Mock httpx responses, test 5+ SMILES cases |

**Subtotal Sprint 2: ~5 giờ**

---

### Sprint 3 — Polish + Demo-Ready

| # | Task | File | Ước tính | Ghi chú |
|---|---|---|---|---|
| 18 | Implement QualityChecker (reflection) | `agents/orchestrator_agent.py` | 45 phút | LlmAgent check final_report quality |
| 19 | FastAPI endpoint wrapper (`/chat`) | `api/chat.py` | 1 giờ | Expose ADK runner qua HTTP cho frontend |
| 20 | `.env` configuration | `.env.example` | 15 phút | MODEL_SERVER_URL, PUBMED_API_KEY, GOOGLE_API_KEY |
| 21 | README + cài đặt nhanh | `README.md` | 30 phút | `docker-compose up` hoặc `adk web` |
| 22 | Demo script với 5 molecules | `demo/run_demo.py` | 30 phút | Aspirin, Paracetamol, PFOA, Bisphenol-A, Caffeine |
| 23 | Benchmark latency | `tests/bench.py` | 30 phút | Đo end-to-end time, xác nhận ~8-16s |

**Subtotal Sprint 3: ~3.5 giờ**

---

### Tổng quan timeline

```
Giờ 0-7:   Sprint 1 — MVP hoạt động (pipeline end-to-end)
Giờ 7-12:  Sprint 2 — Parallel execution + error handling
Giờ 12-15: Sprint 3 — Polish, API, demo
─────────────────────────────────────────────────────
Total: ~15 giờ kỹ thuật cho hệ thống production-ready hackathon
```

---

### Thứ tự file cần tạo (theo dependency)

```
1. tools/__init__.py                     (empty shell trước)
2. tools/tox_tools.py                    (validate + analyze)
3. tools/research_tools.py               (pubchem + pubmed)
4. agents/__init__.py                    (empty)
5. agents/screening_agent.py             (phụ thuộc: tox_tools)
6. agents/researcher_agent.py            (phụ thuộc: research_tools)
7. agents/writer_agent.py                (không tool, chỉ cần state schema)
8. agents/orchestrator_agent.py          (phụ thuộc: tất cả agents trên)
9. main.py hoặc agent.py                 (ADK root entry point)
```

---

### Các lệnh kiểm tra nhanh

```bash
# 1. Test model_server connection
curl http://localhost:8000/health

# 2. Test analyze endpoint trực tiếp
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CC(=O)Oc1ccccc1C(=O)O"}'

# 3. Test Tool Layer
python -c "
from tools.tox_tools import validate_smiles, analyze_molecule
v = validate_smiles('CC(=O)Oc1ccccc1C(=O)O')
print('Validate:', v)
r = analyze_molecule(v['canonical_smiles'])
print('Final verdict:', r.get('final_verdict'))
"

# 4. Chạy ADK web UI (dev mode)
adk web agents/orchestrator_agent.py

# 5. Chạy với Python runner
python main.py --smiles "CC(=O)Oc1ccccc1C(=O)O"
```

---

### Biến môi trường cần thiết

```env
# .env
GOOGLE_API_KEY=your_gemini_api_key_here
MODEL_SERVER_URL=http://localhost:8000
PUBMED_API_KEY=                        # Optional — tăng rate limit từ 3→10 req/s
ADK_AGENT_ENGINE_ID=                   # Optional — Vertex AI Agent Engine
LOG_LEVEL=INFO
```

---

*Tài liệu này được tạo tự động dựa trên đặc tả kiến trúc ToxAgent ngày 31/03/2026.*
*Tham khảo: [Google ADK Documentation](https://google.github.io/adk-docs/), [PubChem PUG REST API](https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest), [NCBI E-utilities](https://www.ncbi.nlm.nih.gov/books/NBK25497/)*
