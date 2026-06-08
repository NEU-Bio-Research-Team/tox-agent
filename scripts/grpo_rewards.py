import re
import json
from typing import Any, Dict, List

# Define the required keys based on _MOLRAG_RESPONSE_SCHEMA in molrag_reasoner.py
# (Nếu sau này bạn chốt chỉ dùng 7 field cốt lõi thì rút gọn cả schema lẫn set này cho khớp.)
REQUIRED_KEYS = {
    "evidence_overview",
    "longform_summary",
    "mechanism_chain",
    "key_substructures",
    "confidence_rationale",
    "analogy_reasoning",
    "risk_modifiers",
    "knowledge_highlights",
    "literature_highlights",
    "suggested_label",
    "confidence",
}

def toxicity_label_reward(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    """Reward function for matching target label (e.g. from dataset metadata)."""
    rewards = []
    # In TRL, GRPO sends metadata in kwargs. Let's look for targets.
    targets = kwargs.get("label_targets", [])
    
    for i, completion in enumerate(completions):
        try:
            # Clean completion and extract JSON
            clean_text = completion.strip()
            # Basic parsing
            payload = json.loads(clean_text)
            predicted_label = str(payload.get("suggested_label", "")).strip().lower()
            
            # Get target label
            target = targets[i].strip().lower() if i < len(targets) else "toxic"
            
            if predicted_label == target:
                rewards.append(1.0)
            else:
                rewards.append(-0.5)
        except Exception:
            rewards.append(-1.0) # Penalty for unparseable completion
            
    return rewards

def json_schema_reward(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    """Reward compliance với _MOLRAG_RESPONSE_SCHEMA — partial credit, gradient mượt."""
    rewards = []
    for completion in completions:
        try:
            clean_text = completion.strip()
            # Strip markdown code fence nếu có
            if clean_text.startswith("```"):
                lines = clean_text.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip().startswith("```"):
                    lines = lines[:-1]
                clean_text = "\n".join(lines).strip()

            payload = json.loads(clean_text)
            present_keys = set(payload.keys())
            overlap = REQUIRED_KEYS.intersection(present_keys)

            # Thưởng theo tỉ lệ field đúng, scale sao cho khớp hoàn toàn = 1.5.
            # Không còn yêu cầu "không thừa không thiếu" nên model xuất đủ 11 field
            # sẽ đạt 1.5 thay vì kẹt ở 7/11 như code cũ.
            score = len(overlap) / len(REQUIRED_KEYS)
            rewards.append(score * 1.5)
        except Exception:
            rewards.append(-1.0)
    return rewards

def mechanism_chain_quality(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    """Reward the depth and realism of the generated mechanism chain (e.g. assays, SMARTS)."""
    rewards = []
    # Common toxicophore keywords and assay markers
    substructure_keywords = re.compile(
        r"(aromatic|amine|aldehyde|nitro|halogen|epoxide|sulfur|phosphorus|alkylation|scaffold|smarts)", 
        re.IGNORECASE
    )
    assay_keywords = re.compile(
        r"(hERG|NR-AR|NR-AhR|SR-MMP|Tox21|clinical|mitochondrial|receptor|channel)",
        re.IGNORECASE
    )
    
    for completion in completions:
        try:
            clean_text = completion.strip()
            payload = json.loads(clean_text)
            
            chain = payload.get("mechanism_chain", [])
            substructures = payload.get("key_substructures", [])
            
            if not isinstance(chain, list) or not chain:
                rewards.append(0.0)
                continue
                
            # Score factors
            chain_length = len(chain)
            has_substructure_matches = any(substructure_keywords.search(s) for s in substructures)
            has_assay_matches = any(assay_keywords.search(step) for step in chain)
            
            # Base score: length (prefer 2-5 reasoning steps)
            score = min(chain_length / 4.0, 1.0)
            
            # Modifiers
            if has_substructure_matches:
                score += 0.25
            if has_assay_matches:
                score += 0.25
                
            rewards.append(min(score, 1.5))
        except Exception:
            rewards.append(0.0)
            
    return rewards

def confidence_calibration(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    """Reward confidence scores that align logically with analog similarities (ECE reduction)."""
    rewards = []
    # Max similarities from prompt metadata or kwargs
    similarities = kwargs.get("max_similarities", [])
    
    for i, completion in enumerate(completions):
        try:
            clean_text = completion.strip()
            payload = json.loads(clean_text)
            confidence = float(payload.get("confidence", 0.5))
            
            sim = float(similarities[i]) if i < len(similarities) else 0.5
            
            # Rules:
            # 1. If similarity is very high (>0.85), confidence should be high (>0.8)
            # 2. If similarity is very low (<0.3), confidence should not be high (<=0.6)
            diff = abs(confidence - sim)
            
            # Reward inversely proportional to difference
            reward = 1.0 - diff
            rewards.append(max(reward, 0.0))
        except Exception:
            rewards.append(0.0)
            
    return rewards
