import type { FinalReport, StructuredRecommendation, RecommendationSection } from './api';

export interface ChatReportContext {
  smiles: string;
  compound_name: string | null;
  risk_level: string;
  executive_summary: string;
  clinical: {
    verdict: string | null;
    probability: number | null;
    confidence: number | null;
    interpretation: string | null;
  };
  mechanism: {
    highest_risk: string | null;
    assay_hits: number | null;
    top_tasks: Array<{ task: string; score: number }>;
  };
  structural: {
    target_task: string | null;
    target_task_score: number | null;
    top_atoms_summary: string;
    explainer_note: string | null;
  };
  molrag: {
    evidence_summary: string | null;
    confidence: number | null;
    confidence_zone: string | null;
    tox_classes: string[];
    top_analog: string | null;
  };
  literature: {
    papers: Array<{ title: string; year: string | number; snippet: string; pmid?: string }>;
    synthesis_text: string | null;
  };
  fusion: {
    final_label: string | null;
    agreement: boolean | null;
    decision_note: string | null;
  };
  ood: {
    flag: boolean | null;
    reason: string | null;
    recommendation: string | null;
  };
  reliability_warning: string | null;
  recommendations: string[];
}

function safeFloat(value: unknown): number | null {
  const n = typeof value === 'number' ? value : parseFloat(String(value));
  return Number.isFinite(n) ? n : null;
}

function safeBool(value: unknown): boolean | null {
  if (typeof value === 'boolean') return value;
  return null;
}

function clampStr(value: string | null | undefined, max: number): string {
  if (!value) return '';
  return value.length > max ? value.slice(0, max) : value;
}

export function buildChatContext(report: FinalReport): ChatReportContext {
  const s = report.sections;

  // Top 5 mechanism tasks by score
  const taskScores = s.mechanism_toxicity?.task_scores ?? {};
  const topTasks = Object.entries(taskScores)
    .sort(([, a], [, b]) => (b as number) - (a as number))
    .slice(0, 5)
    .map(([task, score]) => ({ task, score: score as number }));

  // Top atoms as compact text (drop base64 images entirely)
  const topAtomsSummary = (s.structural_explanation?.top_atoms ?? [])
    .slice(0, 5)
    .map((a) => `${a.element}(${a.importance.toFixed(2)})`)
    .join(', ');

  // Literature: title + year + short snippet only, strip full abstracts
  const papers = (s.literature_context?.relevant_papers ?? [])
    .slice(0, 5)
    .map((p) => ({
      title: p.title ?? '',
      year: p.year ?? '',
      snippet: clampStr(p.abstract_snippet ?? p.snippet ?? p.abstract ?? '', 200),
      pmid: p.pmid,
    }));

  // Flatten recommendations to string array
  let recsFlat: string[] = [];
  const recs = s.recommendations;
  if (Array.isArray(recs)) {
    recsFlat = (recs as Array<string | StructuredRecommendation>).map((r) =>
      typeof r === 'string' ? r : (r as StructuredRecommendation).action ?? JSON.stringify(r),
    );
  } else if (recs && typeof recs === 'object') {
    const section = recs as RecommendationSection;
    if (section.content) recsFlat = [section.content];
  }

  const riskLevel =
    typeof report.risk_level === 'string'
      ? report.risk_level
      : (report.risk_level as { level?: string } | null)?.level ?? 'UNKNOWN';

  return {
    smiles: report.report_metadata.smiles,
    compound_name:
      report.report_metadata.compound_name ??
      report.report_metadata.common_name ??
      null,
    risk_level: riskLevel,
    executive_summary: report.executive_summary,
    clinical: {
      verdict: s.clinical_toxicity?.verdict ?? null,
      probability: safeFloat(s.clinical_toxicity?.probability),
      confidence: safeFloat(s.clinical_toxicity?.confidence),
      interpretation: s.clinical_toxicity?.interpretation ?? null,
    },
    mechanism: {
      highest_risk: s.mechanism_toxicity?.highest_risk ?? null,
      assay_hits: s.mechanism_toxicity?.assay_hits ?? null,
      top_tasks: topTasks,
    },
    structural: {
      // heatmap_base64 and molecule_png_base64 are intentionally omitted
      target_task: s.structural_explanation?.target_task ?? null,
      target_task_score: safeFloat(s.structural_explanation?.target_task_score),
      top_atoms_summary: topAtomsSummary,
      explainer_note: s.structural_explanation?.explainer_note ?? null,
    },
    molrag: {
      evidence_summary:
        s.molrag_evidence?.evidence_summary ??
        s.molrag_evidence?.longform_summary ??
        null,
      confidence: safeFloat(s.molrag_evidence?.confidence),
      confidence_zone: s.molrag_evidence?.molrag_evidence?.confidence_zone ?? null,
      tox_classes: s.molrag_evidence?.tox_classes ?? [],
      top_analog: s.molrag_evidence?.retrieved_examples?.[0]?.name ?? null,
    },
    literature: {
      papers,
      synthesis_text: s.literature_context?.literature_synthesis?.synthesis_text ?? null,
    },
    fusion: {
      final_label: s.fusion_result?.final_label ?? null,
      agreement: safeBool(s.fusion_result?.agreement),
      decision_note: s.fusion_result?.decision_note ?? null,
    },
    ood: {
      flag: safeBool(s.ood_assessment?.flag),
      reason: s.ood_assessment?.reason ?? null,
      recommendation: s.ood_assessment?.recommendation ?? null,
    },
    reliability_warning: s.reliability_warning ?? null,
    recommendations: recsFlat.slice(0, 10),
  };
}
