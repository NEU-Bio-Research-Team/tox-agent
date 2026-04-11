import type { RiskLevel, RiskLevelCode, RiskLevelDetail } from '../lib/api';

const ALLOWED_RISK_LEVELS: RiskLevelCode[] = ['CRITICAL', 'HIGH', 'MODERATE', 'LOW', 'UNKNOWN'];

export interface NormalizedRiskLevel {
  code: RiskLevelCode;
  description: string | null;
}

function toRiskLevelCode(value: unknown): RiskLevelCode {
  if (typeof value !== 'string') {
    return 'UNKNOWN';
  }

  const normalized = value.trim().toUpperCase();
  if (ALLOWED_RISK_LEVELS.includes(normalized as RiskLevelCode)) {
    return normalized as RiskLevelCode;
  }

  return 'UNKNOWN';
}

export function normalizeRiskLevel(riskLevel: RiskLevel | null | undefined): NormalizedRiskLevel {
  if (typeof riskLevel === 'string') {
    return {
      code: toRiskLevelCode(riskLevel),
      description: null,
    };
  }

  if (riskLevel && typeof riskLevel === 'object') {
    const detail = riskLevel as RiskLevelDetail;
    const description = typeof detail.description === 'string' && detail.description.trim().length > 0
      ? detail.description.trim()
      : null;

    return {
      code: toRiskLevelCode(detail.level),
      description,
    };
  }

  return {
    code: 'UNKNOWN',
    description: null,
  };
}
