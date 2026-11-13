import type { Tone } from '../../shared/ui/callout';

/**
 * The tone of a model's class label (ADR-0002).
 *
 * A positive class ("blocker", "toxic", "active") is `warning`: it says what the
 * model predicted, which is not a verdict. A negative class ("non_blocker",
 * "inactive", "not_toxic") is `neutral`. Neither is `danger`, which is reserved
 * for failures of the system, so a prediction is never drawn as an alarm.
 */
export function labelTone(label: string): Tone {
  return /^(non[_\s-]?|not[_\s-]?|no[_\s-]|negative|inactive)/i.test(label.trim()) ? 'neutral' : 'warning';
}
