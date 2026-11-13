import type { Endpoint, PredictCapabilities } from '../../shared/api/types';

export type EndpointCapability = NonNullable<PredictCapabilities['endpoints']>[number];

/** Short names and one-line descriptions of the three independent measurements.
 * They are fixed text, not taken from `display_name`, so the page reads the same
 * whichever deployment answers. */
export const ENDPOINT_LABEL: Record<Endpoint, string> = {
  herg: 'hERG',
  tox21: 'Tox21',
  clintox: 'ClinTox',
};

export const ENDPOINT_DESCRIPTION: Record<Endpoint, string> = {
  herg: 'Chặn kênh hERG',
  tox21: '12 assay độc lập',
  clintox: 'Tín hiệu thử nghiệm lâm sàng',
};

export const ENDPOINT_ORDER: ReadonlyArray<Endpoint> = ['herg', 'tox21', 'clintox'];

/** What to call a model release: the name its manifest gives it, else its
 * identifier. Marked when it is what an unnamed request gets for `endpoint`. */
export function modelLabel(model: { model_id: string; display_name?: string; default_for?: string[] }, endpoint: Endpoint): string {
  const name = model.display_name?.trim() || model.model_id;
  return (model.default_for ?? []).includes(endpoint) ? `${name} · mặc định` : name;
}
