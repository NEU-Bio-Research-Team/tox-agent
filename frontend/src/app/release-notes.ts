import { APP_VERSION } from './build-info';

export const RELEASE_NOTES_EVENT = 'toxagent:open-release-notes';
export const RELEASE_NOTES_STORAGE_KEY = `toxagent.release-notes.seen.${APP_VERSION}`;

export const RELEASE_NOTES_ITEMS = [
  'Added new tox_type_model options: tox21_ensemble_3_best and tox21_pretrained_gin_model for mechanism prediction.',
  'Added tox21_ensemble_3_best as a binary_tox_model option; binary score now supports ensemble averaging across ChemBERTa + MolFormer hERG heads.',
  'Aligned explanation routing with selected tox_type_model; ensemble mode now uses the designated best-member explainer engine.',
  'Integrated serving support for pretrained-GIN Tox21 inference and ensemble-3-best mechanism aggregation.',
  'Kept release notes popup and version badge behavior so build updates remain visible immediately after deployment.',
];
