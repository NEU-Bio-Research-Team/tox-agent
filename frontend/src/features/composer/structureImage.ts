/** What an uploaded structure image must be before it is sent for
 * recognition, shared by the composer's dialog and the analysis page's
 * upload tab so the two cannot drift apart. */
export const ACCEPTED_MIME_TYPES = ['image/png', 'image/jpeg', 'image/webp'] as const;
export type AcceptedMimeType = (typeof ACCEPTED_MIME_TYPES)[number];

//: Mirrors PolicySettings.max_image_bytes — checked again on the backend,
// this is only a fast client-side rejection before spending an upload.
export const MAX_IMAGE_BYTES = 5_000_000;

export interface StagedImage {
  mimeType: AcceptedMimeType;
  dataBase64: string;
  previewUrl: string;
  fileName: string;
  sizeBytes: number;
}

export function isAcceptedMimeType(value: string): value is AcceptedMimeType {
  return (ACCEPTED_MIME_TYPES as readonly string[]).includes(value);
}
