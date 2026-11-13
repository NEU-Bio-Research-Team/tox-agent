import { ArchiveX } from 'lucide-react';
import { ApiError } from '../../../shared/api/types';

/** Keep an artifact retention outcome distinct from a generic 404: the latter
 * may intentionally hide both existence and ownership from an untrusted
 * caller. W4-10 will emit `artifact_expired`/410 only when its tombstone can
 * establish expiry without weakening that boundary. */
export function ArtifactUnavailable({ artifact, error }: { artifact: string; error: unknown }) {
  const expired = error instanceof ApiError && (error.code === 'artifact_expired' || error.status === 410);
  if (expired) {
    return (
      <div className="rounded-xl border border-dashed p-4 text-sm bg-surface-muted">
        <div className="flex items-center gap-2 text-ink">
          <ArchiveX className="h-4 w-4 text-ink-tertiary" />
          <p className="font-medium">Nội dung này đã hết hạn lưu</p>
        </div>
        <p className="mt-2 text-xs text-ink-secondary">
          {artifact} này đã bị xoá theo chính sách lưu giữ dữ liệu. Liên kết vẫn còn nhưng nội dung không thể khôi phục từ giao diện.
        </p>
      </div>
    );
  }
  return (
    <p className="text-sm text-danger">
      Không tải được {artifact} này — có thể nó không còn hoặc bạn không có quyền xem.
    </p>
  );
}
