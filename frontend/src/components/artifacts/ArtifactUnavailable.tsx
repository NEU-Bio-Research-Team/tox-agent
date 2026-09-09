import { ArchiveX } from 'lucide-react';
import { ApiError } from '../../lib/api/types';

/** Keep an artifact retention outcome distinct from a generic 404: the latter
 * may intentionally hide both existence and ownership from an untrusted
 * caller. W4-10 will emit `artifact_expired`/410 only when its tombstone can
 * establish expiry without weakening that boundary. */
export function ArtifactUnavailable({ artifact, error }: { artifact: string; error: unknown }) {
  const expired = error instanceof ApiError && (error.code === 'artifact_expired' || error.status === 410);
  if (expired) {
    return (
      <div className="rounded-xl border border-dashed p-4 text-sm" style={{ borderColor: 'var(--border)', backgroundColor: 'var(--surface-alt)' }}>
        <div className="flex items-center gap-2" style={{ color: 'var(--text)' }}>
          <ArchiveX className="h-4 w-4" style={{ color: 'var(--text-faint)' }} />
          <p className="font-medium">Artifact đã hết hạn</p>
        </div>
        <p className="mt-2 text-xs" style={{ color: 'var(--text-muted)' }}>
          {artifact} này đã bị loại khỏi dữ liệu truy cập theo chính sách lưu giữ. Link provenance được giữ lại, nhưng nội dung không thể khôi phục từ giao diện.
        </p>
      </div>
    );
  }
  return (
    <p className="text-sm" style={{ color: 'var(--accent-red)' }}>
      Không tải được {artifact} này — có thể không tồn tại hoặc bạn không có quyền xem.
    </p>
  );
}
