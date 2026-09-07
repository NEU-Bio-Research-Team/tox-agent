import type { ClarificationTextContent } from '../../lib/api/types';
import { Button } from '../ui/button';

/** Every option the router can send is a UI action, never literal text to
 * resend as a chat message — sending the raw option string back (the
 * previous behavior) produced the exact same clarification again, since the
 * backend never treated "submit_smiles" as anything but ordinary prose. */
const OPTION_LABEL_VI: Record<string, string> = {
  submit_smiles: 'Nhập SMILES',
};

export function ClarificationCard({
  content,
  onAction,
}: {
  content: ClarificationTextContent;
  onAction: (action: string) => void;
}) {
  // `out_of_scope` and `capability_unavailable` both carry their text in
  // `.message`; an ordinary clarification carries it in `.question`. OCR
  // messages are rendered locally because the transport-safe backend text is
  // English today, while the UI must distinguish an unavailable deployment
  // capability from an image that simply could not be recognised.
  const code = content.code;
  const capability = content.capability;
  const reason = content.reason;
  const text =
    code === 'capability_unavailable' && capability === 'structure_recognition'
      ? 'Dịch vụ nhận diện cấu trúc từ ảnh chưa được bật trong môi trường này. Hãy dùng SMILES hoặc vẽ cấu trúc để phân tích ngay.'
      : code === 'structure_recognition_failed' && reason === 'service_unavailable'
        ? 'Dịch vụ nhận diện cấu trúc hiện không truy cập được. Hãy thử lại sau, hoặc nhập SMILES trực tiếp.'
        : code === 'structure_recognition_failed' && reason === 'no_structure_detected'
          ? 'Không nhận diện được cấu trúc hoá học trong ảnh. Hãy dùng ảnh rõ hơn, hoặc nhập SMILES trực tiếp.'
          : code === 'structure_recognition_failed' && reason === 'attachment_unavailable'
            ? 'Ảnh tải lên không còn sẵn sàng để nhận diện. Hãy tải ảnh lên lại hoặc nhập SMILES trực tiếp.'
            : content.message || content.question;

  return (
    <div className="my-2 max-w-lg rounded-xl border p-3" style={{ backgroundColor: 'var(--accent-blue-muted)', borderColor: 'var(--border)' }}>
      <p className="text-sm" style={{ color: 'var(--text)' }}>
        {text}
      </p>
      {content.options && content.options.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-2">
          {content.options.map((option) => (
            <Button key={option} size="sm" variant="outline" onClick={() => onAction(option)}>
              {OPTION_LABEL_VI[option] ?? option}
            </Button>
          ))}
        </div>
      )}
    </div>
  );
}
