import { Navbar } from '../components/shell/Navbar';
import { Footer } from '../components/shell/Footer';

const SECTIONS = [
  {
    title: 'Ba ranh giới triển khai',
    body: 'ToxPred đo (stateless, offline sau provisioning). ToxAgent control plane sở hữu product state, session, validator và audit trail. Agent runtime (OpenCode/DSH) chỉ chạy vòng model-tool, không public ra internet và không có quyền gì ngoài MCP tool đã khai báo cho đúng một run.',
  },
  {
    title: 'Không có aggregate toxicity score',
    body: 'hERG channel blockade, Tox21 assay activity và ClinTox clinical-trial signal là ba phép đo độc lập. Không màn hình nào trong sản phẩm gộp chúng thành một nhãn, một màu hay một con số duy nhất.',
  },
  {
    title: 'Mỗi con số neo vào một observation',
    body: 'Một claim trong câu trả lời luôn mang observation_id và field_path trỏ về đúng giá trị predictor đã trả — validator từ chối bất kỳ con số nào không đối chiếu được với nguồn trước khi câu trả lời được lưu.',
  },
  {
    title: 'Model không tự commit câu trả lời',
    body: 'Text của model chỉ trở thành nội dung sản phẩm sau khi qua validator (số liệu, phân loại, giới hạn bắt buộc, claim cấm). Model được tối đa hai lượt sửa; nếu vẫn sai, hệ thống tự dựng một câu trả lời dự phòng có đánh dấu rõ ràng.',
  },
];

export function AboutPage() {
  return (
    <div style={{ minHeight: '100vh', backgroundColor: 'var(--bg)' }}>
      <Navbar />
      <main className="mx-auto max-w-3xl px-6 py-16">
        <h1 className="mb-8 text-3xl font-bold" style={{ color: 'var(--text)' }}>
          Về ToxAgent
        </h1>
        <div className="space-y-8">
          {SECTIONS.map((section) => (
            <section key={section.title}>
              <h2 className="mb-2 text-lg font-semibold" style={{ color: 'var(--text)' }}>
                {section.title}
              </h2>
              <p className="text-sm leading-relaxed" style={{ color: 'var(--text-muted)' }}>
                {section.body}
              </p>
            </section>
          ))}
        </div>
      </main>
      <Footer />
    </div>
  );
}
