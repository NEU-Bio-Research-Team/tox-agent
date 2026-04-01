import { Copy, RefreshCw } from 'lucide-react';
import { Button } from '../ui/button';

export function AIRecommendationsSection() {
  const handleCopy = () => {
    // Mock copy functionality
    console.log('Copy to clipboard');
  };

  return (
    <section id="recommendations">
      <h2 className="text-2xl font-bold mb-6" style={{ color: 'var(--text)' }}>
        §5 AI Recommendations
      </h2>

      <div 
        className="rounded-r-xl p-8"
        style={{
          background: `linear-gradient(135deg, var(--surface) 0%, var(--bg) 100%)`,
          borderLeft: '3px solid var(--accent-blue)'
        }}
      >
        {/* AI Source Indicator */}
        <div className="flex items-center gap-2 mb-6">
          <svg className="w-3 h-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <rect x="3" y="3" width="18" height="18" rx="2" />
            <path d="M9 9h.01M15 9h.01M9 15h6" />
          </svg>
          <span className="text-xs uppercase tracking-wider" style={{ color: 'var(--text-faint)' }}>
            AI · WriterAgent · Tổng hợp báo cáo
          </span>
        </div>

        {/* Sections */}
        <div className="space-y-6">
          {/* Executive Summary */}
          <div>
            <h3 className="text-xs font-bold uppercase mb-2" style={{ color: 'var(--text-muted)', letterSpacing: '0.08em' }}>
              TÓM TẮT ĐIỀU HÀNH
            </h3>
            <p className="text-base leading-relaxed" style={{ color: 'var(--text)', lineHeight: '1.7' }}>
              Phân tử Aspirin thể hiện xác suất độc tính thấp (p_toxic = 0.23) với độ tin cậy cao (89%). 
              Kết quả phân tích cho thấy phân tử này an toàn trong ứng dụng lâm sàng với liều lượng thích hợp.
            </p>
          </div>

          {/* Clinical Interpretation */}
          <div>
            <h3 className="text-xs font-bold uppercase mb-2" style={{ color: 'var(--text-muted)', letterSpacing: '0.08em' }}>
              ĐÁNH GIÁ LÂM SÀNG
            </h3>
            <p className="text-base leading-relaxed" style={{ color: 'var(--text)', lineHeight: '1.7' }}>
              Dựa trên phân tích 12 đường truyền độc tính Tox21, không có đường truyền nào vượt ngưỡng nguy hiểm (0.7). 
              Các chỉ số SR-HSE và SR-MMP cho thấy mức độ kích hoạt thấp, phù hợp với hồ sơ an toàn đã được chứng minh lâm sàng.
            </p>
          </div>

          {/* Mechanism Analysis */}
          <div>
            <h3 className="text-xs font-bold uppercase mb-2" style={{ color: 'var(--text-muted)', letterSpacing: '0.08em' }}>
              PHÂN TÍCH CƠ CHẾ
            </h3>
            <p className="text-base leading-relaxed" style={{ color: 'var(--text)', lineHeight: '1.7' }}>
              Cơ chế tác dụng chủ yếu thông qua ức chế COX, không liên quan đến các đường truyền độc tính nghiêm trọng. 
              Nhóm acetyl và nhân benzoic không thể hiện dấu hiệu cấu trúc độc hại (toxicophore) trong ngữ cảnh này.
            </p>
          </div>

          {/* Literature Context */}
          <div>
            <h3 className="text-xs font-bold uppercase mb-2" style={{ color: 'var(--text-muted)', letterSpacing: '0.08em' }}>
              BỐI CẢNH VĂN HỌC
            </h3>
            <p className="text-base leading-relaxed" style={{ color: 'var(--text)', lineHeight: '1.7' }}>
              5 nghiên cứu được phân tích cho thấy Aspirin có hồ sơ an toàn tốt trong sử dụng lâm sàng. 
              Dữ liệu PubChem và bioassay xác nhận tính không độc hại ở liều điều trị thông thường.
            </p>
          </div>

          {/* Recommendations */}
          <div>
            <h3 className="text-xs font-bold uppercase mb-3" style={{ color: 'var(--text-muted)', letterSpacing: '0.08em' }}>
              KHUYẾN NGHỊ
            </h3>
            <div className="space-y-2">
              <div className="flex items-start gap-3">
                <span style={{ color: 'var(--accent-green)', fontSize: '18px' }}>✓</span>
                <p className="text-base flex-1" style={{ color: 'var(--text)' }}>
                  Phân tử phù hợp cho phát triển tiếp tục trong ứng dụng lâm sàng
                </p>
              </div>
              <div className="flex items-start gap-3">
                <span style={{ color: 'var(--accent-yellow)', fontSize: '18px' }}>⚠</span>
                <p className="text-base flex-1" style={{ color: 'var(--text)' }}>
                  Theo dõi đường truyền chuyển hóa nhóm acetyl ở bệnh nhân có rối loạn gan
                </p>
              </div>
              <div className="flex items-start gap-3">
                <span style={{ color: 'var(--accent-green)', fontSize: '18px' }}>✓</span>
                <p className="text-base flex-1" style={{ color: 'var(--text)' }}>
                  Xem xét nghiên cứu liều-đáp ứng lâm sàng cho các chỉ định mới
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center gap-3 mt-8 pt-6" style={{ borderTop: '1px solid var(--border)' }}>
          <Button 
            variant="ghost" 
            size="sm"
            className="text-sm"
            style={{ color: 'var(--text-muted)' }}
          >
            <RefreshCw className="w-4 h-4 mr-2" />
            Tạo lại báo cáo
          </Button>
          <Button 
            variant="ghost" 
            size="sm"
            onClick={handleCopy}
            className="text-sm"
            style={{ color: 'var(--text-muted)' }}
          >
            <Copy className="w-4 h-4 mr-2" />
            Sao chép văn bản
          </Button>
        </div>

        {/* Generation Info */}
        <div className="mt-4 pt-4 space-y-1 text-xs" style={{ borderTop: '1px solid var(--border)', color: 'var(--text-faint)' }}>
          <p><span className="font-semibold">Model:</span> Gemini 1.5 Pro</p>
          <p><span className="font-semibold">Temperature:</span> 0.2 (deterministic)</p>
          <p><span className="font-semibold">Format:</span> Structured JSON Schema</p>
        </div>
      </div>
    </section>
  );
}
