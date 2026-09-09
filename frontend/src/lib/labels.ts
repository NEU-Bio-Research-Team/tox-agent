import type { LimitationCode, RunStatus } from './api/types';

export const RUN_STATUS_LABEL_VI: Record<RunStatus, string> = {
  queued: 'đang chờ',
  running: 'đang chạy',
  validating: 'đang kiểm định',
  completed: 'hoàn tất',
  failed: 'thất bại',
  cancelled: 'đã huỷ',
};

export const LANE_LABEL_VI: Record<string, string> = {
  deterministic: 'deterministic',
  agentic: 'agentic',
  mixed: 'mixed',
};

export const INTENT_LABEL_VI: Record<string, string> = {
  analysis: 'phân tích',
  analysis_batch: 'phân tích hàng loạt',
  report_qa: 'hỏi báo cáo',
  evidence_research: 'tìm evidence',
  attribution: 'attribution',
  structure_recognition: 'nhận diện cấu trúc',
  clarification_required: 'cần làm rõ',
  out_of_scope: 'ngoài phạm vi',
};

/** Plan section 9.4 — six codes, each a fixed, non-optional sentence. A
 * limitation is content, not a disclaimer to summarize or shorten. */
export const LIMITATION_LABEL_VI: Record<LimitationCode, string> = {
  uncalibrated_probability:
    'Xác suất từ mô hình chưa được hiệu chỉnh (uncalibrated) — không phải nguy cơ lâm sàng đã được định cỡ.',
  applicability_is_rule_based:
    'Applicability được đánh giá bằng luật cố định (element_rules_v1), không phải phát hiện out-of-distribution đã học.',
  attribution_not_causality:
    'Attribution chỉ cho biết trọng số đóng góp của mô hình cho đúng một endpoint/task, không chứng minh quan hệ nhân quả.',
  endpoint_unavailable: 'Một hoặc nhiều endpoint được hỏi hiện không khả dụng cho phân tử này.',
  evidence_scope_limited:
    'Evidence được trích dẫn có phạm vi giới hạn — không phải tổng quan toàn diện của y văn.',
  screening_not_safety_assessment:
    'Đây là kết quả sàng lọc tính toán, không phải đánh giá an toàn hay khuyến nghị lâm sàng.',
};

export const ERROR_CODE_LABEL_VI: Record<string, string> = {
  invalid_request: 'Yêu cầu không hợp lệ.',
  invalid_smiles: 'SMILES không hợp lệ.',
  not_found: 'Không tìm thấy.',
  session_not_found: 'Không tìm thấy session.',
  analysis_not_found: 'Không tìm thấy phân tích.',
  api_route_not_found: 'API không có route này. Frontend và control-plane có thể đang lệch phiên bản.',
  unauthenticated: 'Token xác thực không hợp lệ hoặc đã hết hạn.',
  forbidden: 'Bạn không có quyền thực hiện thao tác này.',
  conflict: 'Trạng thái đã thay đổi, vui lòng tải lại.',
  endpoint_unavailable: 'Endpoint này hiện không khả dụng.',
  capability_unavailable: 'Bản triển khai này chưa bật tính năng đó.',
  structure_recognition_unavailable: 'Dịch vụ nhận diện cấu trúc hiện không phản hồi.',
  smiles_not_detected: 'Không nhận ra cấu trúc trong ảnh. Thử ảnh rõ hơn hoặc nhập SMILES.',
  predictor_not_ready: 'Predictor chưa sẵn sàng.',
  predictor_protocol_error: 'Predictor trả về phản hồi không hợp lệ.',
  runtime_unavailable: 'Runtime agent hiện không khả dụng.',
  runtime_protocol_error: 'Runtime agent trả về phản hồi không hợp lệ.',
  tool_denied: 'Tool bị từ chối theo capability profile.',
  tool_timeout: 'Tool call vượt quá thời gian cho phép.',
  provider_rate_limited: 'Nhà cung cấp model đang giới hạn tốc độ.',
  evidence_unavailable: 'Evidence hiện không khả dụng.',
  answer_validation_failed: 'Đáp án không qua được kiểm định.',
  deadline_exceeded: 'Run vượt quá thời hạn cho phép.',
  internal_error: 'Lỗi hệ thống nội bộ.',
};

export function errorMessageVi(code: string, fallback: string): string {
  return ERROR_CODE_LABEL_VI[code] ?? fallback;
}
