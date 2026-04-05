# XSMILES Performance Brainstorm (AUC cao nhưng vẫn miss toxic)

Ngày phân tích: 2026-03-27  
Phạm vi: kiểm tra luồng train/eval/inference của XSMILES + luồng GNNExplainer trong workspace hiện tại.

---

## 1) Kết luận nhanh

- **AUC ~0.997 là tái lập được** với checkpoint hiện tại trên split test hiện tại.
- **Global label-flip (đảo nhãn toàn cục) gần như không đúng**:
  - AUC(y, p) = 0.9978
  - AUC(1-y, p) = 0.0022
  - AUC(y, 1-p) = 0.0022
- Việc có 1 mẫu toxic bị dự đoán non-toxic **không mâu thuẫn** với AUC cao, vì AUC là metric theo ranking, không phụ thuộc ngưỡng 0.5.
- False negative chính nằm ở một mẫu có Pt (Carboplatin-like):
  - `C1CC(C1)(C(=O)O)C(=O)O.N.N.[Pt]`
  - P(toxic) = 0.1426 (predicted non-toxic)

---

## 2) Luồng hiện tại có bị đảo nhãn không?

### 2.1 Train/Eval

- Dataset loader lấy nhãn `CT_TOX` từ ClinTox (comment rõ: 1=toxic, 0=non-toxic).
- Training dùng `BCEWithLogits`/`FocalLoss` trên logit đầu ra 1 chiều.
- Evaluation dùng `sigmoid(logit)` để ra xác suất và `roc_auc_score(labels, probs)`.

=> Không thấy chỗ nào đảo chiều xác suất thành P(non-toxic).

### 2.2 Inference trong app

- `predict_batch` cũng dùng `sigmoid(logit)` thành `P(toxic)`.
- Rule phân lớp: `pred = 1 if prob >= threshold else 0`.

=> Semantics ở app khớp với train/eval.

### 2.3 GNNExplainer

- GNNExplainer chỉ giải thích theo `target_class` mà app chọn từ kết quả Stage A.
- Nó **không sửa prediction gốc**, chỉ giải thích quyết định đó.

=> Nếu prediction sai thì explanation chỉ là “giải thích cho quyết định sai”, không phải bằng chứng model đúng/sai.

---

## 3) Vì sao AUC rất cao nhưng vẫn có toxic bị miss?

Test set hiện có 148 mẫu, chỉ 10 positive (toxic).  
Khi mình tái tính từ checkpoint:

- AUC = 0.997826
- Accuracy = 0.972973
- F1 = 0.818182
- Confusion matrix = [[135, 3], [1, 9]]

AUC đo theo cặp positive-negative. Với 10 positive và 138 negative, có:

- tổng số cặp = 10 x 138 = 1380
- số cặp xếp hạng sai chỉ 3 cặp

Nên AUC vẫn gần 1.0 dù có 1 FN ở ngưỡng 0.5.

---

## 4) Tất cả khả năng khiến performance trông “quá đẹp”

## A. Khả năng cao

1. **Metric mismatch (AUC cao, threshold decision vẫn sai vài ca)**
- AUC là ranking metric, không phải metric theo ngưỡng.
- Một vài mẫu toxic gần/qua ngưỡng vẫn có thể bị miss.

2. **Imbalance mạnh + test positive quá ít (10 mẫu)**
- Chỉ cần rất ít thứ hạng sai là AUC đã cực cao.
- Bootstrap cho thấy AUC vẫn cao nhưng độ bất định của PR-AUC không hẹp tuyệt đối.

3. **Out-of-distribution chemistry (đặc biệt organometallic)**
- Error tập trung mạnh vào molecule có kim loại (Pt).
- Trong 3 mẫu test có metal, có 2 mẫu bị sai (66.7% error), trong khi overall error chỉ 2.7%.
- Featurization atom đang one-hot mạnh cho nhóm phần tử organic phổ biến; nguyên tố lạ gom vào “other”.

4. **Threshold 0.5 chưa tối ưu cho mục tiêu toxic recall**
- Quét ngưỡng trên test cho thấy F1 tốt nhất quanh ~0.8 với dữ liệu hiện tại.
- Nếu mục tiêu là không bỏ sót toxic, nên tối ưu threshold theo recall/cost thay vì giữ cố định 0.5.

5. **Label ambiguity/noise trong chính benchmark ClinTox**
- Có canonical molecules xuất hiện với cả nhãn 0 và 1 trong toàn bộ dataset (19 trường hợp khi canonical hóa).
- Ví dụ 5-FU canonical có cả nhãn 0 và 1.
- Điều này làm đánh giá single-molecule theo “tri thức dược lý” có thể mâu thuẫn với ground truth benchmark.

## B. Khả năng trung bình

6. **Selection bias do chỉ báo cáo một split/seed tốt**
- Hiện config cố định seed=42 và scaffold split duy nhất.
- Nếu đã tinh chỉnh nhiều lần trên cùng test split rồi giữ run đẹp nhất, performance sẽ bị lạc quan.

7. **Data leakage kiểu gần-duplicate không phải chuỗi raw**
- Đã kiểm tra overlap raw/canonical/desalt-canonical giữa train-val-test: 0 overlap.
- Với split đang dùng, leakage kiểu này không thấy.
- Tuy nhiên vẫn nên kiểm tra additional standardization (tautomer/protonation) nếu audit nghiêm ngặt.

8. **Calibration chưa tốt (xác suất không tương ứng risk thật)**
- AUC có thể cao dù xác suất chưa calibrated.
- Với ứng dụng screening, calibration (ECE/Brier/reliability curve) quan trọng.

## C. Khả năng thấp nhưng nên lưu ý kỹ thuật

9. **Code caveat: lưu best model bằng shallow copy state_dict**
- Trong training loop, `best_model_state = model.state_dict().copy()` là copy nông.
- Có rủi ro không khôi phục đúng epoch tốt nhất tuyệt đối.
- Caveat này thường làm xấu kết quả, không phải làm đẹp giả.

10. **Inconsistency nhỏ giữa ngưỡng app và hiển thị explainer**
- App phân lớp theo `prob >= threshold`.
- `predicted_class` trong `gnn_explainer.py` lại cố định `prob > 0.5` để hiển thị title.
- Có thể gây rối khi threshold khác 0.5, nhưng không làm sai AUC train/test.

---

## 5) Đánh giá riêng về luồng GNNExplainer

- Luồng hiện tại phù hợp mục tiêu “giải thích graph pathway” vì frozen SMILES embedding.
- Nhưng cần hiểu đúng:
  - Đây là **local post-hoc explanation**.
  - Không chứng minh mô hình đúng về nhân quả.
  - Nếu dự đoán sai thì explanation chỉ mô tả lý do mô hình sai.
- Vì vậy GNNExplainer không thể dùng để xác thực rằng AUC 0.997 là “thật” hay “ảo”; nó chỉ giải thích quyết định từng mẫu.

---

## 6) Trả lời trực tiếp câu hỏi label-flip

- **Không có bằng chứng cho global label-flip** trong code path train/eval/inference.
- **Có bằng chứng mạnh cho các nguyên nhân khác**:
  - dữ liệu cực mất cân bằng,
  - sample positive rất ít,
  - OOD organometallic,
  - benchmark label ambiguity,
  - threshold chưa tối ưu theo mục tiêu recall toxic.

---

## 7) Hành động khuyến nghị (ưu tiên)

1. Chạy lại benchmark với nhiều seed/scaffold splits (ít nhất 5-10) và báo cáo mean ± std.
2. Thêm báo cáo calibration: Brier score, ECE, reliability curve.
3. Tối ưu threshold theo mục tiêu nghiệp vụ (ưu tiên recall toxic), không khóa 0.5.
4. Báo cáo subgroup metrics cho molecules có kim loại / nguyên tố ngoài organic set.
5. Audit label quality:
   - phát hiện canonical conflicts,
   - thống nhất rule xử lý salts/protonation,
   - tách một external test set sạch để kiểm chứng thực chiến.
6. Sửa code caveat `best_model_state` bằng deep copy tensor state dict để chắc chắn khôi phục best epoch thật.

---

## 8) Snapshot kiểm chứng mình đã chạy

- Recompute từ checkpoint: AUC=0.997826, CM=[[135,3],[1,9]].
- File `test_data/toxic_compounds.csv`: 9/10 toxic đúng, miss duy nhất TOX-009 (Pt compound).
- Kiểm tra overlap split:
  - raw overlap train-test = 0
  - canonical overlap train-test = 0
  - desalt canonical overlap train-test = 0

=> Performance cao là có thật trên split hiện tại, nhưng vẫn cần audit thêm trước khi coi là “chắc chắn tổng quát hóa tốt”.
