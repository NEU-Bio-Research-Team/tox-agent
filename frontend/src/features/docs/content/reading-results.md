Mỗi phép đo cho ra một thẻ riêng. Bài này chỉ cho bạn nhìn vào đâu trên thẻ, và những gì thẻ không nói.

## Một thẻ cho mỗi phép đo

hERG và ClinTox cho một xác suất cùng nhãn tương ứng với ngưỡng đã dùng. Tox21 cho một dòng cho mỗi assay. Ba phép đo trả lời ba câu hỏi khác nhau, nên chúng không bao giờ được gộp lại.

```card-anatomy
```

> [!NOTE] Không có kết luận gộp
>
> Ba phép đo không bao giờ được gộp thành một nhãn, một màu hay một con số. Đếm số assay Tox21 đang active cũng không phải thước đo mức độ nghiêm trọng, nên trang không hiển thị con số đó.

## Applicability

Một dòng ghi chú cho biết phân tử có nằm ngoài những gì mô hình được xây dựng cho hay không. Đánh giá này dựa trên luật cố định (`element_rules_v1`), không phải một đo lường out-of-distribution học được. Hãy coi kết quả có ghi chú này là yếu hơn.

## Giải thích nguyên tử

Với hERG và Tox21, bạn có thể hỏi nguyên tử nào ảnh hưởng nhiều nhất đến dự đoán. Phần này cho thấy mô hình chú ý vào đâu, không phải nguyên nhân gây ra tác dụng.

> [!WARNING] Không phải bằng chứng cơ chế
>
> Vùng được tô sáng là gợi ý để diễn giải. Đừng dùng nó để kết luận về cơ chế.

## Phép đo không có sẵn

Nếu triển khai không phục vụ một phép đo, thẻ của nó hiện là không có sẵn thay vì bị ẩn. Chưa chạy hoặc chạy thất bại không phải là kết quả âm tính.
