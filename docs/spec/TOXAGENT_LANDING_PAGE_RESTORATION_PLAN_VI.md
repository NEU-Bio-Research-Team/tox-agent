# Kế hoạch khôi phục landing page ToxAgent

## Mục tiêu

Khôi phục ngôn ngữ thị giác của landing prototype cũ — tím đậm, display type
lớn, section full-height, sticky navigation và scroll progression — vào
frontend đang deploy, nhưng chỉ truyền đạt capability thật của ToxAgent hiện
tại.

Nguồn tham khảo: `legacy/landing-page-prototype/src/app/App.tsx`. Đây là
prototype Figma đã archive, không copy nguyên xi vào production.

## Phạm vi triển khai

1. **Visual foundation**
   - Purple gradient (`#1E0368` làm anchor), nền có depth, glass card.
   - Navbar sticky/translucent khi scroll, anchor navigation và progress bar.
   - Responsive mobile-first; tôn trọng `prefers-reduced-motion`.

2. **Các section cuộn**
   - Hero: ToxAgent, CTA đến Workbench và Quick Predict.
   - Quy trình: SMILES/ảnh → OCR → hERG/Tox21 → XAI.
   - Những đảm bảo sản phẩm: endpoints riêng biệt, provenance, grounded
     answer/validation.
   - Quick Predict + XAI showcase, lấy shape/nội dung từ product thật.
   - Giới hạn khoa học: screening, không safety verdict hay clinical advice.
   - CTA kết thúc và footer.

3. **Nội dung và assets**
   - Không mang sang số liệu chưa kiểm chứng, tên team giả, case study hoặc
     claim latency từ prototype cũ.
   - Chỉ đưa asset archived vào `frontend` khi attribution/license phù hợp;
     thay import `figma:asset` bằng asset local hợp lệ.

4. **Chất lượng**
   - Anchor có keyboard focus, heading hierarchy và contrast hợp lệ.
   - Test desktop/mobile, `npm run typecheck`, frontend tests, build và bundle
     budget.

## Các file chính dự kiến đổi

- `frontend/src/pages/LandingPage.tsx`
- CSS/theme hoặc component landing nhỏ, nếu cần
- test component/E2E cho anchor CTA và navigation scroll

## Tiêu chí hoàn tất

- Landing có thể cuộn qua toàn bộ section trên desktop/mobile, không overflow.
- CTA dẫn đúng `/sessions` và `/predict`.
- Nội dung không đưa claim khoa học/sản phẩm vượt capability runtime.
- Build production và kiểm tra bundle đều pass.
