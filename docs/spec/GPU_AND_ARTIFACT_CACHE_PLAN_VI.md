# Kế hoạch GPU và cache artifact cho ToxPred + ToxOCR

## Hiện trạng đã xác minh

- Host có NVIDIA GeForce RTX 3050 (6 GB VRAM), driver 581.86, CUDA driver API
  13.0.
- Docker daemon hiện chỉ có runtime `runc`; chưa có NVIDIA Container Toolkit.
- ToxPred Dockerfile mặc định `TORCH_VARIANT=cpu`; ToxOCR cài Torch 1.13.1 CPU.
- Mỗi image build từ `python:3.10-slim`, do đó không kế thừa Torch từ host.
  `pip --no-cache-dir` và layer build thất bại trước khi commit làm các wheel
  lớn bị tải lại sau mỗi retry.

## Mục tiêu

- ToxPred inference chạy CUDA bằng Torch 2.6 CUDA wheel.
- ToxOCR/MolScribe inference chạy CUDA bằng Torch 1.13.1 CUDA 11.7 wheel.
  Driver NVIDIA mới hơn vẫn tương thích ngược với CUDA runtime của wheel.
- Wheel Torch và checkpoint MolScribe được provision/cached bền vững, không
  phụ thuộc một lần pip download thành công trong Docker build.
- CPU vẫn là fallback rõ ràng qua profile/env, không tự đổi device im lặng.

## Phase 0 — GPU runtime host (một lần)

1. Cài NVIDIA Container Toolkit theo hướng dẫn distribution của host.
2. Chạy `sudo nvidia-ctk runtime configure --runtime=docker` và restart Docker.
3. Xác minh bằng container CUDA:

   ```bash
   docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
   ```

   Chỉ tiếp tục Phase 1 khi lệnh này hiển thị RTX 3050 trong container.

## Phase 1 — Tách profile CPU/GPU rõ ràng

1. ToxPred:
   - Giữ `TORCH_VARIANT=cpu` cho profile CPU.
   - Profile GPU build với `TORCH_VARIANT=cu124` (hoặc variant được xác nhận
     có wheel Torch 2.6 cho Python 3.10), `TOXPRED_DEVICE=cuda`.
2. ToxOCR:
   - Parameter hoá Torch variant; GPU image dùng `torch==1.13.1+cu117` từ
     PyTorch CUDA 11.7 index, `TOXOCR_DEVICE=cuda`.
   - `MolScribePredictor` chỉ chọn cuda khi `torch.cuda.is_available()`; health
     phải báo device thực tế để không tuyên bố GPU giả.
3. Compose:
   - Thêm GPU reservation/device request cho `toxpred` và `toxocr`.
   - Expose env profile (`TOXAGENT_ACCELERATOR=gpu|cpu`), không publish GPU
     service ra browser.
4. VRAM:
   - RTX 3050 6 GB đủ cho inference một request/lần; đặt concurrency ban đầu
     1 cho OCR và giới hạn predictor, đo peak VRAM trước khi tăng.

## Phase 2 — Wheelhouse/cache bền vững

1. **Dev/local:** tạo persistent wheelhouse bên ngoài Docker build context:

   ```bash
   pip download --dest .cache/wheels \
     --extra-index-url https://download.pytorch.org/whl/cu124 \
     torch==2.6.0+cu124
   pip download --dest .cache/wheels \
     --extra-index-url https://download.pytorch.org/whl/cu117 \
     torch==1.13.1+cu117
   ```

   Download có thể retry/resume riêng; Dockerfile cài từ wheelhouse bằng
   `--find-links` thay vì tải lại. Wheelhouse không commit vào git.
2. **CI/prod:** sửa/cài lại buildx, dùng BuildKit cache mount cho pip và cache
   export qua registry/GitHub Actions. Không dùng `--no-cache-dir` cho layer
   dependency cần recover sau network failure.
3. Pin checksum/manifest cho wheels trong release evidence; cache không được
   biến thành dependency không xác định version.

## Phase 3 — Checkpoint MolScribe

1. Download checkpoint một lần sang persistent volume/object storage, xác minh
   SHA-256 và version/repo ID.
2. Mount read-only vào toxocr và đặt `TOXOCR_CHECKPOINT_PATH`; không tải từ
   Hugging Face trong startup production.
3. `/health/ready` chỉ true sau model load CUDA thành công. Startup logs/health
   ghi model device và checkpoint fingerprint, không log credential HF.

## Phase 4 — Verification và rollback

1. `torch.cuda.is_available()` trong từng image; `nvidia-smi` quan sát VRAM.
2. Chạy một hERG/Tox21 prediction và một ảnh OCSR; xác minh API output bằng
   CPU baseline trong tolerance đã định nghĩa, không thay đổi scientific
   semantics.
3. Load test tuần tự/concurrency 1 → đo latency/cost/VRAM → nới giới hạn nếu
   không OOM.
4. Rollback: `TOXPRED_DEVICE=cpu`, `TOXOCR_DEVICE=cpu` và image CPU tag; giữ
   GPU image tách tag để không rollback bằng rebuild.

## Tiêu chí hoàn tất

- `docker run --gpus all ... nvidia-smi` pass.
- Cả hai service báo device CUDA thật, ready và inference pass.
- Build lặp lại không tải lại Torch nếu wheelhouse/cache còn tồn tại.
- CPU fallback smoke pass và deployment không cần internet để tải checkpoint.
