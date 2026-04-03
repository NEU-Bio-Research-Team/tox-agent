# Project Status Update (2026-04-02)

## 1) Tinh nang dang hoat dong on

### Backend / Cloud Run
- Service dang chay on tren revision `tox-agent-cpu-00078-246`.
- Health check OK: `/health` tra ve `status=healthy`, model clinical + tox21 da load.
- Da tang autoscaling tu `maxScale=1` len `maxScale=3` de giam tinh trang bi nghen instance.

### Agent layer (`/agent/analyze`)
- Luong ADK da hoat dong lai on dinh trong smoke test chinh (`CCO`):
	- `runtime_mode = adk`
	- `runtime_note = null`
	- `validation_status = VALID`
- Recommendation da ve dung tu LLM:
	- `recommendation_source = llm`
	- `recommendation_source_detail = llm_success:vertex_adc:global`

### Structural image / molecular plot
- Van de `molecule_png_base64` null da duoc xu ly trong ket qua test moi nhat:
	- `molecule_png_base64_is_null = false`
	- `heatmap_base64_is_null = false`
- Endpoint `/analyze` (force explain) cung tra ve image day du, khong null.

### Frontend
- Frontend dang phuc vu ban tracking `0.0.2` tren hosting.

## 2) Tinh nang dang gap issue / can theo doi

### Intermittent 429 va timeout o agent endpoint
- Van co kha nang gap `429 Rate exceeded` hoac timeout cho `/agent/analyze` trong cac khoang tai cao.
- Trieu chung nay da giam sau khi tang `maxScale=3`, nhung chua the xem la triet de.

### Phu thuoc quota Vertex AI
- Khi Vertex bi `RESOURCE_EXHAUSTED`, he thong co the phai retry model tier/location.
- Trong truong hop xau nhat van co kha nang roi ve fallback de bao toan ket qua.

### Log noisey (khong block chuc nang)
- Log ADK van co cac dong "Event from an unknown agent".
- Hien tai day la warning noisey, khong thay chan flow chinh.

## 3) Danh gia tong quan hien tai

- Muc do san sang hien tai: **Co the demo/van hanh duoc**.
- 2 van de critical truoc day da dat trang thai **pass trong smoke test moi nhat**:
	1. Agent layer khong con mac dinh deterministic fallback trong flow thanh cong.
	2. Molecular structure plot (`molecule_png_base64`) da co du lieu.

## 4) Viec nen lam tiep theo (uu tien)

1. Tang quan sat production:
	 - Dashboard cho latency, 429 rate, timeout rate cua `/agent/analyze`.
2. Chot chinh sach scaling:
	 - Thu nghiem `maxScale` cao hon (vi du 5) neu luu luong tang.
3. Them smoke test tu dong sau deploy:
	 - Kiem tra bat buoc cac field: `runtime_mode`, `recommendation_source`, `molecule_png_base64`, `heatmap_base64`.

