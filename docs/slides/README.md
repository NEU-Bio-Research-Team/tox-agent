# ToxAgent slide decks

Decks are generated from the specs in [`docs/spec`](../spec) with `python-pptx`, on top of
the team template `Slides_template.pptx` at the repository root.

## Files

| File | Nội dung |
|---|---|
| `deck_lib.py` | Helper layer: chrome của template, palette, textbox/box/table/arrow helpers |
| `build_deck.py` | Nội dung 37 slide của `ToxAgent_03_Harness_Master_Plan.pptx` |
| `assets/` | Ảnh nhúng trong deck (ảnh chụp GitHub và ảnh chụp UI ToxAgent) |

## Rebuild

```bash
pip install python-pptx
python docs/slides/build_deck.py     # ghi ToxAgent_03_Harness_Master_Plan.pptx ở repo root
```

Script đọc `Slides_template.pptx` ở repo root, xoá các slide mẫu, và dựng lại từng slide
trên layout `BLANK` với đúng phần chrome (thanh footer, nhãn trái/phải, số trang) và
bảng màu của template. Font dùng Times New Roman, giống template.

## Nguồn nội dung

Deck bám theo [`docs/spec/TOXAGENT_HARNESS_MASTER_PLAN_VI.md`](../spec/TOXAGENT_HARNESS_MASTER_PLAN_VI.md).
Khi sửa doc, sửa `build_deck.py` tương ứng rồi build lại — không chỉnh trực tiếp file `.pptx`,
vì lần build sau sẽ ghi đè.

## Ảnh trong `assets/`

| File | Nguồn | Ngày chụp |
|---|---|---|
| `gh_opencode.png` | GitHub OpenGraph card của `sst/opencode` | 2026-09-03 |
| `gh_dsh.png` | GitHub OpenGraph card của `deepseek-ai/deepseek-harness` | 2026-09-03 |
| `mcp_gh.png` | GitHub OpenGraph card của `modelcontextprotocol/modelcontextprotocol` | 2026-09-03 |
| `pipeline_crop.png` | Ảnh chụp UI ToxAgent v0.0.6, cắt từ `documents photos/2. Analysis Pipeline.png` | — |
