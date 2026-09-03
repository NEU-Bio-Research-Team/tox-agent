# -*- coding: utf-8 -*-
"""ToxAgent Harness — Master Plan deck (Vietnamese), built on Slides_template.pptx."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deck_lib import *
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

REPO = "/home/minhquang/tox-agent"
IMG  = os.path.join(REPO, "docs", "slides", "assets")
os.chdir(REPO)

D = Deck(left_label="ToxAgent Harness", right_label="NEU Bio-Research Team")
S = D.content
Y0 = 1.02          # top of the content area
BOT = 6.95         # above the footer bar

# ═══════════════════════════════════════════════════ 1. TITLE
D.title_slide(
    title="ToxAgent Harness\nTừ khái niệm đến lộ trình xây lại",
    subtitle_lines=[
        ("NEU Bio-Research Team", 19, True, None),
        ("Presenter: Nguyen Minh Quang", 19, True, None),
        ("Nguồn: docs/spec/TOXAGENT_HARNESS_MASTER_PLAN_VI.md · nhánh agent_test", 15, False, None),
    ],
    badge="Architecture Master Plan",
    corner_lines=["ToxAgent,", "2026"],
)

# ═══════════════════════════════════════════════════ 2. AGENDA
s = S("Nội dung")
rows = [
    ("1", "Harness là gì — và OpenCode / DSH là gì", "Nền tảng khái niệm", BLUE),
    ("2", "ToxAgent hôm nay: sáu sản phẩm, bảy chẩn đoán", "Hiện trạng", ORANGE),
    ("3", "Giữ gì, xoá gì — và kiến trúc đích", "Quyết định", GREEN),
    ("4", "Thuê agent loop từ OpenCode và DSH", "Tích hợp", CYAN),
    ("5", "Lộ trình S0–S7, rủi ro và quyết định cần chốt", "Thực thi", RED),
]
y = 1.35
for num, title, kicker, color in rows:
    box(s, 0.85, y, 0.95, 0.92, [(num, 34, True, WHITE)], fill=color)
    textbox(s, 2.00, y + 0.06, 10.4, 0.45, [(title, 25, True, DARK)])
    textbox(s, 2.00, y + 0.53, 10.4, 0.33, [(kicker.upper(), 14, True, GRAY)])
    y += 1.12
banner(s, 0.85, 6.28, 11.65, 0.62,
       "Một câu: giữ hạt nhân khoa học — thay toàn bộ control plane.", size=20)

# ═══════════════════════════════════════════════════ 3. EXEC SUMMARY
s = S("Tóm tắt điều hành")
box(s, 0.60, 1.15, 12.13, 1.32,
    [("“Giữ và cô lập scientific kernel cùng deterministic analysis contract;", 22, True, WHITE),
     ("thay thế toàn bộ control plane bằng một harness stateful, typed, provenance-first —", 22, True, WHITE),
     ("trong đó agent loop được thuê từ OpenCode hoặc DSH thay vì tự viết.”", 22, True, WHITE)],
    fill=DARK)
cards = [
    (0.60, "1", "Ranh giới “chỉ giữ /predict”\nlà quá hẹp", RED,
     ["Thứ cần giữ là scientific kernel:", "SMILES, model registry, threshold,",
      "OOD, verdict — và /analyze.", "",
      "Bỏ /analyze = đẩy invariant khoa học", "vào orchestration không xác định."]),
    (4.62, "2", "Gần như toàn bộ agent /\ncontrol plane nên viết lại", ORANGE,
     ["ADK, god module, planner if/elif,", "session in-memory, SSE từ call stack,",
      "context phẳng cắt theo ký tự.", "",
      "Đây là phần đang tạo nợ —", "không phải phần tạo giá trị."]),
    (8.64, "3", "Đích: modular monolith,\nhai làn, một kernel", GREEN,
     ["Làn A deterministic để audit.", "Làn B agent loop cho hỏi đáp,",
      "chạy trên OpenCode hoặc DSH.", "",
      "ToxAgent vẫn sở hữu tool plane,", "session, rules và provenance."]),
]
for x, num, head, color, body in cards:
    box(s, x, 2.66, 4.09, 3.30, [], fill=LGRAY, line=color, line_w=1.75)
    box(s, x + 0.20, 2.86, 0.62, 0.62, [(num, 26, True, WHITE)], fill=color)
    textbox(s, x + 0.95, 2.88, 3.00, 0.90,
            [(l, 17, True, color) for l in head.split("\n")], space_after=1)
    textbox(s, x + 0.24, 3.88, 3.65, 2.00,
            [((l or " "), (16 if l else 7), False, DARK) for l in body], space_after=5)
banner(s, 0.60, 6.16, 12.13, 0.66,
       "Không bắt đầu bằng multi-agent, plugin framework, code execution hay một model loop tự viết.",
       size=19, fill=RED)

# ═══════════════════════════════════════════════════ 4. FIRST SLICE
s = S("Nếu chỉ chọn một đường để bắt đầu")
steps = [
    ("1", "Freeze /analyze", "contract + golden test"),
    ("2", "Extract\nToxicologyAnalyzer", "ra khỏi FastAPI"),
    ("3", "Tool gọi\nin-process", "bỏ self-HTTP"),
    ("4", "Session /\nmessage / part", "store bền vững"),
]
steps2 = [
    ("5", "ToxAgent MCP +\nRuntimeGateway", "OpenCode / DSH"),
    ("6", "Migrate\nfrontend", "unified SSE"),
    ("7", "Xoá ADK và\n/agent/* cũ", "deprecation window"),
]
x = 0.62
for num, head, sub in steps:
    box(s, x, 1.30, 2.86, 1.85, [], fill=PALEBG, line=BLUE)
    box(s, x + 0.14, 1.44, 0.52, 0.52, [(num, 22, True, WHITE)], fill=BLUE)
    textbox(s, x + 0.14, 2.06, 2.58, 0.68,
            [(l, 18, True, DARK) for l in head.split("\n")], space_after=0)
    textbox(s, x + 0.14, 2.74, 2.58, 0.30, [(sub, 14, False, GRAY)])
    x += 3.06
arrow(s, 0.62, 3.32, 11.86, 0.26, color=MGRAY)
x = 1.40
for num, head, sub in steps2:
    box(s, x, 3.80, 3.30, 1.85, [], fill=PALEGR, line=GREEN)
    box(s, x + 0.16, 3.94, 0.52, 0.52, [(num, 22, True, WHITE)], fill=GREEN)
    textbox(s, x + 0.16, 4.56, 3.00, 0.68,
            [(l, 18, True, DARK) for l in head.split("\n")], space_after=0)
    textbox(s, x + 0.16, 5.24, 3.00, 0.30, [(sub, 14, False, GRAY)])
    x += 3.60
banner(s, 0.62, 5.88, 11.86, 0.82,
       "Đây là con đường ngắn nhất để “đập đi xây lại” mà không đập luôn phần khoa học đáng giữ nhất.",
       size=20)

# ═══════════════════════════════════════════════════ DIVIDER 1
D.section("PHẦN 1", "Harness là gì",
          "Định nghĩa, bảy trách nhiệm, và hai runtime sẽ cho ToxAgent thuê agent loop")

# ═══════════════════════════════════════════════════ 5. DEFINITION
s = S("Harness là gì")
box(s, 0.60, 1.18, 12.13, 1.05,
    [("Lớp phần mềm bao quanh một LLM, biến nó từ một hàm sinh văn bản", 22, True, WHITE),
     ("thành một hệ thống thực thi có trạng thái, có công cụ và có ràng buộc.", 22, True, WHITE)],
    fill=DARK)
box(s, 0.60, 2.52, 5.90, 1.65,
    [("MODEL quyết định", 15, True, GRAY),
     ("nên làm gì tiếp theo", 26, True, BLUE)], fill=PALEBG, line=BLUE)
box(s, 6.83, 2.52, 5.90, 1.65,
    [("HARNESS quyết định", 15, True, GRAY),
     ("điều gì được phép xảy ra, ghi lại thế nào,", 18, True, ORANGE),
     ("và cái gì vào context lần sau", 18, True, ORANGE)], fill=PALEOR, line=ORANGE)
textbox(s, 0.60, 4.42, 12.13, 0.50,
        [("Ví von: model là động cơ; harness là khung xe, hộp số, phanh và bảng đồng hồ.",
          21, True, DARK)], align=PP_ALIGN.CENTER)
box(s, 0.60, 5.05, 12.13, 1.10,
    [("Động cơ mạnh không tự tạo ra một chiếc xe lái được.", 22, True, WHITE),
     ("ToxAgent đang có động cơ tốt — và đang thiếu phần còn lại của chiếc xe.", 19, False, PALE)],
    fill=BLUE)

# ═══════════════════════════════════════════════════ 6. SEVEN DUTIES
s = S("Bảy trách nhiệm của một harness")
duties = [
    ("1", "Agent loop", "model → tool → observation → model", BLUE),
    ("2", "Tool plane", "registry, schema, timeout, typed error", BLUE),
    ("3", "Context management", "assembly, budget, projection, compaction", CYAN),
    ("4", "State & persistence", "session/message/part, resume, audit", CYAN),
    ("5", "Policy & enforcement", "auth, quota, invariant — bằng code", ORANGE),
    ("6", "Lifecycle & observability", "hooks, usage/cost, tracing", ORANGE),
    ("7", "Interface & streaming", "event feed bền vững, cancel, reconnect bằng Last-Event-ID", GREEN),
]
x, y = 0.62, 1.28
for i, (num, name, sub, color) in enumerate(duties):
    w = 3.90 if i < 6 else 12.10
    box(s, x, y, w, 1.20, [], fill=LGRAY, line=color)
    box(s, x + 0.14, y + 0.14, 0.46, 0.46, [(num, 19, True, WHITE)], fill=color)
    textbox(s, x + 0.70, y + 0.14, w - 0.82, 0.48, [(name, 18.5, True, DARK)])
    textbox(s, x + 0.14, y + 0.70, w - 0.30, 0.40, [(sub, 15, False, GRAY)])
    if i < 6:
        x += 4.10
        if i % 3 == 2:
            x = 0.62; y += 1.38
    else:
        pass
banner(s, 0.62, 5.68, 12.10, 1.12,
       "Bảy quyết định này vẫn đang được ra trong ToxAgent hôm nay — chỉ là rải rác, ẩn danh và không ai sở hữu.",
       size=20)

# ═══════════════════════════════════════════════════ 7. HARNESS VS ...
s = S("Harness khác gì với các khái niệm lân cận")
data = [
    ["Khái niệm", "Là gì", "KHÔNG phải là gì"],
    ["Model / LLM", "Hàm sinh token, có thể phát tool-call request",
     "Không có state, không có quyền, không tự thực thi"],
    ["Agent", "Một cấu hình: prompt + tool surface + policy + model route",
     "Không phải runtime — nhiều agent chạy trên cùng harness"],
    ["Framework\nLangGraph, CrewAI, ADK", "Thư viện để TỰ LẮP một loop",
     "Không chạy được ngay — vẫn phải tự làm cả bảy trách nhiệm"],
    ["Harness\nOpenCode, DSH, Claude Code", "Runtime hoàn chỉnh đã có sẵn bảy trách nhiệm",
     "Không chứa domain logic của bạn"],
    ["MCP", "Giao thức mô tả và gọi tool GIỮA các process",
     "Không phải harness; không làm bus nội bộ cùng process"],
]
table(s, 0.55, 1.22, 12.23, 3.85, data, col_w=[2.6, 4.6, 4.8],
      font_size=16, bold_first_col=True, row_h=[0.44] + [0.66] * 5)
box(s, 0.55, 5.35, 12.23, 1.35,
    [("Thay ADK bằng LangGraph / CrewAI = đổi framework này lấy framework khác.", 21, True, WHITE),
     ("Dùng OpenCode hoặc DSH = thuê nguyên một harness đã có sẵn cả bảy trách nhiệm.", 21, True, PALE)],
    fill=DARK)

# ═══════════════════════════════════════════════════ 8. OPENCODE
s = S("OpenCode — harness mã nguồn mở, kiến trúc client / server")
picture(s, os.path.join(IMG, "gh_opencode.png"), 0.62, 1.22, w=6.10)
textbox(s, 0.62, 4.42, 6.10, 0.34,
        [("Ảnh chụp GitHub · 2026-09-03", 13, False, GRAY)])
items = [
    ("Headless server", True), ("  opencode serve · HTTP API có OpenAPI 3.1 · SSE"),
    ("Session model", True), ("  session → message → part, typed state, event là change feed"),
    ("Agents & permissions", True), ("  mode primary/subagent, steps cap, deny theo tool pattern"),
    ("MCP client", True), ("  kết nối MCP server local và remote"),
]
lines = []
for it in items:
    if isinstance(it, tuple):
        lines.append(("\u2022 " + it[0], 17.5, True, DARK))
    else:
        lines.append((it, 15.5, False, GRAY))
textbox(s, 7.00, 1.28, 5.75, 3.30, lines, space_after=2)
box(s, 7.00, 4.62, 5.75, 2.05,
    [("Vì sao dùng được cho ToxAgent", 17, True, WHITE),
     ("Chính kiến trúc client/server khiến nó dùng được", 17, False, PALE),
     ("làm backend runtime cho một ứng dụng khác —", 17, False, PALE),
     ("không chỉ cho TUI của chính nó.", 17, False, PALE)],
    fill=BLUE, align=PP_ALIGN.LEFT)
textbox(s, 0.62, 4.85, 6.10, 1.85,
        [("Bề mặt API ToxAgent sẽ dùng", 18, True, DARK),
         ("POST /session", 16, False, BLUE),
         ("GET  /event                      (SSE)", 16, False, BLUE),
         ("POST /session/{id}/prompt_async", 16, False, BLUE),
         ("POST /session/{id}/abort         (cancel)", 16, False, BLUE)],
        space_after=3, font=FONT)

# ═══════════════════════════════════════════════════ 9. DSH
s = S("DeepSeek Harness (DSH) — “Everything is a Plugin”")
picture(s, os.path.join(IMG, "gh_dsh.png"), 0.62, 1.22, w=6.10)
textbox(s, 0.62, 4.42, 6.10, 0.34,
        [("Ảnh chụp GitHub · 2026-09-03", 13, False, GRAY)])
lines = []
for head, sub in [
    ("Plugin runtime", "Cordis dependency graph, profile/patch YAML, fail-loud config"),
    ("Python SDK", "spawn runtime qua JSON-RPC stdio, nhận events + final response"),
    ("MCP client", "stdio và Streamable HTTP, namespace mcp__<server>__<tool>"),
    ("Tool restriction", "áp dụng ở CẢ prompt lẫn execution"),
]:
    lines.append(("\u2022 " + head, 17.5, True, DARK))
    lines.append(("  " + sub, 15.5, False, GRAY))
textbox(s, 7.00, 1.28, 5.75, 3.30, lines, space_after=2)
box(s, 7.00, 4.62, 5.75, 2.05,
    [("Giới hạn phải biết trước khi tích hợp", 17, True, WHITE),
     ("Chưa có mid-turn cancel ở SDK → adapter phải công bố", 16, False, WHITE),
     ("cancel_turn = false. Vượt deadline thì đóng worker và", 16, False, WHITE),
     ("ghi runtime.turn.failed — không giả lập “cancel thành công”.", 16, False, WHITE)],
    fill=RED, align=PP_ALIGN.LEFT)
box(s, 0.62, 4.85, 6.10, 1.82,
    [("Điểm mạnh: mức độ tuỳ biến rất sâu.", 18, True, DARK),
     ("Đánh đổi: bề mặt cấu hình lớn hơn", 18, False, GRAY),
     ("và wire protocol còn pre-release churn", 18, False, GRAY),
     ("→ bắt buộc pin version.", 18, True, ORANGE)],
    fill=PALEOR, line=ORANGE, align=PP_ALIGN.LEFT)

# ═══════════════════════════════════════════════════ 10. COMPARE
s = S("Chọn OpenCode hay DSH cho việc gì")
data = [
    ["Nhu cầu", "OpenCode", "DSH", "Khuyến nghị"],
    ["Custom frontend gọi programmatically", "HTTP server, OpenAPI 3.1, SSE",
     "SDK subprocess, JSON-RPC stdio", "OpenCode"],
    ["Cancel một turn", "POST /session/:id/abort", "SDK chưa có prompt-cancel", "OpenCode"],
    ["Worker Python / batch / eval", "Gọi HTTP được", "Python SDK trực tiếp", "DSH"],
    ["Custom composition sâu", "Config / plugin được", "Cordis profile/patch rất linh hoạt", "DSH nếu cần"],
    ["Maturity của embedded wire", "HTTP API dễ tích hợp", "SDK/wire còn churn", "OpenCode"],
    ["Model route chỉ có ở một runtime", "Phụ thuộc auth đã kết nối", "Phụ thuộc profile đã kết nối",
     "Runtime có route thắng"],
]
table(s, 0.55, 1.22, 12.23, 3.55, data, col_w=[3.5, 3.2, 3.2, 2.3],
      font_size=15.5, bold_first_col=True, row_h=[0.42] + [0.52] * 6)
y = 5.05
for x, w, head, body, color in [
    (0.55, 3.95, "OpenCode primary", "cho ToxAgent custom web / chat", BLUE),
    (4.69, 3.95, "DSH primary", "khi model route chỉ có trong DSH profile", ORANGE),
    (8.83, 3.95, "DSH worker", "cho batch experiment, replay, evaluation", GREEN),
]:
    box(s, x, y, w, 0.98, [(head, 20, True, WHITE), (body, 15, False, WHITE)], fill=color)
banner(s, 0.55, 6.18, 12.23, 0.62,
       "Cả hai chạy cùng một test suite — KHÔNG gọi cả hai để ensemble mọi câu trả lời.",
       size=19, fill=DARK)

# ═══════════════════════════════════════════════════ 11. TWO MODEL LAYERS
s = S("Hai lớp model không được nhầm lẫn")
box(s, 0.60, 1.20, 12.13, 0.78,
    [("OpenCode và DSH KHÔNG phải nguồn LLM budget — chúng là runtime.", 22, True, WHITE)],
    fill=RED)
box(s, 0.60, 2.22, 5.90, 2.30, [], fill=PALEBG, line=BLUE, line_w=2)
textbox(s, 0.85, 2.42, 5.40, 0.45, [("LỚP 1", 15, True, GRAY)])
textbox(s, 0.85, 2.78, 5.40, 0.55, [("ToxAgent scientific models", 22, True, BLUE)])
textbox(s, 0.85, 3.38, 5.40, 1.00,
        [("Model dự đoán độc tính của dự án.", 17, False, DARK),
         ("Chạy deterministic — không phải LLM.", 17, False, DARK),
         ("Tài sản khoa học: versioned, reproducible.", 17, False, DARK)], space_after=3)
box(s, 6.83, 2.22, 5.90, 2.30, [], fill=PALEOR, line=ORANGE, line_w=2)
textbox(s, 7.08, 2.42, 5.40, 0.45, [("LỚP 2", 15, True, GRAY)])
textbox(s, 7.08, 2.78, 5.40, 0.55, [("OpenCode / DSH model route", 22, True, ORANGE)])
textbox(s, 7.08, 3.38, 5.40, 1.00,
        [("LLM để hiểu câu hỏi, chọn tool, viết câu trả lời.", 17, False, DARK),
         ("Budget thực tế nằm ở provider phía sau runtime,", 17, False, DARK),
         ("không nằm ở runtime.", 17, False, DARK)], space_after=3)
box(s, 0.60, 4.75, 12.13, 1.95,
    [("Hệ quả vận hành", 19, True, WHITE),
     ("• KHÔNG chuyển credential / OAuth token từ OpenCode hay DSH sang code ToxAgent.", 18, False, WHITE),
     ("• Mỗi runtime tiếp tục sở hữu credential, refresh và wire protocol của nó.", 18, False, WHITE),
     ("• Nếu budget đến từ subscription cá nhân → chỉ local dev và internal eval,", 18, False, WHITE),
     ("   cho tới khi điều khoản provider xác nhận backend automation được phép.", 18, False, WHITE)],
    fill=DARK, align=PP_ALIGN.LEFT)

# ═══════════════════════════════════════════════════ 12. PRINCIPLES
s = S("Mười nguyên tắc thiết kế")
prin = [
    ("1", "Khoa học là hạt nhân, không phải plugin"),
    ("2", "Hai làn, một kernel — làn A không LLM"),
    ("3", "Enforcement nằm ngoài prompt"),
    ("4", "State là source of truth, event chỉ là feed"),
    ("5", "Mọi con số phải truy nguyên được"),
    ("6", "Model chỉ nhìn thấy projection"),
    ("7", "Tool surface nhỏ và có capability profile"),
    ("8", "Runtime là thứ thay thế được"),
    ("9", "Mọi thứ đều versioned"),
    ("10", "Fail loud, không fallback âm thầm"),
]
x, y = 0.62, 1.30
for i, (num, text) in enumerate(prin):
    color = [BLUE, CYAN, ORANGE, GREEN, RED][i % 5]
    box(s, x, y, 5.95, 0.92, [], fill=LGRAY, line=color)
    box(s, x + 0.13, y + 0.16, 0.60, 0.60, [(num, 21, True, WHITE)], fill=color)
    textbox(s, x + 0.86, y + 0.22, 4.95, 0.55, [(text, 18, True, DARK)],
            anchor=MSO_ANCHOR.MIDDLE)
    if i % 2 == 0:
        x = 6.76
    else:
        x = 0.62; y += 1.06

# ═══════════════════════════════════════════════════ DIVIDER 2
D.section("PHẦN 2", "ToxAgent hôm nay",
          "Sáu sản phẩm trong một repo, bảy chẩn đoán kiến trúc")

# ═══════════════════════════════════════════════════ 13. SIX PRODUCTS
s = S("ToxAgent không phải “một agent” — đang chứa sáu sản phẩm")
prods = [
    ("ML inference platform", "Nhiều backend/model, ensemble, binary toxicity, Tox21", BLUE),
    ("Scientific analysis API", "/predict, /explain, /analyze, batch, OOD", BLUE),
    ("Compound input utilities", "SMILES validation, canonicalization, image-to-SMILES", CYAN),
    ("Evidence platform", "PubChem, PubMed, Europe PMC, Semantic Scholar", CYAN),
    ("MolRAG / read-across", "Fingerprint, similar-molecule retrieval, fusion", GREEN),
    ("Report application", "Report projection, evidence QA, grounded chat, export", GREEN),
]
x, y = 0.62, 1.28
for i, (name, sub, color) in enumerate(prods):
    box(s, x, y, 5.95, 1.18, [], fill=LGRAY, line=color)
    box(s, x, y, 0.13, 1.18, [], fill=color, radius=False)
    textbox(s, x + 0.30, y + 0.16, 5.50, 0.44, [(name, 20, True, DARK)])
    textbox(s, x + 0.30, y + 0.66, 5.50, 0.44, [(sub, 15, False, GRAY)])
    if i % 2 == 0:
        x = 6.76
    else:
        x = 0.62; y += 1.32
box(s, 0.62, 5.30, 12.09, 1.42,
    [("Vấn đề không phải thiếu feature.", 22, True, WHITE),
     ("Sáu sản phẩm này chưa có boundary rõ — nên model server đồng thời làm API, model registry,", 18, False, PALE),
     ("workflow engine, chat harness, tool dispatcher, state recovery, rendering và SSE.", 18, False, PALE)],
    fill=DARK)

# ═══════════════════════════════════════════════════ 14. GOD MODULE + DIAGNOSES
s = S("Bảy chẩn đoán — đo trên code hôm nay")
for x, big, small, color in [
    (0.62, "6.000+", ["dòng trong", "model_server/main.py"], RED),
    (3.72, "2", ["runtime chồng nhau", "ADK + deterministic"], ORANGE),
    (6.82, "0", ["bản ghi bền vững", "sau khi process restart"], RED),
    (9.92, "len//4", ["cách đang ước lượng", "token cho quyết định cắt"], ORANGE),
]:
    stat_card(s, x, 1.22, 2.80, 1.50, big, small, big_color=color)
data = [
    ["#", "Chẩn đoán", "Biểu hiện trong code"],
    ["1", "God module", "main.py vừa load model, vừa route, vừa chat, vừa render, vừa SSE"],
    ["2", "“Agent layer” thực ra là workflow stage", "Screening / Researcher / EvidenceQA / Writer là hàm deterministic"],
    ["3", "Hai runtime chồng lên nhau", "adk_available, runtime_mode, state_keys lộ ra public response"],
    ["4", "Chat state không bền vững", "_SESSION_STORE in-memory + client gửi lại report_state"],
    ["5", "Tool plane chưa là một plane", "dispatch if/elif; tool gọi HTTP ngược vào chính process"],
    ["6", "Context sửa ở sai tầng", "chuỗi phẳng, cắt theo ký tự, vá lỗi bằng hậu xử lý chuỗi"],
    ["7", "Drift giữa code, config và docs", "workspace_mode.yaml, README và docs không đồng thuận"],
]
table(s, 0.62, 2.95, 12.09, 3.35, data, col_w=[0.5, 4.4, 7.3],
      font_size=15.5, bold_first_col=True, row_h=[0.40] + [0.42] * 7,
      align_first=PP_ALIGN.CENTER)
banner(s, 0.62, 6.40, 12.09, 0.48,
       "Thêm một harness mới trực tiếp vào file này sẽ tạo lớp orchestration thứ ba.",
       size=18, fill=RED)

# ═══════════════════════════════════════════════════ 15. AGENTS THAT AREN'T
s = S("“Agent” trong ToxAgent thực ra là workflow stage")
picture(s, os.path.join(IMG, "pipeline_crop.png"), 0.62, 1.35, w=6.35)
textbox(s, 0.62, 4.62, 6.35, 0.34,
        [("Ảnh chụp UI ToxAgent v0.0.6 — pipeline hiện tại", 13.5, False, GRAY)])
rows = [
    ("ScreeningAgent", "gọi analysis + optional MolRAG"),
    ("ResearcherAgent", "chạy các provider lookup / search"),
    ("EvidenceQAAgent", "dedupe, chấm relevance, gắn cờ"),
    ("WriterAgent", "chiếu state thành report có cấu trúc"),
]
y = 1.28
for name, what in rows:
    box(s, 7.25, y, 5.50, 0.86, [], fill=LGRAY, line=ORANGE)
    textbox(s, 7.42, y + 0.10, 5.16, 0.38, [(name, 19, True, DARK)])
    textbox(s, 7.42, y + 0.48, 5.16, 0.32, [(what, 15, False, GRAY)])
    y += 0.98
box(s, 7.25, 5.28, 5.50, 1.42,
    [("Không cần identity, memory và agent loop riêng.", 18, True, WHITE),
     ("Giữ chúng dưới dạng agent làm tăng prompt/runtime", 16, False, PALE),
     ("surface nhưng không tạo thêm autonomy hữu ích.", 16, False, PALE)],
    fill=DARK, align=PP_ALIGN.LEFT)

# ═══════════════════════════════════════════════════ 16. MISSING DUTIES
s = S("Đối chiếu: ToxAgent đang đặt bảy trách nhiệm ở đâu")
data = [
    ["Trách nhiệm harness", "ToxAgent hiện tại đặt ở đâu", "Hệ quả"],
    ["Agent loop", "ADK + nhánh deterministic fallback", "Hai runtime semantics cho cùng use case"],
    ["Tool plane", "if/elif trong main.py; tool tự gọi HTTP", "Không lifecycle, không typed error"],
    ["Context management", "chuỗi phẳng, cắt ký tự, len//4", "Mất anchor bằng chứng, không kiểm soát budget"],
    ["State & persistence", "_SESSION_STORE + client rehydration", "Không chịu được restart hay multi-instance"],
    ["Policy & enforcement", "phần lớn nằm trong prompt", "Không cưỡng chế được, không audit được"],
    ["Lifecycle & observability", "rải rác, không thống nhất", "Không đo được cost/latency theo tool"],
    ["Interface & streaming", "SSE sinh trực tiếp từ call stack", "Mất event là mất trạng thái"],
]
table(s, 0.55, 1.25, 12.23, 4.30, data, col_w=[3.3, 4.6, 4.3],
      font_size=16, bold_first_col=True, row_h=[0.46] + [0.55] * 7)
box(s, 0.55, 5.75, 12.23, 0.98,
    [("Mỗi chẩn đoán ở trên là một trong bảy trách nhiệm đang bị thiếu hoặc đặt sai chỗ —", 20, True, WHITE),
     ("không phải bảy lỗi rời rạc.", 20, True, PALE)],
    fill=DARK)

# ═══════════════════════════════════════════════════ DIVIDER 3
D.section("PHẦN 3", "Giữ gì, xoá gì — và kiến trúc đích",
          "Ranh giới giữa tài sản khoa học và nợ kiến trúc")

# ═══════════════════════════════════════════════════ 17. FOUR DECISIONS
s = S("Bốn loại quyết định")
quads = [
    (0.62, 1.22, "GIỮ CONTRACT", GREEN,
     "Bên ngoài tiếp tục thấy hành vi tương thích; bên trong vẫn được refactor.",
     ["/predict", "/analyze", "/explain", "Clinical + Tox21", "OOD assessment"]),
    (6.76, 1.22, "GIỮ LOGIC, BỌC LẠI", BLUE,
     "Thuật toán còn đúng nhưng module/API hiện tại không còn là boundary.",
     ["RDKit canonicalization", "PubChem/PubMed providers", "MolRAG retrieval + fusion",
      "Evidence QA → validator", "Writer → ReportBuilder"]),
    (0.62, 4.00, "VIẾT MỚI", ORANGE,
     "Không cố cứu kiến trúc runtime cũ; chỉ viết adapter migration khi cần.",
     ["Session/message/part store", "Tool registry + runner", "Context assembler",
      "AgentRuntimeGateway", "Unified SSE"]),
    (6.76, 4.00, "XOÁ", RED,
     "Không mang pattern hoặc contract này sang kiến trúc đích.",
     ["ADK declarations + compat", "_SESSION_STORE", "report_state rehydration",
      "planner if/elif + normalizers", "Self-HTTP trong cùng process"]),
]
for x, y, head, color, sub, items in quads:
    box(s, x, y, 5.95, 2.68, [], fill=LGRAY, line=color, line_w=1.75)
    box(s, x, y, 5.95, 0.52, [(head, 19, True, WHITE)], fill=color)
    textbox(s, x + 0.20, y + 0.60, 5.55, 0.50, [(sub, 14.5, False, GRAY)])
    textbox(s, x + 0.20, y + 1.14, 5.55, 1.48,
            [("\u2022 " + i, 15.5, False, DARK) for i in items], space_after=2)

# ═══════════════════════════════════════════════════ 18. /analyze BOUNDARY
s = S("Vì sao ranh giới phải là /analyze, không phải /predict")
box(s, 0.62, 1.22, 5.90, 0.62, [("NẾU CHỈ GIỮ /predict", 20, True, WHITE)], fill=RED)
textbox(s, 0.75, 1.98, 5.65, 2.55,
        [("Harness mới sẽ phải TỰ ghép:", 18, True, DARK),
         ("• clinical prediction", 17, False, DARK),
         ("• Tox21 mechanism", 17, False, DARK),
         ("• threshold / calibration policy", 17, False, DARK),
         ("• OOD gating", 17, False, DARK),
         ("• explanation gating", 17, False, DARK),
         ("• phép tổng hợp tạo final_verdict", 17, False, DARK)], space_after=5)
box(s, 0.62, 4.62, 5.90, 1.05,
    [("→ Logic khoa học rơi vào", 18, True, WHITE),
     ("orchestration KHÔNG xác định", 18, True, WHITE)], fill=RED)
box(s, 6.83, 1.22, 5.90, 0.62, [("GIỮ /analyze LÀM BOUNDARY", 20, True, WHITE)], fill=GREEN)
textbox(s, 6.96, 1.98, 5.65, 2.55,
        [("Một SMILES → một AnalysisResult đầy đủ:", 18, True, DARK),
         ("• reproducible, versioned", 17, False, DARK),
         ("• canonical SMILES + model hash", 17, False, DARK),
         ("• threshold policy version", 17, False, DARK),
         ("• OOD status bắt buộc", 17, False, DARK),
         ("• correlation / run ID", 17, False, DARK),
         ("• benchmark bám vào một contract duy nhất", 17, False, DARK)], space_after=5)
box(s, 6.83, 4.62, 5.90, 1.05,
    [("→ Scientific kernel là", 18, True, WHITE),
     ("một application service in-process", 18, True, WHITE)], fill=GREEN)
banner(s, 0.62, 5.90, 12.11, 0.80,
       "/analyze đóng gói nhiều scientific invariant nhất — nên nó là ranh giới của kernel.",
       size=21)

# ═══════════════════════════════════════════════════ 19. TARGET ARCHITECTURE
s = S("Kiến trúc đích: hai làn, một scientific kernel")
# column 1 — entry
box(s, 0.55, 1.30, 2.45, 0.62, [("Web UI / API client", 16, True, WHITE)], fill=DARK)
box(s, 0.55, 2.08, 2.45, 0.62, [("FastAPI + identity", 16, True, WHITE)], fill=DARK)
box(s, 0.55, 2.86, 2.45, 1.10,
    [("Deterministic", 16, True, WHITE), ("lane router", 16, True, WHITE)], fill=BLUE)
arrow(s, 1.62, 1.94, 0.30, 0.12, color=MGRAY, direction="down")
arrow(s, 1.62, 2.72, 0.30, 0.12, color=MGRAY, direction="down")
# lane A
box(s, 3.35, 1.30, 4.30, 2.05, [], fill=PALEGR, line=GREEN, line_w=1.75)
textbox(s, 3.50, 1.42, 4.00, 0.40, [("LÀN A — deterministic", 17, True, GREEN)])
textbox(s, 3.50, 1.86, 4.00, 1.40,
        [("• analysis · batch · benchmark", 15, False, DARK),
         ("• KHÔNG gọi LLM", 15, True, RED),
         ("• nền để audit", 15, False, DARK),
         ("• ReportBuilder deterministic", 15, False, DARK)], space_after=3)
# lane B
box(s, 3.35, 3.52, 4.30, 2.35, [], fill=PALEOR, line=ORANGE, line_w=1.75)
textbox(s, 3.50, 3.64, 4.00, 0.40, [("LÀN B — agent runtime", 17, True, ORANGE)])
textbox(s, 3.50, 4.08, 4.00, 1.70,
        [("• context assembler + budget", 15, False, DARK),
         ("• model adapter", 15, False, DARK),
         ("• tool runner + capability filter", 15, False, DARK),
         ("• loop do OpenCode / DSH cấp", 15, True, ORANGE)], space_after=3)
arrow(s, 3.04, 3.02, 0.28, 0.26, color=GREEN)
arrow(s, 3.04, 3.62, 0.28, 0.26, color=ORANGE)
# shared column
for y, title, sub, color in [
    (1.30, "Scientific kernel", "SMILES · model registry · threshold · OOD · verdict", BLUE),
    (2.42, "Research + MolRAG", "PubChem · PubMed · analog retrieval · knowledge", BLUE),
    (3.54, "Observation store", "typed result · projection · provenance", CYAN),
    (4.66, "Session / message / part", "durable state · checkpoint · change feed", CYAN),
]:
    box(s, 8.00, y, 4.78, 0.98, [], fill=LGRAY, line=color)
    textbox(s, 8.16, y + 0.10, 4.46, 0.40, [(title, 18, True, DARK)])
    textbox(s, 8.16, y + 0.52, 4.46, 0.38, [(sub, 13.5, False, GRAY)])
arrow(s, 7.69, 2.10, 0.28, 0.26, color=MGRAY)
arrow(s, 7.69, 4.52, 0.28, 0.26, color=MGRAY)
box(s, 8.00, 5.78, 4.78, 0.55, [("→ SSE ra UI (reconnectable)", 16, True, WHITE)], fill=DARK)
box(s, 0.55, 4.42, 2.45, 1.58, [], fill=LGRAY, line=BLUE)
textbox(s, 0.68, 4.52, 2.20, 1.42,
        [("Router quyết định", 14.5, True, BLUE),
         ("• analysis, batch → A", 12.5, False, DARK),
         ("• follow-up → B", 12.5, False, DARK),
         ("• thiếu input → hỏi lại", 12.5, False, DARK),
         ("Model KHÔNG tự chọn làn", 12.5, True, RED)], space_after=3)
banner(s, 0.55, 6.14, 7.10, 0.70,
       "Modular monolith — không phải microservices bắt buộc.", size=18, fill=BLUE)

# ═══════════════════════════════════════════════════ 20. TOOL PLANE
s = S("Tool plane: 9 tool, lọc theo capability profile")
tools = [
    ("resolve_molecule", "tên / SMILES / ảnh → canonical", BLUE),
    ("run_toxicity_analysis", "chạy analysis deterministic đầy đủ", BLUE),
    ("get_report_section", "projection nhỏ của report", BLUE),
    ("lookup_compound", "metadata từ PubChem", CYAN),
    ("search_toxicology_literature", "tìm evidence có cấu trúc", CYAN),
    ("get_article_detail", "abstract / metadata bài đã chọn", CYAN),
    ("find_similar_molecules", "analog / read-across", GREEN),
    ("lookup_structural_alerts", "alert đã chuẩn hoá", GREEN),
    ("explain_mechanism", "context cơ chế theo endpoint", GREEN),
]
x, y = 0.55, 1.25
for i, (name, sub, color) in enumerate(tools):
    box(s, x, y, 4.02, 0.98, [], fill=LGRAY, line=color)
    textbox(s, x + 0.16, y + 0.10, 3.72, 0.40, [(name, 16, True, DARK)], font=MONO)
    textbox(s, x + 0.16, y + 0.54, 3.72, 0.36, [(sub, 14, False, GRAY)])
    x += 4.14
    if i % 3 == 2:
        x = 0.55; y += 1.10
data = [
    ["Capability profile", "Tool được thấy"],
    ["analysis", "resolve_molecule · run_toxicity_analysis"],
    ["report_qa", "get_report_section · article/evidence tools · analog · mechanism"],
    ["literature_review", "lookup_compound · search_literature · get_article_detail"],
    ["read_across", "get_report_section · analog · structural alert · mechanism"],
]
table(s, 0.55, 4.62, 12.23, 1.55, data, col_w=[3.0, 9.2], font_size=15.5,
      bold_first_col=True, row_h=[0.36] + [0.30] * 4)
banner(s, 0.55, 6.28, 12.23, 0.58,
       "Cấm tool phải làm CẢ HAI: loại schema khỏi prompt, VÀ chặn ở execution layer.",
       size=18, fill=RED)

# ═══════════════════════════════════════════════════ 21. RULES / HOOKS / SKILLS
s = S("Rule, Hook và Skill — ba thứ không được lẫn nhau")
cols = [
    (0.55, "RULE / POLICY", BLUE, "Quyết định deterministic phải LUÔN đúng",
     ["Auth / ownership", "Canonical SMILES", "Lane routing",
      "Allowed tool surface", "Deadline / quota", "Numeric provenance",
      "Citation requirement", "No raw blob in context"],
     "Là code / config — không phải câu “hãy luôn…”"),
    (4.69, "HOOK", ORANGE, "Điểm lifecycle nhỏ để quan sát / cưỡng chế / project",
     ["on_request_admitted", "before_model_call", "after_model_call",
      "before_tool_call", "after_tool_call", "before_compaction",
      "before_turn_commit", "on_run_failed"],
     "Fixed typed chain — không dynamic plugin graph"),
    (8.83, "SKILL", GREEN, "Playbook chuyên ngành, nạp theo nhu cầu",
     ["interpret-clinical-risk", "interpret-tox21-mechanisms", "assess-herg-risk",
      "assess-hepatotoxicity", "interpret-ood", "perform-read-across",
      "review-toxicology-literature", "write-toxicity-report"],
     "Procedural knowledge — KHÔNG cấp thêm quyền tool"),
]
for x, head, color, sub, items, foot in cols:
    box(s, x, 1.22, 3.95, 4.72, [], fill=LGRAY, line=color, line_w=1.75)
    box(s, x, 1.22, 3.95, 0.55, [(head, 20, True, WHITE)], fill=color)
    textbox(s, x + 0.18, 1.86, 3.60, 0.62, [(sub, 15, True, GRAY)])
    textbox(s, x + 0.18, 2.58, 3.60, 2.60,
            [("• " + i, 15.5, False, DARK) for i in items], space_after=4)
    textbox(s, x + 0.18, 5.32, 3.60, 0.55, [(foot, 14, True, color)])
box(s, 0.55, 6.08, 12.23, 0.78,
    [("Skill khi model cần ÁP DỤNG tri thức. Hook/Rule khi hành động phải XẢY RA nhất quán.", 20, True, WHITE)],
    fill=DARK)

# ═══════════════════════════════════════════════════ 22. STATE + PROVENANCE
s = S("State bền vững, compaction và provenance")
textbox(s, 0.55, 1.16, 6.00, 0.42, [("Mô hình state — source of truth", 20, True, DARK)])
chain = [("Session", "owner · status · active_analysis · context_epoch", BLUE),
         ("Message", "role · sequence · model · usage", BLUE),
         ("Part", "text | tool_call | tool_result | citation | error", CYAN),
         ("Observation", "payload_ref · model_projection · provenance", CYAN),
         ("Checkpoint", "summary · pinned_observation_ids · token_count", GREEN)]
y = 1.66
for name, sub, color in chain:
    box(s, 0.55, y, 6.00, 0.80, [], fill=LGRAY, line=color)
    textbox(s, 0.72, y + 0.08, 5.66, 0.36, [(name, 17, True, DARK)])
    textbox(s, 0.72, y + 0.44, 5.66, 0.32, [(sub, 13.5, False, GRAY)])
    y += 0.90
box(s, 0.55, 6.20, 6.00, 0.62,
    [("Event chỉ là feed — không replay JSONL để dựng state", 15.5, True, WHITE)], fill=DARK)
textbox(s, 6.90, 1.16, 5.88, 0.42, [("Chuỗi provenance của một con số", 20, True, DARK)])
prov = [("final numeric / scientific claim", RED),
        ("claim index / citation marker", ORANGE),
        ("observation_id + field path", BLUE),
        ("tool / model run + policy version", GREEN)]
y = 1.70
for text, color in prov:
    box(s, 6.90, y, 5.88, 0.66, [(text, 17, True, WHITE)], fill=color)
    if y < 4.0:
        arrow(s, 9.68, y + 0.70, 0.32, 0.22, color=MGRAY, direction="down")
    y += 1.00
textbox(s, 6.90, 4.92, 5.88, 1.20,
        [("Vi phạm → một lần regenerate có feedback typed.", 16, False, DARK),
         ("Vẫn vi phạm → deterministic safe answer từ report", 16, False, DARK),
         ("projection, kèm warning.", 16, False, DARK)], space_after=3)
box(s, 6.90, 6.20, 5.88, 0.62,
    [("Compaction giảm context — KHÔNG xoá audit history", 15.5, True, WHITE)], fill=DARK)

# ═══════════════════════════════════════════════════ DIVIDER 4
D.section("PHẦN 4", "Thuê agent loop từ OpenCode và DSH",
          "ToxAgent sở hữu domain, tools, state và provenance — runtime cho thuê loop và đường tới LLM")

# ═══════════════════════════════════════════════════ 23. RUNTIME GATEWAY
s = S("Kiến trúc runtime-backed cho làn B")
box(s, 0.55, 1.28, 2.55, 0.66, [("ToxAgent UI / CLI", 16, True, WHITE)], fill=DARK)
box(s, 0.55, 2.10, 2.55, 0.66, [("ToxAgent API", 16, True, WHITE)], fill=DARK)
box(s, 0.55, 2.92, 2.55, 0.80, [("Deterministic", 15.5, True, WHITE), ("lane router", 15.5, True, WHITE)], fill=BLUE)
arrow(s, 1.66, 1.96, 0.32, 0.12, color=MGRAY, direction="down")
arrow(s, 1.66, 2.78, 0.32, 0.12, color=MGRAY, direction="down")
box(s, 0.55, 3.92, 2.55, 0.95,
    [("Làn A", 15, True, WHITE), ("Scientific service", 15, True, WHITE)], fill=GREEN)
box(s, 3.45, 2.92, 3.10, 0.95, [("AgentRuntimeGateway", 17, True, WHITE)], fill=ORANGE)
arrow(s, 3.12, 3.20, 0.30, 0.30, color=ORANGE)
box(s, 3.45, 4.20, 3.10, 0.85,
    [("OpenCode adapter", 15.5, True, DARK), ("HTTP + SSE", 13.5, False, GRAY)],
    fill=PALEBG, line=BLUE)
box(s, 3.45, 5.20, 3.10, 0.85,
    [("DSH adapter", 15.5, True, DARK), ("Python SDK · JSON-RPC stdio", 13.5, False, GRAY)],
    fill=PALEOR, line=ORANGE)
arrow(s, 4.82, 3.92, 0.32, 0.22, color=MGRAY, direction="down")
box(s, 6.90, 4.20, 2.70, 0.85, [("Provider /", 14.5, True, DARK), ("model route", 14.5, True, DARK)],
    fill=LGRAY, line=MGRAY)
box(s, 6.90, 5.20, 2.70, 0.85, [("Provider /", 14.5, True, DARK), ("model route", 14.5, True, DARK)],
    fill=LGRAY, line=MGRAY)
arrow(s, 6.60, 4.52, 0.26, 0.22, color=MGRAY)
arrow(s, 6.60, 5.52, 0.26, 0.22, color=MGRAY)
box(s, 3.45, 1.28, 6.15, 1.28,
    [("Cả hai adapter đăng ký CÙNG một ToxAgent MCP server", 17, True, DARK),
     ("→ cùng tool contract, cùng observation, bất kể runtime nào đang chạy", 15, False, GRAY)],
    fill=PALEBG, line=BLUE)
arrow(s, 4.85, 2.62, 0.30, 0.26, color=BLUE, direction="up")
arrow(s, 9.64, 1.78, 0.26, 0.26, color=BLUE)
box(s, 9.95, 1.28, 2.85, 0.90, [("ToxAgent", 17, True, WHITE), ("MCP server", 17, True, WHITE)], fill=BLUE)
box(s, 9.95, 2.38, 2.85, 0.78, [("Scientific service", 15, True, DARK)], fill=LGRAY, line=BLUE)
box(s, 9.95, 3.32, 2.85, 0.78, [("Research + MolRAG", 15, True, DARK)], fill=LGRAY, line=BLUE)
box(s, 9.95, 4.26, 2.85, 0.78, [("Observation store", 15, True, DARK)], fill=LGRAY, line=CYAN)
box(s, 9.95, 5.20, 2.85, 0.85, [("ToxAgent session /", 14.5, True, WHITE), ("message / part store", 14.5, True, WHITE)], fill=DARK)
arrow(s, 11.20, 2.20, 0.32, 0.14, color=MGRAY, direction="down")
banner(s, 0.55, 6.28, 12.25, 0.62,
       "ToxAgent giữ product session, observation, attachment và provenance — runtime chỉ giữ model context.",
       size=18, fill=DARK)

# ═══════════════════════════════════════════════════ 24. THREE LEVELS
s = S("Ba mức tích hợp — làm theo thứ tự")
levels = [
    ("MỨC A", "MCP-first, dùng UI của harness", BLUE,
     ["Chưa cần sửa frontend ToxAgent", "Chưa cần viết agent loop",
      "So sánh ngay model nào gọi tool tốt hơn", "Tool plane kiểm thử độc lập"],
     "→ developer workflow, internal demo, xây eval dataset"),
    ("MỨC B", "Runtime gateway cho frontend ToxAgent", ORANGE,
     ["Map product session ↔ runtime session", "Chọn và pin runtime + model route",
      "Normalize external events", "Commit final answer sau provenance"],
     "→ kiến trúc mục tiêu khi cần giữ UI hiện tại"),
    ("MỨC C", "Direct LLM provider (tương lai)", GREEN,
     ["Thêm adapter thứ ba: DirectModelRuntime", "Scientific kernel không đổi",
      "MCP tools không đổi", "Frontend không đổi"],
     "→ chỉ thay execution provider cho làn B"),
]
x = 0.55
for head, sub, color, items, foot in levels:
    box(s, x, 1.25, 3.95, 4.55, [], fill=LGRAY, line=color, line_w=1.75)
    box(s, x, 1.25, 3.95, 0.55, [(head, 20, True, WHITE)], fill=color)
    textbox(s, x + 0.18, 1.90, 3.60, 0.75, [(sub, 18, True, DARK)])
    textbox(s, x + 0.18, 2.75, 3.60, 2.20,
            [("• " + i, 15.5, False, DARK) for i in items], space_after=7)
    textbox(s, x + 0.18, 5.10, 3.60, 0.62, [(foot, 14.5, True, color)])
    x += 4.14
banner(s, 0.55, 5.98, 12.25, 0.82,
       "Đầu tư quan trọng nhất là MCP server — một server duy nhất phục vụ cả hai runtime.",
       size=21, fill=DARK)

# ═══════════════════════════════════════════════════ 25. MCP SERVER
s = S("ToxAgent MCP server — điểm đầu tư quan trọng nhất")
picture(s, os.path.join(IMG, "mcp_gh.png"), 7.35, 1.22, w=5.42)
textbox(s, 7.35, 3.98, 5.42, 0.34,
        [("Model Context Protocol · ảnh chụp GitHub 2026-09-03", 13, False, GRAY)])
textbox(s, 0.55, 1.20, 6.55, 0.42, [("MCP server PHẢI", 20, True, GREEN)])
textbox(s, 0.55, 1.66, 6.55, 2.55,
        [("• gọi application service, không import FastAPI handler", 16.5, False, DARK),
         ("• trả structured content + observation ID", 16.5, False, DARK),
         ("• có timeout và typed error", 16.5, False, DARK),
         ("• cấp attachment reference có ACL", 16.5, False, DARK),
         ("• ghi model / artifact / policy / evidence version", 16.5, False, DARK),
         ("• chỉ expose read-only, deterministic capability ở MVP", 16.5, False, DARK)],
        space_after=6)
textbox(s, 0.55, 4.35, 6.55, 0.42, [("MCP server KHÔNG ĐƯỢC", 20, True, RED)])
textbox(s, 0.55, 4.81, 6.55, 1.55,
        [("• trả base64 ảnh vào model output", 16.5, False, DARK),
         ("• expose training, filesystem, shell, arbitrary HTTP fetch", 16.5, False, DARK),
         ("• bắt model tự điền session_id hay bearer token trong tool args", 16.5, False, DARK)],
        space_after=6)
box(s, 7.35, 4.45, 5.42, 2.25,
    [("Authorization context phải đến từ", 16.5, True, WHITE),
     ("transport hoặc runtime binding.", 16.5, True, WHITE),
     (" ", 8, False, None),
     ("analysis_id / report_id là domain input hợp lệ —", 15.5, False, PALE),
     ("bearer token thì không.", 15.5, False, PALE)],
    fill=DARK, align=PP_ALIGN.LEFT)
banner(s, 0.55, 6.50, 6.55, 0.42,
       "Một server duy nhất phục vụ cả OpenCode lẫn DSH", size=16, fill=BLUE)

# ═══════════════════════════════════════════════════ 26. LLM COST
s = S("LLM không tham gia mọi use case")
data = [
    ["Use case", "LLM calls mục tiêu"],
    ["/predict · /predict/batch · /explain · /analyze", "0"],
    ["Build structured report mặc định (ReportBuilder)", "0"],
    ["Evidence QA / provenance validation", "0"],
    ["Tóm tắt report bằng ngôn ngữ tự nhiên", "1"],
    ["Một câu hỏi follow-up đơn giản", "1 – 2"],
    ["Research + synthesis có tool calls", "2 – 4 (hard step cap)"],
    ["Retry do vi phạm provenance", "tối đa 1"],
    ["Multi-agent / reflection ensemble", "0 trong MVP"],
]
table(s, 0.55, 1.22, 7.10, 3.90, data, col_w=[5.0, 2.1], font_size=16,
      bold_first_col=False, row_h=[0.42] + [0.43] * 8)
textbox(s, 8.00, 1.22, 4.80, 0.45, [("Đòn bẩy giảm cost", 20, True, DARK)])
textbox(s, 8.00, 1.72, 4.80, 3.45,
        [("• tool roster nhỏ và ổn định trong session", 16, False, DARK),
         ("• không bật tool coding mặc định", 16, False, DARK),
         ("• projection trước, compaction sau", 16, False, DARK),
         ("• không gửi base64 / raw JSON / literature dump", 16, False, DARK),
         ("• pin system prompt, tool order và model", 16, False, DARK),
         ("• output-token cap và steps cap", 16, False, DARK),
         ("• cache theo canonical SMILES + policy version", 16, False, DARK),
         ("• reuse observation thay vì gọi lại tool", 16, False, DARK),
         ("• deterministic fallback thay model reflection", 16, False, DARK)],
        space_after=7)
box(s, 0.55, 5.32, 12.25, 1.42,
    [("Phải đo input · cache read/write · output · reasoning và cost theo TỪNG runtime / model route", 19, True, WHITE),
     ("TRƯỚC khi mở rộng tool catalog — tool schema từng chiếm phần lớn static prefix của session mẫu.", 18, False, PALE)],
    fill=DARK)

# ═══════════════════════════════════════════════════ DIVIDER 5
D.section("PHẦN 5", "Lộ trình, rủi ro và quyết định",
          "Tám giai đoạn S0–S7, mỗi giai đoạn có điều kiện thoát đo được")

# ═══════════════════════════════════════════════════ 27. ROADMAP
s = S("Lộ trình hợp nhất — S0 đến S7")
stages = [
    ("S0", "Đóng băng baseline\n+ inventory budget", BLUE),
    ("S1", "Tách scientific\nkernel", BLUE),
    ("S2", "Tool/observation plane\n+ MCP tối thiểu", CYAN),
    ("S3", "Mở research tools\n+ eval hai runtime", CYAN),
]
stages2 = [
    ("S4", "Session bền vững\n+ unified SSE", ORANGE),
    ("S5", "AgentRuntimeGateway\n+ frontend migration", ORANGE),
    ("S6", "Skills, compaction,\nenforcement", GREEN),
    ("S7", "Dọn runtime cũ\n+ quyết định production", GREEN),
]
x = 0.58
for code, text, color in stages:
    box(s, x, 1.32, 2.92, 1.72, [], fill=LGRAY, line=color, line_w=1.75)
    box(s, x, 1.32, 2.92, 0.48, [(code, 19, True, WHITE)], fill=color)
    textbox(s, x + 0.14, 1.92, 2.64, 1.00,
            [(l, 16, True, DARK) for l in text.split("\n")],
            align=PP_ALIGN.CENTER, space_after=1)
    x += 3.06
arrow(s, 0.58, 3.20, 11.86, 0.24, color=MGRAY)
x = 0.58
for code, text, color in stages2:
    box(s, x, 3.62, 2.92, 1.72, [], fill=LGRAY, line=color, line_w=1.75)
    box(s, x, 3.62, 2.92, 0.48, [(code, 19, True, WHITE)], fill=color)
    textbox(s, x + 0.14, 4.22, 2.64, 1.00,
            [(l, 16, True, DARK) for l in text.split("\n")],
            align=PP_ALIGN.CENTER, space_after=1)
    x += 3.06
box(s, 0.58, 5.52, 5.90, 1.28,
    [("S4 chỉ phụ thuộc S1", 18, True, WHITE),
     ("→ có thể chạy SONG SONG với S2–S3 nếu đủ người", 16, False, PALE)],
    fill=BLUE)
box(s, 6.54, 5.52, 5.90, 1.28,
    [("Strangler, không big-bang", 18, True, WHITE),
     ("Endpoint cũ chạy qua adapter; golden test đi trước refactor", 16, False, PALE)],
    fill=ORANGE)

# ═══════════════════════════════════════════════════ 28. EXIT CRITERIA
s = S("Điều kiện thoát của từng giai đoạn")
data = [
    ["Giai đoạn", "Điều kiện thoát (đo được)"],
    ["S0", "Cùng input + policy + artifact cho output trong tolerance; biết budget nằm ở provider nào"],
    ["S1", "/predict, /explain, /analyze giữ contract; unit test kernel chạy không cần FastAPI"],
    ["S2", "Mọi tool có schema/timeout/typed error; cả hai runtime gọi cùng contract, không bịa số"],
    ["S3", "Tool catalog chỉ tăng khi eval chứng minh cần thiết; có số liệu cost theo runtime"],
    ["S4", "Restart hoặc đổi instance vẫn resume được; bỏ được report_state từ client"],
    ["S5", "Không cần ADK để analyze/chat; cùng frontend API chạy trên hai runtime"],
    ["S6", "Session dài không mất evidence anchors; provenance violation dưới ngưỡng"],
    ["S7", "Một control plane duy nhất; đã chốt primary runtime cho deployment"],
]
table(s, 0.55, 1.25, 12.25, 4.55, data, col_w=[1.6, 10.6], font_size=16,
      bold_first_col=True, row_h=[0.44] + [0.51] * 8, align_first=PP_ALIGN.CENTER)
box(s, 0.55, 6.00, 12.25, 0.82,
    [("Không chuyển giai đoạn bằng cảm tính. Shadow → warn → enforce cho mọi validator mới.", 20, True, WHITE)],
    fill=DARK)

# ═══════════════════════════════════════════════════ 29. RISKS
s = S("Rủi ro cao nhất và cách giảm")
data = [
    ["Rủi ro", "Tác động", "Cách giảm"],
    ["Rewrite làm đổi semantics model", "Rất cao", "Contract / golden test TRƯỚC refactor"],
    ["External evidence provider không ổn định", "Cao", "Provider interface, cache, typed degradation"],
    ["Context summary làm mất uncertainty", "Cao", "Pin observation / citation / policy; verify checkpoint"],
    ["User memory gây contamination khoa học", "Cao", "Explicit preference only; không auto-learn fact"],
    ["Budget đến từ subscription cá nhân", "Cao", "Giới hạn ở local dev / internal eval"],
    ["Tool surface lớn làm sai routing", "Trung bình", "Capability profiles, tối đa ~6–9 tool mỗi call"],
    ["Provenance strict làm giảm UX", "Trung bình", "Shadow → warn → enforce, deterministic fallback"],
    ["Wire protocol runtime thay đổi", "Trung bình", "Pin version; adapter mỏng sau AgentRuntimeProvider"],
]
table(s, 0.55, 1.22, 12.25, 4.05, data, col_w=[4.9, 1.9, 5.4], font_size=15.5,
      bold_first_col=True, row_h=[0.42] + [0.45] * 8)
textbox(s, 0.55, 5.42, 12.25, 0.42, [("Những điều CHƯA nên làm", 21, True, RED)])
nots = ["Tự viết model loop hay thêm LangGraph/CrewAI thay ADK",
        "Tách từng tool thành microservice",
        "Cho model tự chọn làn A hay làn B",
        "Dùng prompt để enforce auth, threshold hay provenance",
        "Cho skill cấp thêm quyền tool",
        "Subagent cho screening/research/writer trong MVP"]
x, y = 0.55, 5.88
for i, t in enumerate(nots):
    textbox(s, x, y, 6.05, 0.34, [("• " + t, 15.5, False, DARK)])
    if i % 2 == 0:
        x = 6.75
    else:
        x = 0.55; y += 0.36

# ═══════════════════════════════════════════════════ 30. DECISIONS TO MAKE
s = S("Quyết định cần chốt trước khi code")
textbox(s, 0.55, 1.18, 6.05, 0.42, [("Đề xuất mặc định", 20, True, GREEN)])
data = [
    ["Câu hỏi", "Mặc định khuyến nghị"],
    ["Store session/message/part", "Firestore trước, sau interface"],
    ["Runtime topology", "Modular monolith trên Cloud Run"],
    ["Agent runtime", "OpenCode primary; DSH worker; không ADK"],
    ["LLM ở làn A / làn B", "Cấm / cho phép có budget"],
    ["Subagents · generic plugins", "Không trong MVP"],
    ["Provenance", "Shadow trước, sau đó strict"],
    ["User memory", "Opt-in preferences only"],
]
table(s, 0.55, 1.62, 6.05, 3.35, data, col_w=[3.0, 3.0], font_size=14.5,
      bold_first_col=True, row_h=[0.38] + [0.42] * 7)
textbox(s, 6.90, 1.18, 5.90, 0.42, [("Cần product / engineering xác nhận", 20, True, ORANGE)])
qs = ["Public scientific API có external consumer chưa?",
      "Cần exact schema /analyze hay chỉ semantic compat?",
      "Report/history retention bao lâu? Xoá theo user?",
      "Raw literature response có cần lưu để audit không?",
      "Image/heatmap chuyển signed URL ngay được không?",
      "Frontend cần stream token-level hay part-level?",
      "Ngưỡng provenance nào CHẶN, ngưỡng nào chỉ warning?"]
y = 1.66
for i, q in enumerate(qs):
    box(s, 6.90, y, 0.48, 0.42, [(str(i + 1), 16, True, WHITE)], fill=ORANGE)
    textbox(s, 7.52, y + 0.04, 5.28, 0.40, [(q, 16, False, DARK)],
            anchor=MSO_ANCHOR.MIDDLE)
    y += 0.48
box(s, 0.55, 5.30, 12.25, 1.45,
    [("ToxAgent sở hữu domain, tools, state và provenance;", 22, True, WHITE),
     ("OpenCode / DSH cho thuê agent loop và đường tới LLM.", 22, True, PALE)],
    fill=DARK)

# ═══════════════════════════════════════════════════ 31. CONCLUSION
s = S("Kết luận")
box(s, 0.55, 1.20, 12.25, 1.55,
    [("Không phải “giữ model API, viết một agent mới bên ngoài”.", 19, False, PALE),
     (" ", 8, False, None),
     ("Giữ và cô lập scientific kernel cùng deterministic analysis contract;", 23, True, WHITE),
     ("thay thế toàn bộ control plane bằng một harness stateful, typed, provenance-first.", 23, True, WHITE)],
    fill=DARK)
textbox(s, 0.55, 2.98, 12.25, 0.44,
        [("Thứ tự đầu tư", 21, True, DARK)], align=PP_ALIGN.CENTER)
order = [("1", "ToxAgent\nMCP server", BLUE),
         ("2", "Agent/profile tối giản\ncho OpenCode & DSH", BLUE),
         ("3", "Eval hai runtime\ntrên cùng prompts", CYAN),
         ("4", "OpenCode-first\nruntime gateway", ORANGE),
         ("5", "DSH adapter cho\nworker / eval", ORANGE),
         ("6", "Direct-provider adapter\nkhi có production budget", GREEN)]
x = 0.55
for num, text, color in order:
    box(s, x, 3.52, 1.95, 1.75, [], fill=LGRAY, line=color, line_w=1.75)
    box(s, x + 0.68, 3.66, 0.60, 0.60, [(num, 21, True, WHITE)], fill=color)
    textbox(s, x + 0.08, 4.34, 1.79, 0.92,
            [(l, 12.5, True, DARK) for l in text.split("\n")],
            align=PP_ALIGN.CENTER, space_after=1)
    x += 2.06
box(s, 0.55, 5.55, 12.25, 1.22,
    [("Tận dụng ngay LLM budget sẵn có, tránh duy trì ADK/custom loop,", 20, True, WHITE),
     ("và vẫn giữ lối thoát khi budget, provider hoặc điều khoản deployment thay đổi.", 20, True, PALE)],
    fill=GREEN)

# ═══════════════════════════════════════════════════ 32. THANK YOU
s = S(None)
textbox(s, 0.17, 2.35, 13.00, 1.00, [("Cảm ơn!", 46, True, DARK)], align=PP_ALIGN.CENTER)
textbox(s, 0.55, 3.55, 12.25, 0.90,
        [("docs/spec/TOXAGENT_HARNESS_MASTER_PLAN_VI.md", 20, True, BLUE),
         ("HARNESS_SYSTEM_DESIGN_VI · HARNESS_USE_CASES_VI · HARNESS_USER_STORIES_VI", 17, False, GRAY)],
        align=PP_ALIGN.CENTER, space_after=8)
box(s, 3.55, 4.85, 6.25, 0.85,
    [("Câu hỏi và phản biện luôn được hoan nghênh", 19, True, WHITE)], fill=DARK)

out = os.path.join(REPO, "ToxAgent_03_Harness_Master_Plan.pptx")
D.save(out)
print("saved:", out, "| slides:", len(D.prs.slides.__iter__.__self__._sldIdLst))
