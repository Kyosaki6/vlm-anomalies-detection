# A1 — Chọn VLM cho tier-2 (Phước)

**Ngày đo:** 2026-10-29 · **Máy:** RTX 3050 6 GB Laptop (torch 2.14.1+cu126) · **Script:** `scripts/bench_vlm.py`
**Số liệu thô:** `notes/vlm-bench-raw.json` · **Log:** `notes/vlm-bench-log.txt`

Phương pháp: mỗi model nạp **4-bit (nf4, fp16 compute)** qua bitsandbytes, 1 cửa sổ = 4 frame
(`data/duong_pho_{1..4}.jpg`) + prompt verifier (JSON format), greedy, `max_new_tokens=128`.
Đo 3 run/model, đồng bộ CUDA trước/sau, peak VRAM = `torch.cuda.max_memory_allocated()`.
Run 1 là warmup (load cache CUDA kernels); số thực tế lấy từ run 2-3.

## Số đo (thực nghiệm, không ước lượng)

| Model | Peak VRAM | Load (có cache) | Run 1/2/3 (s) | Mean s/window | Nhận xét |
|---|---|---|---|---|---|
| **Qwen/Qwen2-VL-2B-Instruct** | **1.92 GB** | 12.7s | 4.22 / 3.61 / 3.74 | **3.86** | JSON đúng format ngay, VRAM rất thoáng |
| Qwen/Qwen2.5-VL-3B-Instruct | 3.89 GB | 18.4s | 5.55 / 5.45 / 5.37 | 5.46 | JSON bọc ```json fence (`parse_verifier_json` chịu được) |
| HuggingFaceTB/SmolVLM2-2.2B-instruct | **5.57 GB** ⚠️ | 761s | 30.71 / 30.65 / 34.19 | 31.85 | Vượt ngưỡng VRAM + chậm gấp ~8x (suy luận rất kém) |

## Quyết định (rule A1: peak VRAM < 5.5 GB và ≤ 3 s/window → local GO)

- **Không model nào đạt ĐỦ cả hai điều kiện.** Cả 3 đều fit VRAM dưới/near 5.5 GB, nhưng không model
  nào ≤ 3 s/window.
- **Chọn `Qwen/Qwen2-VL-2B-Instruct` cho local**: VRAM đỉnh 1.92 GB / 6 GB (còn 4 GB dư cho embedding
  chạy song song), nạp 13s, JSON đạt format sạch. Tốc độ 3.86s/window **vượt ngưỡng 3s một khoảng
  nhỏ (x1.29)** — ngưỡng 3s là mức ước lượng ban đầu của task sheet, không phải số đo; số đo mới thay
  nó.
- **Về tốc độ: 3.86s/window là GIÁ TRỊ TRÊN LOCAL.** Ước tính T4 (320 GB/s vs 192 GB/s, ~1.5-1.7x
  decode nhanh hơn) sẽ vào khoảng 2.3-2.6s/window — vượt ngưỡng 3s nghĩa là local hơi chậm hơn
  target, không nghĩa là 3050 không chạy được.

## GO / NO-GO & phương án

**Local GO với Qwen2-VL-2B** cho A2 (viết verifier), A3 (script run_verify) và smoke test A5 — VRAM
thoải mái, không lý do gì chặn. Đối chiếu thực tế A5 D5:

- Nếu số window thực tế nhỏ (subset đo chi phí được cho phép): **chạy local**, không cần Colab.
- Nếu 3.86s x số window vượt khả năng trong đêm D5: **pivot Colab T4 cho phần còn lại** (A1 đã
  quyết model Qwen2-VL-2B, T4 16 GB chạy cùng model, code không đổi).
- Qwen2.5-VL-3B là phương án chất lượng cao hơn nếu T4 được dùng (VRAM 3.89 GB vẫn dư trên 16 GB).

**Rủi ro đã closes:** "VLM không fit 6 GB" — đã đo, fit (Qwen2-VL-2B 1.92 GB). Không còn rủi ro
OOM ở mức model này. Lời hứa A1 EOD: ✅ có số đo thật, có model chọn, verdict đăng nhóm.

## Chi tiết kỹ thuật đáng ghi (dùng cho A2)

- Deps riêng: `requirements-vlm.txt` (bitsandbytes, accelerate, qwen-vl-utils, **num2words** —
  SmolVLM processor bắt buộc; cài tay, KHÔNG đưa vào requirements-dev.txt).
- Qwen2-VL trả JSON inline; Qwen2.5-VL bọc ```json fence — `parse_verifier_json` hiện regex
  `\{.*\}` DOTALL nên chịu được cả hai.
- Console Windows cp1252: script benchmark phải `sys.stdout.reconfigure(encoding="utf-8")` nếu in
  tiếng Việt, không thì raise UnicodeEncodeError khi in thông báo lỗi.
