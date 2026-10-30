# A1redo — Chọn VLM cho tier-2 trên Colab T4 (Phước)

**Ngày đo:** 2026-10-30 · **Máy:** Colab T4 (14.56 GB, idle 169 MB) · **Git:** `242cdf3` (commit local lúc chạy Colab, chưa push nên không tồn tại trên remote; raw JSON/log T4 giữ nguyên hash này)
**Prompt hash:** `3d71fdba` (`configs/prompts.yaml` verifier, không chỉnh tay) · **Windows hash:** `556e8652` (LF; bản Windows CRLF ra `f112ac22` — cùng nội dung, đã xác minh sha1 sau khi chuẩn hoá dòng)
**Số liệu thô:** `notes/vlm-bench-raw-t4-{qwen2vl2b,qwen25vl3b,smolvlm2}.json` + `-capped-*.json` · **Log:** `notes/vlm-bench-log-t4.txt` · **Phiên bản:** `notes/pip-versions-t4.txt`
**Cửa sổ:** `notes/vlm-bench-windows.json` (16 cửa sổ THẬT: 8 anomaly/8 lớp từ GT UCF test + 8 normal từ calib, seed 0, 4 khung 320x240 cách đều)

## 1. Phương pháp (redo plan §2C/§3)

- Mỗi model **một process** (bitsandbytes leak giữa model trong cùng kernel). **Warmup 2 run** (không tính) + **8 run đo**, cycle qua ≥8 cửa sổ khác nhau; headline = **median post-warmup**.
- Prompt lấy nguyên từ `configs/prompts.yaml` (`verifier.system` + `verifier.user`), greedy, `max_new_tokens=128` (p99 out tokens quan sát ≤ 54).
- VRAM: peak `nvidia-smi memory.used` sample nền 0.2 s (gồm CUDA context — số quyết định "fits in 6 GB") + `torch` allocated/reserved; probe riêng cho cửa sổ nhiều token nhất (worst case).
- Vòng mô phỏng 6 GB: `--vram-cap-gb 5.5` (chỉ giới hạn PyTorch allocator, KHÔNG gồm CUDA context) → cộng **0.7 GB margin** cho 3050 (display + context). **6 GB fit = simulated, không đo trên 3050.**
- Quality: mỗi model sinh 1 lần trên **cả 16 cửa sổ**, parse bằng `vad.verify.vlm.parse_verifier_json` (không re-implement).
- Loại trừ: thời gian tải weight (pre-download bằng `huggingface-cli`; `load_seconds` = warm-cache cold-process load, ghi riêng).

## 2. Bảng A — tốc độ & bộ nhớ (Colab T4, post-warmup)

| Model · variant | Image tok | Prompt tok | Out tok (med) | Prefill s | Decode s | **Median s/window** | p90 | Device peak GB | Torch alloc GB | Load s (warm cache) |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2-VL-2B · images_native | 392 | 134 | 47 | 0.35 | 2.68 | 3.03 | 4.78 | 1.99 | 1.59 | 117.1 |
| Qwen2-VL-2B · video_native | 197 | 125 | 41 | 0.23 | 2.25 | **2.48** | 2.89 | 1.94 | 1.58 | — |
| **Qwen2.5-VL-3B · images_native** | 392 | 134 | 54 | 0.63 | 3.82 | **4.45** | 4.74 | 2.93 | 2.52 | 155.6 |
| Qwen2.5-VL-3B · video_native | 197 | 125 | 51 | 0.38 | 3.63 | **4.00** | 4.48 | 2.86 | 2.49 | — |
| SmolVLM2-2.2B · images_nosplit | 332 | 121 | — | — | — | **CRASH** (xem §6) | — | — | 1.55 (tại crash) | 291.2 |
| SmolVLM2-2.2B · images_split | 4328 | 121 | — | — | — | **CRASH** (xem §6) | — | — | 2.64 (tại crash) | — |

Vòng cap 5.5 GB (3 cửa sổ, gồm worst-case): Qwen2-VL-2B và Qwen2.5-VL-3B **đều chạy sạch, không OOM**; device peak giữ nguyên (1.99 / 2.93 GB).
Clocks: idle 585 → 945–960 MHz khi chạy, temp 57→73 °C, không throttle.

## 3. Bảng B — quality sanity (16 cửa sổ, variant images_native)

| Model | Parse first-try | Consistency | TPR | FPR | Type match | Conf đúng / sai |
|---|---|---|---|---|---|---|
| Qwen2-VL-2B | **2/16 ❌** | 1/2 | n/a | 1/2 | n/a | 1.00 / 0.90 |
| **Qwen2.5-VL-3B** | **16/16 ✅** | **16/16 ✅** | 6/8 | **0/8** | 0/8 | 0.95 / 0.95 |
| SmolVLM2-2.2B | 0/16 ❌ | — | — | — | — | — |

*n=16 là sanity check, KHÔNG phải benchmark (A5 mới là workload thật). Đếm dạng "7/8", không chọn winner trên chênh lệch 1 cửa sổ.*
- **Qwen2-VL-2B thất bại vì chéo template**: `{"is_anomaly": true|false, "event_type": "<Arrest>"...}` — điền placeholder thay vì giá trị (14/16 output hỏng JSON). Câu caption khi điền đúng thì khá (vd "A police officer is seen arresting a suspect"), nhưng rule quality gate: model không giữ được schema thì loại bất kể tốc độ (A2 raise sau 1 retry).
- **Qwen2.5-VL-3B luôn bọc ```json fence** — `parse_verifier_regex` chịu được. Confidence gần như **hằng số 0.95** → không có tín hiệu, không được dùng làm mức tin cậy.
- Bias "Robbery": 5/8 anomaly bị gán Robbery, type_match 0/8 — **event typing để C3 lo** (typer riêng đạt 40.4% top-1); tier-2 chỉ chốt `is_anomaly` + caption.

## 4. Bảng C — variant đầu vào (cùng 16 cửa sổ)

| So sánh | Kết quả | Quyết định |
|---|---|---|
| images_native (320x240) vs images_cap448 (`max_pixels=448²`) | Token giống hệt (392) và latency giống trong nhiễu (3.03 vs 3.05 / 4.45 vs 4.43) — UCF 320x240 nằm DƯỚI cap → cap là no-op | Giữ native, không upscale |
| images vs **video input** (Qwen, temporal patch 2) | Token gần nửa (392→197), median 4.45→4.00 (3B) và 3.03→2.48 (2B); 1/1 sample output video hợp lệ | Video nhanh hơn ~11% (3B); **quality chưa đo trên 16 cửa sổ** → A2 đi images_native, A6 xác nhận video rồi mới đổi |
| SmolVLM2 nosplit vs split | Cả hai CRASH khi generate (§6) | Loại |

## 5. Bảng D — capacity (thay câu hỏi "3 s/window có đạt không")

Ngưỡng 3 s của task-sheet viết cho 3050, T4 là hạ tầng chính → cửa là **capacity check** (redo plan §4). N = số cửa sổ thật chờ **TH6 freeze (hỏi Thịnh)**; dưới đây dùng N ví dụ:

| Setup | s/window (đo được) | 200 windows | 600 windows | 1000 windows | Vượt 2 h/session? |
|---|---|---|---|---|---|
| T4 Colab · 3B images (đo quality) | 4.45 | 15 phút | 45 phút | 74 phút | Không (1600 windows ≈ 119 phút, sát ngưỡng 2 h) |
| T4 Colab · 3B video (chưa đo quality) | 4.00 | 13 phút | 40 phút | 67 phút | Không |
| 3050 local | **NOT MEASURED** (fit simulated) | n/a | n/a | n/a | n/a |

Colab rớt giữa chừng là bình thường: output ghi increment per-window (JSONL) + ghi lại `--out` sau mỗi model → chạy lại chỉ mất tối đa 1 model.

## 6. Verdict (rule §4, áp dụng đúng chữ)

**COLAB T4 GO: Qwen/Qwen2.5-VL-3B-Instruct, median 4.45 s/window on T4 (images_native; video 4.00), 6 GB-viable: yes (simulated, peak 2.93 + 0.7 margin), parse 16/16, capacity 600 windows in 45 min.**

Lý do (từng bước rule §4):
1. **VRAM gate:** 2B ✓ (1.99+0.7=2.69<6.0), 3B ✓ (2.93+0.7=3.63<6.0), SmolVLM2 ✗ (không generate được).
2. **Quality gate (≥14/16 parse và consistency):** 2B ✗ (2/16), **3B ✅ (16/16 cả hai)**, SmolVLM2 ✗ (0/16).
3. **Tốc độ:** 3B median 4.45 s (images) / 4.00 s (video) — không còn ngưỡng 3 s; cửa thực sự là capacity.
4. **Capacity:** 600 windows ≈ 45 phút — vừa một phiên T4 (~3 h) thoải mái; N thật chờ TH6. Cửa cần trả lời là capacity, không phải "3 s/window".
5. Qwen2.5-VL-3B là fallback chất lượng đã được dùng làm primary chính vì 2B **thất bại gate với prompt hiện tại** (hash `3d71fdba`). Nếu A6 sửa prompt, có thể re-test 2B — nhưng **A5 sẽ chạy 3B**.

**SmolVLM2-2.2B: loại.** Cả `images_nosplit` và `images_split` crash generate trên **cả 16 cửa sổ** với transformers 5.20.2: `RuntimeError: shape '[-1, 14, 2048]' is invalid for input of size 663552` (reshape visual token không chia hết cho 14 với input 4 ảnh rời). Không sửa trong timebox A1 (ứng viên yếu nhất, không đáng đánh đổi thời gian). Số cũ "5.57 GB / 31.85 s" trên 3050 giờ được giải thích: image splitting nổ token (4328 vs 332) trên một đường generate không ổn định — **bị thay thế hoàn toàn** bởi số T4 ở trên.

## 7. Giới hạn (limitations)

- 16 cửa sổ từ MỘT dataset (8 video test có GT + 8 calib normal), seed 0 — sanity, không ranking; A5 mới là số liệu nộp.
- **3050: chỉ fit simulated** (allocator cap + 0.7 GB margin). Không có speed claim nào cho 3050 (mục §2C.6 optional chưa chạy).
- Chỉ thử nf4 4-bit; không so fp16/8-bit.
- Video mode chưa có quality table (chỉ 1 sample sạch) — cần A6 xác nhận trước khi đổi A2 sang video.
- Confidence 3B ~ hằng số 0.95: không dùng làm tín hiệu (không threshold trên confidence).
- Prompt của 2B-echo: thất bại gắn với prompt hash `3d71fdba` này; đổi prompt = re-test.
- Số đo 3050 cũ (`notes/vlm-bench-raw-3050-legacy.json`, `notes/vlm-bench-log-3050-legacy.txt`): ảnh tĩnh 4 bức KHÁC KÍCH THƯỚC, mean gồm run warmup (F2/F3 trong review) — chỉ giữ làm lịch sử, không so sánh tốc độ với số T4.

## 8. Handoff cho A2 (copy nguyên vào `vad/verify/vlm.py`)

```
model_id: Qwen/Qwen2.5-VL-3B-Instruct
loader: Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig(nf4, fp16 compute), device_map="cuda", attn_implementation="sdpa"
inputs: 4 ảnh RGB 320x240 (images_native, không upscale, không max_pixels); CHƯA dùng video mode (nhanh hơn 11% nhưng chưa đo quality — chỉ đổi sau khi A6 xác nhận)
prompt: configs/prompts.yaml verifier.system + verifier.user (hash 3d71fdba) — system turn BẮT BUỘC (messages[0])
generation: greedy (do_sample=False), max_new_tokens=128, stop on EOS
known quirks:
  - luôn bọc ```json fence -> parse_verifier_regex chịu được, không cần strip tay
  - confidence gần như hằng số 0.95 -> không dùng làm tín hiệu
  - bias "Robbery" (5/8 anomaly): event_type chỉ tin ở mức caption; typing để C3 lo
  - TPR 6/8, FPR 0/8 trên 16 cửa sổ sanity — không phải benchmark
  - Qwen2-VL-2B echo template "true|false" với prompt này (2/16 parse) — KHÔNG chọn làm default
  - VRAM thực đo T4: peak 2.93 GB (device) / 2.52 GB (torch alloc); 6 GB-viable simulated
```

## 9. Số đo 3050 cũ (lịch sử, đã bị thay bằng số T4 ở trên)

Ngày 2026-10-29 trên RTX 3050 6 GB: Qwen2-VL-2B peak 1.92 GB, 3 run 4.22/3.61/3.74 s (post-warmup mean 3.68), nhưng input là **4 ảnh tĩnh kích thước khác nhau** và prompt viết tay trong script — không so được với số T4 ở trên. Raw: `notes/vlm-bench-raw-3050-legacy.json`, log: `notes/vlm-bench-log-3050-legacy.txt`. Kết luận cũ "local GO" đã bị rút: VLM chạy trên Colab T4 (quyết định redo plan).
