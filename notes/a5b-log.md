# A5b log — tier 2 thật trên 307 cửa sổ UCF-Crime test (Phước)

**Ngày:** 2026-11-01 · **Task:** A5c · **Nguồn:** `results/verify/` (PR #37) + log Colab A5b
**Model:** `Qwen/Qwen2.5-VL-3B-Instruct`, 4-bit nf4 · **Prompt hash:** `3d71fdba` · **Fake:** `false`
**Code:** `run_verify.py` @ `7b3a174` (305 file) + clamp PR #35 (2 file rerun)

## 1. Tổng quan

| Chỉ số | Giá trị |
|---|---|
| Cửa sổ | **307/307**, 0 lỗi |
| Video có cửa sổ | **114** (90 anomaly + 24 normal) |
| Lượt chạy | 305 file từ lượt chính + 2 file rerun sau clamp PR #35 |
| VLM xác nhận | **129/307 = 42.0%** |
| VLM từ chối | **178/307 = 58.0%** |
| Mật độ | 307/290 = **1.06 cửa sổ/video test** |
| Video có >= 1 cửa sổ confirmed | **72/114** (71 anomaly + 1 normal) |

Phân bố theo GT:

| Nhóm | Cửa sổ | Confirmed | Rejected |
|---|---:|---:|---:|
| Video anomaly (90 video) | 183 | **128 (69.9%)** | 55 (30.1%) |
| Video normal (24 video) | 124 | **1 (0.8%)** | 123 (99.2%) |
| **Tổng** | **307** | **129** | **178** |

- Cửa sổ normal duy nhất bị confirmed: **1/124**. ID và caption của cửa sổ này để riêng cho A7 **sau khi A6b chấm xong** — không ghi vào log này để tránh làm lộ một dòng của mẫu blind A6.
- **85/124** cửa sổ normal đến từ 2 video dài: `Normal_Videos_935` (55) và `Normal_Videos_924` (30). FA/hour bị chi phối bởi 2 video này; đây là caveat đã biết từ tier 1 (20.64 vs 9.7 FA/h khi bỏ 935 + 924).
- **37/307** cửa sổ có `is_anomaly=false` nhưng `event_type != Normal` (VLM tự mâu thuẫn). Theo `notes/a6-protocol.md` §1, verdict chỉ lấy `is_anomaly` → 37 cửa sổ này là **rejected**. Ghi footnote trong report, không đổi rule.
- `event_type` của VLM (chỉ để tham khảo, không dùng cho alert): Normal 141, Explosion 35, Robbery 23, Burglary 20, Shoplifting 20, Shooting 17, Arrest 13, Arson 12, Vandalism 7, RoadAccidents 7, Assault 5, Fighting 5, Stealing 2.

## 2. Thời gian và chunk

| Chỉ số | Giá trị |
|---|---|
| Median s/window | **3.71 s** |
| p90 s/window | **14.41 s** (nearest-rank của `timing_report`) |
| Mean s/window | 10.10 s (3100.8 / 307) |
| Min / max | 2.57 s / 457.01 s |
| Tổng thời gian xử lý | **3100.8 s = 51.68 phút** |
| Chunk-call | **480** |
| Cửa sổ bị chia chunk | **65/307 = 21.2%** (max 17 chunk, cửa sổ 490 s) |
| Timestamp first → last | 2026-10-31 07:52:33 → 11:05:33 UTC (span 3h13m, gồm resume/nghỉ; không phải wall time xử lý) |

Phân bố số chunk/cửa sổ: 242 cửa sổ 1 chunk, 31 cửa sổ 2 chunk, 17 cửa sổ 3 chunk, 6 cửa sổ 4 chunk, 11 cửa sổ còn lại 5–17 chunk.

So với A1 (16 cửa sổ sanity, T4, `vlm-selection.md`): median 4.45 s/window → A5b 3.71 s/window. Đây là workload thật (307 cửa sổ, chunk theo D-1), không phải cùng tập cửa sổ nên không so trực tiếp như benchmark.

## 3. VRAM và model load

- **A5b không ghi được peak VRAM.** Log Colab chỉ có progress bar + summary thời gian; per-window metadata trong `results/verify/` không có trường VRAM. Không bịa số.
- Theo fallback của D4: dùng số **A1 benchmark** cho Qwen2.5-VL-3B 4-bit, 4 frame/window, T4:
  - peak device: **2.93 GB** (gồm CUDA context, `nvidia-smi`)
  - peak torch allocated: **2.52 GB**
  - model load warm-cache: **155.6 s**
  - GPU: **Tesla T4 14.56 GB** (notebook A5a assert T4; log A5b không in tên GPU)
  - pip/driver: `notes/pip-versions-t4.txt` (môi trường A1; A5b không chụp lại)
- Nếu report/defense cần đúng số A5b: rerun pilot 10 cửa sổ trên Colab với `torch.cuda.max_memory_allocated()` và cập nhật mục này. Cho đến lúc đó, mọi câu "fits on 6 GB" phải ghi rõ là **A1 benchmark + 0.7 GB margin**, không phải số A5b.

## 4. Clamp biên video (PR #35)

- **48/307** cửa sổ có `end_s > duration` (lệch tối đa 0.933 s).
- 2 cửa sổ từng raise (`RoadAccidents002_x264[0]`, `Vandalism017_x264[1]`) đã rerun sau clamp.
- 46 cửa sổ còn lại vẫn dùng frame trước clamp; chunk count và verdict OR không đổi, fingerprint không đổi nên resume bỏ qua. Đây là hành vi đã biết trong PR #35, không phải bug mới; report phải ghi rõ nếu trích chi tiết frame.

## 5. Kết luận cho report

- Cost rows: `% escalated` = **1.06 cửa sổ/video**; `s/window` median **3.71**, p90 **14.41**; **480** chunk-call; **21.2%** cửa sổ bị chunk; tổng xử lý **51.7 phút**.
- Confirmation rate **42.0%**; normal-video false positive **1/124** cửa sổ.
- Không dùng `confidence` (gần hằng số 0.95) và không dùng `event_type` của VLM làm verdict/alert type; typer C3 quyết định type.
- Caveats phải ghi cùng bảng: **37 cửa sổ tự mâu thuẫn**, **46 cửa sổ chưa clamp frame**, **peak VRAM A5b không đo được** (dùng A1 2.93 GB có footnote).
