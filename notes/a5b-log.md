# A5b log — tier 2 thật trên 307 cửa sổ UCF-Crime test (Phước)

**Ngày:** 2026-11-01 · **Task:** A5c · **Nguồn:** `results/verify/` (PR #37) + log chạy A5b
**Model:** `Qwen/Qwen2.5-VL-3B-Instruct`, 4-bit nf4 · **Prompt hash:** `3d71fdba` · **Fake:** `false`
**Code:** `run_verify.py` @ `7b3a174` (305 file) + clamp PR #35 (2 file rerun)

> Bằng chứng dùng trong note: 307 file `results/verify/**/window_*.json` (đọc lại 2026-11-01),
> log phiên B `notes/a5b-run-resume.log` (đã commit, 18 KB), và pilot VRAM 10 cửa sổ chạy lại cùng ngày (§3.1).
> Mọi số ở §1, §2, §4 tính lại được từ 307 file; riêng VRAM là số đo mới, có ghi rõ cách đo.

## 1. Tổng quan

| Chỉ số | Giá trị |
|---|---|
| Cửa sổ | **307/307**, 0 lỗi |
| Video có cửa sổ | **114/290 = 39.3%** (90 anomaly + 24 normal) |
| Lượt chạy | 305 file từ lượt chính + 2 file rerun sau clamp PR #35 |
| VLM xác nhận | **129/307 = 42.0%** |
| VLM từ chối | **178/307 = 58.0%** |
| Mật độ | 307/290 = **1.06 cửa sổ/video** (đơn vị cửa sổ/video, KHÔNG phải %) |
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

## 2. Thời gian, phiên chạy và chunk

| Chỉ số | Giá trị |
|---|---|
| Median s/window | **3.71 s** |
| p90 s/window | **14.41 s** (nearest-rank của `timing_report`) |
| Mean s/window | 10.10 s (3100.8 / 307) |
| Min / max | 2.57 s / 457.01 s |
| Tổng thời gian xử lý | **3100.8 s = 51.68 phút** |
| Timestamp first → last | 2026-10-31 07:52:33 → 11:05:33 UTC (span 3h13m, có 2 gap; không phải wall time xử lý) |

### 2.1 Hai phiên chạy + 1 lượt rerun

Lượt A5b **không chạy liền một mạch**: nó gồm 2 phiên và 1 lượt rerun. Cùng model, cùng prompt,
cùng `verifier_config` (đã đối chiếu cả 307 file: `4bit`, `max_attempts=2`, `max_new_tokens=128`,
`prompt_hash=3d71fdba`, `k=4`, `max_chunk_s=30`, `commit=7b3a174`), nên **trộn phiên không làm sai verdict**.

| Phiên | Cửa sổ | Thời gian (UTC) | Sum `elapsed_s` | Median s/cửa sổ | Bằng chứng |
|---|---:|---|---:|---:|---|
| A | 107 | 07:52:33 – 08:25:36 | 2028.3 s | 9.73 | file trong `results/verify/`; không có log trong repo |
| B (resume) | 198 | 10:05:21 – 10:21:49 | 997.8 s | 3.24 | `notes/a5b-run-resume.log`: `307 window xét, 198 verify, 107 bỏ qua, 2 lỗi` |
| Rerun clamp | 2 | 11:05:29 – 11:05:33 | 74.7 s | – | `RoadAccidents002[0]` 70.8 s, `Vandalism017[1]` 3.9 s |
| **Tổng** | **307** | | **3100.8 s** | **3.71** | 2028.3 + 997.8 + 74.7 = 3100.8 |

- Cả hai phiên chạy trên **RTX 3050 6GB (máy Phước)**: phiên B để lại log Windows/torch CUDA `notes/a5b-run-resume.log` (đã commit).
- **Gap 1 = 99.75 phút** (08:25:36 → 10:05:21): phiên A dừng giữa chừng (chủ động dừng, không crash, không OOM), sau đó resume và 107 file đã xong được **bỏ qua theo fingerprint** — log phiên B xác nhận đúng 107 skip.
- **Gap 2 = 43.7 phút** (10:21:49 → 11:05:29): chờ lượt rerun clamp 2 cửa sổ, không phải thời gian xử lý.
- Khác biệt tốc độ giữa 2 phiên (~3×: 9.27 vs 3.14 s/chunk-call) **không** do config hay workload — caption dài như nhau (51.0 vs 50.5 ký tự/chunk) và config giống hệt. Nguyên nhân khả dĩ là tải GPU/nhiệt độ lúc chạy; chưa đo lại nên ghi nhận là hạn chế, cost row vì vậy ghi số gộp 307 cửa sổ kèm median từng phiên.

### 2.2 Outlier 457 s

`Explosion017_x264/window_0000` là cửa sổ 18 s, **1 chunk**, nhưng `elapsed_s` = **457.01 s** — chiếm **14.7%** tổng thời gian (22.5% của riêng phiên A) và một mình nó đẩy mean từ 8.64 lên 10.10 s.

- Nguyên nhân: **stall đọc video** (nguồn video/ổ mạng), không phải model.
- Đã loại trừ: reload model (cửa sổ liền trước 9.8 s, liền sau 14.1 s), OOM/offload retry (không có lỗi, không có dòng retry, các cửa sổ sau chạy bình thường).
- **Tổng trừ outlier: 3100.8 − 457.0 = 2643.8 s = 44.06 phút**, mean 8.64 s/cửa sổ. Budget ~52 phút của A6c trong D4-2 là cận trên an toàn (dư ~8 phút).

### 2.3 Chunk

| Chỉ số | Giá trị |
|---|---|
| Chunk-call | **480** |
| Cửa sổ bị chia chunk | **65/307 = 21.2%** (max 17 chunk, cửa sổ dài nhất 490 s) |

Phân bố số chunk/cửa sổ: 242 cửa sổ 1 chunk, 31 cửa sổ 2 chunk, 17 cửa sổ 3 chunk, 6 cửa sổ 4 chunk, 11 cửa sổ còn lại 5–17 chunk.

So với A1 (16 cửa sổ sanity, T4, `vlm-selection.md`): median 4.45 s/window → A5b 3.71 s/window. Đây là workload thật (307 cửa sổ, chunk theo D-1), không phải cùng tập cửa sổ nên không so trực tiếp như benchmark.

## 3. VRAM, GPU và model load

### 3.1 Số đo trên máy sản xuất (pilot 10 cửa sổ, 2026-11-01)

A5b **không** ghi peak VRAM (metadata không có field, log phiên A không có trong repo), nên đã **chạy lại pilot 10 cửa sổ** trên đúng máy RTX 3050 6GB với cùng config production:

| Chỉ số | Giá trị |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 **6GB Laptop** GPU, driver 581.86 |
| Môi trường | Python 3.11.9 · torch 2.14.1+cu126 · transformers 5.17.0 · bitsandbytes 0.50.2 · accelerate 1.15.0 |
| Pilot | 10 cửa sổ / 5 video, **28 chunk-call** (1–10 chunk/cửa sổ), **0 lỗi** |
| Config | 4-bit nf4, `k=4`, `max_chunk_s=30`, `max_new_tokens=128`, `prompt_hash=3d71fdba` — trùng A5b |
| Code | `run_verify.py` @ `48739a3`; khác `7b3a174` **chỉ ở clamp PR #35**, không đổi model/generation |
| Peak VRAM (torch allocated) | **2.52 GB** |
| Peak VRAM (torch reserved) | **2.63 GB** |
| Peak VRAM (nvidia-smi, device) | **2.91 GB** (2983 MB, baseline desktop 173 MB) |
| Headroom | 6.0 − 2.91 = **3.1 GB**; sau pilot còn 2.37 GB trống |
| Model load | **18 s** (bar `Loading weights`, HF cache ấm) |

- Code đo: `torch.cuda.max_memory_allocated()` / `max_memory_reserved()` + sample `nvidia-smi memory.used` mỗi 0.25 s; output ghi vào thư mục tạm, **không ghi vào `results/verify/`**.
- **Tái lập A1**: A1 trên T4 ghi 2.52 GB torch / 2.93 GB device → pilot trên 3050 cho 2.52 / 2.91. Hai GPU khác nhau, cùng model/config, cùng mức VRAM. Từ giờ câu "fits on 6 GB" có số first-party của đúng máy sản xuất, không phải suy từ benchmark.
- Lưu ý: pilot chỉ dùng để đo VRAM. `s/window` của pilot **không** dùng cho cost row (10 cửa sổ, chọn theo video, không phải mẫu của 307).

### 3.2 Cái không đo được (ghi rõ để không ai trích nhầm)

- Peak VRAM **của chính lượt 307 cửa sổ**: không có; dùng số pilot ở §3.1 (cùng máy, cùng config) và ghi rõ là pilot.
- **Retry**: metadata chỉ ghi `max_attempts=2`, không ghi số lần thử; log phiên B không có dòng retry nào → không kết luận được số retry của 307 cửa sổ. 2 lỗi cuối log là lỗi clamp (đã rerun ở PR #35), **không phải retry**.
- **Session restart / resume**: 1 lần dừng giữa chừng (phiên A → B, 107 file skip theo fingerprint) + 1 lượt rerun clamp 2 file. Đây là "session restarts and resume events" mà checklist A5c hỏi.

## 4. Clamp biên video (PR #35)

- **48/307** cửa sổ có `end_s > duration` (lệch tối đa 0.933 s).
- 2 cửa sổ từng raise (`RoadAccidents002_x264[0]`, `Vandalism017_x264[1]`) đã rerun sau clamp.
- 46 cửa sổ còn lại vẫn dùng frame trước clamp; chunk count và verdict OR không đổi, fingerprint không đổi nên resume bỏ qua. Đây là hành vi đã biết trong PR #35, không phải bug mới; report phải ghi rõ nếu trích chi tiết frame.
- **Nguồn của claim "rerun sau clamp"**: `timestamp` của 2 file là 11:05:29 / 11:05:33 (cách phiên B 43.7 phút) và `notes/a5b-run-resume.log` ghi đúng 2 dòng `[lỗi] RoadAccidents002_x264[0]` / `[lỗi] Vandalism017_x264[1]`. Cả 307 file đều ghi `commit: 7b3a174`, nên **`metadata.commit` không chứng minh được** điều này — bằng chứng là log + timestamp, không phải commit.

## 5. Kết luận cho report

Cost rows (mỗi dòng **một đơn vị**, không trộn % với cửa sổ/video):

| Cost cell | Giá trị | Đơn vị |
|---|---|---|
| `% escalated` | **39.3%** (114/290) | % **video** test có >= 1 cửa sổ lên tier 2 |
| Mật độ cửa sổ | **1.06** (307/290) | cửa sổ/video — hàng riêng |
| `s/window` median | **3.71** (p90 **14.41**) | giây/cửa sổ |
| Tổng xử lý | **51.68 phút** (trừ outlier 44.06 phút) | phút |
| Chunk-call | **480** | lần gọi VLM; **21.2%** cửa sổ bị chunk |

- `% escalated` **phải** lấy mẫu số là video: mọi cửa sổ ứng viên đều được chuyển lên tier 2, nên "tỉ lệ cửa sổ escalated" luôn bằng 100% và vô nghĩa. 1.06 cửa sổ/video là hàng riêng, không phải %.
- Confirmation rate **42.0%**; normal-video false positive **1/124** cửa sổ.
- Không dùng `confidence` (gần hằng số 0.95) và không dùng `event_type` của VLM làm verdict/alert type; typer C3 quyết định type.
- Caveats phải ghi cùng bảng: **37 cửa sổ tự mâu thuẫn**, **46 cửa sổ chưa clamp frame**, **peak VRAM là pilot 10 cửa sổ trên RTX 3050 (§3.1), không phải lượt 307 cửa sổ**, **s/window khác ~3× giữa 2 phiên (9.73 vs 3.24) nên cost row ghi số gộp 307 kèm median từng phiên**.
- **Cho TH8**: `report/04-thuc-nghiem.md` dòng "Tỉ lệ cửa sổ chuyển tiếp lên Tầng 2" đang ghi `1.1686 (307 cửa sổ, 1.06/video)` với đơn vị `%` — sửa thành 39.3% (đơn vị % video) và thêm hàng 1.06 cửa sổ/video.
