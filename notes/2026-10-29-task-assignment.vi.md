# Phân công công việc: ai làm gì, cụ thể từng việc

Tài liệu đi kèm với `notes/2026-10-28-plan-zero-shot-video-vad.md` (LÀM GÌ / TẠI SAO). File này là AI LÀM / KHI NÀO.

> **Cập nhật tiến độ Ngày 1 (Cuối ngày Thứ Năm 29/10/2026):**
> - **461 test đã pass** (`python -m pytest -q`), 0 lỗi, không tải model/cần GPU trong CI.
> - **Các PR đã merge hôm nay:** PR #8 (embedder thật B3), PR #9 (schema kết quả C1), PR #10 (pipeline tier-1 TH2, loader E1, tài liệu LAVAD TH1, khung báo cáo TH8), PR #11 (điểm vào alerting C5), PR #12 (đánh giá typer C3), PR #13 (bộ tải UCF + manifests B1, quy ước nhãn B2), PR #14 (hoàn thành embedding Colab).
> - **Bộ cache embedding hợp nhất:** Toàn bộ 350 video (290 test + 60 calib) đã được trích xuất cho cả SigLIP 2 và CLIP B/32 trên Colab T4 (700 file `.npy`). Các shard A4, B5, C4 hoàn thành sớm; cosine fp16 vs fp32 trên video thật đạt chuẩn (> 0.99997, `notes/embedder-benchmark.md`).
> - **Event Typer C3 đã đo đạc:** Top-1 strict 40.4%, lenient 50.0% so với ngẫu nhiên 7.7% trên UCF GT (`results/typing_eval.json`).
> - **Các khoảng trống đã giải quyết:** G1, G2, G3, G4, G6, G7, G8 đều đã xong và có test.
> - **Rủi ro lớn nhất còn lại của Ngày 1:** Task A1 (Phước: đo đạc VLM và quyết định chạy local RTX 3050 hay Colab T4).
> - **Mốc then chốt tiếp theo (Ngày 2):** Thịnh chạy TH3 (`scripts/run_benchmark.py`) và TH4 (cổng tái lập baseline A0 ~53 AUC).

**TIMELINE & VAI TRÒ:**
- Deadline: nộp TRƯỚC ngày 4/11 (hạn chót cứng: cuối ngày 3/11 23:59).
- Nhóm: 4 người (Phước, Huy, Nam, Thịnh).
- Thịnh phụ trách toàn bộ dashboard Streamlit.
- Khung thời gian là 6 ngày (29/10 .. 3/11), nên phạm vi đã được cắt giảm ngay từ đầu (mục 6).

Khung thời gian: Ngày 1 = Thứ Năm 29/10/2026 ... Ngày 6 = Thứ Ba 03/11/2026. **Đóng băng tính năng (feature freeze): cuối D5 (tối 02/11).** D6 = báo cáo, slides, đóng gói, nộp bài (hạn chót 23:59). Không làm gì mới vào D6.
Cách đánh số ngày: D1 29/10 (Thứ Năm), D2 30/10 (Thứ Sáu), D3 31/10 (Thứ Bảy), D4 01/11 (Chủ Nhật), D5 02/11 (Thứ Hai), D6 03/11 (Thứ Ba).

Quy ước mã ID: các task là A1..A7, B1..B7, C1..C7, TH1..TH8 (Thịnh), E1..E7 (dashboard/clips). `D<n>` luôn có nghĩa là NGÀY n (Day n).

## 0. Giả định cần xác nhận vào D1 (thay đổi kế hoạch nếu có gì sai)

1. Thịnh = viết báo cáo + nhánh tier-1 + toàn bộ dashboard Streamlit. Xem mục 6 về các phần đã cắt giảm để đảm bảo tính khả thi.
2. GPU: Phước RTX 3050 6 GB, Huy RTX 3050 4 GB, Nam GTX 1060 3 GB (fp32), Thịnh không có GPU. Colab/Kaggle T4 = dự phòng.
3. Ước tính thời gian bên dưới là phỏng đoán, CHƯA ĐƯỢC ĐO THỰC TẾ. Phép đo đầu tiên (A1, B3) sẽ thay thế các ước tính này.
4. Chưa ai được hỏi xem có đồng ý nhận các vai trò này hay không. Có thể hoán đổi bất cứ thứ gì; giữ nguyên các interface.

## 1. Bảng tóm tắt

| Người | Phần cứng | Nhánh chính | Làm thêm | Ngày nặng nhất |
|---|---|---|---|---|
| Phước | 3050 6 GB | **A. Tier-2 VLM** (bắt buộc, rủi ro lớn nhất) | embedding shard 1 (40%) | D1-D3, D5 |
| Huy | 3050 4 GB | **B. Dữ liệu + nối embedder** (đường găng / critical path) | embedding shard 2 (35%), tự quay clip (E3/E4) | D1-D2, D3-D5 |
| Nam | 1060 3 GB | **C. Cảnh báo, schema kết quả, bộ phân loại sự kiện (event typer)** | embedding shard 3 (25%, fp32), build_results (C7) | D1-D5 |
| Thịnh | không có | **D. Chấm điểm tier-1, benchmark, ablations** + **E. Dashboard** | Báo cáo tiếng Việt + slides | D1-D6 (không có ngày nghỉ) |

Thịnh giữ số lượng task nhiều nhất; mục 6 cắt giảm phạm vi (ShanghaiTech là stretch, bỏ A7, ít clip hơn, UI chỉ ở chế độ replay) để khối lượng công việc trong tầm kiểm soát.

## 2. Các khoảng trống phát hiện trong skeleton đã commit (mỗi mục giờ là một task cụ thể)

| # | Khoảng trống | Task | Trạng thái (cuối D1) |
|---|---|---|---|
| G1 | `cooldown_s` có trong `policy.yaml` nhưng không có code nào đọc | C2 | **ĐÃ XONG** (`vad/graph/alerts.py`, commit `9cf6561`) |
| G2 | `co_events` không bao giờ được caller tính toán, nên việc leo thang cảnh báo (tai nạn + cháy nổ) không thể kích hoạt end-to-end | C2 | **ĐÃ XONG** (`vad/graph/alerts.py`, commit `9cf6561`) |
| G3 | `GraphPolicy.responders_in(zone)` bỏ qua `zone` và trả về toàn bộ đơn vị ứng phó: một placeholder nhìn như thật | C2 (triển khai hoặc xoá) | **ĐÃ XONG** (`vad/graph/policy.py` truy vấn độ phủ thật, commit `9cf6561`) |
| G4 | Chưa có schema kết quả và chưa có module kết nối (glue) giữa tier-1, tier-2, cảnh báo và dashboard | C1, TH2, C5 | **ĐÃ XONG** (`vad/results.py` ở PR #9, `vad/alerting.py` ở PR #11, `vad/pipeline.py` ở PR #10) |
| G5 | ShanghaiTech lưu dạng các THƯ MỤC khung hình; sampler hiện chỉ đọc file video | B4 (chỉ làm dạng STRETCH) | Chưa làm (stretch) |
| G6 | Baseline A0 cần OpenAI CLIP ViT-B/32 nhưng hiện chỉ có stub của SigLIP embedder | B3 (làm cho tổng quát với model) | **ĐÃ XONG** (`vad/embed/siglip.py` hỗ trợ đa model, PR #8) |
| G7 | Dashboard hiện chỉ chạy trên dữ liệu giả (fake data) | E1, E2 | **ĐÃ XONG** (`app/data.py` nạp file JSON `RunResult` thật, PR #10) |
| G8 | Baseline A0 cần embedding CLIP B/32 của CÙNG 290 video: tốn một lượt trích xuất embedding thứ hai. Việc giải mã video 30 fps có thể là phần tốn kém nhất chứ không phải model, do đó nên trích xuất cả hai model trong MỘT lượt decode video | B3 | **ĐÃ XONG** (chế độ một lượt decode cho cả hai model trong `scripts/precompute_embeddings.py`, 700 file đã cache) |

## 3. Danh sách task theo từng người

Định dạng: ID | Thời gian | Task | TIÊU CHÍ NGHIỆM THU (làm sao để biết đã xong).

### A. Phước: tier-2 VLM (+ GPU)

| ID | Thời gian | Task | Tiêu chí nghiệm thu | Trạng thái (cuối D1) |
|---|---|---|---|---|
| A1 | D1 | Đo thực tế, không đoán mò: tải 2-3 VLM sinh ngôn ngữ ứng viên ở định dạng 4-bit (các ứng viên để kiểm chứng, ví dụ: Qwen2-VL-2B, Qwen2.5-VL-3B, SmolVLM2), ghi nhận mức VRAM đỉnh (peak VRAM) và số giây trên mỗi cửa sổ cho 4 khung hình. Viết `notes/vlm-selection.md` | Chọn được một model với peak VRAM đo được < 5.5 GB. Nếu không model nào vừa, quyết định dùng Colab T4 cho toàn bộ A5 ngay lúc này | **`[x]` ĐÃ XONG (D2, merged PR #15).** `notes/vlm-selection.md`: Colab T4 GO cho Qwen2.5-VL-3B (4.45 s/window, 2.93 GB peak, 16/16 parse). Handoff cho A2 sẵn sàng. |
| A2 | D1-D2 | Cài đặt `VLMVerifier.verify` trong `vad/verify/vlm.py`: frames + prompt từ `configs/prompts.yaml` -> `parse_verifier_json`; retry 1 lần nếu parse lỗi, sau đó raise ngoại lệ (không bao giờ tạo kết quả giả) | Unit test với model runner được mock; chạy thử (smoke run) trên 3 clip trả về `VerifyResult` thật | **`[ ]` ĐANG CHỜ.** `parse_verifier_json` đã có; stub `VLMVerifier.verify` ném `NotImplementedError` đúng chuẩn. |
| A3 | D2-D3 | `scripts/run_verify.py`: đọc `results/windows/<video>.json`, chọn k khung hình cách đều nhau cho mỗi window, ghi MỘT file json cho mỗi window, hỗ trợ resume (chạy tiếp khi bị ngắt) | Kill giữa chừng, chạy lại, script bỏ qua các window đã xong | **`[ ]` ĐANG CHỜ (D2-D3).** Đợi candidate windows từ TH6 / D4. |
| A4 | D2 (chạy qua đêm) | Chạy embedding shard 1 ngay khi B3 xong (`--shard 1/4`, tuỳ chọn `2/4`) | Số lượng file `cache/embeddings/*.npy` khớp với kích thước shard | **`[x]` ĐÃ XONG (THAY THẾ).** Toàn bộ shard (350 video × 2 model = 700 file) đã hoàn thành trên Colab T4. |
| A5 | D5 | Chạy tier 2 trên các candidate window (chấp nhận một TẬP CON có chi phí đo được, xem mục 6). Dùng Colab T4 nếu thời gian đo ở A1 cho thấy 3050 quá chậm. Bắt đầu ngay khi Thịnh bàn giao các window (cuối D4) | Log: tỉ lệ window leo thang (%), số giây/window, peak VRAM. Cung cấp số liệu cho dòng ablation A6 | **`[ ]` ĐANG CHỜ (D5).** |
| A6 | D5 | Kiểm tra thủ công 20 window; tinh chỉnh verifier prompt ĐÚNG MỘT LẦN; so sánh giữa chỉ có yes/no vs caption+type | Bản ghi chú ngắn kèm precision trước/sau trên 20 window đó | **`[ ]` ĐANG CHỜ (D5).** |
| A7 | D5-D6 | Tuỳ chọn: phân tích dương tính giả (false-positive) của VLM, 10 trường hợp có chú thích cho báo cáo. Chỉ làm nếu A5/A6 đã hoàn thành | 10 trường hợp có chú thích, hoặc tuyên bố bỏ qua rõ ràng | **`[ ]` TUỲ CHỌN (D5-D6).** |

Người chịu trách nhiệm rủi ro cho: VLM không vừa 6 GB VRAM. Quy tắc quyết định: kết quả A1 vào cuối D1 là quyết định cuối cùng; không để đến D5 mới phát hiện ra.

### B. Huy: data + kết nối embedder (đường găng) + clip tự quay

| ID | Thời gian | Task | Tiêu chí nghiệm thu | Trạng thái (cuối D1) |
|---|---|---|---|---|
| B3 | D1-D2 (**làm trước, mọi người chờ cái này**) | Nối embedder thật trong `vad/embed/siglip.py`, tái sử dụng `search.load_model`, `embed_images`, `embed_texts`. Làm cho model-generic (theo HF id) để CÙNG một class phục vụ được cả SigLIP 2 và OpenAI CLIP B/32 (cần cho A0). Cài đặt việc chạy thật trong `scripts/precompute_embeddings.py` (sampler -> batch -> `save_embeddings`, có thể resume) | 3 video được trích xuất embedding trên GPU của Huy; cosine giữa fp16 và fp32 > 0.999; ghi lại tốc độ ảnh/giây; chạy được với tham số `--shard`; nhận 2 model trong một lượt decode (SigLIP 2 + CLIP B/32). **Bắt buộc phải merge trước giữa trưa D2**, nếu không toàn bộ dây chuyền bị trễ | **`[x]` ĐÃ XONG.** Nam làm thay và merge sớm (PR #8). Chạy 1 lượt decode 2 model. Đã đo cosine fp16 vs fp32 trên video thật trên Colab T4 (min cos 0.99997, `notes/embedder-benchmark.md`). |
| B1 | D1 (tải dữ liệu trong lúc code B3) | Kiểm tra dung lượng ổ đĩa trước. Tải tập UCF-Crime test + file chú thích thời gian + ~60 video train-normal (dùng hiệu chuẩn). Tạo `manifests/ucf_test.csv`, `manifests/ucf_calib_normal.csv` dùng `probe_video` để lấy `n_frames`/`fps` | 290 dòng test (140 bất thường + 150 bình thường) và ~60 dòng calib; không trùng lặp ID; `precompute --dry-run` chạy tốt trên cả hai | **`[x]` ĐÃ XONG.** PR #13 đã merge (`0868fec`). 290 dòng test + 60 dòng calib; script tải và tạo manifest chạy tốt; dry-run xanh. |
| B2 | D2 | Đối chiếu quy ước nhãn với code của LAVAD / RTFM: `[start, end]` là lấy cận (inclusive) hay không, index từ 0 hay 1, 30 fps? Sửa `vad/datasets/ucf_crime.py` và test tương ứng nếu kết quả khác. Viết `notes/ucf-gt-convention.md` dẫn nguồn | Note dẫn chính xác các dòng code nguồn; toàn bộ 140 tên video có chú thích đều tồn tại trong manifest | **`[x]` ĐÃ XONG SỚM (D1).** PR #13 đã merge (`0868fec`). `notes/ucf-gt-convention.md` xác nhận inclusive, 0-indexed, 30 fps; khớp LAVAD/RTFM; không cần sửa parser. |
| B5 | D2 (chạy qua đêm) | Embedding shard 2 (35%) | Hoàn thành shard, các file cache có mặt đầy đủ | **`[x]` ĐÃ XONG (THAY THẾ).** Toàn bộ shard đã hoàn thành qua lượt chạy hợp nhất trên Colab T4. |
| E3 | D2-D4 | Tự quay clip đường phố Việt Nam: **khoảng 5** clip dài 10-30 giây, ít nhất 2 clip bình thường; các clip "sự cố" phải là DÀN DỰNG với sự đồng thuận, không quay người thật khi chưa có phép. Gán nhãn thủ công giây bắt đầu/kết thúc. `manifests/own_demo.csv` + file nhãn. Chỉ dùng cho demo, KHÔNG dùng làm benchmark | Commit manifest + nhãn | **`[ ]` ĐANG CHỜ (D2-D4).** Task chính tiếp theo của Huy. |
| E4 | D5 | Chạy các clip tự quay qua pipeline trên GPU của Huy | JSON kết quả cho từng clip | **`[ ]` ĐANG CHỜ (D5).** |
| B7 | D5 | Năng lực dự phòng: hỗ trợ Phước chạy batch tier-2 (chia window theo hash, tương tự như chia shard) | n/a | **`[ ]` ĐANG CHỜ (D5).** |
| B4 | **STRETCH** | ShanghaiTech: chỉ làm nếu tier 1 đóng băng vào D4 mà vẫn còn thời gian. `vad/io/frames.py`, `vad/datasets/shanghaitech.py`. Nếu không thì bỏ qua | Xem mục 6 | **`[ ]` STRETCH.** |
| B6 | **STRETCH** | Trích xuất embedding ShanghaiTech + ngân hàng normal-memory. Phụ thuộc vào B4 | Xem mục 6 | **`[ ]` STRETCH.** |

### C. Nam: cảnh báo, schema kết quả, bộ phân loại sự kiện (event typer)

| ID | Thời gian | Task | Tiêu chí nghiệm thu | Trạng thái (cuối D1) |
|---|---|---|---|---|
| C1 | D1 (**làm trước, mở đường cho dashboard**) | `vad/results.py`: dataclass có phiên bản (`schema_version`) + đọc/ghi JSON cho một lần chạy của video: sample_fps, mảng điểm, window, kết quả verify, cảnh báo, metadata (model, commit, timestamp). Cố định cấu trúc lưu trên ổ cứng trong thư mục `results/`. Commit một file JSON mẫu viết tay | Test chuyển đổi hai chiều (round-trip test); Thịnh xác nhận có thể code ứng dụng dựa trên schema đó. **Merge trước cuối D1** | **`[x]` ĐÃ XONG.** PR #9 đã merge (`c1a0580`). `RunResult` dataclass, test round-trip, `results/sample_run.json`. Thịnh đã chấp nhận schema. |
| C2 | D2 | Xử lý nợ kỹ thuật của policy G1-G3: cài đặt `cooldown_s` theo từng (camera, event); viết hàm tính `co_events` từ các window chồng lấn thời gian; biến `responders_in(zone)` thành hàm chạy thật hoặc xoá nó đi | Test: cảnh báo thứ hai nằm trong thời gian cooldown bị chặn và được phép sau khi hết cooldown; hai window chồng lấn kích hoạt leo thang cảnh báo end-to-end | **`[x]` ĐÃ XONG SỚM (D1).** Đã merge ở `9cf6561`. Cooldown, leo thang `co_events`, và truy vấn độ phủ thực tế trong `vad/graph/policy.py` & `alerts.py`. |
| C4 | D2 (chạy qua đêm) | Embedding shard 3 (25%) trên card 1060. Kiểm tra D1: `torch.cuda.get_arch_list()` có chứa `sm_61` không. Nếu không, ghim bản torch cũ hơn hoặc chuyển shard cho một lượt chạy trên Colab | Hoàn thành shard, hoặc bàn giao vào D1 | **`[x]` ĐÃ XONG.** Đã xác nhận `sm_61` trên GTX 1060; toàn bộ shard đã hoàn thành qua lượt chạy hợp nhất trên Colab T4. |
| C3 | D3 | Zero-shot event typer: lấy giá trị trung bình theo window của cached embeddings so với 13 prompt `event_typing` -> `EventType`. Đánh giá trên các đoạn UCF GT: độ chính xác top-1 trên 13 lớp | Độ chính xác so sánh với mức ngẫu nhiên 1/13; giữ TÁCH BIỆT với AUC phát hiện bất thường. Ma trận nhầm lẫn (confusion matrix) là tuỳ chọn | **`[x]` ĐÃ XONG SỚM (D1).** Đã merge ở PR #12 (`9bf274d`). `vad/score/typing.py`, `scripts/eval_typer.py`. Kết quả trên UCF GT: strict 40.4%, lenient 50.0% so với ngẫu nhiên 7.7% (`results/typing_eval.json`). |
| C5 | D4 | `vad/alerting.py`: windows + kết quả verify + ngữ cảnh (zone, hour) -> cảnh báo đã hợp nhất (GraphPolicy + `merge_alerts` + cooldown) | Bảng kiểm thử kịch bản có ít nhất 10 ca, tất cả đều pass (xanh) | **`[x]` ĐÃ XONG SỚM (D1).** Đã merge ở PR #11 (`9751e31`). Bảng kiểm thử kịch bản 10+ ca đều xanh. |
| C7 | D5 | `scripts/build_results.py`: cached embeddings + windows + đầu ra tier-2 -> cảnh báo -> `results/*.json` cho toàn bộ video | Dashboard đọc được kết quả thật | **`[ ]` ĐANG CHỜ (D5).** |
| C6 | **TUỲ CHỌN** | Một truy vấn độ phủ (coverage query) thực tế + xuất đồ thị con của policy sang DOT. Là mục đầu tiên cần cắt nếu thiếu thời gian | Xem mục 6 | **`[x]` XONG MỘT PHẦN.** Truy vấn độ phủ thật (`responders_in`, `coverage_gaps`, `events_requiring`) đã xong; xuất DOT là tuỳ chọn. |

### D. Thịnh (mã task TH1-TH8, E1-E7): tier-1, benchmark, ablations + báo cáo + toàn bộ dashboard

| ID | Thời gian | Task | Tiêu chí nghiệm thu | Trạng thái (cuối D1) |
|---|---|---|---|---|
| TH1 | D1 | Đọc kỹ giao thức của LAVAD (prompt, CLIP B/32, cách gộp AUC); chuẩn bị notebook Colab để chạy một shard; xác nhận CI vẫn xanh | Một trang ghi chú giao thức; notebook Colab chạy được `--dry-run` | **`[x]` ĐÃ XONG.** PR #10 đã merge (`4ac605b`). `notes/lavad-protocol.md`, `notebooks/colab_shard.ipynb`. Đã chạy thật toàn bộ trên Colab T4. |
| TH2 | D1-D2 | `vad/pipeline.py` (tier-1): cached embeddings -> điểm text -> memory tuỳ chọn -> phép biến đổi chuỗi thời gian (temporal ops) -> candidate windows, tất cả các công tắc ablation nằm trong một config dataclass | Test với `FakeEmbedder`; cùng một config phải tái lập được cùng số điểm | **`[x]` ĐÃ XONG.** PR #10 đã merge (`4ac605b`). `PipelineConfig`, `ABLATIONS` A0..A5, `run_tier1`, 23 unit test đều xanh. |
| TH3 | D2-D3 | `scripts/run_benchmark.py`: với mỗi video trong manifest: tải cache, chấm điểm, `expand_to_frames`, GT, micro AUC + bootstrap CI theo cấp video; ghi `results/ablation_*.json` | Chạy end-to-end thành công trên tập con 5 video trước | **`[ ]` TIẾP THEO (D2).** Đã thông đường: manifest, bộ nạp GT, pipeline, và toàn bộ 700 embedding cache đã sẵn sàng. |
| TH4 | D3 (**gate**) | Tái lập baseline A0: CLIP B/32, 2 prompt baseline. Đồng thời kiểm tra chéo giữa các máy (cùng 1 video trích xuất trên 2 máy khác nhau, cosine > 0.999) | A0 nằm trong khoảng lệch 3 điểm so với mốc 53. Nếu không: DỪNG mọi việc khác, debug lại giao thức cùng B2 | **`[ ]` ĐANG CHỜ (D2-D3).** Kiểm tra chéo không cần nữa (chạy cùng 1 GPU T4). Cổng A0 hướng tới mốc ~53 AUC qua TH3. |
| E1 | D2-D3 | Bộ nạp dữ liệu dashboard: `app/data.py` đọc `results/*.json`; giữ dữ liệu giả phía sau cờ `--demo` rõ ràng kèm biểu ngữ thông báo | Dashboard chạy được trên JSON `RunResult` mẫu của Nam | **`[x]` ĐÃ XONG SỚM (D1).** PR #10 đã merge (`4ac605b`). `app/data.py` đọc `results/*.json` và `results/sample_run.json`. |
| E2 | D3-D5 | Toàn bộ ứng dụng Streamlit (`app/streamlit_app.py`): trình phát video, timeline điểm số kèm vạch ngưỡng, danh sách cảnh báo với chú thích VLM, mức độ nghiêm trọng/lực lượng phản ứng/quy trình xử lý, đường dẫn giải thích, chế độ replay. Giao diện tiếng Anh | Chạy được trên JSON mẫu vào D3, trên kết quả thật vào D5 | **`[~]` SKELETON ĐÃ CHẠY.** Đọc được sample JSON; tiếp tục hoàn thiện giao diện D3-D5. |
| TH5 | D4 | Ablations A1-A5 + hiệu chuẩn ngưỡng trên `calib_normal` tại một mức FPR mục tiêu | Bảng kết quả kèm CI, các ngưỡng được ghi lại đầy đủ | **`[ ]` ĐANG CHỜ (D4).** |
| TH6 | Cuối D4 | Đóng băng tier 1: `notes/tier1-results.md`; xuất `results/windows/*.json` cho A3/A5 | Phước có các window vào cuối D4 | **`[ ]` ĐANG CHỜ (D4 EOD).** |
| TH7 | D5 | Chỉ số A6 (hậu VLM) và các chỉ số vận hành TỐI THIỂU: FA/giờ (báo động giả mỗi giờ) và độ trễ tại ngưỡng đã hiệu chuẩn. AUC theo từng lớp chỉ làm nếu còn thời gian | Bảng số liệu; phân tích theo lớp là tuỳ chọn | **`[ ]` ĐANG CHỜ (D5).** |
| TH8 | D1-D6 | **Báo cáo (tiếng Việt)**: hoàn thành khung báo cáo vào D2 (nghiên cứu liên quan + phương pháp), cập nhật các chương kết quả khi có số liệu, bản v1 vào D5, slides + bản cuối vào D6 | Bản v1 được cả 4 người xem xét vào D5 | **`[x]` KHUNG ĐÃ XONG.** PR #10 đã merge (`4ac605b`). `report/` 01..06 + `refs.md` với các vị trí đánh dấu `TBD`. |
| E5 | **TUỲ CHỌN** | Khung nhìn đồ thị chính sách (render file DOT từ C6) trên dashboard | Đồ thị hiển thị trên UI | **`[ ]` TUỲ CHỌN.** |
| E6 | D5 | Hình ảnh cho báo cáo: biểu đồ timeline điểm số cho 5 video dạng PNG trong `results/figures/`. Bộ sưu tập ca thất bại (failure-case gallery) là TUỲ CHỌN. Thêm `matplotlib` vào `requirements-app.txt` nếu dùng | 5 file PNG | **`[ ]` ĐANG CHỜ (D5).** |
| E7 | D6 | Video quay màn hình 2-3 phút làm phương án dự phòng khi không có GPU; hướng dẫn chạy dashboard trong `vad/README.md` | Video phát bình thường; các bước trong README chạy được trên bản clone sạch | **`[ ]` ĐANG CHỜ (D6).** |

## 4. Đường găng (critical path) và các điểm bàn giao (handoffs)

```
B3 embedder (giữa trưa D2) ─┬─> shards A4/B5/C4 (qua đêm D2) ─> TH3 benchmark ─> TH4 GATE (A0 ~53, D3) ─> TH5 ablations (D4) ─> TH6 windows (cuối D4)
B2 quy ước nhãn (D2) ──────┘                                                                                     │
C1 schema (cuối D1) ───────────────> dashboard E1/E2 (Thịnh)                                                     v
A1 đo đạc VLM (D1) ─> A2/A3 (D1-D3) ──────────────────────────────────────────> A5 tier-2 (D5) ─> C7 build_results (D5) ─> dashboard chạy thật (D5)
                                                                                                                      └─> đóng băng cuối D5 ─> D6 báo cáo + nộp
```

Dây chuyền này chỉ có ĐÚNG 1 ngày dự phòng. Nếu B3 không được merge trước giữa trưa D2, hoặc TH4 thất bại vào D3, kích hoạt danh sách cắt giảm ngay lập tức.

| Điểm bàn giao | Từ -> Đến | Điều kiện tiên quyết trước khi người tiếp theo bắt đầu | Trạng thái (cuối D1) |
|---|---|---|---|
| H1 | Huy B3 -> Phước A4, Nam C4, Huy B5 | Embedder thật + script precompute hỗ trợ resume đã merge vào `main` (giữa trưa D2) | **ĐÃ XONG** (PR #8). Toàn bộ shard đã xong trên Colab (350 video × 2 model = 700 file `.npy`). |
| H2 | Huy B1/B2 -> Thịnh TH3/TH4 | Manifests đã commit, ghi chú quy ước nhãn đã viết xong | **ĐÃ XONG** (PR #13, `manifests/ucf_test.csv`, `notes/ucf-gt-convention.md`). |
| H3 | Nam C1 -> Thịnh E1/E2 | `vad/results.py` đã merge, có JSON mẫu trong `results/` (cuối D1) | **ĐÃ XONG** (PR #9, Thịnh chấp nhận, loader `app/data.py` merge ở PR #10). |
| H4 | Thịnh TH6 -> Phước A3/A5 | `results/windows/*.json` cho tất cả video benchmark (cuối D4) | Đang chờ (lịch cuối D4; mở khoá bởi TH3/TH4). |
| H5 | Phước A5 -> Nam C7 -> Thịnh E2 | Có file JSON tier-2 cho từng window | Đang chờ (lịch D5). |

## 5. Lịch làm việc (ai làm gì mỗi ngày)

| Ngày | Phước | Huy | Nam | Thịnh |
|---|---|---|---|---|
| D1 29/10 | A1 đo VLM, bắt đầu A2 | bắt đầu B3, B1 tải dữ liệu | **C1 schema (merge trước cuối ngày)**, C4 kiểm tra kiến trúc | TH1, bắt đầu TH2, khung báo cáo |
| D2 30/10 | A2 xong, A3, A4 chạy qua đêm | **B3 merge trước giữa trưa**, B2, B5 chạy qua đêm, bắt đầu E3 | C2 xử lý nợ policy, C4 chạy qua đêm | TH2 xong, TH3, E1 loader dashboard |
| D3 31/10 | A3 xong | E3 quay clip | C3 typer | **TH4 gate**, E2 giao diện dashboard |
| D4 01/11 | chuẩn bị prompt, đợi windows | E3 xong, B4 chỉ làm nếu tier 1 đúng tiến độ | C5 alerting | TH5 ablations, **TH6 đóng băng cuối ngày**, E2 |
| D5 02/11 | **A5, A6** | B7 hỗ trợ A5, E4 clip tự quay | **C7 build_results** | TH7, E2 chạy thật, E6 hình ảnh, **báo cáo v1**. **ĐÓNG BĂNG CUỐI NGÀY** |
| D6 03/11 | A7 tuỳ chọn, đóng gói | đóng gói | đóng gói, README | báo cáo + slides bản cuối, E7 quay video, **nộp bài (hạn chót 23:59)** |

## 6. Cắt giảm phạm vi (quyết định ngay từ đầu vì dời deadline) và cân đối tải

Quyết định dứt khoát ngay từ đầu, không phải "chờ trễ hạn mới tính":
- **ShanghaiTech (B4, B6) là STRETCH.** UCF-Crime test là benchmark bắt buộc duy nhất. Chỉ bắt đầu B4 nếu tier 1 đóng băng vào D4 mà Huy rảnh. Nếu không, báo cáo sẽ ghi "nằm ngoài phạm vi do giới hạn thời gian" và bỏ phần thảo luận gán nhãn "training-free + normal reference".
- **A7 ablation bản small-256 bị huỷ.** Các ablation bắt buộc là A0..A5 cộng với dòng A6 (sau VLM).
- **Clip tự quay: khoảng 5 clip**, không phải 10-20.
- **Phân tích theo từng lớp, ma trận nhầm lẫn, render đồ thị C6 (E5), bộ sưu tập ca thất bại: tuỳ chọn.** Chỉ làm sau khi mọi thứ khác đã hoàn tất.
- **Tier 2 có thể chạy trên một tập con** của các candidate window, có báo cáo chi phí đo đạc cụ thể. Tuyệt đối không cắt bỏ hoàn toàn (ràng buộc đề tài VLM).
- Báo cáo được viết liên tục từ D1, không để dồn đến cuối (TH8).

Các rủi ro quá tải cụ thể:
- Thịnh: TH1-TH8 cộng toàn bộ dashboard (E1, E2, E6, E7). Các cắt giảm trên giữ cho phần việc này tập trung vào tính năng cốt lõi. Chế độ replay giúp tránh phải viết pipeline inference thời gian thực.
- Nam: pipeline backend C1-C7 và engine policy. Nếu gấp, cắt phần render đồ thị C6 và ma trận nhầm lẫn C3.
- Phước: điểm nghẽn rủi ro duy nhất (single point of failure) cho VLM. Quy tắc: A1 vào D1 quyết định chạy local hay Colab. Huy (B7) là cặp dự phòng.

Thứ tự cắt giảm nếu trễ hạn tiếp (cắt từ trên xuống, không bao giờ đụng vào phần dưới cùng):
1. Các mục tuỳ chọn kể trên (C6/E5, gallery ca thất bại, phân tích theo lớp, task A7)
2. Cắt bỏ toàn bộ ShanghaiTech (vốn đã là stretch)
3. Giảm clip tự quay xuống còn tối đa 3 clip
4. Độ phủ của Tier-2 thu hẹp xuống ít video hơn (nhưng vẫn phải đo đạc và báo cáo trung thực)
5. TUYỆT ĐỐI KHÔNG CẮT: Tier-2 (ràng buộc đề tài), gate A0 (TH4), bootstrap CI cấp video, video demo dự phòng (E7)

## 7. Quy ước làm việc (working agreements)

- Mỗi task một branch: `feat/<tên>-<id>` (ví dụ `feat/huy-b3-embedder`). Merge vào `main` qua PR; CI bắt buộc phải xanh.
  Với thời gian 6 ngày, PR được review trong ngày hoặc tác giả tự merge sau khi CI xanh và thông báo cho nhóm.
- `vad/types.py` và `vad/interfaces.py` đóng băng sau cuối D1: muốn sửa đổi phải được sự đồng ý của tất cả người dùng.
- Định dạng commit: `type(scope): message`, bằng tiếng Anh (khớp với lịch sử git của repo).
- Họp standup 10 phút hàng ngày: hôm qua làm gì, hôm nay làm gì, có bị nghẽn gì không. Bị nghẽn (blocker) phải đăng báo ngay khi gặp, không chờ đến lúc họp standup.
- Mỗi task chỉ được tính là "xong" khi tiêu chí nghiệm thu của nó thoả mãn VÀ `python -m pytest -q` xanh. Các hàm stub phải giữ nguyên dạng stub (`NotImplementedError`) cho đến khi làm thật; không ai được đưa ra kết quả giả mạo.

## 8. Checklist D1 (mọi người, 15 phút)

| Ai | Lệnh chạy | Báo cáo | Kết quả / Trạng thái (cuối D1) |
|---|---|---|---|
| tất cả | `git pull`, `pip install -r requirements-dev.txt`, `python -m pytest -q` | 246 passed? | **461 passed** (tăng từ 246 test skeleton, toàn bộ xanh, không cần GPU). |
| Nam | `python -c "import torch; print(torch.cuda.get_arch_list())"` | có chứa `sm_61`? | **Đã xác nhận `sm_61`**; toàn bộ shard đã hoàn thành qua lượt chạy Colab T4. |
| Huy, Phước, Nam | `nvidia-smi --query-gpu=name,memory.total --format=csv` và `df -h .` | dung lượng ổ đĩa trống trước khi tải | **Đã xác nhận**; dữ liệu 9.2 GB tải & xử lý trên Colab T4. |
| Phước | Kiểm tra VRAM từ task A1 | chạy VLM local: tiếp tục / huỷ (go / no-go) | **CHƯA XONG (`[ ]`)** — rủi ro lớn nhất còn lại của Ngày 1. |
| Thịnh | CI xanh trên `main`; chuẩn bị notebook Colab | gửi link | **ĐÃ XONG**; CI xanh, Colab T4 chạy xong 700 file cache + kiểm tra fp16/fp32. |
| Nam + Thịnh | Thống nhất file JSON `RunResult` mẫu (C1 mở đường cho E1) | chấp thuận schema | **ĐÃ XONG**; schema merge ở PR #9, chấp thuận và tích hợp ở PR #10 (`app/data.py`). |

## 9. Câu hỏi mở

1. Mọi người có chấp nhận vai trò được phân công không? Có task nào muốn hoán đổi?
2. Ai có tài khoản Colab/Kaggle còn hạn ngạch GPU? (chạy shard dự phòng, và phương án dự phòng cho tier-2)
3. Giảng viên có yêu cầu mẫu báo cáo (template) hay quy định số trang không? (khung báo cáo đang chờ thông tin này)
4. Dashboard khi báo cáo/bảo vệ có được phép tải kết quả lưu sẵn từ ổ đĩa không, hay bắt buộc phải chạy model trực tiếp (live)?
