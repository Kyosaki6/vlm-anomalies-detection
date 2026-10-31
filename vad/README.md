# vad/ — Zero-Shot Video Anomaly & Surveillance Alerting

Kế hoạch đầy đủ: `notes/2026-10-28-plan-zero-shot-video-vad.md`; phân công: `notes/2026-10-29-task-assignment.md`.
**Deadline: nộp trước 04/11/2026** (hạn chót cuối ngày 03/11 23:59); freeze tính năng tối 02/11. Nhóm 4 người (Phước, Huy, Nam, Thịnh).
File này chỉ nói **cái gì đã chạy thật, cái gì còn là việc của bạn**.

## Trạng thái từng module

| Module | Owner | Trạng thái | Ghi chú |
|---|---|---|---|
| `vad/types.py`, `vad/interfaces.py` | cả nhóm | **xong** | Hợp đồng. Đổi = báo cả nhóm. |
| `vad/fakes.py` | cả nhóm | **xong** | `FakeEmbedder`, `FakeVerifier` để làm việc không cần GPU. |
| `vad/datasets/ucf_crime.py` | Huy (T3) | **parser xong + test** | Còn: chốt quy ước biên frame (đóng/mở) với LAVAD/RTFM, loader ShanghaiTech, manifest thật. |
| `vad/io/video.py` | Thịnh (T1) | **xong + test** | Sampler tuần tự (grab), căn khớp `expand_to_frames`; test sinh video thật bằng OpenCV. Chưa thử trên video UCF-Crime thật. |
| `vad/datasets/sharding.py` | Thịnh (T1) | **xong + test** | Chia shard bằng SHA-256, ổn định giữa các máy. |
| `vad/embed/cache.py` | Thịnh (T1) | **xong + test** | `.npy` fp16, ghi nguyên tử, resumable. |
| `vad/embed/siglip.py` | Thịnh (T1) | **xong + test (PR #8)** | Model thật, lazy load, fp32. **Model-generic**: cùng class chạy SigLIP 2 (padding 64) và OpenAI CLIP B/32 (77 token) cho baseline A0. Đo GTX 1060: **~14 ảnh/s**, VRAM 1.7–2.25 GB, dim 768, `logit_scale` thật **112.85**. Bẫy 5.x: `get_*_features` trả object → lấy `.pooler_output`; text không có `attention_mask`; bs=64 sụp còn 1.8 ảnh/s. |
| `scripts/precompute_embeddings.py` | Thịnh (T1) | **chạy thật (PR #8)** | `process_video` stream theo lô, lỗi 1 video không làm sập shard, resume theo cache. **`--model` lặp lại được**: SigLIP 2 + CLIP B/32 trong MỘT lượt giải mã video (cache ghi sau khi mọi model xong, không lệch nhau). Đã chạy end-to-end: video 10s → 20 khung → `.npy` (20, 768). Video hỏng báo lỗi, không nạp model. |
| `scripts/sanity_embedder.py` | Nam (T4 hỗ trợ) | **mới + chạy được** | Ảnh trong `data/` → text score + hạng lớp. Bắt lỗi plumbing trong 1 phút, không cần dataset. KHÔNG phải số liệu báo cáo. |
| `vad/score/text.py` | Thịnh (T1) | **xong + test** | Điểm text (ensemble prompt) + điểm memory, thuần numpy. |
| `vad/temporal/ops.py` | Thịnh (T1) | **xong + test** | normalize / EMA / median / top-k pool / candidate windows / căn frame. |
| `vad/eval/metrics.py` | Thịnh (T1) | **xong + test** | AUC frame-level micro + bootstrap CI theo **video**. Chưa có `run_benchmark.py`. |
| `vad/verify/vlm.py` | Phước (T2) | **xong + test (A2)** | `VLMVerifier.verify` nối model thật (lazy, 4-bit nf4), prompt `verifier` từ YAML (cache ở `__init__`), retry 1 lần (lần 2 thêm nudge "reply with JSON only" — không phải no-op với greedy) rồi `RuntimeError` — không bao giờ trả kết quả bịa. `device` nối thẳng vào `device_map`; 4-bit không có CUDA fail TRƯỚC khi tải processor. `is_anomaly` đọc được chuỗi "false" (bug `bool("false")==True` đã sửa). Default theo A1redo: `Qwen/Qwen2.5-VL-3B-Instruct` (T4: 4.45 s/cửa sổ, parse 16/16). Test dùng mock runner, không tải model. **Smoke run 3 clip thật**: PASS trên T4 (bảng kết quả trong PR #16). |
| `vad/graph/policy.py` + `configs/policy.yaml` | Nam (T4) | **xong + test** | Zone/giờ/leo thang/gộp cảnh báo + đường đi giải thích. Coverage query thật: `responders_in`, `events_requiring`, `coverage_gaps` (zone khai báo `available_responders`). |
| `vad/graph/alerts.py` | Nam (T4) | **xong + test** | `Alert.camera` (cooldown theo camera, lùi về video_id nếu chưa biết). `apply_cooldown` (dùng `hygiene.cooldown_s`, trước đây là khoá chết) + `build_alert_feed`: cửa sổ chồng lấn -> leo thang, rồi gộp, rồi cooldown. Đây là thứ dashboard ăn. |
| `vad/score/typing.py` | Nam (T4) | **xong + test** | Event typer zero-shot 13 lớp (cosine với prompt trong `configs/prompts.yaml`, không train). `aggregate='mean'|'max'`, `include_normal`. **Chưa đo accuracy trên UCF** (cần embedding thật). |
| `vad/results.py` | Nam (T4/C1) | **xong + test** | Hợp đồng file kết quả: `RunResult` <-> `results/<video_id>.json` + `results/windows/<video_id>.json`. `schema_version` chặn file lệch phiên bản; ghi nguyên tử; trường lạ được bỏ qua. File mẫu `results/sample_run.json` (dữ liệu GIẢ, tự khai trong `metadata.note`). |
| `app/streamlit_app.py` | Thịnh (T5) | **skeleton chạy trên DỮ LIỆU GIẢ** | Cần đọc kết quả thật từ `results/` — schema đã có ở `vad/results.py`. |

## Chạy

```bash
pip install -r requirements-dev.txt            # test (CI cũng dùng)
pip install -r requirements-app.txt            # thêm streamlit cho dashboard
python -m pytest -q                            # 364 test, không cần GPU, không tải model
streamlit run app/streamlit_app.py             # dashboard demo (dữ liệu giả)
python scripts/precompute_embeddings.py --manifest manifests/<file>.csv --shard 2/4 --dry-run
```

## GPU từng máy (dùng để chia shard `--shard i/N`)

| Máy | GPU | Dtype khi tính | Vai trò |
|---|---|---|---|
| Phước | RTX 3050 6 GB | float16 | VLM (T2) + 1-2 shard embedding |
| Huy | RTX 3050 4 GB | float16 | 1 shard embedding. VLM không chắc vừa 4 GB, không nằm trên đường găng. |
| Nam | GTX 1060 3 GB | **float32** (Pascal fp16 rất chậm) | 1 shard embedding. Kiểm `torch.cuda.get_arch_list()` có `sm_61`. |
| Thịnh | không có | — | CPU + Colab T4 làm dự phòng |

## Luật chung

1. **Prompt phát hiện phải generic** (`configs/prompts.yaml`). Tên lớp (fighting, accident...) chỉ dùng ở event typing. Vi phạm = rò nhãn, như lỗi của bản 15 ảnh cũ.
2. **Không calibrate threshold trên tập test.** Dùng `split=calib_normal`.
3. **CI theo VIDEO, không theo frame** (`video_bootstrap_auc_ci`).
4. Mọi shard dùng **cùng model, cùng fps, cùng tiền xử lý**. Ngày 4 kiểm cosine > 0.999 giữa hai card cho cùng một video.
5. Test không được tải model hay cần GPU. Thứ gì cần GPU thì có bản `Fake*` để test.
6. Thư mục test là `tests/vad_suite/` (KHÔNG đặt là `tests/vad/`: nó che mất package `vad` khi pytest chạy).

## Quy ước manifest (CSV, `manifests/*.csv`)

`video_id,path,dataset,split[,n_frames,fps]` — `dataset` ∈ `ucf_crime|shanghaitech|own`, `split` ∈ `test|calib_normal|demo`. `video_id` không đuôi file, không trùng.

