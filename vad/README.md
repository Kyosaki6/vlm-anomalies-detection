# vad/ — Zero-Shot Video Anomaly & Surveillance Alerting

Kế hoạch đầy đủ: `.omo/plans/zero-shot-video-vad.md`. File này chỉ nói **cái gì đã chạy thật, cái gì còn là việc của bạn**.

## Trạng thái từng module

| Module | Owner | Trạng thái | Ghi chú |
|---|---|---|---|
| `vad/types.py`, `vad/interfaces.py` | cả nhóm | **xong** | Hợp đồng. Đổi = báo cả nhóm. |
| `vad/fakes.py` | cả nhóm | **xong** | `FakeEmbedder`, `FakeVerifier` để làm việc không cần GPU. |
| `vad/datasets/ucf_crime.py` | Huy (T3) | **parser xong + test** | Còn: chốt quy ước biên frame (đóng/mở) với LAVAD/RTFM, loader ShanghaiTech, manifest thật. |
| `vad/io/video.py` | Thịnh (T1) | **xong + test** | Sampler tuần tự (grab), căn khớp `expand_to_frames`; test sinh video thật bằng OpenCV. Chưa thử trên video UCF-Crime thật. |
| `vad/datasets/sharding.py` | Thịnh (T1) | **xong + test** | Chia shard bằng SHA-256, ổn định giữa các máy. |
| `vad/embed/cache.py` | Thịnh (T1) | **xong + test** | `.npy` fp16, ghi nguyên tử, resumable. |
| `vad/embed/siglip.py` | Thịnh (T1) | **STUB** (`NotImplementedError`) | Nối vào `search.py` cũ. 1060: dùng fp32. |
| `scripts/precompute_embeddings.py` | Thịnh (T1) | **`--dry-run` chạy; chạy thật = STUB** | Sampler đã có; còn thiếu embedder thật. |
| `vad/score/text.py` | Thịnh (T1) | **xong + test** | Điểm text (ensemble prompt) + điểm memory, thuần numpy. |
| `vad/temporal/ops.py` | Thịnh (T1) | **xong + test** | normalize / EMA / median / top-k pool / candidate windows / căn frame. |
| `vad/eval/metrics.py` | Thịnh (T1) | **xong + test** | AUC frame-level micro + bootstrap CI theo **video**. Chưa có `run_benchmark.py`. |
| `vad/verify/vlm.py` | Phước (T2) | **parser JSON xong; `VLMVerifier` = STUB** | **Chưa chọn model.** Phải đo VRAM thật, không đoán. |
| `vad/graph/policy.py` + `configs/policy.yaml` | Nam (T4) | **chạy được + test** | Zone/giờ/leo thang/gộp cảnh báo + đường đi giải thích. Chưa có event typer zero-shot, chưa có coverage query thật (`responders_in` là placeholder). |
| `app/streamlit_app.py` | Thành (T5) | **skeleton chạy trên DỮ LIỆU GIẢ** | Cần đọc kết quả thật từ `results/`. |

## Chạy

```bash
pip install -r requirements-dev.txt            # test (CI cũng dùng)
pip install -r requirements-app.txt            # thêm streamlit cho dashboard
python -m pytest -q                            # 236 test, không cần GPU, không tải model
streamlit run app/streamlit_app.py             # dashboard demo (dữ liệu giả)
python scripts/precompute_embeddings.py --manifest manifests/<file>.csv --shard 2/4 --dry-run
```

## GPU từng máy (dùng để chia shard `--shard i/N`)

| Máy | GPU | Dtype khi tính | Vai trò |
|---|---|---|---|
| Phước | RTX 3050 6 GB | float16 | VLM (T2) + 1-2 shard embedding |
| Huy | RTX 3050 4 GB | float16 | 1 shard embedding. VLM không chắc vừa 4 GB, không nằm trên đường găng. |
| Nam | GTX 1060 3 GB | **float32** (Pascal fp16 rất chậm) | 1 shard embedding. Kiểm `torch.cuda.get_arch_list()` có `sm_61`. |
| Thịnh, Thành | không có | — | CPU + Colab T4 làm dự phòng |

## Luật chung

1. **Prompt phát hiện phải generic** (`configs/prompts.yaml`). Tên lớp (fighting, accident...) chỉ dùng ở event typing. Vi phạm = rò nhãn, như lỗi của bản 15 ảnh cũ.
2. **Không calibrate threshold trên tập test.** Dùng `split=calib_normal`.
3. **CI theo VIDEO, không theo frame** (`video_bootstrap_auc_ci`).
4. Mọi shard dùng **cùng model, cùng fps, cùng tiền xử lý**. Ngày 4 kiểm cosine > 0.999 giữa hai card cho cùng một video.
5. Test không được tải model hay cần GPU. Thứ gì cần GPU thì có bản `Fake*` để test.
6. Thư mục test là `tests/vad_suite/` (KHÔNG đặt là `tests/vad/`: nó che mất package `vad` khi pytest chạy).

## Quy ước manifest (CSV, `manifests/*.csv`)

`video_id,path,dataset,split[,n_frames,fps]` — `dataset` ∈ `ucf_crime|shanghaitech|own`, `split` ∈ `test|calib_normal|demo`. `video_id` không đuôi file, không trùng.
