# Kiểm tra lại AUC tier-1 trên máy local — 31/10/2026

**Kết luận ngắn:** con số tier-1 dùng cho báo cáo — **73.00 % [64.45, 81.15]** — đã được tính lại trên máy
Nam bằng CPU, không cần GPU, và khớp **từng bit** với `results/ablation_A4c.json` đã đóng băng. Bản
`results/run_A4c` do C7c sinh ra (thứ sẽ giao cho dashboard) cũng cho đúng 73.00 khi tính lại; lệch 7.1e-8
chỉ vì điểm được làm tròn 6 chữ số khi ghi JSON (`SCORE_DECIMALS = 6` trong `vad/results.py`).

## Vì sao phải làm

Máy Nam chỉ có cache embedding, manifest và `results/ablation_A4c.json`; thư mục `datasets/` trống, nên
không thể tự kiểm tra con số 73.00 (script `run_benchmark.py` cần file annotation GT). Nếu không kiểm tra,
cả nhóm chỉ biết "Colab của Thịnh ra 73.00" — đủ để báo cáo, nhưng không đủ để trả lời nếu hội đồng hỏi
"làm sao biết con số này đúng?".

## Đã tải gì (không tải video)

Ba file meta từ Hugging Face `AllenXeon/ucf_crime`, thư mục `UCF_Crimes-Train-Test-Split/` — đúng nguồn mà
`scripts/download_ucf_crime.py` dùng (`REPO_ID`, `META_FILES`):

| File (trong `datasets/ucf_crime/`) | Bytes | SHA-256 |
|---|---:|---|
| `Temporal_Anomaly_Annotation_for_Testing_Videos.txt` | 15 574 | `3b9542413f2ed9e94f73bf0488c151b3d7d595a8d2ad30f524e9243a2ae2a17c` |
| `Anomaly_Test.txt` | 13 056 | `afcb2a6bffe7286607e08c864425964d8784cbd5b1f263e03b76025e11548de7` |
| `Anomaly_Train.txt` | 70 394 | `ea91e03de7581bfa6b139e663ebb7511ab1651c33427f85741d3f497a2027744` |

File annotation có 290 dòng, đủ 290 video trong `manifests/ucf_test.csv` (không thiếu video nào).
`datasets/` nằm trong `.gitignore` nên ba file này **không vào git**.

**Lỗ hổng còn lại:** `scripts/download_ucf_crime.py` chưa ghim revision của dataset HF. Nếu mirror sửa file,
con số có thể lệch trong lần chạy sau. Việc nên làm sau freeze: thêm `revision=<commit>` khi tải.

## Cách 1 — chạy lại benchmark từ cache (kiểm cả pipeline điểm)

```powershell
cd vlm-anomalies-detection
python scripts/run_benchmark.py --manifest manifests/ucf_test.csv --ablation A4c `
    --calib-manifest manifests/ucf_calib_normal.csv --n-bootstraps 1000 `
    --out-dir "$env:TEMP\bench_a4c_check"
```

Kết quả: `A4c: AUC micro 73.00  CI95 [64.45, 81.15]  macro 63.35  1111808 frame  17.06s`.
So với `results/ablation_A4c.json`: `auc_micro`, `ci_lo`, `ci_hi`, `auc_macro_mean_per_video`,
`n_frames`, `n_anomalous_frames` giống **hệt từng bit**; 290/290 AUC từng video giống hệt.

Ý nghĩa: toàn bộ pipeline tier-1 (cache -> điểm -> nhãn GT -> AUC + bootstrap) chạy lại được trên máy này
trong ~17 giây, không GPU. Con số trong báo cáo không phụ thuộc vào một lần chạy Colab may mắn.

## Cách 2 — kiểm tra chính file C7c giao cho dashboard

Đọc `results/run_A4c/<video_id>.json`, bung điểm 1 fps về đúng `n_frames` của video, lấy nhãn từ GT rồi tính
lại cùng công thức:

```python
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path("vlm-anomalies-detection")          # sửa nếu chạy từ chỗ khác
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import run_benchmark as rb
from vad.datasets.ucf_crime import load_gt
from vad.eval.metrics import fast_frame_level_auc, fast_video_bootstrap_auc_ci
from vad.temporal.ops import expand_to_frames

rows = rb.read_benchmark_manifest(ROOT / "manifests/ucf_test.csv")
ann = rb.load_annotations(rb.DEFAULT_ANNOTATIONS)
scores, labels = [], []
for r in rows:
    d = json.loads((ROOT / "results/run_A4c" / f"{r['video_id']}.json").read_text(encoding="utf-8"))
    scores.append(expand_to_frames(np.asarray(d["anomaly_scores"], float), 1.0, r["fps"], r["n_frames"]))
    labels.append(load_gt(ann, r["video_id"], r["n_frames"]))
print(fast_frame_level_auc(scores, labels))
print(fast_video_bootstrap_auc_ci(scores, labels, n_bootstraps=1000, seed=42))
```

Kết quả:

| Chỉ số | Tính lại từ `results/run_A4c` | File đóng băng | Lệch |
|---|---:|---:|---:|
| AUC micro | 0.729998135921125 | 0.7299980645982048 | 7.1e-8 |
| CI dưới | 0.644533096915353 | 0.6445330315156348 | 6.5e-8 |
| CI trên | 0.8115136755686703 | 0.8115136016658212 | 7.4e-8 |
| AUC từng video (140 video bất thường) | — | — | 0 (cả 140) |

Lệch cỡ 1e-7 là do điểm trong file JSON bị làm tròn 6 chữ số thập phân khi ghi, không phải lỗi pipeline:
khi in ra 2 chữ số như báo cáo, hai bên đều là 73.00 / [64.45, 81.15].

## Điều kết quả này KHÔNG chứng minh

- Không kiểm tier-2 (VLM, event typer), không kiểm ShanghaiTech, không kiểm dashboard.
- Không chứng minh quy ước GT "segment đóng cả hai đầu, 30 fps" đúng theo bài báo gốc. Nó chỉ chứng minh
  file GT trên máy **đúng là file mà lần chạy đóng băng đã dùng** — vì cùng file, cùng code, ra cùng số.
  Quy ước vẫn là giả định, ghi ở `notes/ucf-gt-convention.md`.

## Câu nói khi bảo vệ

"Tier-1 đạt AUC 73.00 % (CI 95 % 64.45–81.15) trên 290 video UCF-Crime test, 1.11 triệu frame. Con số này
nhóm em tính lại được trên laptop trong 17 giây từ cache embedding, không cần GPU, và khớp từng bit với
file kết quả đã lưu."
