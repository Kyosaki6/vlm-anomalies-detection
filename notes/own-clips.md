# E3 — clip demo: trạng thái và checklist

**Trạng thái: BỘ CLIP DEMO 7 FILE — NGUỒN iStock PREVIEW, CHƯA ĐẠT E3 GỐC (`valid_for_e3=false`).**

Ngày 2026-10-31, Huy xác nhận cả 7 clip được tải từ trang tìm kiếm iStock dưới đây, chỉ là **bản preview có watermark,
chưa mua licence**. Đây không phải clip tự quay hay cảnh dàn dựng có consent. Vì vậy mọi dòng nhãn vẫn
`valid_for_e3=false`; không dùng các clip này trong báo cáo/bảo vệ như bằng chứng "clip tự quay", không dùng cho bất kỳ
số liệu benchmark nào. Video không được commit (`/datasets/` bị ignore); chỉ dùng cục bộ cho demo dashboard.

- Nguồn (theo xác nhận của nhóm): https://www.istockphoto.com/vi/search/2/film?family=creative&phrase=fight%20street
- Licence: chưa mua; bản preview/comp có watermark. Chưa có asset ID hoặc hoá đơn cho từng clip.
- Ngoại lệ cần kiểm tra: `own_normal_03` là 1920x1080 (6 clip còn lại 768x432).
- `own_fight_01` trên máy Huy là **bản local 607 frame (24.28 s)**, SHA-256 `d3be684e...`; bản canonical trên Drive là
  303 frame (12.14 s), SHA-256 `0f9f74e8...`. E4b dùng bản local theo quyết định của Huy; `python scripts/fetch_demo.py --check`
  sẽ báo lệch cho clip này. Muốn tái lập từ Drive phải upload lại bản 607 frame và cập nhật `EXPECTED_SHA256`, hoặc quay lại
  bản canonical rồi re-embed.

## 1. Lấy video

Video không nằm trong git (`/datasets/` bị ignore). Tải bằng:

```bash
pip install gdown                      # chỉ script này cần
python scripts/fetch_demo.py           # -> datasets/own/*.mp4, kiểm SHA-256 từng file
python scripts/fetch_demo.py --check   # chỉ kiểm file đang có
```

Script từ chối file lạ, thiếu file, hoặc SHA-256 lệch (`EXPECTED_SHA256` trong `scripts/fetch_demo.py`). Đổi bộ clip
trên Drive thì phải cập nhật bảng đó và chạy lại `build_manifest.py`.

| File | Thời lượng | Nhãn | Ghi chú |
|---|---:|---|---|
| `own_normal_01.mp4` | 18.28 s | `Normal` | iStock preview, giao thông TP.HCM, watermark, metadata tác giả `Witthaya` |
| `own_normal_02.mp4` | 21.02 s | `Normal` | iStock preview, vòng xoay nhìn từ trên cao, watermark iStock, tác giả `kwhi02` |
| `own_normal_03.mp4` | 6.16 s | `Normal` | iStock preview (theo xác nhận), 1920x1080, ngắn hơn yêu cầu E3 (10–30 s) |
| `own_fight_01.mp4` | 24.28 s | `Fighting` 0.0–12.0 | iStock preview; bản local 607 frame, SHA khác bản canonical 303 frame trên Drive; nhãn 0–12.0 s đã xem lại |
| `own_fight_02.mp4` | 12.68 s | `Fighting` 0.0–12.68 | iStock preview; yêu cầu ghi 0–13 s nhưng clip chỉ dài 12.68 s: kẹp `end_s` về thời lượng |
| `own_fight_03.mp4` | 15.42 s | `Fighting` 5.0–15.0 | iStock preview, watermark |
| `own_steal_01.mp4` | 13.85 s | `Stealing` 1.0–10.0 | iStock preview, giật túi, người thật mặt rõ, watermark, không consent |

Lưu ý: `own_steal_01` trước đây gán `Robbery` (1.0–8.0). Nay đổi sang `Stealing` 1.0–10.0 theo nhóm.

## 2. File đã tạo

Deliverable chính thức (có thể commit):

- `manifests/own_demo.csv` — 7 dòng, `dataset=own`, `split=demo`, `path` tương đối theo gốc repo.
- `manifests/own_demo_labels.csv` — nhãn theo giây, có cột `source`, `valid_for_e3`.
- `scripts/fetch_demo.py` + `tests/vad_suite/test_fetch_demo.py` — tải và kiểm bộ clip.

**Nhãn chỉ để đối chiếu tay: hệ thống hiện không có loader đọc `own_demo_labels.csv`.** Chỉ `own_demo.csv` được
`scripts/precompute_embeddings.py` đọc. Khi cần dùng nhãn trong code, phải viết loader riêng (và validate
`event_type` qua `EventType.from_label`). Clip bình thường để trống `start_s`/`end_s`.

Kiểm tra:

```bash
python scripts/fetch_demo.py
python scripts/precompute_embeddings.py --manifest manifests/own_demo.csv --dry-run
```

`--dry-run` chỉ liệt kê, không mở video. Dùng `python scripts/fetch_demo.py --check` và `ffprobe` để chắc file thật.

## 3. Nếu có thời gian, quay lại để đạt E3

Yêu cầu E3: khoảng 5 clip, **10–30 giây**, **ít nhất 2 clip bình thường**, cảnh sự cố phải **dàn dựng, có consent**,
gán nhãn start/end bằng giây. Không dùng cảnh bạo lực thật, không quay người lạ chưa xin phép.

Đề xuất 5 clip:

| File | Loại | Cách quay an toàn |
|---|---|---|
| `own_normal_01.mp4` | Normal | Đường phố/chợ/hẻm yên tĩnh, máy đứng yên, 15–20 s |
| `own_normal_02.mp4` | Normal | Bãi xe/sân chung cư, 15 s |
| `own_fight_01.mp4` | Fighting | 2 người bạn giả vờ cãi/đánh, **không va chạm, không vũ khí** |
| `own_steal_01.mp4` | Stealing | Một người lấy túi/điện thoại để trên ghế |
| `own_vandalism_01.mp4` | Vandalism | Một người đẩy đổ thùng rác/cọc tiêu |

Quy tắc quay:

- Quay ngang, 720p trở lên, 30 fps, MP4/H.264. Convert nếu điện thoại xuất HEVC/.mov.
- Để 2–3 giây đầu/cuối làm ngữ cảnh; sự cố không bắt đầu ở giây 0.
- Không quay mặt người lạ, biển số, số nhà. Cảnh dàn dựng phải có người tham gia đồng ý.
- Ghi lại ngày, người tham gia (tên viết tắt), hình thức consent (lời nói/giấy) vào mục 5.

Khi đã có 5 clip tự quay trong `datasets/own/`:

```bash
python scripts/build_manifest.py \
  --root datasets/own \
  --dataset own \
  --split demo \
  --expect 5 \
  --out manifests/own_demo.csv
```

`manifests/own_demo_labels.csv` dùng schema:

```csv
video_id,event_type,start_s,end_s,staged,consent,source,valid_for_e3,notes
own_normal_01,Normal,,,false,,self_recorded,true,cho buoi sang
own_normal_02,Normal,,,false,,self_recorded,true,bai xe chung cu
own_fight_01,Fighting,4.2,11.8,true,"verbal 2026-10-30, A/B",self_recorded,true,dan dung, khong va cham
own_steal_01,Stealing,3.5,10.0,true,"verbal 2026-10-30, C",self_recorded,true,lay tui tren ghe
own_vandalism_01,Vandalism,2.0,8.5,true,"verbal 2026-10-30, D",self_recorded,true,day do thung rac
```

Quy ước:

- `event_type` là giá trị trong `vad/types.py` `EventType` (`Normal`, `Fighting`, `Stealing`, `Vandalism`,
  `RoadAccidents`, ...).
- Clip bình thường: `start_s`, `end_s` để trống (ngoài nhãn = bình thường).
- Clip sự cố: một dòng cho mỗi đoạn; `start_s < end_s`, nằm trong thời lượng clip.
- `staged=true` và `consent` bắt buộc với clip sự cố.
- Không dùng `dataset`/`split` khác: phải là `dataset=own`, `split=demo`.
- Không đổi tên file sau khi tạo manifest; nếu đổi, chạy lại `build_manifest.py`.

Checklist:

- [ ] 5 clip, mỗi clip 10–30 giây.
- [ ] Ít nhất 2 clip `Normal`.
- [ ] Không còn file stock/watermark; tất cả là clip tự quay.
- [ ] `manifests/own_demo.csv` có 5 dòng, `dataset=own`, `split=demo`, `path` tương đối.
- [ ] `manifests/own_demo_labels.csv` phủ đủ 5 `video_id`; interval hợp lệ.
- [ ] Clip sự cố có `staged=true` + `consent`, `valid_for_e3=true`.
- [ ] `--dry-run` chạy không lỗi.
- [ ] Không commit video; chỉ commit manifest, labels, notes.

## 4. Ghi chú consent (điền khi quay clip mới)

- Ngày/địa điểm:
- Người tham gia (viết tắt):
- Hình thức consent:
- Cảnh dàn dựng, không có bạo lực thật:
- Người quay:
- Checksum:
  ```bash
  sha256sum datasets/own/*.mp4
  ```

## 5. Liên hệ E4

E4a (2026-10-31) đã chạy SigLIP 2 trên RTX 3050, fps 1, fp16: 7 file `.npy` trong `cache/embeddings/`.
Cross-card check (Colab T4 vs RTX 3050) trên `own_fight_02`: 13 frame, `min_cos=0.99998450`,
`mean_cos=0.99999279`, PASS `> 0.999`. Cache `own_fight_01` có 25 mẫu, khớp bản local 607 frame / 25 fps.

E4b dùng tier-1 A4c với tau đóng băng `1.1685914439041403` và calib stats từ `manifests/ucf_calib_normal.csv`;
tier-2 chạy `scripts/run_verify.py` với `Qwen/Qwen2.5-VL-3B-Instruct` (4-bit), rồi build vào `results/run_demo/`.
Mọi file demo mang note "demo, not benchmark", không dùng cho AUC hay bất kỳ số liệu benchmark nào.

### 5.1 Kết quả E4b (2026-11-01, local RTX 3050)

`results/run_demo/` có đủ 7 file; 4 clip có candidate window, 3 clip Normal không có window:
`own_fight_01` (11–12 s, score 1.1796), `own_fight_02` (4–13 s, 1.4370), `own_fight_03` (7–16 s, 1.4722),
`own_steal_01` (4–9 s, 1.3767). Tier-2 thật (không fake): 3/4 window được VLM xác nhận, tạo 3 alert.

| Clip | Nhãn tay | Window | VLM `is_anomaly` / type | Typer type | Alert |
|---|---|---|---|---|---|
| `own_fight_01` | Fighting | 11–12 s | true / Fighting | Abuse | high |
| `own_fight_02` | Fighting | 4–13 s | **false** / Fighting | Fighting (không áp vì bị reject) | none |
| `own_fight_03` | Fighting | 7–16 s | true / Assault | Vandalism | low |
| `own_steal_01` | Stealing | 4–9 s | true / Assault | Assault | high |

`own_fight_02` là ví dụ thất bại có ích cho defense: tier-1 score cao nhất (1.437) và caption mô tả đánh nhau,
nhưng VLM trả `is_anomaly=false`; theo protocol A6, window bị loại và không sinh alert.

Chi phí tier-2 trên 4 window: window đầu 594.63 s (cold start + tải/nạp model), 3 window sau median 4.03 s,
p90 16.74 s; không window nào bị chia chunk. Đây là số demo, không phải benchmark.

### 5.2 E4c — typer trên 4 candidate window

Chạy typer zero-shot trên cả 4 candidate window: chỉ `own_fight_02` đúng nhãn tay (`Fighting`); các window còn lại
typer trả `Abuse` (`own_fight_01`), `Vandalism` (`own_fight_03`), `Assault` (`own_steal_01`) — **1/4 top-1**, và
**0/3** nếu chỉ tính các window VLM đã xác nhận. Đây là quan sát định tính trên n=4 clip demo, không phải benchmark;
khớp với kết luận C7e rằng typer zero-shot còn yếu trên các lớp hành vi.

Sau khi có manifest chính thức, E4 chạy:

```bash
python scripts/precompute_embeddings.py \
  --manifest manifests/own_demo.csv \
  --shard 1/1 --fps 1 \
  --model google/siglip2-base-patch16-384 \
  --model openai/clip-vit-base-patch32
```

Sau đó dùng `scripts/build_results.py` / pipeline để ra JSON cho dashboard. Clip demo **không** dùng để tính AUC
hay bất kỳ số liệu benchmark nào.
