# E3 — clip demo: trạng thái và checklist

**Trạng thái: DÙNG TẠM 5 CLIP STOCK PREVIEW THEO YÊU CẦU — KHÔNG ĐẠT E3 GỐC.**

Nhóm quyết định dùng 5 file đang có trong `datasets/own/` cho demo/E4. Đây là **stock preview tải về**, không phải
clip tự quay, nên không đáp ứng yêu cầu "clip tự quay, cảnh dàn dựng có consent" của E3. Các file chính thức đã được
tạo nhưng mọi dòng đều đánh dấu `source=stock_preview`, `valid_for_e3=false`, `staged=false`, `consent` để trống.
Không dùng các clip này trong báo cáo/bảo vệ như bằng chứng "clip tự quay" nếu chưa có license.

## 1. Nội dung thực tế của 5 clip

| File | Thời lượng | Nội dung thực tế | Vấn đề |
|---|---:|---|---|
| `own_normal_01.mp4` | 18.28 s | Giao thông TP.HCM (Landmark 81) | Stock preview, metadata ghi tác giả `Witthaya`; không phải clip tự quay |
| `own_normal_02.mp4` | 21.02 s | Vòng xoay nhìn từ trên cao | Stock preview, metadata ghi tác giả `kwhi02`; có watermark iStock |
| `own_fight_01.mp4` | 24.28 s | Người đánh nhau trên đường phố|  `Fighting`; metadata copyright |
| `own_steal_01.mp4` | 13.85 s | Giật túi có người thật, mặt rõ | Stock preview, watermark iStock, metadata copyright; không có consent |
| `own_vandalism_01.mp4` | 16.08 s | Thùng rác đã bị đổ, cảnh tĩnh | Không có hành vi xảy ra trong clip (chỉ hậu quả); metadata copyright |

Ba file có metadata `This video is subject to copyright.`; hai file còn lại có tên tác giả stock. Đây là **preview có
watermark**, không phải footage được cấp phép. Dùng trong bài nộp là rủi ro bản quyền.

## 2. File đã tạo

Deliverable chính thức (có thể commit):

- `manifests/own_demo.csv` — 5 dòng, `dataset=own`, `split=demo`, `path` tương đối theo gốc repo.
- `manifests/own_demo_labels.csv` — nhãn quan sát được, có cột `source`, `valid_for_e3`, ghi rõ cảnh nào không hợp lệ.

Bản local (gitignored, cùng nội dung, dùng để thử nhanh):

- `manifests/own_demo.local.csv`
- `manifests/own_demo_labels.local.csv`

Kiểm tra:

```bash
python scripts/precompute_embeddings.py --manifest manifests/own_demo.csv --dry-run
```

Nhãn hiện tại:

| Video | `event_type` | `start_s..end_s` | Ghi chú |
|---|---|---:|---|
| `own_normal_01` | `Normal` | — | Giao thông bình thường |
| `own_normal_02` | `Normal` | — | Vòng xoay bình thường |
| `own_fight_01` | `Unknown` | — | Chim đánh nhau, không phải người; không gán `Fighting` |
| `own_steal_01` | `Robbery` | 1.0..8.0 | Giật túi có vũ lực |
| `own_vandalism_01` | `Vandalism` | 0.0..16.0 | Chỉ hậu quả, không thấy người gây ra |

`own_fight_01` cố ý để `Unknown`: gán `Fighting` cho clip chim sẽ làm sai nhãn và làm hỏng demo event typing.

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
- Clip bình thường: `start_s`, `end_s` để trống.
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
