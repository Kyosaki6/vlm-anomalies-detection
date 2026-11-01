# C7e — VLM gọi tên loại sự kiện tốt đến đâu, so với typer? (01/11/2026)

Lệnh đã chạy (máy Nam, CPU, đọc cache embedding có sẵn, không nạp video):

```powershell
python scripts/eval_vlm_typing.py --annotation datasets/ucf_crime/Temporal_Anomaly_Annotation_for_Testing_Videos.txt
```

Kết quả: `results/vlm_typing_eval.json`. Giao thức được viết trong docstring của
`scripts/eval_vlm_typing.py` TRƯỚC khi chạy; sau đó có **làm rõ quy ước mép đoạn GT** (mục "Soát lại"
cuối file) — làm rõ này không đổi mẫu chấm (vẫn đúng 62 cửa sổ đó) và không đổi hai con số 52/62,
15/62, nên không phải sửa giao thức sau khi thấy kết quả.

## Đơn vị chấm

Một **cửa sổ tier-1** thoả cả ba: VLM trả `is_anomaly = true`; video là video bất thường theo
annotation; cửa sổ có **ít nhất 1 mẫu cache (1 giây)** nằm trong đoạn GT — đoạn GT hiểu là
`[start, end+1)` frame như `frame_labels` (xem mục "Soát lại" bên dưới về mép 1 frame). Nhãn đúng là
**lớp của video** (13 lớp UCF).
Hai dự đoán trên ĐÚNG cùng cửa sổ đó: nhãn `event_type` THÔ của VLM (`results/verify/...`) và
`ZeroShotEventTyper.classify` trên trung bình embedding của cửa sổ (cùng `window_sample_slice` mà
C7d dùng để gán nhãn cảnh báo).

## Số

| | Số đúng / n | Tỉ lệ |
|---|---|---|
| VLM (nguyên tên lớp) | 52 / 62 | **83,9 %** |
| typer zero-shot (nguyên tên lớp) | 15 / 62 | **24,2 %** |
| đoán mò đều 13 lớp | — | 7,7 % |
| đoán theo lớp phổ biến nhất trong mẫu (Shooting, 16 ca) | 16 / 62 | 25,8 % |
| VLM trả `Normal`/`Unknown` | 0 / 62 | 0,0 % |

Lenient (gộp nhóm tên gần nghĩa như C3): VLM 87,1 %, typer 38,7 %.

62 cửa sổ / 56 video. Bị loại và **đếm riêng**: 64 cửa sổ VLM xác nhận nhưng không giao GT, 2 cửa sổ
chỉ chạm mép GT 33 ms (dưới 1 mẫu), 1 cửa sổ xác nhận trên video toàn bình thường, 178 cửa sổ bị VLM
loại, 0 cửa sổ thiếu nhãn VLM, 0 video thiếu file kết quả (62 + 64 + 2 + 1 + 178 = 307).

## Sáu điều phải đọc kèm, nếu không con số 83,9 % sẽ bị hiểu sai

1. **Đây là mẫu CÓ ĐIỀU KIỆN, không phải toàn bộ 307 cửa sổ.** Chỉ 62 cửa sổ lọt cả hai điều kiện
   (VLM xác nhận + giao GT). Không suy ra "VLM gọi tên đúng 84 % mọi sự kiện UCF".
2. **Mẫu lệch về lớp dễ nhìn.** Shooting 16, Explosion 12, Burglary 9, Arson 8 = 45/62 = 73 %.
   Với các lớp này một dòng chữ "someone shooting a gun" gần như đọc từ khung hình. Typer 24,2 %
   thực ra **ngang mức đoán theo lớp phổ biến nhất** (25,8 %), tức trên mẫu này typer không có
   thông tin lớp nào đáng kể.
3. **VLM được cho sẵn danh sách tên lớp trong prompt** (`configs/prompts.yaml`, mục `verifier`) —
   đây là bài chọn trong 14 nhãn (13 lớp + `Normal`), không phải tự sinh nhãn. Prompt KHÔNG chứa
   nhãn của video, nên không phải rò rỉ GT, nhưng phải nói rõ là closed-set.
4. **Đây không phải độ chính xác của tier-2.** Mọi cửa sổ trong mẫu đều đã được VLM xác nhận là bất
   thường; câu hỏi "cửa sổ này có thật sự bất thường không" là việc của TH7c (precision, FA/giờ).
5. **Mẫu được chọn bằng chính phán quyết của VLM.** Điều kiện vào mẫu là `is_anomaly = true` do VLM
   trả, nên những cửa sổ VLM lúng túng (bị loại, 178 cửa sổ) đã không còn trong mẫu. Con số này là
   "VLM gọi tên đúng khi nó đã tự tin có sự cố", không phải "VLM gọi tên đúng trên mọi sự cố".
6. **Không đặt cạnh C3 (40,4 % / 50,0 %).** C3 chấm trên đoạn GT; C7e chấm trên cửa sổ tier-1 dài
   6-46 s và chỉ trên mẫu VLM đã xác nhận. Khác đơn vị, khác mẫu, không so trực tiếp được.

## Vì sao typer vẫn là bên quyết định nhãn cảnh báo (D-2)

Nếu chỉ đọc bảng trên thì có vẻ D-2 chọn sai bên. Lý do của D-2 là **tính tất định và tái lập được**:
vector 13 câu cố định, cùng không gian embedding với tier-1, không phụ thuộc model sinh (đổi model
VLM là nhãn đổi), và kiểm lại được bằng cache. Nhưng báo cáo phải nói thẳng: **trên mẫu này VLM gọi
tên tốt hơn typer**, và D-2 không được viện lý do "typer chính xác hơn". Nếu hội đồng hỏi, câu trả lời
đúng là: "chọn typer vì tất định và rẻ, không phải vì nó gọi tên giỏi hơn".

So cặp trên đúng 62 cửa sổ (đếm từ `results/vlm_typing_eval.json`): VLM đúng và typer sai = **37**,
typer đúng và VLM sai = **0**. Đọc cặp này để nói "VLM hơn typer" thay vì so hai KTC rời nhau.
Hướng nhầm của VLM (10 ca sai): 6/10 bị gọi là `Robbery` (`Shooting`->`Robbery` 4, `Arrest` 1, `Stealing` 1), 3/10 bị gọi là `Explosion` (`Arson` 2, `Shooting` 1), 1/10 là `Shooting`->`Shoplifting`. Một câu về hướng nhầm này trong báo cáo đáng giá hơn bảng confusion thô.

Ghi chú thêm (giả thuyết, chưa đủ bằng chứng): cửa sổ typer gọi đúng có trung vị 25,0 giây, gọi sai
38,0 giây — khớp với chuyện trung bình embedding của cửa sổ dài làm loãng sự kiện. VLM không có xu
hướng đó (37,5 vs 32,5 giây). n = 62, chưa nên viết thành kết luận. **Không đưa vào báo cáo.**

## Bẫy đã sập khi làm, ghi lại để người sau không sập lại

Lần chạy đầu đọc `verify_results` từ `results/run_A4c/<video>.json` và ra **VLM 15/62 = typer 15/62**
— hai dòng bằng nhau tuyệt đối. Nguyên nhân: từ C7b (D-2), `apply_event_typing` **ghi đè**
`event_type` trong file kết quả bằng nhãn typer; nhãn VLM thô chỉ còn trong
`results/verify/<video>/window_NNNN.json` và trong `metadata.event_typing`. Script giờ đọc file thô,
có test `test_vlm_label_comes_from_raw_verdicts_not_from_the_typer_relabelled_run_file` để khóa lại.

Con số `0/8` trong biên bản sanity cũ (A5a) là 8 cửa sổ, không cùng đơn vị và không cùng mẫu — đừng
đặt cạnh 52/62 như hai phép đo của cùng một thứ.

## Soát lại ngày 01/11 (sau khi mở PR): hai lỗi biên, một rò rỉ đường dẫn

1. **Mép phải của đoạn GT hụt đúng 1 frame.** `frame_labels` đánh dấu frame `start..end` ĐÓNG hai
   đầu, nên quãng thời gian là `[start/30, (end+1)/30)`; bản đầu lấy `end/30`, hụt 33 ms. Hệ quả đo
   được: hai cửa sổ chỉ chạm mép đoạn GT đúng 33 ms (`Explosion027_x264` cửa sổ 6–26 s với GT
   3,5–6,0 s; `Robbery048_x264` cửa sổ 31–46 s với GT 15,0–31,0 s) bị tính là "không giao" trong khi
   cách đánh nhãn của `frame_labels` nói chúng có chạm. Nếu ai đó sửa hàm theo hướng ngược lại, mẫu
   chấm thành **64 ca** thay vì 62 mà không có gì báo.
   Cách chốt: GT là khoảng nửa mở `[start, end+1)` **và** cửa sổ phải có ≥ 1 mẫu cache (1 s ở 1 fps)
   nằm trong đoạn GT. Hai cửa sổ chạm mép 33 ms rơi vào nhóm mới `confirmed_overlap_too_small`; 62 ca
   còn lại **giữ nguyên từng cửa sổ**, nên 52/62 và 15/62 không đổi. Lý do phải đổi chứ không chỉ ghi
   chú: một quy tắc mà người viết lại có thể ra 64 ca thì không tái lập được.
2. **Số trong file kết quả từng là đường dẫn tuyệt đối của máy Nam** (`E:\Personal\Hoc\...`) — file
   này nằm trong git công khai. Giờ ghi đường dẫn tương đối với repo (`results/run_A4c`,
   `results/verify`), có test `test_display_path_is_repo_relative_not_an_absolute_machine_path`.
3. **`--limit` từng ghi đè kết quả đầy đủ**: chạy `--limit 20` mà không truyền `--out` là ghi vào
   đúng `results/vlm_typing_eval.json`. Giờ trả mã 2 và bắt buộc `--out`.

Đã soát độc lập (script riêng, tự parse annotation và đọc `.npy`/`.npz` bằng numpy, không dùng
`vad`/`build_results`): ra đúng 62 ca, 52/62, 15/62, và **0** trường lệch (`gt`/`vlm`/`typer`/
`overlap_s`) so với file đã commit. Nhãn typer tính lại từ ma trận embedding cũng khớp 62/62 ca.

## Khoảng tin cậy 95 % (resample theo VIDEO, 56 video, 2 000 lần)

| | Tỉ lệ | KTC 95 % |
|---|---|---|
| VLM | 83,9 % | [74,6 ; 92,8] |
| typer | 24,2 % | [14,1 ; 35,6] |

Đọc kèm: KTC của typer **phủ mức 25,8 %** của "luôn đoán Shooting", tức trên mẫu này không chứng minh
được typer hơn đoán theo lớp phổ biến nhất; KTC của VLM thì tách hẳn khỏi mức đó. Resample theo cửa
sổ (sai) sẽ cho KTC hẹp hơn và che mất điều này — xem `bootstrap_accuracy_ci`.
Quy tắc gốc: `AGENTS.md`, mục ANTI-PATTERNS ("DO NOT report CIs resampled over frames; resample over videos").
