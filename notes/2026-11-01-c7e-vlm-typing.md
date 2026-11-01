# C7e — VLM gọi tên loại sự kiện tốt đến đâu, so với typer? (01/11/2026)

Lệnh đã chạy (máy Nam, CPU, đọc cache embedding có sẵn, không nạp video):

```powershell
python scripts/eval_vlm_typing.py --annotation datasets/ucf_crime/Temporal_Anomaly_Annotation_for_Testing_Videos.txt
```

Kết quả: `results/vlm_typing_eval.json`. Giao thức được viết trong docstring của
`scripts/eval_vlm_typing.py` TRƯỚC khi chạy và không sửa sau khi thấy số.

## Đơn vị chấm

Một **cửa sổ tier-1** thoả cả ba: VLM trả `is_anomaly = true`; video là video bất thường theo
annotation; cửa sổ giao với đoạn GT ít nhất 0 giây. Nhãn đúng là **lớp của video** (13 lớp UCF).
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

62 cửa sổ / 56 video. Bị loại và **đếm riêng**: 66 cửa sổ VLM xác nhận nhưng không giao GT,
1 cửa sổ xác nhận trên video toàn bình thường, 178 cửa sổ bị VLM loại,
0 cửa sổ thiếu nhãn VLM, 0 video thiếu file kết quả (62 + 66 + 1 + 178 = 307).

## Bốn điều phải đọc kèm, nếu không con số 83,9 % sẽ bị hiểu sai

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

## Vì sao typer vẫn là bên quyết định nhãn cảnh báo (D-2)

Nếu chỉ đọc bảng trên thì có vẻ D-2 chọn sai bên. Lý do của D-2 là **tính tất định và tái lập được**:
vector 13 câu cố định, cùng không gian embedding với tier-1, không phụ thuộc model sinh (đổi model
VLM là nhãn đổi), và kiểm lại được bằng cache. Nhưng báo cáo phải nói thẳng: **trên mẫu này VLM gọi
tên tốt hơn typer**, và D-2 không được viện lý do "typer chính xác hơn". Nếu hội đồng hỏi, câu trả lời
đúng là: "chọn typer vì tất định và rẻ, không phải vì nó gọi tên giỏi hơn".

Ghi chú thêm (giả thuyết, chưa đủ bằng chứng): cửa sổ typer gọi đúng có trung vị 25,0 giây, gọi sai
38,0 giây — khớp với chuyện trung bình embedding của cửa sổ dài làm loãng sự kiện. VLM không có xu
hướng đó (37,5 vs 32,5 giây). n = 62, chưa nên viết thành kết luận.

## Bẫy đã sập khi làm, ghi lại để người sau không sập lại

Lần chạy đầu đọc `verify_results` từ `results/run_A4c/<video>.json` và ra **VLM 15/62 = typer 15/62**
— hai dòng bằng nhau tuyệt đối. Nguyên nhân: từ C7b (D-2), `apply_event_typing` **ghi đè**
`event_type` trong file kết quả bằng nhãn typer; nhãn VLM thô chỉ còn trong
`results/verify/<video>/window_NNNN.json` và trong `metadata.event_typing`. Script giờ đọc file thô,
có test `test_vlm_label_comes_from_raw_verdicts_not_from_the_typer_relabelled_run_file` để khóa lại.

Con số `0/8` trong biên bản sanity cũ (A5a) là 8 cửa sổ, không cùng đơn vị và không cùng mẫu — đừng
đặt cạnh 52/62 như hai phép đo của cùng một thứ.
