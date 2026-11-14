# Độ chính xác 2 cấp và IoU thời gian (14/11/2026)

Bổ sung cho C7e (`notes/2026-11-01-c7e-vlm-typing.md`). Không đổi mẫu chấm, không đổi giao thức
chọn ca. Chỉ thêm hai cách đọc trên đúng 62 cửa sổ đã chấm.

Lệnh đã chạy (đọc cache embedding có sẵn, không nạp video):

```bash
python scripts/eval_vlm_typing.py \
    --annotation datasets/ucf_crime/Temporal_Anomaly_Annotation_for_Testing_Videos.txt \
    --out results/vlm_typing_eval_v2.json
```

File gốc `results/vlm_typing_eval.json` không bị ghi đè. Số liệu dưới đây lấy từ
`results/vlm_typing_eval_v2.json`.

## Mẫu chấm — không đổi

62 cửa sổ / 56 video. Điều kiện vào mẫu giữ nguyên như C7e: VLM trả `is_anomaly = true`, video là
video bất thường, cửa sổ có ít nhất 1 mẫu cache (1 giây) nằm trong đoạn GT.

Bị loại và đếm riêng, khớp biên bản C7e: 64 cửa sổ xác nhận nhưng không giao GT, 2 cửa sổ chỉ chạm
mép GT dưới 1 giây, 1 cửa sổ xác nhận trên video toàn bình thường, 178 cửa sổ bị VLM loại.
62 + 64 + 2 + 1 + 178 = 307.

Hai con số gốc không đổi: VLM đúng nguyên tên 52/62 = 83.9% [74.6, 92.8], typer 15/62 = 24.2%
[14.1, 35.6], đoán mò 7.7%.

## Độ chính xác 2 cấp

Cấp 1 = đúng **nhóm** gần nghĩa (`LENIENT_GROUPS` của C3): Assault gọi thành Fighting vẫn qua cấp 1.
Cấp 2 = đúng **nguyên tên lớp**, và chỉ tính trong những ca đã qua cấp 1. Mẫu số của cấp 2 là số ca
qua cấp 1, không phải 62.

| | Cấp 1 — đúng nhóm | Cấp 2 — đúng tên trong số đã qua cấp 1 |
|---|---|---|
| VLM | 54/62 = 87.1% | 52/54 = 96.3% |
| typer | 24/62 = 38.7% | 15/24 = 62.5% |

VLM hầu như không bao giờ đúng nhóm mà sai tên: 54 ca đúng nhóm thì 52 đúng nguyên lớp, chỉ 2 ca
đúng nhóm nhưng sai tên. Typer đúng nhóm rồi vẫn sai tên 9/24 ca.

Cấp 1 của VLM chính là con số lenient 87.1% đã có trong C7e. Phần mới là cấp 2 có điều kiện: nó nói
rằng khi VLM đã vào đúng nhóm, nó gần như luôn gọi đúng tên.

## IoU thời gian

IoU giữa cửa sổ dự đoán và **hợp** các đoạn GT của video, tính bằng giây. Đoạn GT chồng nhau được
gộp trước khi chia, nên IoU không vượt 1. Trung bình không trọng số theo cửa sổ: cửa sổ dài không
được át cửa sổ ngắn.

Trên 62 ca: trung bình **0.332**, trung vị **0.226**, chỉ **16/62 = 25.8%** đạt IoU ≥ 0.5.

## IoU tách theo đúng/sai lớp

Câu hỏi: cửa sổ gọi đúng tên có khít đoạn GT hơn cửa sổ gọi sai không? Không.

| Nhóm | n | IoU trung bình | Trung vị | IoU ≥ 0.5 |
|---|---:|---:|---:|---:|
| VLM đúng tên | 52 | 0.320 | 0.224 | 23.1% (12/52) |
| VLM sai tên | 10 | 0.395 | 0.327 | 40.0% (4/10) |
| typer đúng tên | 15 | 0.245 | 0.218 | 6.7% (1/15) |
| typer sai tên | 47 | 0.360 | 0.251 | 31.9% (15/47) |

15 ca typer đúng tên trùng khít 15 ca mà cả VLM lẫn typer đều đúng: không có ca nào typer đúng mà
VLM sai. Nhóm đó có IoU thấp nhất trong bảng.

Độ dài cửa sổ không khác nhau giữa hai nhóm. VLM đúng tên: trung vị độ dài 37.5 giây, phần giao GT
8.0 giây. VLM sai tên: 32.5 giây và 7.6 giây.

## Kết luận

Định vị không phải thứ kéo theo nhãn. Gọi đúng tên không làm cửa sổ khít đoạn sự cố hơn; nhóm gọi
sai còn có IoU cao hơn, dù n = 10 nên không kết luận chiều ngược lại.

Nút thắt là độ dài cửa sổ tier-1. Cửa sổ dài 6–46 giây được chấm như một nhát cắt, trong khi đoạn
GT ngắn hơn nhiều, nên IoU trung vị chỉ 0.226 dù nhãn đúng 83.9%. Muốn IoU cao phải siết biên cửa
sổ, không phải sửa bước gọi tên.

## Giới hạn, đọc kèm

1. IoU ở đây chỉ trên 62 ca đã lọt mẫu C7e, tức những cửa sổ VLM đã xác nhận và có giao GT. 64 cửa
   sổ xác nhận nhưng không giao GT có IoU = 0 và không nằm trong trung bình 0.332. Đưa chúng vào
   thì trung bình tụt xuống, và đó là một câu hỏi khác (độ chính xác định vị của mọi cửa sổ xác
   nhận), không phải câu hỏi của file này.
2. Mọi cảnh báo ở mục "Sáu điều phải đọc kèm" của C7e vẫn đúng: mẫu có điều kiện, lệch về lớp dễ
   nhìn, VLM được cho sẵn danh sách lớp, và mẫu được chọn bằng chính phán quyết của VLM.
3. Nhóm VLM sai tên chỉ có 10 ca. Chiều "sai tên thì IoU cao hơn" chưa đủ để thành kết luận; chiều
   được dùng ở trên là chiều âm — đúng tên không kéo IoU lên.
