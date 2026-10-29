# Báo Cáo: Zero-Shot Video Anomaly & Surveillance Alerting

Thư mục này chứa khung báo cáo đồ án môn học (bằng tiếng Việt). Các thuật ngữ kỹ thuật, tên biến, tên mô hình và định danh mã nguồn được giữ nguyên dạng tiếng Anh.

## 1. Cấu trúc tài liệu

Báo cáo được chia thành các chương tương ứng với từng file Markdown:

- `01-gioi-thieu.md`: Đặt vấn đề, mục tiêu, đóng góp và phạm vi hệ thống (kèm các cắt giảm phạm vi theo kế hoạch).
- `02-nghien-cuu-lien-quan.md`: Tổng quan các phương pháp CLIP, SigLIP 2, LAVAD, RTFM và cơ chế xác thực bằng Generative VLM.
- `03-phuong-phap.md`: Kiến trúc hệ thống hai tầng (Tier-1 scoring + Tier-2 VLM), quy trình xử lý thời gian, policy graph và các nguyên tắc thiết kế chống rò rỉ nhãn.
- `04-thuc-nghiem.md`: Thiết lập thực nghiệm, tập dữ liệu UCF-Crime test 290 video, chỉ số AUC micro frame-level kèm bootstrap CI theo video, bảng ablation A0 - A6 và các chỉ số vận hành.
- `05-ket-luan.md`: Kết luận, các hạn chế thực tế và hướng phát triển tiếp theo.
- `06-phu-luc.md`: Lệnh tái lập thực nghiệm, bảng phần cứng và thông số kỹ thuật các máy trạm.
- `refs.md`: Danh mục tài liệu tham khảo.

## 2. Hướng dẫn biên dịch và xuất báo cáo

Báo cáo có thể ghép và xuất thành file PDF hoặc Microsoft Word bằng Pandoc:

### Xuất file Word (.docx)
```bash
pandoc -s -o report.docx \
  report/01-gioi-thieu.md \
  report/02-nghien-cuu-lien-quan.md \
  report/03-phuong-phap.md \
  report/04-thuc-nghiem.md \
  report/05-ket-luan.md \
  report/06-phu-luc.md \
  report/refs.md \
  --toc
```

### Xuất file PDF qua XeLaTeX
```bash
pandoc -s -o report.pdf \
  report/01-gioi-thieu.md \
  report/02-nghien-cuu-lien-quan.md \
  report/03-phuong-phap.md \
  report/04-thuc-nghiem.md \
  report/05-ket-luan.md \
  report/06-phu-luc.md \
  report/refs.md \
  --toc --pdf-engine=xelatex -V geometry:margin=2.5cm -V mainfont="DejaVu Serif"
```

## 3. Trạng thái từng chương và nguồn dữ liệu

| File | Chương | Trạng thái hiện tại | Nguồn số liệu / Nhiệm vụ phụ trách |
|---|---|---|---|
| `01-gioi-thieu.md` | Giới thiệu | Khung hoàn chỉnh | Thịnh (TH8) |
| `02-nghien-cuu-lien-quan.md` | Nghiên cứu liên quan | Khung hoàn chỉnh (đã chốt số liệu nền) | Thịnh (TH1, TH8) |
| `03-phuong-phap.md` | Phương pháp đề xuất | Khung hoàn chỉnh | Thịnh (TH2, TH8) + Nam (C1, C2, C5) + Phước (A2) |
| `04-thuc-nghiem.md` | Thực nghiệm & Đánh giá | Bảng rỗng (TBD) | Cần số liệu từ TH3, TH4 (A0 gate), TH5 (A1 - A5), TH7 (A6, FA/h, latency), C3 (typer top-1), A5 (VLM) |
| `05-ket-luan.md` | Kết luận & Hướng phát triển | Khung hoàn chỉnh | Thịnh (TH8) |
| `06-phu-luc.md` | Phụ lục kỹ thuật | Khung hoàn chỉnh | Huy (B1, B3) + Nam (C4) + Phước (A1) + Thịnh |
| `refs.md` | Tài liệu tham khảo | Khung danh mục | Thịnh (TH8) |
