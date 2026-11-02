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

Báo cáo được ghép từ các file Markdown bằng **Pandoc** (đã kiểm chứng ngày 02/11/2026: 32 trang, mục lục có
số trang, 5 hình in cỡ rộng, bảng không tràn lề).

### Xuất file Word (.docx)

```bash
pandoc -s --toc --metadata toc-title="Mục lục" --resource-path=report \
  --metadata title="Báo cáo đồ án cơ sở KHMT - Phát hiện bất thường video và điều phối cảnh báo" \
  --metadata lang=vi-VN -o report.docx \
  report/01-gioi-thieu.md report/02-nghien-cuu-lien-quan.md report/03-phuong-phap.md \
  report/04-thuc-nghiem.md report/05-ket-luan.md report/06-phu-luc.md report/refs.md
```

### Xuất file PDF

Đường đã chạy được trên máy Windows có Microsoft Word (không cần cài LaTeX):

1. Mở `report.docx` bằng Word.
2. Bấm chuột phải vào mục lục → **Update Field** → *Update entire table* (Pandoc tạo mục lục dạng field nên
   số trang chỉ đúng sau bước này).
3. Chèn số trang: **Insert → Page Number → Bottom of Page → Plain Number 2**, rồi căn giữa.
4. **File → Save As → PDF** (hoặc *Export → Create PDF/XPS*).

Đường thay thế không cần Word (cần cài MiKTeX/TeX Live, hiện **chưa có** trên máy nào của nhóm):

```bash
pandoc -s --toc -o report.pdf --pdf-engine=xelatex -V geometry:margin=2.5cm -V mainfont="DejaVu Serif" \
  report/01-gioi-thieu.md report/02-nghien-cuu-lien-quan.md report/03-phuong-phap.md \
  report/04-thuc-nghiem.md report/05-ket-luan.md report/06-phu-luc.md report/refs.md
```

> **Lưu ý định dạng:** hình trong chương 4 phải nằm **ngoài bảng** và có `{width=95%}`. Nếu đặt hình trong ô
> bảng, Pandoc/Word ép hình vào cột hẹp và hình chỉ còn ~1.3 inch, không đọc được (lỗi đã gặp ở bản nháp
> 02/11/2026). Bảng có từ 5 cột trở lên cần chỉnh độ rộng cột trong Word sau khi xuất.


## 3. Trạng thái từng chương và nguồn dữ liệu

| File | Chương | Trạng thái hiện tại | Nguồn số liệu / Nhiệm vụ phụ trách |
|---|---|---|---|
| `01-gioi-thieu.md` | Giới thiệu | Khung hoàn chỉnh | Thịnh (TH8) |
| `02-nghien-cuu-lien-quan.md` | Nghiên cứu liên quan | Khung hoàn chỉnh (đã chốt số liệu nền) | Thịnh (TH1, TH8) |
| `03-phuong-phap.md` | Phương pháp đề xuất | Khung hoàn chỉnh | Thịnh (TH2, TH8) + Nam (C1, C2, C5) + Phước (A2) |
| `04-thuc-nghiem.md` | Thực nghiệm & Đánh giá | Đã điền số liệu thật (A0 - A6, tầng 2, typing, vận hành) | Nguồn: `results/ablation_*.json`, `results/vlm_typing_eval.json`, `results/typing_eval.json`, `notes/a5b-log.md`, `notes/a6-handcheck.md`, `notes/a7-fp-analysis.md`. Còn chờ: hàng demo E4 (clip iStock, không tính vào benchmark) |
| `05-ket-luan.md` | Kết luận & Hướng phát triển | Khung hoàn chỉnh | Thịnh (TH8) |
| `06-phu-luc.md` | Phụ lục kỹ thuật | Khung hoàn chỉnh | Huy (B1, B3) + Nam (C4) + Phước (A1) + Thịnh |
| `refs.md` | Tài liệu tham khảo | Khung danh mục | Thịnh (TH8) |
