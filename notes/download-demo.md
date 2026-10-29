# Tải dataset demo (E3)

Folder Drive: https://drive.google.com/drive/folders/1I01ZQn6-7GW7Rm96Lozf0IQEkPtCP7lA?usp=drive_link

## Cách 1 — gdown (khuyến nghị)

```bash
pip install gdown

gdown --folder "https://drive.google.com/drive/folders/1I01ZQn6-7GW7Rm96Lozf0IQEkPtCP7lA?usp=drive_link" \
  -O datasets/own --continue
```

`--continue` để chạy lại thì bỏ qua file đã tải xong và tải tiếp file dở.

Nếu folder chưa public (lỗi 401), chọn một trong hai:

- Drive → chuột phải folder → **Share** → **Anyone with the link** → Viewer, rồi chạy lại lệnh trên.
- Hoặc lấy cookie từ trình duyệt đang đăng nhập Google:

```bash
gdown --folder "https://drive.google.com/drive/folders/1I01ZQn6-7GW7Rm96Lozf0IQEkPtCP7lA?usp=drive_link" \
  -O datasets/own --cookies-from-browser chrome --continue
```

## Cách 2 — tải tay

Mở link → **Download all** → giải nén → chép các file `.mp4` vào `datasets/own/`.

## Kiểm tra

```bash
ls -lh datasets/own/
```

File phải giữ đúng tên trong `manifests/own_demo.csv` (ví dụ `own_normal_01.mp4`). Nếu gdown tạo thư mục con, dồn file lên:

```bash
find datasets/own -mindepth 2 -name '*.mp4' -exec mv -n {} datasets/own/ \;
```
