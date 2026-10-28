# T4 — Event typer + Alert feed (2026-10-29)

Trạng thái: **xong phần code + test, chưa có số trên dữ liệu thật** (cần embedding SigLIP2 thật, hiện còn stub).
`python -m pytest -q` → **267 passed, 1 skipped** (trước file này: 236 passed). Không cần GPU, không tải model.

## Đã làm gì

### 1. `vad/score/typing.py` — event typer zero-shot

Trả lời: *"cửa sổ này bất thường kiểu gì?"* (13 lớp UCF-Crime). Chạy **sau** phát hiện, đo **riêng** (plan §7).

Cách hoạt động, từng bước:

1. Đọc `configs/prompts.yaml` → khoá `event_typing`. Có `template` và 13 `labels`, mỗi lớp một câu mô tả
   (vd `Fighting: "people fighting"`).
2. Ghép template: `"a surveillance video frame showing {label}"` → 13 câu. Đưa qua `embedder.embed_texts`
   → 13 vector, L2-normalize **một lần** ở `__init__` (không tính lại mỗi cửa sổ).
3. Với một cửa sổ có T khung hình, `classify(frame_embs)`:
   - `aggregate="mean"` (mặc định): trung bình T vector rồi normalize → "*cả cửa sổ đang diễn ra gì*".
   - `aggregate="max"`: mỗi lớp lấy cosine cao nhất trong T khung → "*có khoảnh khắc nào giống lớp này*".
4. Cosine với 13 vector lớp, nhân `logit_scale`, softmax → xác suất. Lấy top-1 + top-k.

Test chốt hành vi quan trọng nhất: `mean` và `max` **thật sự cho kết quả khác nhau** trên cùng dữ liệu
(1 khung rất giống lớp A giữa 5 khung hơi giống lớp B). Đó là lý do có hai chế độ, và là một dòng ablation đẹp.

**Cảnh báo trung thực:** `confidence` là softmax thô, **chưa calibrate**. Với `logit_scale=100`, chênh lệch
cosine 0.01 đã đẩy xác suất gần 1.0 → confidence gần như luôn bão hoà. **Khi báo cáo phải dùng top-1
accuracy, không được trình bày confidence như xác suất đúng.** (Argmax không bị ảnh hưởng vì logit_scale
là hàm tăng đơn điệu.) Muốn có confidence dùng được thì phải calibrate trên `calib_normal` — chưa làm.

### 2. `vad/graph/policy.py` — coverage query thật (thay placeholder)

`responders_in` cũ trả **mọi** responder bất kể zone → vô nghĩa. Nay:

| Hàm | Trả lời câu hỏi | Cài đặt |
|---|---|---|
| `responders_in(zone)` | "Zone này có ai trực?" | đọc `zones[..].available_responders` |
| `events_requiring(responder)` | "Ai tham gia sự kiện nào?" | duyệt ngược cạnh `requires` trong graph |
| `coverage_gaps(zone)` | "Zone này **thiếu** ai cho sự kiện nào?" | hiệu của `responders_in` với responder từng sự kiện cần |

`zones` trong `configs/policy.yaml` nay có `available_responders` (dữ liệu minh hoạ cho demo, không phải
bảng phân công thật — ghi rõ trong comment file). Zone không khai báo → `coverage_gaps` trả `{}`
("chưa có dữ liệu"), **không** đoán là "không ai ứng cứu".

Đây là câu trả lời cho câu hỏi bảo vệ *"tại sao dùng graph chứ không phải bảng if-else?"*: cùng một cấu trúc
trả lời được cả ba câu trên, không phải viết tay một truy vấn riêng cho mỗi câu.

### 3. `vad/graph/alerts.py` — từ danh sách sự kiện ra "bản tin cảnh báo"

`policy.to_alert` chỉ xử lý **một** sự kiện. Dashboard cần **cả video**. Ba bước:

1. **Leo thang theo thời gian thật**: hai cửa sổ **chồng lấn thời gian** trong cùng video → mỗi cái nhận
   sự kiện kia làm `co_events`. Vd tai nạn 12-24s + nổ 18-30s → chồng lấn → tự thêm `fire_dept`, nâng
   severity. Không chồng lấn thì không leo thang. (Trước đây `co_events` là tham số nhưng **chưa ai truyền**
   → escalation trong graph chưa bao giờ chạy thật.)
2. `merge_alerts` (có sẵn): cùng loại cách nhau < `merge_gap_s` → một sự cố.
3. `apply_cooldown` (**mới**): cùng video + cùng loại, bắt đầu < `cooldown_s` sau cảnh báo đã giữ → bỏ.
   `hygiene.cooldown_s` trước đây là **khoá chết trong YAML**, không code nào đọc.

Chi tiết dễ sai: cảnh báo bị bỏ **không** làm mới cooldown (mốc tính từ cảnh báo đã giữ), nếu không một sự
cố kéo dài sẽ tự chặn chính nó mãi mãi. Có test riêng cho việc này.

Ghi chú: `Alert` chưa có trường `camera`, nên cooldown tạm khoá theo `(video_id, event_type)`. Muốn có
camera thật phải sửa `vad/types.py` — file hợp đồng chung, **phải báo cả nhóm** trước khi sửa.

## Chạy thử (không cần GPU)

```python
import numpy as np
from vad.fakes import FakeEmbedder, FakeVerifier
from vad.graph.alerts import build_alert_feed
from vad.graph.policy import GraphPolicy
from vad.score.typing import ZeroShotEventTyper
from vad.types import EventType, VerifyResult, Window

typer = ZeroShotEventTyper(FakeEmbedder(dim=32))
emb = FakeEmbedder(dim=32).embed_images([np.arange(8), np.arange(8) + 1])
r = typer.classify(emb)
print(r.event_type.value, [(e.value, round(p, 3)) for e, p in r.top_k])

policy = GraphPolicy()
fake = FakeVerifier(event_type=EventType.ROAD_ACCIDENT, confidence=0.88)
det = [
    (Window("RoadAccidents001", 12, 24, 4.1), fake.verify([], {})),
    (Window("RoadAccidents001", 18, 30, 5.3),
     VerifyResult(True, EventType.EXPLOSION, "a vehicle catches fire", 0.91)),
]
for a in build_alert_feed(det, policy, zone="highway", hour=23):
    print(a.event_type.value, a.severity, a.responders, " -> ".join(a.path))
```

Kết quả thật của lần chạy này:

```
ALERT [12-24s] RoadAccidents  severity=critical
  responders: traffic_police, ambulance, fire_dept
  path: event:RoadAccidents -> severity:high -> time_band:night(+1) -> escalates_with:Explosion -> protocol:P-ACCIDENT

coverage_gaps("parking_lot") = {'Arson': ('fire_dept',), 'Explosion': ('fire_dept','ambulance'),
                                'RoadAccidents': ('traffic_police','ambulance'), 'Shooting': ('ambulance',)}
events_requiring("security") = ['Assault','Fighting','Shoplifting','Stealing','Unknown','Vandalism']
```

## Còn thiếu (việc tiếp theo của T4)

- [ ] **Đo accuracy typer trên UCF thật** — cần embedding thật (đang bị chặn bởi `vad/embed/siglip.py` stub).
      Không có số này thì không được ghi gì về typer lên slide.
- [ ] Ablation `mean` vs `max` (2 dòng, chạy sau khi có cache).
- [ ] Calibrate `logit_scale`/confidence trên `calib_normal` — nếu muốn dùng confidence để lọc.
- [ ] `Alert.camera` (cần báo nhóm vì `vad/types.py` là hợp đồng chung) → cooldown theo camera thật.
- [ ] Event typer chưa có Protocol trong `vad/interfaces.py` — hiện chỉ là class; nếu T2/T5 cần `FakeTyper`
      thì nên thêm (cũng là sửa file hợp đồng, báo nhóm).

## Câu hỏi có thể bị hỏi khi bảo vệ

**"Typer này học từ đâu?"** — Không học. Zero-shot: so cosine giữa embedding khung hình và câu mô tả lớp.
Không có gradient, không có tham số train.

**"Sao không gộp luôn vào phát hiện?"** — Vì phát hiện phải dùng prompt **generic**; nhét tên lớp vào là
rò nhãn (đúng lỗi của bản 15 ảnh cũ). Hai việc đo riêng: detection = "có bất thường không" (AUC),
typing = "bất thường kiểu gì" (accuracy 13 lớp).

**"Confidence 1.0 nghĩa là chắc chắn 100%?"** — Không. Là softmax thô chưa calibrate. Báo cáo dùng accuracy.

**"Cảnh báo trùng lặp xử lý sao?"** — Ba lớp: leo thang (chồng lấn thời gian), gộp (`merge_gap_s`),
cooldown (`cooldown_s`). Test riêng cho từng lớp và cho trường hợp sự cố kéo dài.

**"Vì sao graph?"** — Vì coverage query: `coverage_gaps("parking_lot")` trả về "zone này thiếu xe cứu thương
cho tai nạn" bằng cách duyệt graph, không cần một `if` cho mỗi cặp (sự kiện × zone). Bảng if-else làm được
việc tương tự nhưng phải viết tay từng câu hỏi.
