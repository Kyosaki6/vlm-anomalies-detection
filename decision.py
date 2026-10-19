# -*- coding: utf-8 -*-
"""decision.py — quy tắc quyết định + chỉ số đánh giá (dùng bởi main.py và eval.py).

Hai câu hỏi hệ thống phải trả lời:

1. **Ảnh này giống sự kiện nào?**  -> ``search()`` (retrieval, top-k).
2. **Ảnh này có bất thường không?** -> ``anomaly_score()`` + ngưỡng.

Quy ước của nhóm (ghi rõ để không overclaim):

- 5 ảnh ``duong_pho_*`` được coi là **bình thường** (theo đề bài ENG-02).
- 10 ảnh ``danh_nhau_*`` + ``tai_nan_xe_*`` là **bất thường**.
- Điểm bất thường = ``1 - max cosine similarity`` tới **gallery chỉ gồm ảnh
  bình thường**. Càng xa ảnh bình thường => điểm càng cao.
- Ngưỡng mặc định ``DEFAULT_ANOMALY_THRESHOLD`` là ngưỡng **minh hoạ** cho bộ
  15 ảnh, chọn tròn giữa phân bố bình thường và bất thường; ``eval.py`` báo
  thêm kết quả leave-one-out để cho thấy ngưỡng nhạy thế nào với dữ liệu nhỏ.
  Đây không phải ngưỡng tổng quát cho camera thật.

Vì sao không dùng "độ khớp %" làm xác suất bất thường? Vì gallery có lẫn cả
ảnh bất thường, nên ảnh tai nạn vẫn có "độ khớp" cao với ảnh tai nạn khác.
Đo thử: AUC của ``max cosine tới toàn bộ gallery`` chỉ ~0,64 — không dùng được.
Điểm bất thường đúng phải so với **ảnh bình thường**, không phải ảnh giống nhất.
"""

from pathlib import Path

__all__ = [
    "NORMAL_PREFIX",
    "DEFAULT_ANOMALY_THRESHOLD",
    "category_of",
    "is_normal_category",
    "decide_abnormal",
    "binary_metrics",
    "roc_auc",
    "threshold_sweep",
    "best_f1",
    "loo_threshold_eval",
    "confusion_matrix",
]

NORMAL_PREFIX = "duong_pho"

# Ngưỡng minh hoạ cho bộ 15 ảnh (19/10/2026): chọn tròn 0.30, nằm giữa
# phân bố bình thường (cao nhất ~0.267) và bất thường (thấp nhất ~0.315,
# trừ 1 ảnh ngoại lệ 0.250). Không phải ngưỡng tổng quát cho camera thật —
# xem notes/2026-10-19-eval-binary.md mục 6.2 để thấy độ nhạy của ngưỡng.
DEFAULT_ANOMALY_THRESHOLD = 0.30


def category_of(filename: str) -> str:
    """'danh_nhau_3.jpg' -> 'danh_nhau' (chỉ bỏ số thứ tự ở cuối tên file).

    Dùng regex thay vì cắt theo '_' cuối: tên nhóm có thể chứa '_' (tai_nan_xe).
    """
    import re
    stem = Path(filename).stem
    match = re.match(r"^(.*)_(\d+)$", stem)
    return match.group(1) if match else stem


def is_normal_category(category: str) -> bool:
    """Nhóm 'duong_pho' là bình thường; mọi nhóm khác là bất thường."""
    return category == NORMAL_PREFIX


def decide_abnormal(anomaly_score: float, threshold: float = DEFAULT_ANOMALY_THRESHOLD) -> bool:
    """True nếu điểm bất thường >= ngưỡng."""
    return anomaly_score >= threshold


def binary_metrics(y_true, y_pred) -> dict:
    """Chỉ số phân loại nhị phân. True = bất thường (positive).

    Trả về dict gồm tp/fp/tn/fn + accuracy/precision/recall/f1/fpr/specificity.
    Chia cho 0 thì trả 0.0 (thay vì crash) — với bộ dữ liệu nhỏ điều này xảy ra.
    """
    pairs = list(zip(y_true, y_pred))
    tp = sum(1 for t, p in pairs if t and p)
    fp = sum(1 for t, p in pairs if not t and p)
    tn = sum(1 for t, p in pairs if not t and not p)
    fn = sum(1 for t, p in pairs if t and not p)

    def div(a, b):
        return a / b if b else 0.0

    precision = div(tp, tp + fp)
    recall = div(tp, tp + fn)
    return {
        "n": len(pairs),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": div(tp + tn, len(pairs)),
        "precision": precision,
        "recall": recall,
        "f1": div(2 * precision * recall, precision + recall),
        "fpr": div(fp, fp + tn),
        "specificity": div(tn, tn + fp),
    }


def roc_auc(scores, labels) -> float:
    """AUC theo công thức Mann–Whitney (có xử lý điểm trùng).

    labels: True = bất thường (positive). Trả NaN nếu chỉ có một lớp.
    """
    pairs = sorted(zip(scores, labels))
    n = len(pairs)
    n_pos = sum(1 for _, y in pairs if y)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # Gán hạng trung bình cho các điểm bằng nhau (ties).
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1

    pos_rank_sum = sum(ranks[k] for k in range(n) if pairs[k][1])
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def threshold_sweep(scores, labels, steps=None) -> list:
    """Quét ngưỡng, trả về list dict (đã sắp theo ngưỡng tăng dần).

    Ứng viên mặc định = mọi điểm score duy nhất + 0.0 (để có trường hợp
    "không ảnh nào bất thường").
    """
    scores = list(scores)
    labels = list(labels)
    uniq = sorted(set(round(s, 6) for s in scores))
    if steps is not None:
        candidates = [round(i / steps, 6) for i in range(steps + 1)]
    else:
        # Điểm giữa hai score liên tiếp cũng là ngưỡng hợp lệ — nếu chỉ lấy
        # đúng các score quan sát được, ta bỏ lỡ khoảng tách tốt nhất.
        midpoints = [round((a + b) / 2.0, 6) for a, b in zip(uniq, uniq[1:])]
        candidates = uniq + midpoints
    candidates = sorted(set([0.0] + candidates + ([uniq[-1] + 1e-6] if uniq else [])))

    rows = []
    for t in candidates:
        m = binary_metrics(labels, [s >= t for s in scores])
        rows.append({"threshold": t, **m})
    return rows


def best_f1(scores, labels) -> dict:
    """Ngưỡng cho F1 cao nhất; hoà thì chọn ngưỡng cao hơn (ít báo động nhầm hơn)."""
    rows = threshold_sweep(scores, labels)
    return max(rows, key=lambda r: (round(r["f1"], 6), r["threshold"]))


def loo_threshold_eval(scores, labels) -> dict:
    """Chọn ngưỡng kiểu leave-one-out — cách trung thực với bộ dữ liệu nhỏ.

    Với từng ảnh i: chọn ngưỡng tốt nhất trên 14 ảnh còn lại, rồi dự đoán ảnh i.
    Nhờ vậy ảnh i không tham gia chọn ngưỡng cho chính nó.
    """
    scores = list(scores)
    labels = list(labels)
    n = len(scores)
    y_pred = []
    chosen = []
    for i in range(n):
        rest_scores = [s for k, s in enumerate(scores) if k != i]
        rest_labels = [y for k, y in enumerate(labels) if k != i]
        if len(set(rest_labels)) < 2:
            t = DEFAULT_ANOMALY_THRESHOLD
        else:
            t = best_f1(rest_scores, rest_labels)["threshold"]
        chosen.append(t)
        y_pred.append(scores[i] >= t)
    return {
        "metrics": binary_metrics(labels, y_pred),
        "thresholds": chosen,
        "mean_threshold": sum(chosen) / n if n else 0.0,
    }


def confusion_matrix(y_true_cat, y_pred_cat, labels=None) -> dict:
    """Ma trận nhầm lẫn 3 nhóm: dict[(true, pred)] = count."""
    labels = list(labels) if labels is not None else sorted(set(y_true_cat) | set(y_pred_cat))
    matrix = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true_cat, y_pred_cat):
        matrix.setdefault(t, {x: 0 for x in labels})
        matrix[t][p] = matrix[t].get(p, 0) + 1
    return matrix
