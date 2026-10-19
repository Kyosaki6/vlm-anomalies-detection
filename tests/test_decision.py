# -*- coding: utf-8 -*-
"""tests/test_decision.py — kiểm tra toán của phần đánh giá.

Các hàm này là chỗ dễ sai nhất (F1, AUC, ngưỡng), nên test bằng số biết trước.
"""

import pytest

import decision


def test_category_of():
    assert decision.category_of("danh_nhau_3.jpg") == "danh_nhau"
    assert decision.category_of("tai_nan_xe_12.jpg") == "tai_nan_xe"
    assert decision.category_of("duong_pho_1.jpg") == "duong_pho"
    assert decision.category_of("khong_co_so.jpg") == "khong_co_so"


def test_is_normal_category():
    assert decision.is_normal_category("duong_pho")
    assert not decision.is_normal_category("tai_nan_xe")


def test_decide_abnormal_boundary():
    assert decision.decide_abnormal(0.30, 0.267)
    assert decision.decide_abnormal(0.267, 0.267)  # >= là bất thường
    assert not decision.decide_abnormal(0.20, 0.267)


def test_binary_metrics_hand_computed():
    # 2 TP, 1 FP, 1 TN, 1 FN
    y_true = [True, True, True, False, False]
    y_pred = [True, True, False, True, False]
    m = decision.binary_metrics(y_true, y_pred)
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (2, 1, 1, 1)
    assert m["accuracy"] == pytest.approx(3 / 5)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["fpr"] == pytest.approx(1 / 2)


def test_binary_metrics_all_correct():
    m = decision.binary_metrics([True, False], [True, False])
    assert m["f1"] == 1.0 and m["fpr"] == 0.0 and m["accuracy"] == 1.0


def test_roc_auc_perfect_and_reversed():
    # Tách hoàn hảo: bất thường điểm cao hơn hẳn.
    assert decision.roc_auc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1.0
    assert decision.roc_auc([0.1, 0.2, 0.8, 0.9], [True, True, False, False]) == 0.0


def test_roc_auc_ties_is_half():
    assert decision.roc_auc([0.5, 0.5, 0.5, 0.5], [True, True, False, False]) == 0.5


def test_roc_auc_single_class_is_nan():
    import math
    assert math.isnan(decision.roc_auc([0.1, 0.2], [True, True]))


def test_best_f1_finds_separable_threshold():
    scores = [0.9, 0.8, 0.3, 0.2]
    labels = [True, True, False, False]
    best = decision.best_f1(scores, labels)
    assert best["f1"] == 1.0
    assert 0.3 < best["threshold"] <= 0.8


def test_threshold_sweep_monotonic_recall():
    scores = [0.9, 0.7, 0.4, 0.1]
    labels = [True, True, False, False]
    rows = decision.threshold_sweep(scores, labels)
    recalls = [r["recall"] for r in rows]
    # Ngưỡng càng cao, recall càng giảm (hoặc giữ nguyên).
    assert recalls == sorted(recalls, reverse=True)


def test_loo_threshold_eval_does_not_use_self():
    scores = [0.9, 0.8, 0.3, 0.2]
    labels = [True, True, False, False]
    result = decision.loo_threshold_eval(scores, labels)
    # Với 4 điểm, LOO có thể sai 1 ảnh (0.8 khi ngưỡng học từ 3 ảnh còn lại).
    assert result["metrics"]["accuracy"] >= 0.75
    assert len(result["thresholds"]) == 4


def test_confusion_matrix_counts():
    cm = decision.confusion_matrix(
        ["a", "a", "b"], ["a", "b", "b"], ["a", "b"]
    )
    assert cm["a"]["a"] == 1
    assert cm["a"]["b"] == 1
    assert cm["b"]["b"] == 1
