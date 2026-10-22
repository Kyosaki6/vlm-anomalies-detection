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


def _accuracy(y_true, y_pred):
    return sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)


def test_bootstrap_ci_deterministic_with_fixed_seed():
    y_true = [1, 0, 1, 1, 0, 1, 0, 0]
    y_pred = [1, 0, 0, 1, 0, 1, 1, 0]
    first = decision.bootstrap_ci(_accuracy, y_true, y_pred, seed=7)
    second = decision.bootstrap_ci(_accuracy, y_true, y_pred, seed=7)
    assert first == second


def test_bootstrap_ci_bounds_contain_point_estimate():
    y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
    y_pred = [1, 0, 0, 1, 0, 1, 1, 0, 1, 1]
    low, high = decision.bootstrap_ci(_accuracy, y_true, y_pred)
    point = _accuracy(y_true, y_pred)
    assert 0.0 <= low <= point <= high <= 1.0


def test_bootstrap_ci_perfect_predictions_collapse_to_one():
    low, high = decision.bootstrap_ci(_accuracy, [1, 0, 1], [1, 0, 1])
    assert (low, high) == (1.0, 1.0)


def test_bootstrap_ci_wider_confidence_is_not_narrower():
    y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
    y_pred = [1, 0, 0, 1, 0, 1, 1, 0, 1, 0]
    l90, h90 = decision.bootstrap_ci(_accuracy, y_true, y_pred, ci=0.90)
    l99, h99 = decision.bootstrap_ci(_accuracy, y_true, y_pred, ci=0.99)
    assert l99 <= l90 and h90 <= h99


def test_bootstrap_ci_works_with_roc_auc():
    labels = [True, True, True, False, False, False]
    scores = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
    low, high = decision.bootstrap_ci(lambda yt, ys: decision.roc_auc(ys, yt), labels, scores)
    assert high == 1.0
    assert 0.0 <= low <= 1.0


def test_bootstrap_ci_all_nan_returns_nan():
    low, high = decision.bootstrap_ci(lambda yt, ys: float("nan"), [1, 0], [1, 0], n_bootstraps=5)
    assert low != low and high != high


@pytest.mark.parametrize("kwargs", [
    {"y_true": [], "y_pred": []},
    {"y_true": [1, 0], "y_pred": [1]},
    {"y_true": [1], "y_pred": [1], "n_bootstraps": 0},
    {"y_true": [1], "y_pred": [1], "ci": 1.0},
    {"y_true": [1], "y_pred": [1], "ci": 0.0},
])
def test_bootstrap_ci_invalid_input_raises(kwargs):
    y_true = kwargs.pop("y_true")
    y_pred = kwargs.pop("y_pred")
    with pytest.raises(ValueError):
        decision.bootstrap_ci(_accuracy, y_true, y_pred, **kwargs)


def test_bootstrap_ci_non_callable_metric_raises():
    with pytest.raises(ValueError):
        decision.bootstrap_ci(None, [1], [1])


def test_action_recall_all_match():
    actions = ["a", "b", "c"]
    assert decision.action_recall(actions, list(actions)) == 1.0


def test_action_recall_none_match():
    assert decision.action_recall(["a", "b"], ["x", "y"]) == 0.0


def test_action_recall_partial_match():
    assert decision.action_recall(["a", "b", "c", "d"], ["a", "x", "c", "y"]) == pytest.approx(0.5)


def test_action_recall_shared_action_counts_as_correct():
    shared = "Báo CSGT, gọi cấp cứu và đặt cọc tiêu cảnh báo"
    assert decision.action_recall([shared, "khác"], [shared, "khác"]) == 1.0


def test_action_recall_empty_is_zero():
    assert decision.action_recall([], []) == 0.0


def test_action_recall_length_mismatch_raises():
    with pytest.raises(ValueError):
        decision.action_recall(["a"], ["a", "b"])
