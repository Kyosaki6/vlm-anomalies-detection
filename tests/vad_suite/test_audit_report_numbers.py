"""Test kiểm toán số liệu báo cáo (`scripts/audit_report_numbers.py`).

Không cần GPU, không cần video, không tải model: chỉ kiểm phần toán học (AUC có
xử lý hoà) và bộ đếm PASS/FAIL. Phần đọc dữ liệu thật chạy bằng tay trên máy có
`results/` + file nhãn UCF.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.audit_report_numbers import AuditError, Auditor, auc_pooled


def test_auc_perfect_separation_is_one() -> None:
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert auc_pooled(scores, labels) == pytest.approx(1.0)


def test_auc_reversed_is_zero() -> None:
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([0, 0, 1, 1])
    assert auc_pooled(scores, labels) == pytest.approx(0.0)


def test_auc_all_ties_is_half() -> None:
    """Mọi điểm bằng nhau: hạng trung bình cho AUC 0.5, không được chia cho 0."""
    scores = np.array([1.0, 1.0, 1.0, 1.0])
    labels = np.array([0, 1, 0, 1])
    assert auc_pooled(scores, labels) == pytest.approx(0.5)


def test_auc_known_mixed_case() -> None:
    """Ví dụ tính tay: dương [0.4, 0.9], âm [0.2, 0.6] -> 3/4 cặp đúng."""
    scores = np.array([0.4, 0.9, 0.2, 0.6])
    labels = np.array([1, 1, 0, 0])
    assert auc_pooled(scores, labels) == pytest.approx(0.75)


def test_auc_single_class_is_nan() -> None:
    scores = np.array([0.1, 0.2])
    assert np.isnan(auc_pooled(scores, np.array([1, 1])))
    assert np.isnan(auc_pooled(scores, np.array([0, 0])))


def test_auc_rejects_length_mismatch() -> None:
    with pytest.raises(AuditError):
        auc_pooled(np.array([0.1, 0.2]), np.array([1]))


def test_auditor_returns_zero_when_everything_matches(capsys: pytest.CaptureFixture[str]) -> None:
    auditor = Auditor()
    auditor.check_value("a", 1.0, 1.0, 0.001)
    auditor.check("b", "x", "x", True)
    assert auditor.report() == 0
    assert "2/2 PASS" in capsys.readouterr().out


def test_auditor_returns_one_and_names_the_failure(capsys: pytest.CaptureFixture[str]) -> None:
    """Một số lệch phải làm script thoát khác 0 và in tên mục — không im lặng."""
    auditor = Auditor()
    auditor.check_value("latency_tier1_s", 2.03, 2.50, 0.005)
    assert auditor.report() == 1
    out = capsys.readouterr().out
    assert "FAIL: latency_tier1_s" in out
    assert "0/1 PASS" in out
