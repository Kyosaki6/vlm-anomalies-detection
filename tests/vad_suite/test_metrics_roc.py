# -*- coding: utf-8 -*-
"""Toán nền của `vad.eval.metrics`: `roc_auc` (Mann-Whitney, có ties) và `_percentile`.

Chuyển từ tests/test_decision.py khi bỏ pipeline 15 ảnh cũ; số biết trước, tính tay.
"""
from __future__ import annotations

import math

import pytest

from vad.eval.metrics import _percentile, roc_auc


def test_roc_auc_perfect_and_reversed():
    assert roc_auc([0.9, 0.8, 0.2, 0.1], [True, True, False, False]) == 1.0
    assert roc_auc([0.1, 0.2, 0.8, 0.9], [True, True, False, False]) == 0.0


def test_roc_auc_all_ties_is_half():
    assert roc_auc([0.5, 0.5, 0.5, 0.5], [True, True, False, False]) == 0.5


def test_roc_auc_partial_ties_hand_computed():
    # pos = {0.5, 0.9}, neg = {0.5, 0.1}: cặp (0.5,0.5)=0.5, (0.5,0.1)=1, (0.9,0.5)=1, (0.9,0.1)=1 -> 3.5/4
    assert roc_auc([0.5, 0.9, 0.5, 0.1], [True, True, False, False]) == pytest.approx(0.875)


def test_roc_auc_single_class_is_nan():
    assert math.isnan(roc_auc([0.1, 0.2], [True, True]))
    assert math.isnan(roc_auc([0.1, 0.2], [False, False]))


def test_percentile_linear_interpolation():
    vals = [0.0, 1.0, 2.0, 3.0, 4.0]
    assert _percentile(vals, 0.0) == 0.0
    assert _percentile(vals, 1.0) == 4.0
    assert _percentile(vals, 0.5) == 2.0
    assert _percentile(vals, 0.125) == pytest.approx(0.5)


def test_percentile_single_value():
    assert _percentile([7.0], 0.3) == 7.0
