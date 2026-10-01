# -*- coding: utf-8 -*-
"""
tests/test_graph.py — pytest cho graph.py

Kiểm tra:
  - Cả 15 id trong events.json: get_action() trả str khác rỗng,
    get_event() trả tên sự kiện khác rỗng.
  - Id không tồn tại ("99") -> get_action() và get_event() trả None.
"""

import pytest
from graph import get_action, get_event

# 15 id lấy từ events.json (id dạng str)
ALL_IDS = [str(i) for i in range(1, 16)]


@pytest.mark.parametrize("event_id", ALL_IDS)
def test_get_action_returns_nonempty_string(event_id):
    result = get_action(event_id)
    assert isinstance(result, str), f"get_action({event_id!r}) phải là str, nhận {type(result)}"
    assert result.strip() != "", f"get_action({event_id!r}) không được rỗng"


@pytest.mark.parametrize("event_id", ALL_IDS)
def test_get_event_returns_nonempty_string(event_id):
    result = get_event(event_id)
    assert isinstance(result, str), f"get_event({event_id!r}) phải là str, nhận {type(result)}"
    assert result.strip() != "", f"get_event({event_id!r}) không được rỗng"


def test_get_action_invalid_id_returns_none():
    assert get_action("99") is None


def test_get_event_invalid_id_returns_none():
    assert get_event("99") is None
