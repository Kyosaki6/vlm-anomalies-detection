# -*- coding: utf-8 -*-
"""
tests/test_graph.py — pytest cho graph.py

Khác bản cũ: không chỉ kiểm tra "trả về chuỗi khác rỗng" mà kiểm tra
**đúng nội dung** lấy từ events.json. Một hàm trả về "abc" cũng qua được
bản test cũ — bản này thì không.
"""

import json
from pathlib import Path

import pytest

import graph

EVENTS_FILE = Path(__file__).resolve().parent.parent / "events.json"


def load_events():
    with open(EVENTS_FILE, encoding="utf-8-sig") as f:
        return json.load(f)


EVENTS = load_events()
IDS = [str(ev["id"]) for ev in EVENTS]


@pytest.mark.parametrize("event_id", IDS)
def test_get_action_returns_exact_value(event_id):
    expected = next(ev["cach_xu_ly"] for ev in EVENTS if str(ev["id"]) == event_id)
    assert graph.get_action(event_id) == expected


@pytest.mark.parametrize("event_id", IDS)
def test_get_event_returns_exact_value(event_id):
    expected = next(ev["ten_su_kien"] for ev in EVENTS if str(ev["id"]) == event_id)
    assert graph.get_event(event_id) == expected


@pytest.mark.parametrize("event_id", ["99", "0", "-1", "abc", ""])
def test_unknown_id_returns_none(event_id):
    assert graph.get_action(event_id) is None
    assert graph.get_event(event_id) is None


@pytest.mark.parametrize("event_id", IDS)
def test_graph_has_edge_event_to_action(event_id):
    event = graph.get_event(event_id)
    action = graph.get_action(event_id)
    assert graph.GRAPH.has_edge(event, action)
