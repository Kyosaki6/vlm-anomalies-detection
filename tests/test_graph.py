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


@pytest.mark.parametrize("event_id", IDS)
def test_get_event_by_id_returns_exact_metadata(event_id):
    expected = next(ev for ev in EVENTS if str(ev["id"]) == event_id)
    res = graph.get_event_by_id(event_id)
    assert res is not None
    assert res["id"] == str(expected["id"])
    assert res["file"] == expected["file"]
    assert res["ten_su_kien"] == expected["ten_su_kien"]
    assert res["cach_xu_ly"] == expected["cach_xu_ly"]
    assert res["category"] == expected["category"]
    assert res["is_normal"] == expected["is_normal"]
    assert res["severity"] == expected["severity"]
    assert res["responder"] == expected["responder"]


@pytest.mark.parametrize("event_id", ["99", "0", "-1", "abc", ""])
def test_get_event_by_id_unknown_returns_none(event_id):
    assert graph.get_event_by_id(event_id) is None


@pytest.mark.parametrize("category, expected_count", [
    ("danh_nhau", 5),
    ("duong_pho", 5),
    ("tai_nan_xe", 5),
])
def test_get_events_by_category_counts(category, expected_count):
    events = graph.get_events_by_category(category)
    assert len(events) == expected_count
    assert all(ev["category"] == category for ev in events)


@pytest.mark.parametrize("invalid_category", ["", "unknown", "invalid", "cctv"])
def test_get_events_by_category_invalid_returns_empty(invalid_category):
    assert graph.get_events_by_category(invalid_category) == []


@pytest.mark.parametrize("event_id", IDS)
def test_get_action_details_returns_exact_details(event_id):
    expected = next(ev for ev in EVENTS if str(ev["id"]) == event_id)
    details = graph.get_action_details(event_id)
    assert details is not None
    assert details["event_id"] == str(expected["id"])
    assert details["action"] == expected["cach_xu_ly"]
    assert details["severity"] == expected["severity"]
    assert details["responder"] == expected["responder"]
    assert details["category"] == expected["category"]


def test_get_action_details_unknown_returns_none():
    assert graph.get_action_details("999") is None


@pytest.mark.parametrize("event_id", IDS)
def test_graph_node_attributes(event_id):
    expected = next(ev for ev in EVENTS if str(ev["id"]) == event_id)
    event_name = expected["ten_su_kien"]
    assert event_name in graph.GRAPH.nodes
    node = graph.GRAPH.nodes[event_name]
    assert node["kind"] == "su_kien"
    assert node["id"] == str(expected["id"])
    assert node["category"] == expected["category"]
    assert node["is_normal"] == expected["is_normal"]
    assert node["severity"] == expected["severity"]
    assert node["responder"] == expected["responder"]
    assert node["file"] == expected["file"]
