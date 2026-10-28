# -*- coding: utf-8 -*-
"""Dashboard (English UI) — chế độ REPLAY: đọc kết quả đã tính sẵn, không cần GPU hay tải model.

    streamlit run app/streamlit_app.py

Skeleton này chạy trên dữ liệu GIẢ để track T5 (Thành) dựng UI ngay; thay `load_demo_run` bằng đọc từ results/.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402

from vad.fakes import FakeVerifier  # noqa: E402
from vad.graph.policy import GraphPolicy  # noqa: E402
from vad.temporal.ops import candidate_windows, ema, robust_normalize  # noqa: E402
from vad.types import EventType  # noqa: E402


def load_demo_run(seed: int = 0, seconds: int = 120, fps: float = 1.0):
    """DỮ LIỆU GIẢ: nền nhiễu + một đoạn cao. TODO(T5): đọc scores/captions/alerts từ results/."""
    rng = np.random.default_rng(seed)
    scores = rng.normal(0.0, 1.0, int(seconds * fps))
    scores[45:60] += 4.0
    return scores, fps


def main() -> None:
    st.set_page_config(page_title="Zero-Shot Video Anomaly Alerting", layout="wide")
    st.title("Zero-Shot Video Anomaly & Surveillance Alerting")
    st.caption("Replay mode: showing precomputed results. DEMO DATA — placeholder until real runs are wired in.")

    with st.sidebar:
        st.header("Settings")
        zone = st.selectbox("Zone", ["default", "school_gate", "market", "highway", "parking_lot"])
        hour = st.slider("Hour of day", 0, 23, 22)
        threshold = st.slider("Anomaly threshold (z-score)", 0.0, 6.0, 2.5, 0.1)

    scores, fps = load_demo_run()
    smooth = ema(robust_normalize(scores), alpha=0.4)
    st.subheader("Anomaly score timeline")
    st.line_chart(smooth)

    windows = candidate_windows(smooth, sample_fps=fps, threshold=threshold, video_id="demo_video")
    st.subheader("Alerts")
    if not windows:
        st.info("No suspicious window above the threshold.")
    policy, verifier = GraphPolicy(), FakeVerifier(EventType.FIGHTING)
    for w in windows:
        alert = policy.to_alert(w, verifier.verify([], {"video_id": w.video_id}), zone=zone, hour=hour)
        if alert is None:
            continue
        with st.container(border=True):
            st.markdown(f"**{alert.event_type.value}** · severity **{alert.severity}** · "
                        f"{alert.start_s:.0f}s–{alert.end_s:.0f}s")
            st.write(f"Responders: {', '.join(alert.responders) or '—'} · Protocol: {alert.protocol}")
            st.write(f"VLM caption: {alert.caption}")
            st.caption("Why: " + " → ".join(alert.path))


if __name__ == "__main__":
    main()
