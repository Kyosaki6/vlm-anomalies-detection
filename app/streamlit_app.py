# -*- coding: utf-8 -*-
"""Dashboard (English UI) — chế độ REPLAY: đọc kết quả đã tính sẵn trong results/, không cần GPU hay model.

    streamlit run app/streamlit_app.py              # đọc results/*.json
    streamlit run app/streamlit_app.py -- --demo    # dữ liệu GIẢ có banner (không cần results/)
    VAD_DEMO=1 streamlit run app/streamlit_app.py   # như trên, bằng biến môi trường

Dữ liệu giả CHỈ xuất hiện khi bật --demo, và luôn có banner. File kết quả có `metadata.note`
(vd `results/sample_run.json`) cũng hiện banner với nội dung nguyên văn.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import data  # noqa: E402


def demo_requested(argv: list[str] | None = None) -> bool:
    args = sys.argv[1:] if argv is None else argv
    return "--demo" in args or os.environ.get("VAD_DEMO", "").strip() not in ("", "0")


def _timeline_chart(run, threshold: float):
    import altair as alt

    t, s = data.timeline(run)
    df = pd.DataFrame({"time (s)": t, "score": s})
    line = alt.Chart(df).mark_line().encode(x="time (s)", y="score")
    rule = alt.Chart(pd.DataFrame({"y": [threshold]})).mark_rule(color="red", strokeDash=[6, 4]).encode(y="y")
    return line + rule


def main() -> None:
    st.set_page_config(page_title="Zero-Shot Video Anomaly Alerting", layout="wide")
    st.title("Zero-Shot Video Anomaly & Surveillance Alerting")
    st.caption("Replay mode: showing precomputed results from results/. No model runs here.")

    demo = demo_requested()
    override = os.environ.get("VAD_RESULTS_DIR", "").strip()
    results_dir = Path(override) if override else data.default_results_dir()

    with st.sidebar:
        st.header("Settings")
        if demo:
            run = data.demo_run()
        else:
            ids = data.list_video_ids(results_dir)
            if not ids:
                st.error(f"No result files in {results_dir}. Run the pipeline or start with `-- --demo`.")
                st.stop()
            video_id = st.selectbox("Video", ids)
            try:
                run = data.load_run(video_id, results_dir)
            except (ValueError, FileNotFoundError) as e:
                st.error(f"Cannot load {video_id}: {e}")
                st.stop()
        lo, hi = (float(min(run.anomaly_scores, default=0.0)), float(max(run.anomaly_scores, default=1.0)))
        hi = max(hi, lo + 1e-6)
        threshold = st.slider("Score threshold", lo, hi, (lo + hi) / 2, (hi - lo) / 100)
        min_sev = st.selectbox("Minimum severity", ["low", "medium", "high", "critical"])

    if data.is_demo(run):
        st.warning(data.DEMO_NOTE)
    elif data.is_placeholder(run):
        st.warning(f"Placeholder result: {data.run_note(run)}")

    meta = run.metadata
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Video", run.video_id)
    c2.metric("Duration", f"{data.duration_s(run):.0f} s")
    c3.metric("Crossings >= threshold", data.threshold_crossings(run, threshold))
    c4.metric("Alerts", len(run.alerts))
    st.caption(f"model: {meta.get('model', '?')} · commit: {meta.get('commit', '?')} · "
               f"run at: {meta.get('timestamp', '?')} · sample_fps: {run.sample_fps:g}")

    st.subheader("Anomaly score timeline")
    st.altair_chart(_timeline_chart(run, threshold), use_container_width=True)

    st.subheader("Candidate windows (tier 1) and verification (tier 2)")
    rows = []
    for w, v in data.detections(run):
        rows.append({"start (s)": w.start_s, "end (s)": w.end_s, "tier-1 score": round(w.score, 3),
                     "VLM verdict": "not verified" if v is None else ("anomaly" if v.is_anomaly else "normal"),
                     "type": "" if v is None else v.event_type.value,
                     "confidence": None if v is None else v.confidence,
                     "caption": "" if v is None else v.caption})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No candidate windows in this run.")

    st.subheader("Alerts")
    alerts = data.alerts_by_severity(run.alerts, min_sev)
    if not alerts:
        st.info("No alerts at this severity." if run.alerts else "No alerts in this run.")
    for a in alerts:
        with st.container(border=True):
            st.markdown(f"**{a.event_type.value}** · severity **{a.severity}** · {a.start_s:.0f}s–{a.end_s:.0f}s"
                        f"{' · camera ' + a.camera if a.camera else ''}")
            st.write(f"Responders: {', '.join(a.responders) or '—'} · Protocol: {a.protocol}")
            st.write(f"VLM caption: {a.caption or '—'} (confidence {a.confidence:.2f})")
            st.caption("Why: " + " → ".join(a.path))


if __name__ == "__main__":
    main()
