# -*- coding: utf-8 -*-
"""Dashboard (English UI) — chế độ REPLAY: đọc kết quả đã tính sẵn trong results/, không cần GPU hay model.

    streamlit run app/streamlit_app.py              # đọc results/run_A4c/ (+ results/ablation_*.json cho tab Overview)
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

VERDICT_COLORS = {"confirmed": "#2e9e5b", "rejected": "#d9534f", "not verified": "#9aa0a6"}


def demo_requested(argv: list[str] | None = None) -> bool:
    args = sys.argv[1:] if argv is None else argv
    return "--demo" in args or os.environ.get("VAD_DEMO", "").strip() not in ("", "0")


@st.cache_data(show_spinner=False)
def _summaries(root: str) -> list[dict]:
    return data.video_summaries(root)


def _timeline_chart(run, threshold: float):
    import altair as alt

    t, s = data.timeline(run)
    df = pd.DataFrame({"time (s)": t, "score": s})
    layers = []
    bands = data.window_bands(run)
    if bands:
        layers.append(
            alt.Chart(pd.DataFrame(bands)).mark_rect(opacity=0.25).encode(
                x="start (s):Q", x2="end (s):Q",
                color=alt.Color("verdict:N", title="tier-2 verdict",
                                scale=alt.Scale(domain=list(VERDICT_COLORS), range=list(VERDICT_COLORS.values()))),
                tooltip=["start (s)", "end (s)", "tier-1 score", "verdict"]))
    layers.append(alt.Chart(df).mark_line().encode(x="time (s):Q", y="score:Q"))
    layers.append(alt.Chart(pd.DataFrame({"y": [threshold]})).mark_rule(color="red", strokeDash=[6, 4]).encode(y="y:Q"))
    return alt.layer(*layers)


def _alert_card(run, alert) -> None:
    c = data.alert_card(run, alert)
    with st.container(border=True):
        st.markdown(f"**{c['type']}** · severity **{c['severity']}** · {c['start_s']:.0f}s–{c['end_s']:.0f}s"
                    f"{' · camera ' + c['camera'] if c['camera'] else ''}")
        st.write(f"Responders: {', '.join(c['responders']) or '—'} · Protocol: {c['protocol']}")
        st.write(f"VLM caption: {c['caption'] or '—'} (confidence {c['confidence']:.2f})")
        if c["vlm_type"] and c["vlm_type"] != c["type"]:
            st.caption(f"Alert type comes from the zero-shot typer. VLM said: {c['vlm_type']} (info only).")
        st.caption("Why: " + " → ".join(c["path"]))


def _replay_tab(run, threshold: float, min_sev: str) -> None:
    meta = run.metadata
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Video", run.video_id)
    c2.metric("Duration", f"{data.duration_s(run):.0f} s")
    c3.metric("Crossings >= threshold", data.threshold_crossings(run, threshold))
    c4.metric("Alerts", len(run.alerts))
    c5.metric("VLM rejected", data.vlm_rejected_count(run),
              help="Candidate windows the tier-2 VLM judged not anomalous.")
    st.caption(f"model: {meta.get('model', '?')} · commit: {meta.get('commit', '?')} · "
               f"run at: {meta.get('timestamp', '?')} · sample_fps: {run.sample_fps:g} · "
               f"tier 2: {meta.get('tier2', '?')}")

    video_path = data.find_video_path(run.video_id)
    wins = data.detections(run)
    start_s = 0
    if wins:
        labels = ["Start of video"] + [f"{w.start_s:.0f}s–{w.end_s:.0f}s · {data.verdict_label(v)}" for w, v in wins]
        pick = st.selectbox("Jump to window", range(len(labels)), format_func=labels.__getitem__)
        start_s = 0 if pick == 0 else int(wins[pick - 1][0].start_s)
    if video_path is None:
        st.info(data.VIDEO_MISSING_TEXT)
    else:
        st.video(str(video_path), start_time=start_s)

    st.subheader("Anomaly score timeline")
    st.altair_chart(_timeline_chart(run, threshold), use_container_width=True)
    st.caption("Shaded bands: tier-1 candidate windows, coloured by tier-2 verdict "
               "(green confirmed, red rejected, grey not verified). Red dashed line: score threshold.")

    st.subheader("Candidate windows (tier 1) and verification (tier 2)")
    rows = data.detection_rows(run)
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No candidate windows in this run.")

    st.subheader("Alerts")
    alerts = data.alerts_by_severity(run.alerts, min_sev)
    if not alerts:
        st.info("No alerts at this severity." if run.alerts else "No alerts in this run.")
    for a in alerts:
        _alert_card(run, a)


def _table(rows: list[dict]) -> None:
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _overview_tab(root: Path) -> None:
    st.subheader("Ablation: frame-level AUC with 95% bootstrap CI over videos")
    rows, skipped = data.load_ablations(root)
    if not rows:
        st.warning(f"Not run: no ablation_*.json in {root}.")
    else:
        _table(rows)
    if skipped:
        st.warning("Skipped unreadable ablation files: " + ", ".join(skipped))

    st.subheader("Tier 2 (A6): operating point before and after VLM verification")
    st.info(data.PRE_REGISTERED_CLAIM)
    a6 = data.load_a6(root)
    if a6 is None:
        st.warning("A6 not run: ablation_A6.json is missing or incomplete. No numbers shown.")
        return
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("A6 AUC", f"{100 * a6['auc_micro']:.2f}%",
              help=f"95% CI [{100 * a6['ci_lo']:.2f}, {100 * a6['ci_hi']:.2f}]")
    m2.metric("Delta AUC vs A4c", f"{100 * a6['delta_auc']:+.2f} pts")
    m3.metric("Windows confirmed", f"{a6.get('n_confirmed', '?')} / {a6.get('n_windows', '?')}")
    m4.metric("Verifier", str(a6.get("verifier", "?")).split("/")[-1])
    lo, hi = a6.get("delta_auc_ci_lo"), a6.get("delta_auc_ci_hi")
    if lo is not None and hi is not None:
        st.caption(f"Paired delta AUC 95% CI: [{100 * lo:+.2f}, {100 * hi:+.2f}] pts, resampled over videos. "
                   f"Threshold {a6.get('threshold', 0):.4f} ({a6.get('threshold_source', '?')}).")
    _table(data.operating_rows(a6))
    rs = a6.get("rejected_split")
    if rs:
        st.caption(f"Rejected windows: {rs.get('rejected_in_normal_videos', '?')} in normal videos, "
                   f"{rs.get('rejected_in_anomalous_videos', '?')} in anomalous videos; "
                   f"{rs.get('rejected_overlapping_gt', '?')} overlapped ground truth (lost true positives).")
    st.subheader("Tier 2 cost")
    cost = data.cost_rows(a6)
    if cost:
        _table(cost)
    else:
        st.warning("No cost block in ablation_A6.json.")


def main() -> None:
    st.set_page_config(page_title="Zero-Shot Video Anomaly Alerting", layout="wide")
    st.title("Zero-Shot Video Anomaly & Surveillance Alerting")
    st.caption("Replay mode: showing precomputed results from results/. No model runs here.")

    demo = demo_requested()
    override = os.environ.get("VAD_RESULTS_DIR", "").strip()
    results_dir = Path(override) if override else data.default_results_dir()
    abl_override = os.environ.get("VAD_ABLATION_DIR", "").strip()
    ablation_dir = Path(abl_override) if abl_override else data.DEFAULT_RESULTS_DIR

    with st.sidebar:
        st.header("Settings")
        if demo:
            run = data.demo_run()
        else:
            all_videos = st.toggle("Show all videos", value=False,
                                   help="Off: only videos with at least one candidate window.")
            summaries = _summaries(str(results_dir))
            if not summaries:
                st.error(f"No result files in {results_dir}. Run the pipeline or start with `-- --demo`.")
                st.stop()
            ids = data.filter_video_ids(summaries, with_windows_only=not all_videos)
            if not ids:
                st.error("No video with candidate windows. Turn on 'Show all videos'.")
                st.stop()
            st.caption(f"{len(ids)} of {len(summaries)} videos listed")
            video_id = st.selectbox("Video", ids)
            try:
                run = data.load_run(video_id, results_dir)
            except (ValueError, FileNotFoundError) as e:
                st.error(f"Cannot load {video_id}: {e}")
                st.stop()
        lo, hi = (float(min(run.anomaly_scores, default=0.0)), float(max(run.anomaly_scores, default=1.0)))
        tau, tau_src = data.default_threshold(run)
        if tau is None:
            default = (lo + hi) / 2
            st.caption("No threshold in this result file: slider starts at the midpoint of the scores.")
        else:
            default = tau
            lo, hi = min(lo, tau), max(hi, tau)
            st.caption(f"Default = frozen tau {tau:.4f} (source: {tau_src}).")
        hi = max(hi, lo + 1e-6)
        threshold = st.slider("Score threshold", lo, hi, default, (hi - lo) / 100, key=f"thr-{run.video_id}")
        min_sev = st.selectbox("Minimum severity", ["low", "medium", "high", "critical"])

    if data.is_demo(run):
        st.warning(data.DEMO_NOTE)
    elif data.is_placeholder(run):
        st.warning(f"Placeholder result: {data.run_note(run)}")

    replay, overview = st.tabs(["Replay", "Overview"])
    with replay:
        _replay_tab(run, threshold, min_sev)
    with overview:
        _overview_tab(ablation_dir)


if __name__ == "__main__":
    main()
