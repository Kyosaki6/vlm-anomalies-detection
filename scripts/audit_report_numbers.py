"""Kiểm lại (audit) mọi con số in trong báo cáo, từ dữ liệu thô của repo.

Mục đích: không con số nào trong `report/` được phép chỉ "có trong một file JSON"
mà không kiểm lại được. Script đọc dữ liệu thô (`results/run_A4c/`,
`results/verify/`, `manifests/`) rồi tự tính lại các chỉ số và so sánh với giá
trị đã in. Khi có `--annotations`, script tái lập luôn các chỉ số phụ thuộc GT
(AUC frame-level, precision, incident recall, latency) từ điểm thô và file nhãn.

Cách chạy (máy có dữ liệu UCF):

    python scripts/audit_report_numbers.py \
        --annotations datasets/ucf_crime/Temporal_Anomaly_Annotation_for_Testing_Videos.txt

Không cần GPU, không cần video: chỉ đọc JSON + file nhãn.

Đầu ra: mỗi dòng một con số `report=<in trong báo cáo> audited=<tính lại> PASS/FAIL`,
cuối cùng là số mục FAIL (0 = khớp toàn bộ).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import pathlib
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # cho phép `python scripts/...` không cần cài package
    sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"
MANIFEST = ROOT / "manifests" / "ucf_test.csv"

# Giá trị đang in trong báo cáo (report/*.md). Sửa một bên thì phải sửa bên kia.
CLAIMS: Mapping[str, float] = {
    "n_videos": 290,
    "n_abnormal": 140,
    "n_normal": 150,
    "n_frames": 1111808,
    "n_windows": 307,
    "n_videos_with_window": 114,
    "fa_windows_tier1": 124,
    "normal_hours": 6.01,
    "fa_per_hour_tier1": 20.64,
    "normal_hours_excl": 4.01,
    "fa_per_hour_tier1_excl": 9.73,
    "n_verify_files": 307,
    "n_confirmed": 129,
    "n_rejected": 178,
    "confirm_rate_pct": 42.0,
    "fa_windows_tier2": 1,
    "fa_per_hour_tier2": 0.17,
    "fa_reduction_pct": 99.2,
    "precision_tier1_pct": 28.0,
    "recall_tier1_pct": 51.9,
    "latency_tier1_s": 2.03,
    "precision_tier2_pct": 49.6,
    "recall_tier2_pct": 39.7,
    "latency_tier2_s": 1.74,
    "auc_tier1_pct": 73.00,
    "auc_tier1_lo_pct": 64.45,
    "auc_tier1_hi_pct": 81.15,
    "auc_tier2_pct": 67.90,
    "auc_tier2_lo_pct": 58.39,
    "auc_tier2_hi_pct": 77.49,
    "delta_auc_pts": -5.10,
    "delta_lo_pts": -11.42,
    "delta_hi_pts": -0.25,
    "auc_macro_pct": 63.35,
    "cost_median_s": 3.71,
    "cost_p90_s": 14.41,
    "windows_per_video": 1.06,
}

ABLATION_CLAIMS: Mapping[str, float] = {
    "A0": 48.77, "A1": 65.87, "A2": 72.39, "A3": 58.99,
    "A4": 58.41, "A5": 57.08, "A6": 67.90,
}

LONG_NORMAL_VIDEOS = ("Normal_Videos_935_x264", "Normal_Videos_924_x264")


class AuditError(RuntimeError):
    """Dữ liệu vào không đủ để kiểm lại."""


def read_json(path: pathlib.Path) -> Any:
    with open(path, encoding="utf-8-sig") as fh:  # utf-8-sig: chịu được BOM kiểu Windows
        return json.load(fh)


def read_manifest(path: pathlib.Path = MANIFEST) -> list[dict[str, str]]:
    with open(path, encoding="utf-8-sig") as fh:
        return [row for row in csv.DictReader(fh) if row.get("video_id")]


def auc_pooled(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC nhị phân (Mann-Whitney U) có xử lý hoà bằng hạng trung bình.

    Tương đương diện tích dưới đường ROC khi điểm bằng nhau được tính nửa điểm.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(np.int64)
    if scores.size != labels.size:
        raise AuditError("scores và labels phải cùng độ dài")
    n_pos = int((labels == 1).sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ordered = scores[order]
    ranks = np.empty(ordered.size, dtype=np.float64)
    i = 0
    while i < ordered.size:
        j = i
        while j + 1 < ordered.size and ordered[j + 1] == ordered[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    ranked = np.empty(ordered.size, dtype=np.float64)
    ranked[order] = ranks
    return float((ranked[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


class Auditor:
    """Gom các phép so sánh + đếm lỗi, in ra một bảng duy nhất."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str, bool]] = []

    def check(self, name: str, claimed: str, audited: str, ok: bool) -> None:
        self.rows.append((name, claimed, audited, ok))

    def check_number(self, key: str, audited: float, tol: float, nd: int = 2) -> None:
        self.check_value(key, CLAIMS[key], audited, tol, nd)

    def check_value(self, name: str, claimed: float, audited: float, tol: float,
                    nd: int = 2) -> None:
        ok = abs(audited - claimed) <= tol
        fmt = f"{{:.{nd}f}}"
        self.check(name, fmt.format(claimed), fmt.format(audited), ok)

    def report(self) -> int:
        width = max(len(r[0]) for r in self.rows)
        for name, claimed, audited, ok in self.rows:
            print(f"{name:{width}s} report={claimed:>11s} audited={audited:>11s} "
                  f"{'PASS' if ok else 'FAIL'}")
        failed = [r for r in self.rows if not r[3]]
        print(f"\n{len(self.rows) - len(failed)}/{len(self.rows)} PASS")
        for name, claimed, audited, _ in failed:
            print(f"FAIL: {name}: report {claimed} != audited {audited}")
        return 1 if failed else 0


# --------------------------------------------------------------------- phần 1: đếm và số học

def audit_counts(aud: Auditor) -> dict[str, Any]:
    rows = read_manifest()
    meta = {r["video_id"]: (int(r["n_frames"]), float(r["fps"])) for r in rows}
    normal_ids = {r["video_id"] for r in rows if "Normal_Videos" in r["path"]}

    aud.check_number("n_videos", float(len(rows)), 0.0, 0)
    aud.check_number("n_abnormal", float(len(rows) - len(normal_ids)), 0.0, 0)
    aud.check_number("n_normal", float(len(normal_ids)), 0.0, 0)
    aud.check_number("n_frames", float(sum(n for n, _ in meta.values())), 0.0, 0)

    windows: dict[str, list[dict[str, Any]]] = {}
    for path in glob.glob(str(RESULTS / "run_A4c" / "windows" / "*.json")):
        run = read_json(pathlib.Path(path))
        windows[run["video_id"]] = list(run["candidate_windows"])
    n_win = sum(len(w) for w in windows.values())
    aud.check_number("n_windows", float(n_win), 0.0, 0)
    aud.check_number("n_videos_with_window", float(sum(1 for w in windows.values() if w)), 0.0, 0)
    aud.check_number("windows_per_video", n_win / max(len(windows), 1), 0.005)

    fa1 = sum(len(v) for vid, v in windows.items() if vid in normal_ids)
    hours = sum(meta[vid][0] / meta[vid][1] for vid in normal_ids) / 3600.0
    aud.check_number("fa_windows_tier1", float(fa1), 0.0, 0)
    aud.check_number("normal_hours", hours, 0.005)
    aud.check_number("fa_per_hour_tier1", fa1 / hours, 0.005)

    hours_excl = sum(meta[vid][0] / meta[vid][1] for vid in normal_ids
                     if vid not in LONG_NORMAL_VIDEOS) / 3600.0
    fa1_excl = sum(len(v) for vid, v in windows.items()
                   if vid in normal_ids and vid not in LONG_NORMAL_VIDEOS)
    aud.check_number("normal_hours_excl", hours_excl, 0.005)
    aud.check_number("fa_per_hour_tier1_excl", fa1_excl / hours_excl, 0.005)

    confirmed = rejected = confirmed_normal = files = 0
    verdicts: dict[str, set[int]] = {}
    for path in glob.glob(str(RESULTS / "verify" / "*" / "window_*.json")):
        data = read_json(pathlib.Path(path))
        files += 1
        is_anomaly = bool((data.get("verify_results") or [{}])[0].get("is_anomaly"))
        if is_anomaly:
            confirmed += 1
            verdicts.setdefault(data["video_id"], set()).add(int(data["metadata"]["window_index"]))
            if data["video_id"] in normal_ids:
                confirmed_normal += 1
        else:
            rejected += 1

    aud.check_number("n_verify_files", float(files), 0.0, 0)
    aud.check_number("n_confirmed", float(confirmed), 0.0, 0)
    aud.check_number("n_rejected", float(rejected), 0.0, 0)
    aud.check_number("confirm_rate_pct", 100.0 * confirmed / max(files, 1), 0.05, 1)
    aud.check_number("fa_windows_tier2", float(confirmed_normal), 0.0, 0)
    aud.check_number("fa_per_hour_tier2", confirmed_normal / hours, 0.005)
    aud.check_number("fa_reduction_pct",
                     100.0 * (1 - (confirmed_normal / hours) / (fa1 / hours)), 0.05, 1)

    a4c = read_json(RESULTS / "ablation_A4c.json")
    a6 = read_json(RESULTS / "ablation_A6.json")
    before = a6["operating_before"]
    after = a6["operating_after"]

    aud.check("precision_tier1 (86/307)", f"{before['true_positive_windows']}/{before['windows']}",
              "86/307", before["true_positive_windows"] == 86 and before["windows"] == 307)
    aud.check("recall_tier1 (81/156)", f"{before['incidents_detected']}/{before['incidents']}",
              "81/156", before["incidents_detected"] == 81 and before["incidents"] == 156)

    aud.check_number("precision_tier1_pct", 100.0 * before["window_precision"], 0.05, 1)
    aud.check_number("recall_tier1_pct", 100.0 * before["incident_recall"], 0.05, 1)
    aud.check_number("latency_tier1_s", before["mean_latency_s"], 0.005)
    aud.check_number("precision_tier2_pct", 100.0 * after["window_precision"], 0.05, 1)
    aud.check_number("recall_tier2_pct", 100.0 * after["incident_recall"], 0.05, 1)
    aud.check_number("latency_tier2_s", after["mean_latency_s"], 0.005)

    aud.check_number("auc_tier1_pct", 100.0 * a4c["auc_micro"], 0.005)
    aud.check_number("auc_tier1_lo_pct", 100.0 * a4c["ci_lo"], 0.005)
    aud.check_number("auc_tier1_hi_pct", 100.0 * a4c["ci_hi"], 0.005)
    aud.check_number("auc_tier2_pct", 100.0 * a6["auc_micro"], 0.005)
    aud.check_number("auc_tier2_lo_pct", 100.0 * a6["ci_lo"], 0.005)
    aud.check_number("auc_tier2_hi_pct", 100.0 * a6["ci_hi"], 0.005)
    aud.check_number("delta_auc_pts", 100.0 * a6["delta_auc"], 0.005)
    aud.check_number("delta_lo_pts", 100.0 * a6["delta_auc_ci_lo"], 0.005)
    aud.check_number("delta_hi_pts", 100.0 * a6["delta_auc_ci_hi"], 0.005)
    aud.check_number("auc_macro_pct", 100.0 * a4c["auc_macro_mean_per_video"], 0.005)

    cost = a6["cost"]
    aud.check_number("cost_median_s", cost["s_per_window_median"], 0.005)
    aud.check_number("cost_p90_s", cost["s_per_window_p90"], 0.005)

    for label, claimed in ABLATION_CLAIMS.items():
        data = read_json(RESULTS / f"ablation_{label}.json")
        value = 100.0 * data["auc_micro"]
        aud.check(f"auc_{label}_pct", f"{claimed:.2f}", f"{value:.2f}", abs(value - claimed) <= 0.005)

    return {"rows": rows, "meta": meta, "normal_ids": normal_ids,
            "windows": windows, "verdicts": verdicts, "annotations_path": None}


# ------------------------------------------------- phần 2: tái lập chỉ số phụ thuộc nhãn GT

def audit_ground_truth(aud: Auditor, state: dict[str, Any], annotations_path: pathlib.Path) -> None:
    from vad.datasets.ucf_crime import load_annotations, load_gt

    annotations = load_annotations(annotations_path)
    rows: Sequence[Mapping[str, str]] = state["rows"]
    meta: Mapping[str, tuple[int, float]] = state["meta"]
    normal_ids: set[str] = state["normal_ids"]

    pooled_scores: list[np.ndarray] = []
    pooled_labels: list[np.ndarray] = []
    per_video: list[float] = []
    for row in rows:
        vid = row["video_id"]
        run = read_json(RESULTS / "run_A4c" / f"{vid}.json")
        scores = np.asarray(run["anomaly_scores"], dtype=np.float64)
        if scores.size == 0:
            continue
        n_frames, fps = meta[vid]
        # expand_to_frames (LAVAD): frame f lấy điểm của mẫu floor(f / fps * sample_fps)
        idx = np.floor(np.arange(n_frames) / fps * 1.0).astype(int)
        expanded = scores[np.clip(idx, 0, scores.size - 1)]
        labels = load_gt(annotations, vid, n_frames).astype(np.int64)
        pooled_scores.append(expanded)
        pooled_labels.append(labels)
        if 0 < labels.sum() < labels.size:
            per_video.append(auc_pooled(expanded, labels))

    micro = auc_pooled(np.concatenate(pooled_scores), np.concatenate(pooled_labels))
    macro = float(np.mean(per_video))
    aud.check_value("auc_tier1_pct (từ điểm thô + GT)", CLAIMS["auc_tier1_pct"], 100.0 * micro, 0.005)
    aud.check_value("auc_macro_pct (từ điểm thô + GT)", CLAIMS["auc_macro_pct"], 100.0 * macro, 0.01)

    def spans(vid: str) -> list[tuple[float, float]]:
        ann = annotations.get(vid)
        if ann is None:
            return []
        fps = meta[vid][1]
        # đoạn GT đóng hai đầu -> khoảng nửa mở [start/fps, (end+1)/fps)
        return [(a / fps, (b + 1) / fps) for a, b in ann.segments]

    def overlaps(window: Mapping[str, Any], g0: float, g1: float) -> bool:
        return window["start_s"] < g1 and window["end_s"] > g0

    def operating(kept: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, float]:
        tp = total = incidents = detected = 0
        latencies: list[float] = []
        for vid, wins in kept.items():
            total += len(wins)
            if vid in normal_ids:
                continue
            segs = spans(vid)
            for win in wins:
                if any(overlaps(win, g0, g1) for g0, g1 in segs):
                    tp += 1
            for g0, g1 in segs:
                incidents += 1
                hit = [w for w in wins if overlaps(w, g0, g1)]
                if hit:
                    detected += 1
                    latencies.append(max(0.0, min(w["start_s"] for w in hit) - g0))
        return {"precision": tp / total if total else 0.0,
                "recall": detected / incidents if incidents else 0.0,
                "latency": float(np.mean(latencies)) if latencies else float("nan"),
                "incidents": float(incidents)}

    tier1 = operating(state["windows"])
    aud.check_value("precision_tier1_pct (tái lập)", CLAIMS["precision_tier1_pct"], 100.0 * tier1["precision"], 0.05, 1)
    aud.check_value("recall_tier1_pct (tái lập)", CLAIMS["recall_tier1_pct"], 100.0 * tier1["recall"], 0.05, 1)
    aud.check_value("latency_tier1_s (tái lập)", CLAIMS["latency_tier1_s"], tier1["latency"], 0.005)
    aud.check_value("incidents", 156.0, tier1["incidents"], 0.0, 0)

    verdicts = state["verdicts"]
    kept2 = {vid: [w for i, w in enumerate(wins) if i in verdicts.get(vid, set())]
             for vid, wins in state["windows"].items()}
    tier2 = operating(kept2)
    aud.check_value("precision_tier2_pct (tái lập)", CLAIMS["precision_tier2_pct"], 100.0 * tier2["precision"], 0.05, 1)
    aud.check_value("recall_tier2_pct (tái lập)", CLAIMS["recall_tier2_pct"], 100.0 * tier2["recall"], 0.05, 1)
    aud.check_value("latency_tier2_s (tái lập)", CLAIMS["latency_tier2_s"], tier2["latency"], 0.005)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", default=None,
                        help="Temporal_Anomaly_Annotation_for_Testing_Videos.txt; bỏ trống thì chỉ "
                             "kiểm các con số đếm được và so khớp file JSON")
    args = parser.parse_args(argv)

    auditor = Auditor()
    state = audit_counts(auditor)
    if args.annotations:
        audit_ground_truth(auditor, state, pathlib.Path(args.annotations))
    else:
        print("(bỏ qua phần phụ thuộc GT: không có --annotations)")
    return auditor.report()


if __name__ == "__main__":
    raise SystemExit(main())
