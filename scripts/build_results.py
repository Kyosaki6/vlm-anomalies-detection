# -*- coding: utf-8 -*-
"""Ghép một lần chạy hoàn chỉnh (task C7, Nam): embedding đã cache -> điểm tier-1 -> cửa sổ ứng viên
-> (tuỳ chọn) kết quả tier-2 -> cảnh báo -> `results/<video_id>.json` (đúng hợp đồng ở `vad/results.py`).

    python scripts/build_results.py --manifest manifests/ucf_test.csv --ablation A1 \
        --threshold 0.5 --verifier fake --limit 5
    python scripts/build_results.py --manifest manifests/own_demo.csv --ablation A3 \
        --threshold 2.5 --verify-dir results/tier2 --limit 5

Vì sao có script này: `run_benchmark.py` trả lời "các ablation được bao nhiêu AUC"; nó không sinh ra thứ
dashboard đọc. Script này là mảnh nối cuối: chạy tier-1 trên cache, cắt cửa sổ bằng một ngưỡng, gắn kết
quả tier-2 (nếu đã có), dựng "bản tin cảnh báo" bằng `vad/graph/alerts.py`, rồi ghi file.

BA ĐIỀU KHÔNG ĐƯỢC LÀM Ở ĐÂY:
  1. **Không tự calibrate ngưỡng.** Ngưỡng phải do người dùng truyền vào, kèm `--threshold-source`.
     `placeholder` = chưa calibrate (ghi vào `metadata.note`, dashboard hiện banner), `calib_normal` =
     lấy từ manifest calib (bắt buộc kèm `--calib-manifest` để có chứng cứ). Không có nguồn từ tập test.
  2. **Không bịa tier-2.** Không có kết quả tier-2 thì `verify_results` và `alerts` để rỗng và metadata
     ghi `tier="1"`. `--verifier fake` chỉ để dựng hình dạng file cho dashboard; nó TỰ KHAI là giả.
  3. **Không bỏ qua video im lặng.** Thiếu cache, lệch độ dài cache vs `n_frames/fps`, điểm NaN -> dừng.

Nạp model: chỉ khi cache prompt chưa có (dùng lại `cache/prompts/` của `run_benchmark.py`, CPU là đủ).
`--dry-run` không nạp model, không ghi file — chỉ đếm và liệt kê video thiếu cache.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_benchmark import (BenchmarkInputError, expected_samples, git_commit,  # noqa: E402
                           load_or_embed_prompts, read_benchmark_manifest, select_rows)
from vad.embed.cache import DEFAULT_CACHE_DIR, has_cached, load_embeddings  # noqa: E402
from vad.fakes import FakeVerifier  # noqa: E402
from vad.graph.alerts import build_alert_feed  # noqa: E402
from vad.graph.policy import DEFAULT_POLICY, GraphPolicy  # noqa: E402
from vad.pipeline import ABLATIONS, DEFAULT_PROMPTS, PipelineConfig, build_memory_bank, score_cached_video  # noqa: E402
from vad.results import DEFAULT_RESULTS_DIR, RunResult, make_metadata  # noqa: E402
from vad.types import Alert, VerifyResult  # noqa: E402

DEFAULT_PROMPT_CACHE = ROOT / "cache" / "prompts"
LENGTH_TOLERANCE = 2                    # số mẫu lệch cho phép (làm tròn fps), giống run_benchmark
THRESHOLD_SOURCES = ("placeholder", "calib_normal")
TIER2_SOURCES = ("none", "fake", "json")
PLACEHOLDER_TIER2_NOTE = ("TIER-2 GIẢ: nhãn sự kiện do FakeVerifier sinh, KHÔNG phải VLM. "
                          "Chỉ dùng để dựng hình dạng file cho dashboard.")


class BuildResultsError(ValueError):
    """Đầu vào không dựng được file kết quả (thiếu cache, lệch độ dài, ngưỡng không rõ nguồn...)."""


# --- ngưỡng & ghi chú ----------------------------------------------------------------

def threshold_note(threshold: float, source: str, calib_manifest: str = "") -> str:
    """Ghi chú cảnh báo cho `metadata.note` — RỖNG khi ngưỡng đã calibrate đúng bài.

    `note` khác rỗng là dấu "file này chưa phải kết quả" (`app.data.is_placeholder` đọc đúng nó), nên
    chỉ ngưỡng giả thuyết mới được ghi vào đây. Xuất xứ của ngưỡng thật nằm ở `metadata.threshold_source`
    + `metadata.calib_manifest`, để dashboard không dán nhãn "giả" lên một kết quả thật.
    """
    if source not in THRESHOLD_SOURCES:
        raise BuildResultsError(f"threshold_source phải thuộc {THRESHOLD_SOURCES}, nhận {source!r}")
    if source == "placeholder":
        return (f"NGƯỠNG GIẢ THUYẾT ({threshold:g}): chưa calibrate trên calib_normal, "
                "đầu ra đúng cấu trúc nhưng KHÔNG phải kết quả.")
    if not calib_manifest:
        raise BuildResultsError("threshold_source='calib_normal' cần calib_manifest để có chứng cứ")
    return ""


def _join_notes(*parts: str) -> str:
    return " ".join(p.strip() for p in parts if p and p.strip())


def infer_ablation(config: PipelineConfig) -> str:
    """Config -> tên preset ('A0'..'A5'), khớp sau khi bỏ qua `threshold`. Không khớp preset nào -> ''.

    Để file kết quả ghi được "A3" thay vì bắt người đọc so 16 trường config, và để một config tự chế
    không bị dán nhầm tên preset.
    """
    for name, preset in ABLATIONS.items():
        if replace(config, threshold=None) == replace(preset, threshold=None):
            return name
    return ""


# --- tier-2 ---------------------------------------------------------------------------

def verify_windows(windows: Sequence[Any], verifier: Any, video_id: str) -> tuple[VerifyResult, ...]:
    """Gọi verifier cho từng cửa sổ. `frames=[]` — CHỈ dùng cho FakeVerifier/verifier đọc kết quả sẵn.

    VLM thật cần khung hình giải mã từ video; việc đó thuộc `scripts/run_verify.py` (A3), nó ghi JSON
    từng cửa sổ, rồi truyền lại qua `load_verify_results`. Đặt ở đây để không ai tưởng đường này chạy
    được với VLM thật.
    """
    out: list[VerifyResult] = []
    for w in windows:
        out.append(verifier.verify([], {"video_id": video_id, "start_s": w.start_s,
                                        "end_s": w.end_s, "tier": "2"}))
    return tuple(out)


def load_verify_results(path: str | Path, video_id: str) -> tuple[VerifyResult, ...]:
    """Đọc kết quả tier-2 của một video từ file ĐÚNG định dạng `results/windows/<video_id>.json`.

    Dùng lại `RunResult.from_dict` thay vì tự parse: file sai `schema_version` hay sai kiểu thì lỗi
    ở đúng một chỗ (hợp đồng ở `vad/results.py`), không phải hai chỗ lệch nhau.
    """
    p = Path(path)
    if not p.is_file():
        raise BuildResultsError(f"Chưa có file tier-2 cho {video_id!r}: {p}")
    with open(p, encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "verify_results" not in data:
        raise BuildResultsError(f"{p.name}: cần JSON object có khoá 'verify_results' "
                                "(định dạng của results/windows/<video_id>.json)")
    payload = {**data, "anomaly_scores": []}
    payload.setdefault("sample_fps", 1.0)
    try:
        return RunResult.from_dict(payload).verify_results
    except (KeyError, TypeError, ValueError) as e:
        raise BuildResultsError(f"{p.name}: không đọc được tier-2 ({e})") from e


# --- dựng một video -------------------------------------------------------------------

def build_video_result(video_id: str, config: PipelineConfig, normal_embs: np.ndarray,
                       abnormal_embs: np.ndarray, cache_dir: str | Path = DEFAULT_CACHE_DIR,
                       verify_results: Sequence[VerifyResult] | None = None,
                       policy: GraphPolicy | None = None, camera: str | None = None,
                       zone: str | None = None, hour: int | None = None, ablation: str = "",
                       commit: str = "", extra_note: str = "", tier2_source: str = "none",
                       threshold_source: str = "placeholder", calib_manifest: str = "",
                       normal_bank: np.ndarray | None = None) -> RunResult:
    """Một video -> RunResult đầy đủ (điểm, cửa sổ, tier-2 nếu có, cảnh báo, metadata).

    `config.threshold` quyết định có cắt cửa sổ hay không: None = chỉ có đường điểm (dashboard vẫn vẽ
    được timeline), không có cửa sổ nên cũng không có cảnh báo.
    """
    if tier2_source not in TIER2_SOURCES:
        raise BuildResultsError(f"tier2_source phải thuộc {TIER2_SOURCES}, nhận {tier2_source!r}")

    tier1 = score_cached_video(video_id, config, normal_embs, abnormal_embs, cache_dir,
                               normal_bank=normal_bank)
    windows = tier1.windows
    verify = tuple(verify_results or ())

    if verify and len(verify) != len(windows):
        # tier-2 dựng bằng ngưỡng khác -> số cửa sổ lệch -> ghép bừa sẽ gán nhãn sai cửa sổ
        raise BuildResultsError(
            f"{video_id}: {len(verify)} kết quả tier-2 nhưng {len(windows)} cửa sổ — lệch cặp. "
            "Chạy lại tier-2 trên đúng ngưỡng hiện tại, không ghép bừa."
        )

    alerts: tuple[Alert, ...] = ()
    if verify:
        cam = camera if camera is not None else video_id     # một video = một camera (cooldown lùi về video_id)
        alerts = tuple(build_alert_feed(list(zip(windows, verify)), policy or GraphPolicy(),
                                        zone=zone, hour=hour, camera=cam))

    notes = []
    if config.threshold is not None:
        notes.append(threshold_note(config.threshold, threshold_source, calib_manifest))
    if tier2_source == "fake":
        notes.append(PLACEHOLDER_TIER2_NOTE)
    if extra_note:
        notes.append(extra_note)

    metadata = make_metadata(
        model=config.model_id, commit=commit, tier="1+2" if verify else "1",
        ablation=ablation or infer_ablation(config), sample_fps=config.sample_fps, threshold=config.threshold,
        threshold_source=threshold_source if config.threshold is not None else None,
        calib_manifest=calib_manifest or None,
        camera=camera or video_id, zone=zone or "default", hour=hour,
        tier2=tier2_source, config=asdict(config),
        note=_join_notes(*notes),
    )
    return RunResult(video_id=video_id, sample_fps=config.sample_fps, anomaly_scores=tier1.scores,
                     candidate_windows=windows, verify_results=verify, alerts=alerts, metadata=metadata)


# --- dựng cả manifest ------------------------------------------------------------------

def check_cache(rows: Sequence[dict[str, Any]], config: PipelineConfig, cache_dir: str | Path) -> None:
    """Thiếu cache = dừng và liệt kê đủ mọi video, không lặng lẽ bỏ qua (file kết quả thiếu video là lỗi)."""
    missing = [r["video_id"] for r in rows
               if not has_cached(config.model_id, r["video_id"], config.sample_fps, cache_dir)]
    if missing:
        raise BuildResultsError(f"Thiếu cache embedding ({config.model_id}, fps {config.sample_fps:g}) "
                                f"cho {len(missing)}/{len(rows)} video: {missing}")


def check_lengths(video_id: str, scores: np.ndarray, row: dict[str, Any], sample_fps: float) -> None:
    """Cache dài/ngắn hơn video thật quá dung sai = cache của video khác hoặc manifest sai -> dừng.

    Điểm rỗng/NaN không cần kiểm ở đây: `RunResult` đã từ chối lúc dựng (và đó là chỗ duy nhất nên từ chối).
    """
    want = expected_samples(row["n_frames"], row["fps"], sample_fps)
    if abs(len(scores) - want) > LENGTH_TOLERANCE:
        raise BuildResultsError(f"{video_id}: cache có {len(scores)} mẫu, cần ~{want} "
                                f"({row['n_frames']} frame @ {row['fps']:g} fps, sample_fps {sample_fps:g})")


def build_from_manifest(rows: Sequence[dict[str, Any]], config: PipelineConfig, normal_embs: np.ndarray,
                        abnormal_embs: np.ndarray, cache_dir: str | Path = DEFAULT_CACHE_DIR,
                        out_dir: str | Path = DEFAULT_RESULTS_DIR, verify_dir: str | Path | None = None,
                        verifier: Any | None = None, policy: GraphPolicy | None = None,
                        camera: str | None = None, zone: str | None = None, hour: int | None = None,
                        ablation: str = "", commit: str = "", extra_note: str = "",
                        threshold_source: str = "placeholder", calib_manifest: str = "",
                        normal_bank: np.ndarray | None = None,
                        write: bool = True) -> list[RunResult]:
    """Chạy mọi video trong `rows` và (mặc định) ghi `results/<id>.json` + `results/windows/<id>.json`."""
    check_cache(rows, config, cache_dir)
    tier2_source = "fake" if verifier is not None else ("json" if verify_dir is not None else "none")
    results: list[RunResult] = []
    for row in rows:
        vid = row["video_id"]
        verify: Sequence[VerifyResult] | None = None
        if verifier is not None:
            # cắt cửa sổ trước bằng chính config, rồi mới gọi verifier -> hai bên chắc chắn cùng số cửa sổ
            probe = score_cached_video(vid, config, normal_embs, abnormal_embs, cache_dir,
                                       normal_bank=normal_bank)
            verify = verify_windows(probe.windows, verifier, vid)
        elif verify_dir is not None:
            verify = load_verify_results(Path(verify_dir) / f"{vid}.json", vid)

        res = build_video_result(vid, config, normal_embs, abnormal_embs, cache_dir, verify_results=verify,
                                 policy=policy, camera=camera, zone=zone, hour=hour, ablation=ablation,
                                 commit=commit, extra_note=extra_note, tier2_source=tier2_source,
                                 threshold_source=threshold_source, calib_manifest=calib_manifest,
                                 normal_bank=normal_bank)
        check_lengths(vid, np.asarray(res.anomaly_scores), row, res.sample_fps)
        if write:
            res.to_json(root=out_dir)
            res.save_windows(root=out_dir)
        results.append(res)
    return results


def summary_line(res: RunResult, written: bool = True) -> str:
    """Một dòng cho log: đủ để thấy file này có phải kết quả thật hay không."""
    note = str(res.metadata.get("note", "")).strip()
    flag = "PLACEHOLDER" if note else "ok"
    return (f"{res.video_id:<28} {len(res.anomaly_scores):>5} mẫu  {len(res.candidate_windows):>3} cửa sổ  "
            f"{len(res.alerts):>2} cảnh báo  tier={res.metadata.get('tier', '?')}  [{flag}]"
            + ("" if written else "  (không ghi file)"))


# --- CLI -------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--ablation", default="A1", help=f"một trong {sorted(ABLATIONS)}")
    p.add_argument("--threshold", type=float, default=None,
                   help="ngưỡng cắt cửa sổ; bỏ trống = chỉ có đường điểm, không có cửa sổ/cảnh báo")
    p.add_argument("--threshold-source", choices=THRESHOLD_SOURCES, default="placeholder")
    p.add_argument("--calib-manifest", default=None,
                   help="manifest calib_normal: bắt buộc khi --threshold-source calib_normal, và cho ablation dùng memory")
    p.add_argument("--verifier", choices=("none", "fake"), default="none",
                   help="fake = FakeVerifier (chỉ để dựng hình dạng file, tự khai là giả)")
    p.add_argument("--verify-dir", default=None,
                   help="thư mục chứa <video_id>.json (định dạng results/windows/) từ tier-2 thật")
    p.add_argument("--limit", type=int, default=None, help="chỉ chạy N video cách đều (chạy thử)")
    p.add_argument("--out-dir", default=str(DEFAULT_RESULTS_DIR))
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    p.add_argument("--prompt-cache-dir", default=str(DEFAULT_PROMPT_CACHE))
    p.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    p.add_argument("--policy", default=str(DEFAULT_POLICY))
    p.add_argument("--camera", default=None, help="tên camera cho mọi video (mặc định: video_id)")
    p.add_argument("--zone", default=None, help="zone trong policy.yaml (mặc định: default)")
    p.add_argument("--hour", type=int, default=None, help="giờ 0-23, để policy nâng severity ban đêm")
    p.add_argument("--bank-max-frames", type=int, default=20000)
    p.add_argument("--device", default="cpu", help="thiết bị cho text tower khi cache prompt còn thiếu")
    p.add_argument("--note", default="", help="ghi chú thêm, nối vào metadata.note")
    p.add_argument("--dry-run", action="store_true", help="không nạp model, không ghi file")
    args = p.parse_args(argv)

    if args.ablation not in ABLATIONS:
        p.error(f"ablation không có: {args.ablation}; hợp lệ: {sorted(ABLATIONS)}")
    if args.verifier != "none" and args.verify_dir:
        p.error("chọn MỘT nguồn tier-2: --verifier fake hoặc --verify-dir")
    if args.threshold is None and (args.verifier != "none" or args.verify_dir):
        p.error("tier-2 gắn với cửa sổ: cần --threshold")
    if args.threshold is not None and not math.isfinite(args.threshold):
        p.error("--threshold phải hữu hạn")
    if args.threshold_source == "calib_normal" and not args.calib_manifest:
        p.error("--threshold-source calib_normal cần --calib-manifest làm chứng cứ")
    if args.hour is not None and not 0 <= args.hour <= 23:
        p.error("--hour phải trong 0..23")

    config = ABLATIONS[args.ablation]
    if args.threshold is not None:
        config = replace(config, threshold=float(args.threshold))

    try:
        rows = select_rows(read_benchmark_manifest(args.manifest), args.limit)
    except BenchmarkInputError as e:
        raise BuildResultsError(f"manifest {args.manifest}: {e}") from e

    print(f"{len(rows)} video, ablation {args.ablation}, model {config.model_id}, ngưỡng {config.threshold}")
    if args.dry_run:
        try:
            check_cache(rows, config, args.cache_dir)
        except BuildResultsError as e:
            print(f"DRY-RUN: {e}")
            return 1
        print(f"DRY-RUN: đủ cache cho cả {len(rows)} video; không nạp model, không ghi file")
        return 0

    try:
        normal_embs, abnormal_embs = load_or_embed_prompts(config, args.prompts, args.prompt_cache_dir,
                                                           device=args.device)
        bank = None
        if config.use_memory:
            if not args.calib_manifest:
                p.error(f"{args.ablation} dùng normal memory: cần --calib-manifest manifests/ucf_calib_normal.csv")
            calib_rows = read_benchmark_manifest(args.calib_manifest)
            missing = [r["video_id"] for r in calib_rows
                       if not has_cached(config.model_id, r["video_id"], config.sample_fps, args.cache_dir)]
            if missing:
                raise BuildResultsError(f"Thiếu cache calib_normal cho memory bank ({len(missing)} video): {missing}")
            bank = build_memory_bank([load_embeddings(config.model_id, r["video_id"], config.sample_fps,
                                                      args.cache_dir) for r in calib_rows],
                                     max_frames=args.bank_max_frames)
        results = build_from_manifest(
            rows, config, normal_embs, abnormal_embs, cache_dir=args.cache_dir, out_dir=args.out_dir,
            verify_dir=args.verify_dir, verifier=FakeVerifier() if args.verifier == "fake" else None,
            policy=GraphPolicy(), camera=args.camera, zone=args.zone, hour=args.hour,
            ablation=args.ablation, commit=git_commit(), extra_note=args.note,
            threshold_source=args.threshold_source, calib_manifest=args.calib_manifest or "",
            normal_bank=bank,
        )
    except BenchmarkInputError as e:
        raise BuildResultsError(str(e)) from e

    for res in results:
        print("  " + summary_line(res))
    print(f"đã ghi {len(results)} file vào {args.out_dir} (+ windows/)")
    placeholder = [r.video_id for r in results if str(r.metadata.get("note", "")).strip()]
    if placeholder:
        print(f"CẢNH BÁO: {len(placeholder)} file có metadata.note (ngưỡng giả định hoặc tier-2 giả) — "
              "dashboard sẽ hiện banner, không dùng làm số báo cáo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
