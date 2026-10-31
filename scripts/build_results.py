# -*- coding: utf-8 -*-
"""Ghép một lần chạy hoàn chỉnh (task C7, Nam): embedding đã cache -> điểm tier-1 -> cửa sổ ứng viên
-> (tuỳ chọn) kết quả tier-2 -> cảnh báo -> `results/<video_id>.json` (đúng hợp đồng ở `vad/results.py`).

    python scripts/build_results.py --manifest manifests/ucf_test.csv --ablation A1 \
        --threshold 0.5 --verifier fake --limit 5
    python scripts/build_results.py --manifest manifests/own_demo.csv --ablation A3 \
        --threshold 2.5 --verify-dir results/tier2 --limit 5

`--verify-dir` nhận HAI kiểu bố cục, vì tier-2 có hai nơi ghi ra:
  - `<verify_dir>/<video_id>.json`                -- một file cho cả video (định dạng `RunResult.load_windows`)
  - `<verify_dir>/<video_id>/window_NNNN.json`    -- một file cho mỗi cửa sổ, đúng thứ `scripts/run_verify.py`
    ghi ra. Kiểu thứ hai được kiểm chặt hơn: window_index liên tục từ 0, tên file khớp index, mọi file cùng
    một cấu hình verifier, và fingerprint tính lại từ cửa sổ tier-1 hiện tại phải khớp fingerprint đã lưu.
    Video không có cửa sổ nào thì không cần file tier-2 (đúng: không có gì để verify).

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

from run_benchmark import (BenchmarkInputError, build_calib_stats, expected_samples, git_commit,  # noqa: E402
                           load_or_embed_prompts, read_benchmark_manifest, select_rows)
from vad.embed.cache import DEFAULT_CACHE_DIR, has_cached, load_embeddings  # noqa: E402
from vad.fakes import FakeVerifier  # noqa: E402
from vad.graph.alerts import build_alert_feed  # noqa: E402
from vad.graph.policy import DEFAULT_POLICY, GraphPolicy  # noqa: E402
from vad.pipeline import ABLATIONS, DEFAULT_PROMPTS, CalibStats, PipelineConfig, build_memory_bank, score_cached_video  # noqa: E402
from vad.results import DEFAULT_RESULTS_DIR, RunResult, make_metadata  # noqa: E402
from vad.types import Alert, VerifyResult, Window  # noqa: E402

DEFAULT_PROMPT_CACHE = ROOT / "cache" / "prompts"
LENGTH_TOLERANCE = 2                    # số mẫu lệch cho phép (làm tròn fps), giống run_benchmark
PAIR_TOLERANCE = 1e-3                   # sai số giây khi đối chiếu cửa sổ tính lại vs cửa sổ trong file tier-2
THRESHOLD_SOURCES = ("placeholder", "calib_normal")
TIER2_SOURCES = ("none", "fake", "json")
VERIFY_WINDOW_GLOB = "window_*.json"
PLACEHOLDER_TIER2_NOTE = ("TIER-2 GIẢ: nhãn sự kiện do FakeVerifier sinh, KHÔNG phải VLM. "
                          "FakeVerifier luôn trả cùng một nhãn (Fighting, 0.9) nên MỌI cảnh báo trong "
                          "file này đều là Fighting — không đọc như kết quả. "
                          "Chỉ dùng để dựng hình dạng file cho dashboard.")
FAKE_VERIFY_FILE_NOTE = ("TIER-2 GIẢ: file trong --verify-dir tự khai `fake=true` (do FakeVerifier sinh), "
                         "KHÔNG phải VLM thật — cảnh báo chỉ để dựng hình dạng file, không đọc như kết quả.")


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


def load_verify_results(path: str | Path,
                        video_id: str) -> tuple[tuple[Window, ...], tuple[VerifyResult, ...]]:
    """Đọc file tier-2 -> (cửa sổ đã lưu, kết quả). Trả CẢ cửa sổ để lát nữa đối chiếu từng cặp.

    Dùng lại `RunResult.from_dict` thay vì tự parse: file sai `schema_version` hay sai kiểu thì lỗi
    ở đúng một chỗ (hợp đồng ở `vad/results.py`), không phải hai chỗ lệch nhau.

    Trả cả cửa sổ là điểm sửa từ review PR #20: chỉ so số lượng thì file tier-2 của ngưỡng/ablation
    khác (cùng số cửa sổ nhưng mốc thời gian khác) vẫn lọt, và nhãn của cửa sổ cũ bị dán sang cửa sổ mới.
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
        saved = RunResult.from_dict(payload)
    except (KeyError, TypeError, ValueError) as e:
        raise BuildResultsError(f"{p.name}: không đọc được tier-2 ({e})") from e
    return saved.candidate_windows, saved.verify_results


def resolve_verify_source(verify_dir: str | Path, video_id: str) -> tuple[str, Path]:
    """Chọn nguồn tier-2 cho MỘT video: `"dir"` (mỗi cửa sổ một file), `"file"` (mỗi video một file) hay `"missing"`.

    `missing` KHÔNG phải lỗi ở đây: video không có cửa sổ nào thì không có gì để verify, và tier-2 không
    sinh thư mục cho nó. Việc "video CÓ cửa sổ mà không có kết quả" được `build_video_result` chặn, vì chỉ
    lúc đó mới biết số cửa sổ thật.

    Có cả hai kiểu cùng lúc là lỗi: hai nguồn có thể nói hai chuyện khác nhau, script không được đoán.
    """
    base = Path(verify_dir)
    per_video = base / f"{video_id}.json"
    per_window = base / video_id
    has_file = per_video.is_file()
    has_dir = per_window.is_dir()
    if has_file and has_dir:
        raise BuildResultsError(
            f"{video_id}: có cả {per_video} và thư mục {per_window} — hai nguồn tier-2 cùng lúc, "
            "chọn một (xoá/đổi tên cái còn lại), không đoán cái nào đúng.")
    if has_dir:
        return "dir", per_window
    if has_file:
        return "file", per_video
    return "missing", per_video


def _read_json_object(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise BuildResultsError(f"{path.name}: phải là JSON object, nhận {type(data).__name__}")
    return data


def _check_window_file(path: Path, saved: RunResult, window: Window) -> tuple[int, tuple[Any, ...]]:
    """Kiểm một file tier-2 theo cửa sổ; trả `(window_index, chữ ký cấu hình verifier)`.

    Vì sao cần fingerprint: `check_pairs_match` chỉ so `start_s`/`end_s`. Hai lần chạy khác cấu hình
    (smoothing khác, verifier khác) có thể cho cùng mốc giây nhưng khác `score`/`k`/prompt — nhãn VLM của
    lần này sẽ bị dán sang cửa sổ của lần kia. `run_verify.make_fingerprint` băm cả window (kể cả `score`)
    lẫn cấu hình verifier, nên tính lại bằng cửa sổ tier-1 hiện tại là phép so đủ chặt.
    """
    meta = saved.metadata
    index = meta.get("window_index")
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise BuildResultsError(f"{path.name}: metadata.window_index thiếu hoặc không hợp lệ ({index!r})")
    if path.name != f"window_{index:04d}.json":
        raise BuildResultsError(
            f"{path.name}: tên file không khớp window_index={index} (phải là window_{index:04d}.json) — "
            "file bị đổi tên hoặc trộn từ lần chạy khác, không ghép bừa.")
    k = meta.get("k")
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise BuildResultsError(f"{path.name}: metadata.k thiếu hoặc không hợp lệ ({k!r})")
    max_chunk_s = meta.get("max_chunk_s")
    if not isinstance(max_chunk_s, (int, float)) or isinstance(max_chunk_s, bool) \
            or not math.isfinite(max_chunk_s) or max_chunk_s <= 0:
        raise BuildResultsError(f"{path.name}: metadata.max_chunk_s thiếu hoặc không hợp lệ ({max_chunk_s!r})")
    name = meta.get("model")
    if not isinstance(name, str) or not name.strip():
        raise BuildResultsError(f"{path.name}: metadata.model (tên verifier) thiếu — không tính lại được fingerprint")
    config = meta.get("verifier_config") or {}
    stored = meta.get("fingerprint")
    if not isinstance(stored, str) or not stored:
        raise BuildResultsError(
            f"{path.name}: metadata.fingerprint thiếu — không kiểm tra được file có đúng cửa sổ này không")

    from run_verify import make_fingerprint      # lazy: không cần cv2 khi chỉ dùng kiểu <video_id>.json

    expected = make_fingerprint(window, k, name, config, float(max_chunk_s))
    if expected != stored:
        raise BuildResultsError(
            f"{path.name}: fingerprint không khớp cửa sổ tính lại [{window.start_s:g}, {window.end_s:g}] "
            f"score {window.score:.6g} — file tier-2 của lần chạy khác (khác cửa sổ/verifier/tham số), "
            "không ghép bừa.")
    signature = (name, k, float(max_chunk_s), json.dumps(config, sort_keys=True))
    return index, signature


def load_verify_window_dir(path: str | Path, video_id: str
                           ) -> tuple[tuple[Window, ...], tuple[VerifyResult, ...], bool]:
    """Đọc thư mục tier-2 kiểu `run_verify.py`: `<dir>/<video_id>/window_NNNN.json`.

    Trả `(cửa sổ, kết quả, có_file_giả)`. Mỗi file phải đúng MỘT cửa sổ + MỘT kết quả; cửa sổ được sắp
    theo `window_index` chứ không theo thứ tự file trên đĩa (Windows/Linux/metadata đều có thể đổi thứ tự).
    """
    d = Path(path)
    if not d.is_dir():
        raise BuildResultsError(f"Chưa có thư mục tier-2 cho {video_id!r}: {d}")
    files = sorted(d.glob(VERIFY_WINDOW_GLOB))
    if not files:
        raise BuildResultsError(f"{d}: không có file {VERIFY_WINDOW_GLOB} nào — tier-2 chưa chạy?")

    by_index: dict[int, tuple[Window, VerifyResult, Path]] = {}
    signatures: set[tuple[Any, ...]] = set()
    fake = False
    for p in files:
        payload = {**_read_json_object(p), "anomaly_scores": []}
        payload.setdefault("sample_fps", 1.0)
        try:
            saved = RunResult.from_dict(payload)
        except (KeyError, TypeError, ValueError) as e:
            raise BuildResultsError(f"{p.name}: không đọc được tier-2 ({e})") from e
        if saved.video_id != video_id:
            raise BuildResultsError(
                f"{p.name}: video_id {saved.video_id!r} khác thư mục {video_id!r} — file của video khác.")
        if len(saved.candidate_windows) != 1 or len(saved.verify_results) != 1:
            raise BuildResultsError(
                f"{p.name}: file tier-2 theo cửa sổ phải có đúng 1 cửa sổ + 1 kết quả, nhận "
                f"{len(saved.candidate_windows)} cửa sổ + {len(saved.verify_results)} kết quả.")
        window, result = saved.candidate_windows[0], saved.verify_results[0]
        index, signature = _check_window_file(p, saved, window)
        if index in by_index:
            # Không thể xảy ra khi tên file đã khớp index, nhưng giữ để lỗi rõ nếu ai đó nới kiểm tên file.
            raise BuildResultsError(f"{d}: hai file cùng window_index={index} ({by_index[index][2].name}, {p.name})")
        by_index[index] = (window, result, p)
        signatures.add(signature)
        fake = fake or bool(saved.metadata.get("fake"))

    if len(signatures) > 1:
        raise BuildResultsError(
            f"{d}: các file tier-2 không cùng một cấu hình verifier ({len(signatures)} kiểu: "
            f"{sorted(s[0] for s in signatures)}) — trộn kết quả của hai lần chạy, không ghép bừa.")
    missing = [i for i in range(max(by_index) + 1) if i not in by_index]
    if missing:
        raise BuildResultsError(
            f"{d}: thiếu window_index {missing} (đang có {sorted(by_index)}) — tier-2 chưa chạy hết, "
            "ghép sẽ gán nhãn lệch cửa sổ.")
    ordered = [by_index[i] for i in range(len(by_index))]
    return (tuple(w for w, _, _ in ordered), tuple(v for _, v, _ in ordered), fake)


# --- dựng một video -------------------------------------------------------------------

def check_pairs_match(video_id: str, windows: Sequence[Window],
                      saved: Sequence[Window]) -> None:
    """Từng cặp cửa sổ phải cùng mốc thời gian; lệch quá `PAIR_TOLERANCE` giây -> lỗi.

    Bắt đúng lỗ mà review PR #20 phát hiện: file tier-2 của ngưỡng/ablation khác có thể cùng SỐ cửa sổ
    nhưng khác mốc giây, và nhãn VLM của cửa sổ cũ sẽ bị dán sang cửa sổ mới mà không ai biết.
    """
    if len(saved) != len(windows):
        raise BuildResultsError(
            f"{video_id}: file tier-2 có {len(saved)} cửa sổ nhưng tính lại được {len(windows)} — lệch cặp.")
    for i, (w, s) in enumerate(zip(windows, saved)):
        if abs(w.start_s - s.start_s) > PAIR_TOLERANCE or abs(w.end_s - s.end_s) > PAIR_TOLERANCE:
            raise BuildResultsError(
                f"{video_id}: cửa sổ #{i} tính lại là [{w.start_s:g}, {w.end_s:g}] nhưng file tier-2 ghi "
                f"[{s.start_s:g}, {s.end_s:g}] — tier-2 chạy trên ngưỡng/ablation khác, không ghép bừa.")

def build_video_result(video_id: str, config: PipelineConfig, normal_embs: np.ndarray,
                       abnormal_embs: np.ndarray, cache_dir: str | Path = DEFAULT_CACHE_DIR,
                       verify_results: Sequence[VerifyResult] | None = None,
                       policy: GraphPolicy | None = None, camera: str | None = None,
                       zone: str | None = None, hour: int | None = None, ablation: str = "",
                       commit: str = "", extra_note: str = "", tier2_source: str = "none",
                       threshold_source: str = "placeholder", calib_manifest: str = "",
                       normal_bank: np.ndarray | None = None,
                       saved_windows: Sequence[Window] | None = None,
                       calib_stats: CalibStats | None = None,
                       tier2_fake: bool = False) -> RunResult:
    """Một video -> RunResult đầy đủ (điểm, cửa sổ, tier-2 nếu có, cảnh báo, metadata).

    `config.threshold` quyết định có cắt cửa sổ hay không: None = chỉ có đường điểm (dashboard vẫn vẽ
    được timeline), không có cửa sổ nên cũng không có cảnh báo.

    `saved_windows` = cửa sổ ghi trong file tier-2 (`--verify-dir`). Có nó thì từng cặp cửa sổ ↔ kết quả
    được đối chiếu mốc thời gian; `None` (FakeVerifier) thì bỏ qua vì cửa sổ tính từ chính config.
    """
    if tier2_source not in TIER2_SOURCES:
        raise BuildResultsError(f"tier2_source phải thuộc {TIER2_SOURCES}, nhận {tier2_source!r}")

    tier1 = score_cached_video(video_id, config, normal_embs, abnormal_embs, cache_dir,
                               normal_bank=normal_bank, calib_stats=calib_stats)
    windows = tier1.windows
    verify = tuple(verify_results or ())

    if tier2_source == "json" and windows and not verify:
        # file tier-2 rỗng trong khi video CÓ cửa sổ -> metadata "tier2=json" sẽ nói dối.
        # Video thật sự 0 cửa sổ thì file rỗng là hợp lệ, nên điều kiện `windows` là cần.
        raise BuildResultsError(
            f"{video_id}: Chưa có file tier-2 (hoặc file không có kết quả nào) trong khi video có "
            f"{len(windows)} cửa sổ — metadata sẽ ghi tier2=json trong khi không có tier-2. "
            "Sửa file hoặc bỏ --verify-dir."
        )

    if verify and len(verify) != len(windows):
        # tier-2 dựng bằng ngưỡng khác -> số cửa sổ lệch -> ghép bừa sẽ gán nhãn sai cửa sổ
        raise BuildResultsError(
            f"{video_id}: {len(verify)} kết quả tier-2 nhưng {len(windows)} cửa sổ — lệch cặp. "
            "Chạy lại tier-2 trên đúng ngưỡng hiện tại, không ghép bừa."
        )

    if verify and saved_windows is not None:
        # cùng SỐ cửa sổ chưa đủ: mốc thời gian phải trùng, nếu không là tier-2 của lần chạy khác.
        check_pairs_match(video_id, windows, saved_windows)

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
    if tier2_fake:
        notes.append(FAKE_VERIFY_FILE_NOTE)
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
                        calib_stats: CalibStats | None = None,
                        write: bool = True) -> list[RunResult]:
    """Chạy mọi video trong `rows` và (mặc định) ghi `results/<id>.json` + `results/windows/<id>.json`."""
    check_cache(rows, config, cache_dir)
    if verify_dir is not None and not Path(verify_dir).is_dir():
        # Sai đường dẫn thì mọi video đều "không có tier-2"; nếu manifest tình cờ không có cửa sổ nào,
        # lượt chạy sẽ "thành công" mà metadata ghi tier2=json. Bắt ở đây, không để lọt.
        raise BuildResultsError(f"--verify-dir không tồn tại hoặc không phải thư mục: {verify_dir}")
    tier2_source = "fake" if verifier is not None else ("json" if verify_dir is not None else "none")
    results: list[RunResult] = []
    for row in rows:
        vid = row["video_id"]
        verify: Sequence[VerifyResult] | None = None
        saved_windows: Sequence[Window] | None = None
        fake_tier2 = False
        if verifier is not None:
            # cắt cửa sổ trước bằng chính config, rồi mới gọi verifier -> hai bên chắc chắn cùng số cửa sổ
            probe = score_cached_video(vid, config, normal_embs, abnormal_embs, cache_dir,
                                       normal_bank=normal_bank, calib_stats=calib_stats)
            verify = verify_windows(probe.windows, verifier, vid)
        elif verify_dir is not None:
            kind, source = resolve_verify_source(verify_dir, vid)
            if kind == "dir":
                saved_windows, verify, fake_tier2 = load_verify_window_dir(source, vid)
            elif kind == "file":
                saved_windows, verify = load_verify_results(source, vid)
            # kind == "missing": video không có cửa sổ thì hợp lệ; có cửa sổ thì build_video_result báo lỗi.

        res = build_video_result(vid, config, normal_embs, abnormal_embs, cache_dir, verify_results=verify,
                                 policy=policy, camera=camera, zone=zone, hour=hour, ablation=ablation,
                                 commit=commit, extra_note=extra_note, tier2_source=tier2_source,
                                 threshold_source=threshold_source, calib_manifest=calib_manifest,
                                 normal_bank=normal_bank, saved_windows=saved_windows,
                                 calib_stats=calib_stats, tier2_fake=fake_tier2)
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
                   help="thư mục tier-2 thật: <video_id>.json (một file/video) hoặc "
                        "<video_id>/window_NNNN.json (một file/cửa sổ, đúng thứ scripts/run_verify.py ghi)")
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
        stats = None
        if config.per_video_norm and config.norm_ref == "calib":
            if not args.calib_manifest:
                p.error(f"{args.ablation} chuẩn hoá theo calib_normal: cần --calib-manifest manifests/ucf_calib_normal.csv")
            stats = build_calib_stats(args.calib_manifest, config, args.cache_dir, (normal_embs, abnormal_embs),
                                      args.bank_max_frames)
        results = build_from_manifest(
            rows, config, normal_embs, abnormal_embs, cache_dir=args.cache_dir, out_dir=args.out_dir,
            verify_dir=args.verify_dir, verifier=FakeVerifier() if args.verifier == "fake" else None,
            policy=GraphPolicy(), camera=args.camera, zone=args.zone, hour=args.hour,
            ablation=args.ablation, commit=git_commit(), extra_note=args.note,
            threshold_source=args.threshold_source, calib_manifest=args.calib_manifest or "",
            normal_bank=bank, calib_stats=stats,
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
