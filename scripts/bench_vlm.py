# -*- coding: utf-8 -*-
"""A1redo (Phước): benchmark VLM tier-2 trên Colab T4 với cửa sổ THẬT + prompt THẬT.

Sửa theo `notes/2026-10-29-a1-redo-plan.md` (F1-F13):
  - F2: warmup tách riêng (--warmup), thống kê median/mean/p90 chỉ trên run đo; log out-token + s/token.
  - F3: đầu vào là 16 cửa sổ thật từ scripts/make_bench_windows.py (--windows), không còn ảnh chụp riêng lẻ.
  - F4: prompt lấy từ configs/prompts.yaml (verifier.system + verifier.user), ghi sha1[:8] vào header.
  - F5: quality sanity trên đủ 16 cửa sổ qua vad.verify.vlm.parse_verifier_json (parse/consistency/TPR/FPR/type_match).
  - F7: visual token tường minh: images native 320x240 vs cap 448^2; SmolVLM2 có/nếu không image splitting.
  - F8/F9: VRAM = torch allocated + reserved + device-level nvidia-smi (thread 0.2s); --vram-cap-gb mô
    phỏng card 6 GB; gc.collect + empty_cache giữa các variant.
  - F1/F12: header đủ phiên bản/commit/hashes; log tee được; requirements-vlm.txt bỏ qwen-vl-utils.

Mỗi model MỘT process (Colab: 3 lệnh python riêng, xem notebooks/a1_vlm_bench_colab.ipynb):

    python scripts/bench_vlm.py --dry-run --windows notes/vlm-bench-windows.json
    python -u scripts/bench_vlm.py --model Qwen/Qwen2-VL-2B-Instruct \
        --windows notes/vlm-bench-windows.json --warmup 2 --runs 8 \
        --out raw_qwen2vl2b.json --jsonl qwen2vl2b.jsonl 2>&1 | tee log_qwen2vl2b.txt
    # vòng mô phỏng 6 GB (chỉ 3 cửa sổ gồm worst-case, không quality):
    python -u scripts/bench_vlm.py --model Qwen/Qwen2-VL-2B-Instruct --windows notes/vlm-bench-windows.json \
        --vram-cap-gb 5.5 --runs 3 --skip-quality --out raw_capped.json ...
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vad.verify.vlm import parse_verifier_json  # noqa: E402
from vad.types import EventType  # noqa: E402

# Ứng viên lớp 2-3B (task sheet A1) + variant đầu vào theo redo plan §3.3.
# transformers 5.x: class riêng từng gia đình; Auto* không luôn map đúng SmolVLM2.
CANDIDATES: dict[str, dict[str, Any]] = {
    "Qwen/Qwen2-VL-2B-Instruct": {
        "class": "Qwen2VLForConditionalGeneration",
        "variants": ["images_native", "images_cap448", "video_native"],
    },
    "Qwen/Qwen2.5-VL-3B-Instruct": {
        "class": "Qwen2_5_VLForConditionalGeneration",
        "variants": ["images_native", "images_cap448", "video_native"],
    },
    "HuggingFaceTB/SmolVLM2-2.2B-instruct": {
        "class": "SmolVLMForConditionalGeneration",
        "variants": ["images_nosplit", "images_split"],
    },
}
MAX_PIXELS_448 = 448 * 448


# ---------------------------------------------------------------- nvidia-smi helpers
def _nvidia_smi(query: str) -> list[str] | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0:
            return None
        return [line.strip() for line in out.stdout.strip().splitlines() if line.strip()]
    except Exception:
        return None


class DevicePeakSampler:
    """Sample `memory.used` (MB) của GPU từ nvidia-smi, nền, mỗi 0.2s trong khi inference."""

    def __init__(self, interval_s: float = 0.2):
        self.interval_s = interval_s
        self.peak_mb: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.available = True

    def _loop(self) -> None:
        while not self._stop.is_set():
            rows = _nvidia_smi("memory.used")
            if rows:
                try:
                    mb = float(rows[0])
                except ValueError:
                    mb = None
                if mb is not None:
                    self.peak_mb = mb if self.peak_mb is None else max(self.peak_mb, mb)
            else:
                self.available = False
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "DevicePeakSampler":
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)


def device_idle_mb() -> float | None:
    rows = _nvidia_smi("memory.used")
    if not rows:
        return None
    try:
        return float(rows[0])
    except ValueError:
        return None


# ---------------------------------------------------------------- prompt + windows
def load_verifier_prompt(path: str | Path) -> tuple[str, str, str]:
    """configs/prompts.yaml -> (system, user, sha1[:8]). KHÔNG copy prompt vào script (F4)."""
    import yaml

    text = Path(path).read_text(encoding="utf-8-sig")
    data = yaml.safe_load(text).get("verifier")
    if not data or "system" not in data or "user" not in data:
        raise ValueError(f"configs/prompts.yaml không có khoá verifier.system/user: {path}")
    system, user = str(data["system"]), str(data["user"])
    digest = hashlib.sha1((system + user).encode("utf-8")).hexdigest()[:8]
    return system, user, digest


def load_windows(path: str | Path, root: Path) -> tuple[list[dict], str]:
    """Đọc + kiểm tra file cửa sổ. Trả về (windows, sha1_of_file[:8]). Thiếu/khỏng -> ValueError."""
    p = Path(path)
    raw_bytes = p.read_bytes()
    data = json.loads(raw_bytes.decode("utf-8-sig"))
    wins = data.get("windows") if isinstance(data, dict) else data
    if not wins:
        raise ValueError(f"Không có windows trong {path}")
    for w in wins:
        for key in ("window_id", "video_id", "label", "gt_event", "frame_paths"):
            if key not in w:
                raise ValueError(f"Cửa sổ thiếu khoá {key!r}: {w.get('window_id')}")
        if w["label"] not in ("anomaly", "normal"):
            raise ValueError(f"label không hợp lệ {w['label']!r} ở {w['window_id']}")
        if len(w["frame_paths"]) != 4:
            raise ValueError(f"{w['window_id']} có {len(w['frame_paths'])} frame (spec: 4)")
        missing = [f for f in w["frame_paths"] if not (root / f).exists()]
        if missing:
            raise ValueError(f"{w['window_id']} thiếu frame: {missing}")
    ids = [w["window_id"] for w in wins]
    if len(ids) != len(set(ids)):
        raise ValueError("window_id trùng lặp")
    return wins, hashlib.sha1(raw_bytes).hexdigest()[:8]


# ---------------------------------------------------------------- model load + inference
def load_model(model_id: str, class_name: str, attn: str, vram_cap_gb: float | None):
    """Nạp VLM 4-bit (nf4, fp16 compute). TRƯỚC khi nạp: --vram-cap-gb -> set_per_process_memory_fraction.

    Trả về (model, processor, load_seconds).
    """
    import torch
    import transformers
    from transformers import BitsAndBytesConfig

    if vram_cap_gb is not None:
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        frac = min(vram_cap_gb / total_gb, 1.0)
        torch.cuda.set_per_process_memory_fraction(frac, device=0)
        print(f"  [cap] set_per_process_memory_fraction({frac:.3f}) = {vram_cap_gb} GB / "
              f"{total_gb:.1f} GB", flush=True)

    cls = getattr(transformers, class_name)
    processor = transformers.AutoProcessor.from_pretrained(model_id)
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model = cls.from_pretrained(
        model_id,
        quantization_config=quant,
        device_map="cuda",
        torch_dtype=torch.float16,
        attn_implementation=attn,
    )
    model.eval()
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0
    return model, processor, load_s


def build_inputs(processor, window: dict, variant: str, system: str, user: str,
                 device: str | None = None):
    """Cửa sổ + variant -> (inputs, token_info). device=None -> giữ trên CPU (đếm token không đụng GPU)."""
    import numpy as np
    from PIL import Image

    frames = [Image.open(p).convert("RGB") for p in window["frame_paths"]]
    is_video = variant.startswith("video")
    if is_video:
        content: list[dict] = [{"type": "video"}] + [{"type": "text", "text": user}]
    else:
        content = [{"type": "image"} for _ in frames] + [{"type": "text", "text": user}]
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": content},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Truyền thẳng vào processor(__call__): transformers 5.17 vẫn nhận (có warning khuyên
    # processor_kwargs; thử processor_kwargs bị BỎ qua với SmolVLM do_image_splitting — đã kiểm
    # bằng dry-run: nosplit 332 vs split 4328 image token chỉ đúng khi truyền thẳng).
    pk: dict[str, Any] = {}
    if variant in ("images_cap448", "video_cap448"):
        pk["max_pixels"] = MAX_PIXELS_448
    if variant == "images_nosplit":
        pk["do_image_splitting"] = False
    if is_video:
        inputs = processor(text=[text], videos=[np.stack(frames)], return_tensors="pt", **pk)
    else:
        inputs = processor(text=[text], images=frames, return_tensors="pt", **pk)

    raw_ids = processor.tokenizer.encode(text, add_special_tokens=False)
    n_ph = text.count("<image>") + text.count("<video>")
    total = int(inputs["input_ids"].shape[1])
    token_info = {
        "prompt_tokens": len(raw_ids) - n_ph,     # chữ, không tính placeholder
        "image_tokens": total - len(raw_ids),     # phần mở rộng của placeholder (mỗi frame >= 1)
        "total_tokens": total,
        "n_placeholders": n_ph,
    }
    if token_info["image_tokens"] < 0:
        token_info["negative_delta"] = True      # template không nhét placeholder dạng 1 token; giữ số thật
    return (inputs.to(device) if device else inputs), token_info


def timed_generate(model, processor, inputs, max_new_tokens: int) -> dict:
    """Một lần sinh có phân loại thời gian (§3.6): prefill (max_new_tokens=1) + full; decode = total - prefill."""
    import torch

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        model.generate(**inputs, max_new_tokens=1, do_sample=False)
    torch.cuda.synchronize()
    prefill_s = time.perf_counter() - t0

    torch.cuda.synchronize()
    t1 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    torch.cuda.synchronize()
    total_s = time.perf_counter() - t1

    gen = out[0][inputs["input_ids"].shape[1]:]
    raw = processor.batch_decode([gen], skip_special_tokens=True)[0].strip()
    out_tokens = int(gen.shape[0])
    decode_s = max(total_s - prefill_s, 0.0)
    return {
        "raw": raw,
        "prefill_s": round(prefill_s, 3),
        "total_s": round(total_s, 3),
        "decode_s": round(decode_s, 3),
        "out_tokens": out_tokens,
        "s_per_token": round(decode_s / out_tokens, 4) if out_tokens else None,
    }


def run_stats(times: list[float]) -> dict:
    """Thống kê trên run ĐÃ loại warmup (F2). p90 = nearest-rank: sorted[ceil(0.9n) - 1]."""
    if not times:
        return {}
    import math

    ordered = sorted(times)
    p90_idx = min(len(ordered) - 1, max(0, math.ceil(0.9 * len(ordered)) - 1))
    return {
        "n_measured": len(times),
        "median_s": round(statistics.median(times), 2),
        "mean_s": round(sum(times) / len(times), 2),
        "p90_s": round(ordered[p90_idx], 2),
        "min_s": round(ordered[0], 2),
        "max_s": round(ordered[-1], 2),
        "runs_s": [round(t, 2) for t in times],
    }


# ---------------------------------------------------------------- quality (F5)
def quality_pass(model, processor, windows, variant, system, user, max_new_tokens: int,
                 jsonl_path: str | None, model_id: str) -> dict:
    """Sinh 1 lần trên MỖI cửa sổ, parse bằng parse_verifier_json (không re-implement)."""
    outputs: list[dict] = []
    parsed_rows: list[tuple[dict, Any]] = []
    for w in windows:
        row: dict = {"window_id": w["window_id"], "label": w["label"], "gt_event": w["gt_event"]}
        try:
            inputs, _info = build_inputs(processor, w, variant, system, user, device="cuda")
            gen = timed_generate(model, processor, inputs, max_new_tokens)
            row["raw"] = gen["raw"]
            try:
                parsed = parse_verifier_json(gen["raw"])
                row["parsed"] = {
                    "is_anomaly": parsed.is_anomaly,
                    "event_type": parsed.event_type.value,
                    "caption": parsed.caption,
                    "confidence": parsed.confidence,
                }
                parsed_rows.append((w, parsed))
            except ValueError as e:
                row["error"] = f"parse: {e}"
        except Exception as e:
            row["error"] = f"generate: {type(e).__name__}: {e}"
        outputs.append(row)
        if jsonl_path:
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": time.strftime("%H:%M:%S"), "model": model_id,
                                    "variant": variant, **row}, ensure_ascii=False) + "\n")

    n = len(outputs)
    n_ok_first = sum(1 for r in outputs if "parsed" in r)
    n_consist = sum(1 for _, p in parsed_rows
                    if p.is_anomaly == (p.event_type != EventType.NORMAL))
    anom = [(w, p) for w, p in parsed_rows if w["label"] == "anomaly"]
    norm = [(w, p) for w, p in parsed_rows if w["label"] == "normal"]
    tp = sum(1 for _, p in anom if p.is_anomaly)
    fp = sum(1 for _, p in norm if p.is_anomaly)
    type_hits = sum(1 for w, p in anom if p.event_type.value == w["gt_event"])
    conf_ok = [p.confidence for w, p in parsed_rows
               if p.is_anomaly == (w["label"] == "anomaly")]
    conf_bad = [p.confidence for w, p in parsed_rows
                if p.is_anomaly != (w["label"] == "anomaly")]
    return {
        "n_windows": n,
        "parse_ok_first_try": f"{n_ok_first}/{n}",
        "consistency": f"{n_consist}/{n_ok_first or n}",
        "tpr": f"{tp}/{len(anom)}",
        "fpr": f"{fp}/{len(norm)}",
        "type_match": f"{type_hits}/{len(anom)}",
        "mean_conf_correct": round(sum(conf_ok) / len(conf_ok), 2) if conf_ok else None,
        "mean_conf_wrong": round(sum(conf_bad) / len(conf_bad), 2) if conf_bad else None,
        "outputs": outputs,
    }


# ---------------------------------------------------------------- variant / model
def bench_variant(model, processor, windows, variant: str, args, system: str, user: str,
                  jsonl_path: str | None, model_id: str) -> dict:
    """Timing (warmup tách) + token info + VRAM (torch + device) + worst-case probe + quality (variant chính)."""
    import torch

    row: dict = {"variant": variant, "ok": False}
    n_windows = len(windows)

    # Đếm token trên CPU cho đủ cửa sổ (dùng tìm worst-case + bảng token; không giữ trên GPU).
    token_counts: list[tuple[dict, dict]] = []
    for w in windows:
        inputs_cpu, info = build_inputs(processor, w, variant, system, user)
        token_counts.append((w, info))
        del inputs_cpu
    worst_w, worst_info = max(token_counts, key=lambda t: t[1]["total_tokens"])
    row["tokens"] = {
        "worst_window": worst_w["window_id"],
        "worst_total_tokens": worst_info["total_tokens"],
        "worst_prompt_tokens": worst_info["prompt_tokens"],
        "worst_image_tokens": worst_info["image_tokens"],
    }

    if n_windows < 8 and not args.skip_quality:
        print(f"  [cảnh báo] chỉ {n_windows} cửa sổ (< 8 distinct cho timing)", flush=True)

    times: list[float] = []
    decode_times: list[float] = []
    prefill_times: list[float] = []
    preprocess_times: list[float] = []
    out_tokens: list[int] = []
    sample_raw = ""
    sampler = DevicePeakSampler()
    try:
        with sampler:
            total_iters = args.warmup + args.runs
            for i in range(total_iters):
                w = windows[i % n_windows]
                t_pre0 = time.perf_counter()
                inputs, _info = build_inputs(processor, w, variant, system, user, device="cuda")
                preprocess_s = time.perf_counter() - t_pre0
                gen = timed_generate(model, processor, inputs, args.max_new_tokens)
                is_warmup = i < args.warmup
                tag = "warmup" if is_warmup else f"run {i - args.warmup + 1}/{args.runs}"
                print(f"  {variant} {tag}: total {gen['total_s']}s "
                      f"(prefill {gen['prefill_s']}s, decode {gen['decode_s']}s, "
                      f"{gen['out_tokens']} tok)", flush=True)
                if not is_warmup:
                    times.append(gen["total_s"])
                    decode_times.append(gen["decode_s"])
                    prefill_times.append(gen["prefill_s"])
                    preprocess_times.append(preprocess_s)
                    out_tokens.append(gen["out_tokens"])
                    if not sample_raw:
                        sample_raw = gen["raw"]
                del inputs
                gc.collect()

            # Worst-case probe: 1 cửa sổ nhiều token nhất, đo VRAM riêng (F8: OOM phải tìm thấy ở đây).
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            probe = DevicePeakSampler()
            with probe:
                inputs, _ = build_inputs(processor, worst_w, variant, system, user, device="cuda")
                gen = timed_generate(model, processor, inputs, args.max_new_tokens)
            row["worst_case_probe"] = {
                "window_id": worst_w["window_id"],
                "total_tokens": worst_info["total_tokens"],
                "device_peak_gb": round(probe.peak_mb / 1024, 2) if probe.peak_mb else None,
                "torch_alloc_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
                "torch_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 2),
            }
            del inputs
            gc.collect()
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
        print(f"  VARIANT THẤT BẠI ({variant}): {row['error']}", flush=True)
        return row

    row["ok"] = True
    row["timing"] = run_stats(times)
    row["timing"]["median_preprocess_s"] = round(statistics.median(preprocess_times), 3) if preprocess_times else None
    row["timing"]["median_prefill_s"] = round(statistics.median(prefill_times), 3) if prefill_times else None
    row["timing"]["median_decode_s"] = round(statistics.median(decode_times), 3) if decode_times else None
    row["timing"]["median_out_tokens"] = int(statistics.median(out_tokens)) if out_tokens else None
    row["vram"] = {
        "device_peak_gb": round(sampler.peak_mb / 1024, 2) if sampler.peak_mb else None,
        "device_peak_note": "nvidia-smi memory.used (gồm CUDA context) — số quyết định 'fits in 6 GB'"
                            if sampler.available else "nvidia-smi không đọc được trên máy này",
        "sample_output": sample_raw[:400],
    }
    return row


def bench_model(model_id: str, spec: dict, windows: list[dict], args,
                system: str, user: str, prompt_hash: str) -> dict:
    import torch

    variants = args.variant or spec["variants"]
    bad = [v for v in variants if v not in spec["variants"]]
    if bad:
        raise SystemExit(f"{model_id}: variant không tồn tại {bad}; hợp lệ: {spec['variants']}")

    result: dict = {"model_id": model_id, "class": spec["class"], "ok": False, "variants": []}
    idle = device_idle_mb()
    result["idle_device_mb"] = idle
    clocks_before = _nvidia_smi("clocks.sm")
    try:
        model, processor, load_s = load_model(model_id, spec["class"], args.attn, args.vram_cap_gb)
    except Exception as e:  # OOM khi nạp = số đo hợp lệ, không crash cả bench
        result["error"] = f"load failed: {type(e).__name__}: {e}"
        print(f"  LOAD THẤT BẠI: {result['error']}", flush=True)
        return result

    result["ok"] = True
    result["load_seconds"] = round(load_s, 1)
    result["attn_implementation"] = getattr(model.config, "_attn_implementation", None)

    # Chọn subset cửa sổ cho vòng cap (§2C.4): worst-case + 2 cửa sổ đầu, không lặp.
    run_windows = windows
    if args.vram_cap_gb is not None:
        # Chưa có token counts variant-agnostic: approximate bằng windows[0:2] + worst theo pixel (cùng 320x240 -> windows[0])
        run_windows = windows[:3]

    for variant in variants:
        print(f"\n--- {model_id} :: {variant} ---", flush=True)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            vrow = bench_variant(model, processor, run_windows, variant, args, system, user,
                                 args.jsonl, model_id)
        except Exception as e:
            vrow = {"variant": variant, "ok": False, "error": f"{type(e).__name__}: {e}"}
        vrow["torch_alloc_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 2)
        vrow["torch_reserved_peak_gb"] = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
        result["variants"].append(vrow)

        if variant == variants[0] and not args.skip_quality:
            print(f"--- quality pass (16 cửa sổ, variant chính {variant}) ---", flush=True)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                q = quality_pass(model, processor, windows, variant, system, user,
                                 args.max_new_tokens, args.jsonl, model_id)
                result["quality"] = {"variant": variant, **q}
                print(f"  parse first-try {q['parse_ok_first_try']}, consistency {q['consistency']}, "
                      f"TPR {q['tpr']}, FPR {q['fpr']}, type_match {q['type_match']}", flush=True)
            except Exception as e:
                result["quality"] = {"variant": variant, "error": f"{type(e).__name__}: {e}"}
                print(f"  QUALITY THẤT BẠI: {e}", flush=True)

    result["clocks_sm_before"] = clocks_before
    result["clocks_sm_after"] = _nvidia_smi("clocks.sm")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------- dry-run (CPU, không GPU)
def dry_run(args, system: str, user: str, prompt_hash: str, windows_hash: str) -> int:
    import transformers
    from PIL import Image

    print("=== DRY RUN (CPU, không sinh inference) ===")
    print(f"prompt hash: {prompt_hash} | windows hash: {windows_hash}")
    print(f"prompt system: {system!r}")
    print(f"prompt user (200 đầu): {user[:200]!r}")

    for w in args._windows:
        for f in w["frame_paths"]:
            img = Image.open(Path(args.root) / f)
            img.verify()
    print(f"windows OK: {len(args._windows)} cửa sổ, 4 frame mỗi cửa sổ, ảnh decode được")

    model_ids = args.model or list(CANDIDATES)
    for mid in model_ids:
        spec = CANDIDATES[mid]
        processor = transformers.AutoProcessor.from_pretrained(mid)
        for variant in (args.variant or spec["variants"]):
            worst = None
            for w in args._windows:
                _, info = build_inputs(processor, w, variant, system, user)
                if worst is None or info["total_tokens"] > worst[1]["total_tokens"]:
                    worst = (w, info)
            w, info = worst
            print(f"  {mid:45s} {variant:15s} worst={w['window_id']} "
                  f"prompt={info['prompt_tokens']} image={info['image_tokens']} "
                  f"total={info['total_tokens']}"
                  + ("  [negative_delta: placeholder không nhét 1 token]" if info.get("negative_delta") else ""))
    return 0


# ---------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="A1redo: benchmark VLM 4-bit trên T4 với cửa sổ thật")
    ap.add_argument("--model", action="append", default=None,
                    help="HF id (lặp lại được); mặc định 3 ứng viên; MỖI model một process trên Colab")
    ap.add_argument("--variant", action="append", default=None,
                    help="lọc variant (lặp lại được); mặc định: mọi variant của model")
    ap.add_argument("--windows", default="notes/vlm-bench-windows.json")
    ap.add_argument("--root", default=".", help="thư mục gốc resolve frame_paths")
    ap.add_argument("--prompts", default="configs/prompts.yaml")
    ap.add_argument("--warmup", type=int, default=2, help="số run warmup, KHÔNG tính (F2)")
    ap.add_argument("--runs", type=int, default=8, help="số run đo (redo plan: 8)")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--vram-cap-gb", type=float, default=None,
                    help="mô phỏng card (set_per_process_memory_fraction) — vòng 6 GB, chỉ 3 cửa sổ")
    ap.add_argument("--skip-quality", action="store_true", help="bỏ quality pass (vòng cap dùng)")
    ap.add_argument("--attn", choices=["sdpa", "eager"], default="sdpa")
    ap.add_argument("--out", default=None, help="ghi JSON kết quả (viết lại sau MỖI model)")
    ap.add_argument("--jsonl", default=None, help="append từng cửa sổ quality (chống mất khi Colab rớt)")
    ap.add_argument("--dry-run", action="store_true", help="validate windows + đếm token, KHÔNG cần GPU")
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    system, user, prompt_hash = load_verifier_prompt(root / args.prompts)
    windows, windows_hash = load_windows(root / args.windows, root)
    args._windows = windows
    n_anom = sum(1 for w in windows if w["label"] == "anomaly")
    print(f"windows: {len(windows)} ({n_anom} anomaly / {len(windows) - n_anom} normal), "
          f"hash {windows_hash}; prompt hash {prompt_hash}", flush=True)

    if args.dry_run:
        return dry_run(args, system, user, prompt_hash, windows_hash)

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("Cần GPU thật (torch.cuda.is_available() = False); --dry-run thì không cần GPU")

    import accelerate
    import bitsandbytes
    import transformers

    props = torch.cuda.get_device_properties(0)
    clocks = _nvidia_smi("clocks.sm,clocks.max.sm,temperature.gpu")
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
    except Exception:
        commit = None
    header = {
        "gpu": props.name,
        "gpu_total_vram_gb": round(props.total_memory / 1024**3, 2),
        "clocks_sm_max_temp": clocks,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "bitsandbytes": getattr(bitsandbytes, "__version__", None),
        "accelerate": getattr(accelerate, "__version__", None),
        "git_commit": commit,
        "attn_implementation": args.attn,
        "quant": "nf4, fp16 compute (BitsAndBytesConfig)",
        "prompt_hash": prompt_hash,
        "windows_hash": windows_hash,
        "warmup": args.warmup,
        "runs": args.runs,
        "max_new_tokens": args.max_new_tokens,
        "vram_cap_gb": args.vram_cap_gb,
        "skip_quality": args.skip_quality,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    print(f"GPU: {header['gpu']} ({header['gpu_total_vram_gb']} GB), torch {header['torch']}, "
          f"transformers {header['transformers']}", flush=True)

    model_ids = args.model or list(CANDIDATES)
    if len(model_ids) > 1:
        print("CẢNH BÁO: đo nhiều model trong 1 process (bitsandbytes có thể leak). "
              "Colab: chạy MỖI model một lệnh python.", flush=True)

    results: list[dict] = []
    for mid in model_ids:
        if mid not in CANDIDATES:
            raise SystemExit(f"Không có class mặc định cho {mid!r}; thêm vào CANDIDATES")
        res = bench_model(mid, CANDIDATES[mid], windows, args, system, user, prompt_hash)
        results.append(res)
        if args.out:  # ghi ngay sau mỗi model (F: Colab rớt không mất model đã đo)
            payload = {"header": header, "results": results}
            tmp = Path(str(args.out) + ".tmp")
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, args.out)

    payload = {"header": header, "results": results}
    print("\n=== TỔNG KẾT ===")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
