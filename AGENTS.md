# PROJECT KNOWLEDGE BASE

**Updated:** 2026-10-28 · **Branch:** main · Plan: `notes/2026-10-28-plan-zero-shot-video-vad.md`

## OVERVIEW
Two codebases live in one repo:

1. **`vad/` (NEW, active)** — Zero-Shot Video Anomaly & Surveillance Alerting. Video -> SigLIP 2 frame embeddings
   (cached) -> generic-prompt + normal-memory scoring -> temporal smoothing -> candidate windows -> generative VLM
   verification (tier 2, mandatory: the topic was registered as VLM) -> alert policy graph -> Streamlit dashboard.
   Benchmarks: UCF-Crime test + ShanghaiTech test, frame-level AUC with bootstrap CI over **videos**.
2. **Legacy 15-image pipeline** (`main.py`, `search.py`, `graph.py`, `decision.py`, `eval.py`, `events.json`, `data/`)
   — kept as a smoke fixture and for `decision.py` metrics. It is NOT evidence for the new system.

Deadline: submit before **2026-11-04** (hard cutoff: end of 2026-11-03 23:59; feature freeze 2026-11-02 evening, 11-03 = report + packaging).
Team = 4 (Phước, Huy, Nam, Thịnh). Dashboard owner: Thịnh (whole Streamlit dashboard). Scope cuts: `notes/2026-10-29-task-assignment.md` §6.
Report + defense in Vietnamese, dashboard in English.

## STRUCTURE
```
vad/            types.py, interfaces.py (Protocols), fakes.py  — contracts shared by all tracks
  datasets/     ucf_crime.py (GT parser), sharding.py (SHA-256 shard split)
  io/           video.py (sequential-grab sampler)
  embed/        cache.py (fp16 .npy, atomic), siglip.py (real, model-generic: SigLIP 2 + CLIP)
  score/        text.py (prompt + memory scores, numpy)
  temporal/     ops.py (normalize/EMA/median/top-k/candidate windows/frame alignment)
  verify/       vlm.py (JSON parser real; VLMVerifier STUB)
  graph/        policy.py (YAML -> NetworkX alert policy) ; configs/policy.yaml
  eval/         metrics.py (frame-level AUC, video-level bootstrap, FA/hour, latency)
scripts/        precompute_embeddings.py (--shard i/N ; repeat --model for one-decode multi-model; --dry-run works)
app/            streamlit_app.py (replay mode, PLACEHOLDER data)
configs/        policy.yaml, prompts.yaml
tests/          legacy tests + tests/vad_suite/ (246 total)
notes/          experiment logs + the plan
```

## WHERE TO LOOK
| Task | Location |
|---|---|
| Per-module status, who owns what, what is a stub | `vad/README.md` |
| Full plan, timeline, risks, decisions log | `notes/2026-10-28-plan-zero-shot-video-vad.md` |
| Detection / typing / verifier prompts | `configs/prompts.yaml` |
| Alert rules (zones, time bands, escalation) | `configs/policy.yaml` |
| UCF-Crime annotation parsing | `vad/datasets/ucf_crime.py` |
| ROC-AUC with ties, bootstrap, threshold math | `decision.py` (reused by `vad/eval/metrics.py`) |

## STILL STUBS (raise `NotImplementedError` on purpose, never return fake output)
`VLMVerifier` (vad/verify/vlm.py) only. `SigLIPEmbedder` and the real run of
`scripts/precompute_embeddings.py` now work (branch `exp/siglip-embedder`). Not yet built: ShanghaiTech loader,
`run_benchmark.py`, dashboard reading real `results/`.

## CONVENTIONS
- Python 3.10 compatible (CI runs 3.10 + 3.11 on Ubuntu + Windows). No `StrEnum`; use `from __future__ import annotations`.
- Read JSON/text with `utf-8-sig` (Windows BOM tolerance). Event ids are strings.
- Code comments/docstrings are bilingual: Vietnamese prose, technical terms verbatim.
- Tests never download models or need a GPU; anything GPU-bound has a `Fake*` in `vad/fakes.py`.
- Test dir is `tests/vad_suite/` — NOT `tests/vad/` (that name shadows the `vad` package under pytest).
- Embedding cache keyed by model + video_id + sample_fps; every shard must use the same model, fps, preprocessing.

## ANTI-PATTERNS (THIS PROJECT)
- DO NOT put class names ("fighting", "road accident") in detection prompts: label leak. Typing stage only.
- DO NOT calibrate thresholds on the test split; use `split=calib_normal`.
- DO NOT report CIs resampled over frames; resample over videos (frames are autocorrelated).
- DO NOT use top-1 gallery similarity as anomaly probability (legacy finding; AUC ~0.64).
- DO NOT measure retrieval with the query inside the gallery (false 100%).
- DO NOT claim zero-shot for ShanghaiTech: label it "training-free + normal reference".
- DO NOT pretend a stub works. A missing backend must raise, so the dashboard can never show fabricated results.
- DO NOT use bare ignore patterns like `cache/`: they swallow `vad/embed/cache.py`. Anchor to root (`/cache/`).

## COMMANDS
```bash
pip install -r requirements-dev.txt        # tests / CI
pip install -r requirements-app.txt        # + streamlit
python -m pytest -q                        # 246 tests, no GPU, no model download
streamlit run app/streamlit_app.py         # dashboard (placeholder data for now)
python scripts/precompute_embeddings.py --manifest manifests/<f>.csv --shard 2/4 --dry-run

# legacy pipeline
python main.py data/danh_nhau_1.jpg
python eval.py --gallery data --device cpu
```

## HARDWARE (shard planning)
Phước RTX 3050 6 GB (fp16, also VLM) · Huy RTX 3050 4 GB (fp16) · Nam GTX 1060 3 GB (fp32; check `sm_61` in torch) ·
Thịnh no GPU (CPU + Colab T4). Heavy runs = embedding cache + tier-2 batch only; everything downstream is CPU.

## NOTES
- Published zero-shot frame-level AUC on UCF-Crime (LAVAD, CVPR 2024): ZS-CLIP 53.2, LLaVA-1.5 72.8, LAVAD 80.3.
  Naive per-frame prompting is near chance; the contribution is the pipeline (normalisation, temporal, memory, VLM).
- Day-4 validity gate: reproduce ZS-CLIP ~53 AUC before trusting any other number.
- UCF-Crime annotation segments assumed inclusive `[start, end]`, 0-indexed, 30 fps — UNVERIFIED against LAVAD/RTFM.
