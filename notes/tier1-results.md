# Tier 1 FROZEN — 2026-10-30 (TH6, done on D2, plan said D4 EOD)

Freeze = no more edits to `vad/pipeline.py` scoring semantics, `configs/prompts.yaml` detection prompts, or the A4c threshold.
Changing any of them invalidates `results/windows/` and `results/threshold_A4c.json` (`export_windows.py` refuses a threshold file whose config differs).

## What is frozen

| Item | Value |
|---|---|
| Config | `ABLATIONS["A4c"]`: SigLIP 2 base 384, ensemble prompts (4+4), normalised with fixed `calib_normal` median/MAD (0.6131 / 0.1700), top-k pool 5 s k=3, EMA 0.3 |
| Threshold | tau = **1.1686**, (1 - 5%) quantile on 60 `calib_normal` videos (`results/threshold_A4c.json`) |
| Window rule | score >= tau, merge gap 2 s, min length 1 s, window score = max in window |
| Tier-1 quality | micro AUC **73.00** [64.45, 81.15] over videos (`results/ablation_A4c.json`) |
| Windows | **307** in 114 of 290 test videos (`results/windows/<video_id>.json` + `results/windows/_index.json`) |
| Reproduce | `python scripts/calibrate_threshold.py ... --ablation A4c` then `python scripts/export_windows.py --manifest manifests/ucf_test.csv --ablation A4c --threshold-file results/threshold_A4c.json` |

## Handoff to tier 2 (Phước A5 / Huy A3)

- Read `results/windows/_index.json`: flat list, fixed order, `video_id` + `window_index` is the key (resume by key). Per-video files are standard `RunResult.load_windows`.
- Timestamps are seconds in the ORIGINAL video; metadata carries `fps` and `n_frames` (UCF is mostly 30 fps, some 25).
- Load estimate: 307 windows x 4.45 s = **about 23 min on a Colab T4**. No per-video cap needed.
- No test labels are in these files; windows were produced without looking at GT.

## Two things tier 2 must know (found while freezing, NOT fixed)

1. **Long windows.** Median 10 s, but p90 = 69 s, 34 windows > 60 s, 11 > 120 s, max 490 s (Arrest030, Normal_Videos_924). 4 evenly spaced frames over 490 s is a weak input.
   Suggested tier-2 rule (A3 owner): if duration > 30 s, sample the 4 frames around the window's peak-score second instead of evenly, or split into <= 30 s chunks and OR the verdicts. Decide before A5, record it in the report.
2. **Two videos dominate the false alarms.** `Normal_Videos_935_x264` (55 windows) and `Normal_Videos_924_x264` (30 windows) are 60-minute normal videos: 85 of the 124 false-alarm windows, from 2.0 of the 6.01 normal hours.
   FA/hour is 20.64 overall, but **9.7/hour on the other 148 normal videos** (39 windows / 4.01 h). Report both numbers; do not drop the two videos, do not tune tau on them.

## Honest limits

- tau is tuned for 5% FPR on train-normal; test normals show 10.57% (distribution shift, see `notes/a0-gate.md`).
- Recall is 51.9% (81/156 incidents). Tier 2 can only remove windows, so it cannot raise this number.
- A5/A5c (normal memory) hurt AUC and are not part of the frozen system; they stay in the ablation table as negative results.
