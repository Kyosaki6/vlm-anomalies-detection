# A0 gate (TH4) — 2026-10-30

**Verdict: FAIL by the pre-set band, but the target is inside the CI. 1.4 points below the band, not a bug-sized gap.**

| | value |
|---|---|
| Target (LAVAD Table 1, ZS-CLIP) | 53.16 |
| Accept band (set before running) | 50.2 to 56.2 |
| **A0 micro frame-level AUC** | **48.77** |
| Video-level bootstrap CI95 (1000 resamples, seed 42) | [38.30, 59.40] |
| Macro (mean of per-video AUC), for reference | 49.57 |
| Data | 290 test videos (140 anomalous + 150 normal), 1,111,808 frames |
| Config | CLIP ViT-B/32, `detection.baseline` prompts, raw score, no normalisation, 1 fps cache, `logit_scale` 100 |
| Run time | 30 s scoring + bootstrap (CPU, cache only) |
| Command | `python scripts/run_benchmark.py --manifest manifests/ucf_test.csv --ablation A0` |
| Commit | `a7dc5dc` + uncommitted `scripts/run_benchmark.py` |

The band is NOT widened after the fact. The decision (accept as reproduced, or keep digging) belongs to the team.

## Debug table (day2 §1), cheapest first

| # | Suspect | Result |
|---|---|---|
| 1 | Score direction | Clean. AUC 48.8, not a clean inverse (an inverted scorer would give ~51). Unit test: inverted prompts give 0.0, perfect give 1.0. |
| 2 | GT alignment | Clean. Shifting labels by +1, -1, +16, -16, +30 frames gives 48.56 to 48.91. The AUC barely moves, so `expand_to_frames` and the inclusive 0-indexed convention are not the problem. |
| 3 | Sampling phase (1 fps vs LAVAD stride 16) | Clean on a 60-video subset (`--limit 60`, evenly spaced, 29 anomalous + 31 normal, CPU). Same videos: 1 fps = 50.36 [37.05, 62.37], stride 16 = 50.21 [37.06, 62.30]. Difference 0.15 points, so sampling density is NOT the cause of the gap to 53.16. Script: `scripts/a0_stride_check.py` (re-decodes, cache in `cache/stride16/`, result `results/a0_stride16_limit60.json`). Full 290 videos at stride 16 (5826 s CPU): **48.43 [37.90, 59.09]**, macro 47.85, vs 48.77 at 1 fps. Same conclusion. |
| 4 | Pooling | Clean. Micro (48.77) is what is reported; macro (49.57) printed beside it. LAVAD protocol is micro. |
| 5 | `logit_scale` | Clean. Scales 1, 10, 100 give an identical 48.77. The sigmoid is monotone and the clip creates no ties here. |
| 6 | Prompt text | Clean. Cached prompts are exactly "a standard scene" / "a scene with suspicious or potentially criminal activities", same as the paper. |

Other things checked: cache embeddings are fp16 with norm ~1.0001 and `load_embeddings` renormalises; prompt vectors are unit norm.

## Likely explanation (not proven)

- ~~1 fps vs every-16th-frame sampling~~: ruled out on all 290 videos (48.43 at stride 16 vs 48.77 at 1 fps, check #3).
- The paper gives no CI. Ours is 21 points wide, so 53.16 vs 48.77 is within noise at the video level.
- The paper states preprocessing only as "CLIP ViT-B/32"; the HF CLIP processor (resize 224 + center crop) may differ from the authors' pipeline on 4:3 frames.

## Open decision for the standup

1. Accept as reproduced (target inside the CI, plumbing proven by 1 and 2), note the deviation in the report. Or
2. ~~Spend a Colab run on check #3~~ DONE locally on CPU for all 290 videos: sampling is not the cause (48.43). Remaining unexplained gap (~4.7 points to 53.16) is inside the CI; only HF CLIP preprocessing vs the authors' pipeline is untested. Recommendation: option 1, accept as reproduced within noise and report the deviation honestly.

## First look at A1 to A3 (same loop, cache only)

| ID | micro AUC | CI95 (videos) | macro |
|---|---|---|---|
| A0 | 48.77 | [38.30, 59.40] | 49.57 |
| A1 | 65.87 | [57.93, 73.44] | 56.21 |
| A2 | 72.39 | [64.14, 80.39] | 64.12 |
| A3 | 58.99 | [54.89, 63.24] | 64.12 |

**A3 drops 13 points in micro AUC while macro is unchanged (64.12).** Per-video robust z-score is monotone inside each
video, so per-video AUC cannot change; but micro AUC pools frames across videos, and per-video normalisation erases the
cross-video offset (an anomalous video scores higher overall than a normal one). That offset is part of what micro AUC
rewards. This is the same design question as day2 §1 "FA/hour": per-video normalisation hurts the headline metric and
also makes FA/hour measure the normalisation. Decide (a) keep and report honestly, or (b) normalise with `calib_normal`
statistics, before TH5 on D3 morning. A4/A5 inherit this.

## A3 to A5 and the normalisation decision (2026-10-30, TH5)

Decision: **(b)**, normalise with fixed `calib_normal` median/MAD (`norm_ref="calib"`, ablations `A3c/A4c/A5c`).
Per-video variants A3/A4/A5 stay as controls.

| ID | norm | micro AUC | CI95 (videos) | macro |
|---|---|---|---|---|
| A3 | per video | 58.99 | [54.89, 63.24] | 64.12 |
| A4 | per video | 58.41 | [53.51, 63.42] | 63.35 |
| A5 | per video | 57.08 | [52.14, 62.28] | 58.87 |
| A3c | calib | 72.39 | [64.14, 80.39] | 64.12 |
| A4c | calib | 73.00 | [64.45, 81.15] | 63.35 |
| A5c | calib | 64.95 | [55.19, 74.61] | 57.87 |

- A3c == A2 exactly: a global affine map cannot change micro AUC. This is the sanity check that the new path is wired right.
- A4c vs A3c: +0.6, inside the CI. Temporal pooling/EMA does not clearly help micro AUC.
- A5c: normal memory hurts (-8.1 vs A4c). Report as a negative result; do not tune it on test.
- Memory stats for calib are leave-one-video-out (a video inside its own bank scores 0 and collapses the MAD).
- Best tier-1 candidate for threshold calibration and tier-2 windows: **A4c**.
