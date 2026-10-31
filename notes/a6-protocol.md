# A6 protocol (tier 2 evaluation) — written BEFORE any tier-2 number exists

Task TH7a · decision D-3 · author: Thịnh · date: 2026-10-31
Status when written: `results/verify/` does not exist; A5b has not run; no VLM verdict on the 307 frozen windows has been seen by anyone writing this note.
Consumer: `scripts/eval_tier2.py` (TH7b) implements exactly this note. If the script and this note disagree, the note wins and the script is the bug.

## 0. Rules of the game

1. This note is committed BEFORE A5b finishes. Its commit hash is quoted in the report next to the A6 row.
2. After a tier-2 number exists, only changes that FIX A BUG (script does not implement the note) are allowed. Changing a definition, a metric, a tau or a tie rule after seeing a number is tuning on test and is forbidden. A change of intent goes into the report as a deviation, with the old and new number side by side.
3. Fixed inputs, all frozen (`notes/tier1-results.md`): config `ABLATIONS["A4c"]`, tau = 1.1686, the 307 windows in `results/windows/`, UCF-Crime test (290 videos), GT convention of `notes/ucf-gt-convention.md`.
4. No new tau. Tier 2 is evaluated at the same tau as tier 1. Nothing is calibrated on test.
5. Only `VerifyResult` produced by the real `VLMVerifier` counts. Output of `FakeVerifier` / `run_verify.py --fake` must be self-declared (`metadata`) and `eval_tier2.py` must refuse it for the A6 row (it may be used only for unit tests).

## 1. Window verdict

- Unit: one window = one `results/verify/<video_id>/window_NNNN.json` produced by `run_verify.py`.
- **Verdict = `is_anomaly` of the merged `VerifyResult`**, which already is the OR over chunks (decision D-1: window > 30 s is split into chunks <= 30 s, 4 frames each, `is_anomaly` = OR of chunks).
- Confirmed = `is_anomaly == True`. Rejected = `is_anomaly == False`.
- NOT used for the verdict: `confidence` (it is near constant, `min_confidence` is inert; see `notes/2026-10-31-day3-tasks.md` §6), `event_type` (D-2: the typer decides the type, the VLM type is reported as its own row only), `caption`.
- Every one of the 307 windows must have a verdict. A missing file, a fingerprint / start / end that does not match the frozen window, or a duplicated `window_index` raises. There is no default verdict, no "treat as confirmed", no "treat as rejected". Partial runs are not evaluated: A6 is reported on all 307 windows or not reported as A6 (a subset gets a different row name and states its size).

## 2. Frame score after tier 2

Start from the frozen A4c per-sample score `s[t]` (final output of `score_cached_video`, i.e. after calib-normalisation, top-k pool and EMA, 1 sample per second).

- For every REJECTED window `[start_s, end_s)`, set `s'[t] = 0.0` for every sample `t` whose second falls in the window: `floor(start_s * sample_fps) <= t < ceil(end_s * sample_fps)`, clipped to the array. This is the same span `candidate_windows` produced (`[s/fps, (e+1)/fps)`).
- 0.0 is the `calib_normal` median in normalised units (A4c normalises with the fixed calib median/MAD 0.6131 / 0.1700, so a typical normal sample scores 0). It is below tau = 1.1686, so a demoted window also vanishes from `candidate_windows` at the same tau. That is checked in a test.
- Confirmed windows: `s'[t] = s[t]` (untouched; the VLM never raises a score).
- Samples outside every window: `s'[t] = s[t]` (untouched). Videos with no window are untouched.
- Then expand to frames with the unchanged `expand_to_frames` and score with `fast_frame_level_auc` (micro-pooled over the 290 videos) and `fast_video_bootstrap_auc_ci` (seed 42, 1000 resamples, resampled over VIDEOS, never frames), identical to `results/ablation_A4c.json`.
- Primary A6 quality number = micro AUC of `s'` with its CI. It is compared with A4c's 73.00 [64.45, 81.15].
- Secondary (if TH7b has time): paired difference AUC(A6) - AUC(A4c) with a bootstrap CI over the same video resamples. The unpaired CIs overlap heavily by construction and say nothing about the change.
- Ties: demotion creates many samples with exactly 0.0. `fast_frame_level_auc` handles ties (half credit); no jitter is added.

## 3. Operating metrics (same tau, before vs after)

Use the overlap rule of `operating_point` in `scripts/calibrate_threshold.py`: GT segment `[a, b]` frames -> `g0 = a / fps`, `g1 = (b + 1) / fps`; a window overlaps it iff `w.start_s < g1 and w.end_s > g0`. "After" = the set of CONFIRMED windows only. "Before" = all 307 windows (must reproduce the A4c operating point exactly, which is the identity check).

| Metric | Definition | Expected direction |
|---|---|---|
| FA/hour | windows in normal videos (no GT segment) / total normal hours; `false_alarms_per_hour` | down |
| FA/hour excl. 935 + 924 | same, dropping `Normal_Videos_935_x264` and `Normal_Videos_924_x264` as a second number (not instead of; tier-1 note already reports 20.64 vs 9.7) | down |
| Window precision | windows that overlap >= 1 GT segment / windows. A window in a normal video is always a false positive. Report as count / count | up |
| Incident recall | incidents (GT segments) hit by >= 1 surviving window / incidents (156 incidents, tier 1 hit 81 = 51.9%) | stays or down |
| Detection latency | median and mean of `detection_latency_s(g0, first surviving window start)` over incidents detected AFTER tier 2, with n shown | roughly unchanged |
| Windows rejected | rejected / 307, split into windows in anomalous videos vs normal videos, and overlapping a GT segment vs not | informational |
| Cost | windows per video (307 / 290 = 1.06), VLM chunk-calls, seconds of video shown to the VLM, median / p90 s per window and peak VRAM from the A5b log | informational |

Rules:

- Latency is computed over a changing set of incidents. State n for before and after next to the numbers, and also give the latency restricted to the incidents detected both before and after, so a drop in n is not read as a latency change.
- A rejected window that overlapped a GT segment is a lost true positive. Report that count explicitly: it is the price of the precision gain.
- Precision and FA/hour are window-based, AUC is frame-based. They answer different questions and are never merged into one number.

## 4. What is guaranteed and what is not (stated upfront)

1. **Recall can only stay or fall.** Tier 2 only removes windows and never adds one. If the script ever reports recall after > recall before, it has a bug.
2. **AUC moves only through the frames inside the 307 windows.** Everything else is bit-identical. The 307 windows cover a small part of all frames, so a large AUC change is not expected, and a small or zero change is a legitimate result, not a failure. Direction is not guaranteed: a rejected window that overlaps a GT segment lowers AUC, a rejected window on normal frames raises it.
3. **The claim under test is FA/hour and window precision**, not AUC. The report must say so in the same sentence as the A6 row.
4. If A6 shows no gain (or a loss), it is reported as is. No prompt or rule is adjusted to rescue the number (see §5 for the single allowed prompt tuning).

## 5. Interaction with the A6 hand check (task-sheet A6)

- The task sheet allows the verifier prompt to be tuned ONCE after the 20-window hand check. The 20 windows (A6a, `notes/a6-handcheck-sample.csv`) are drawn from the same test windows, so tuning on them is mild test leakage. It is disclosed, not hidden.
- The A6 row of the ablation table is the run with the prompt used in A5b (before the tuning). It is the headline number.
- If the prompt is tuned and the 307 windows are re-run, that result is a separate row named `A6-tuned`, labelled as tuned after a hand check on 20 test windows, and its before/after precision on the 20 is reported in the A6 short note. The two rows are never merged and the tuned row never replaces the first one.
- The rule of §1-§3 does not change between the two rows.

## 6. Output contract for `results/ablation_A6.json`

Required keys (names may follow the `ablation_A4c.json` style): `ablation` = `"A6"`, `source_ablation` = `"A4c"`, `threshold`, `threshold_source` = `"calib_normal"`, `verifier` (model id), `prompt_hash`, `verify_dir`, `n_windows` = 307, `n_confirmed`, `n_rejected`, `protocol_note` = path + commit hash of this file, `auc_micro`, `ci_lo`, `ci_hi`, `auc_micro_a4c`, `delta_auc` (+ paired CI if computed), `operating_before`, `operating_after` (every metric of §3 with its numerator and denominator), `lost_true_positive_windows`, `cost` (from the A5b log), `commit`, `timestamp`.

## 7. Tests TH7b must pass (identity first)

1. All windows confirmed -> `s' == s` exactly for every video, AUC and operating point equal A4c (`operating_before == operating_after`).
2. All windows rejected -> every sample inside a window is 0.0, samples outside are untouched, no window survives at tau, recall after = 0, FA/hour after = 0.
3. Missing verify file, mismatched window, duplicate `window_index`, `FakeVerifier` output offered as real -> raises.
4. Mixed case on a tiny synthetic video: one confirmed, one rejected overlapping GT, one rejected on a normal video -> hand-computed precision, recall, FA/hour.
5. Recall after <= recall before (property check on random verdicts).
6. Window span for demotion matches `candidate_windows` span (including a window that ends on the last sample).
