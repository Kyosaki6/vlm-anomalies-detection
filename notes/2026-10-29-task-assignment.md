> **Latest commit (checklist verified against): `f385cd8` — Merge pull request #47 from Kyosaki6/docs/thinh-day4-plan** (`origin/main`, 2026-11-01 +07).

# Task assignment: who does what, exactly

Companion to `notes/2026-10-28-plan-zero-shot-video-vad.md` (the WHAT/WHY). This file is the WHO/WHEN.

> **Day 4 Progress Update (Sun 2026-11-01 midday), checked against `origin/main`:**
> - **824 tests passing** (`python -m pytest -q` on `origin/main`), 0 failures, no GPU, no model download.
> - **Tier 2 real evaluation complete (A6):** TH7c merged (PR #46): micro AUC 67.90% [58.39, 77.49] vs A4c 73.00% (delta -5.10%, paired 95% CI [-11.42, -0.25]%), FA/hour 20.64 -> 0.17 (-99.2%), normal FA windows 124 -> 1, precision 28.0% -> 49.6% (+77% relative).
> - **Tier 2 real build complete (C7d):** Rebuilt `results/run_A4c/` with `--verify-dir results/verify` merged (PR #40): 129 confirmed windows -> 101 alerts, typer decides alert type.
> - **VLM typing evaluation complete (C7e):** Merged (PR #41): `results/vlm_typing_eval.json`, `notes/2026-11-01-c7e-vlm-typing.md`.
> - **Merged PRs today:** PR #36 (A6a sample), #40 (C7d real build), #41 (C7e typing eval), #46 (TH7c A6 eval), #47 (Day 4 sync).
> - **Open PRs:** PR #42 (C7f figures), PR #43 (A5c run log, review changes requested), PR #44 (A6b hand check, 6/10 precision), PR #45 (A7 FP analysis).
> - **Next critical milestone:** Thịnh implements E2b (Streamlit dashboard on real data) and fills Chapter 4 in TH8 report.

> **Day 1 Progress Update (Thu 2026-10-29 EOD):**
> - **461 tests passing** (`python -m pytest -q`), 0 failures, no GPU/model download in CI.
> - **Merged PRs today:** PR #8 (real embedder B3), PR #9 (results schema C1), PR #10 (tier-1 pipeline TH2, loader E1, LAVAD protocol TH1, report skeleton TH8), PR #11 (alerting entrypoint C5), PR #12 (event typer eval C3), PR #13 (UCF downloader & manifests B1, GT convention note B2), PR #14 (Colab embeddings done).
> - **Unified Embedding Cache:** All 350 videos (290 test + 60 calib) precomputed for SigLIP 2 + CLIP B/32 on Colab T4 (700 `.npy` files). Shards A4, B5, C4 complete ahead of time; residual fp16-vs-fp32 cosine verified (> 0.99997, `notes/embedder-benchmark.md`).
> - **Event Typer C3 evaluated:** Strict top-1 40.4%, lenient 50.0% vs chance 7.7% on full UCF GT (`results/typing_eval.json`).
> - **Gaps resolved:** G1, G2, G3, G4, G6, G7, G8 all implemented and verified.
> - **Biggest open Day 1 risk:** Task A1 (Phước: VLM benchmark and local RTX 3050 vs Colab T4 decision).
> - **Next critical milestone (Day 2):** Thịnh executes TH3 (`scripts/run_benchmark.py`) and TH4 (A0 baseline reproduction gate ~53 AUC).

**TIMELINE & ROLES:**
- Deadline: submit BEFORE Nov 4 (hard cutoff: end of Nov 3 23:59).
- Team: 4 people (Phước, Huy, Nam, Thịnh).
- Thịnh owns the entire Streamlit dashboard.
- Window is 6 days (Oct 29 .. Nov 3), so scope was cut up front (section 6).

Window: Day 1 = Thu 2026-10-29 ... Day 6 = Tue 2026-11-03. **Feature freeze: end of D5 (Nov 2 evening).** D6 = report, slides,
packaging, submit (cutoff 23:59). Nothing new on D6.
Day numbering: D1 Oct 29 (Thu), D2 Oct 30, D3 Oct 31 (Sat), D4 Nov 1 (Sun), D5 Nov 2 (Mon), D6 Nov 3 (Tue).

ID scheme: tasks are A1..A7, B1..B7, C1..C7, TH1..TH8 (Thịnh), E1..E7 (dashboard/clips). `D<n>` always means DAY n.

## 0. Assumptions to confirm on D1 (change the plan if any is wrong)

1. Thịnh = report writer + tier-1 track + whole Streamlit dashboard. See section 6 for what was cut to make that achievable.
2. GPUs: Phước RTX 3050 6 GB, Huy RTX 3050 4 GB, Nam GTX 1060 3 GB (fp32), Thịnh none. Colab/Kaggle T4 = spare.
3. Time estimates below are guesses, NOT measured. The first measurement (A1, B3) replaces them.
4. Nobody has been asked whether they accept these roles. Swap anything; keep the interfaces.

## 1. Summary table

| Person | Hardware | Main track | Also | Heaviest days |
|---|---|---|---|---|
| Phước | 3050 6 GB | **A. Tier-2 VLM** (mandatory, biggest risk) | embedding shard 1 (40%) | D1-D3, D5 |
| Huy | 3050 4 GB | **B. Data + embedder wiring** (critical path) | embedding shard 2 (35%), own clips (E3/E4) | D1-D2, D3-D5 |
| Nam | 1060 3 GB | **C. Alerts, results schema, event typer** | embedding shard 3 (25%, fp32), build_results (C7) | D1-D5 |
| Thịnh | none | **D. Tier-1 scoring, benchmark, ablations** + **E. Dashboard** | Vietnamese report + slides | D1-D6 (no slack day) |

Thịnh holds the highest task count; section 6 cuts scope (ShanghaiTech stretch, A7 dropped, fewer clips, replay-only UI) so it stays manageable.

## 2. Gaps found in the committed skeleton (each is now an explicit task)

| # | Gap | Task | Status (synced D4 start) |
|---|---|---|---|
| G1 | `cooldown_s` is in `policy.yaml` but no code reads it | C2 | **DONE** (`vad/graph/alerts.py`, commit `9cf6561`) |
| G2 | `co_events` is never computed by any caller, so escalation (accident + explosion) cannot fire end to end | C2 | **DONE** (`vad/graph/alerts.py`, commit `9cf6561`) |
| G3 | `GraphPolicy.responders_in(zone)` ignores `zone` and returns every responder: a placeholder that looks real | C2 (implement or delete) | **DONE** (`vad/graph/policy.py` real coverage queries, commit `9cf6561`) |
| G4 | No results schema and no glue module between tier-1, tier-2, alerts and the dashboard | C1, TH2, C5 | **DONE** (`vad/results.py` in PR #9, `vad/alerting.py` in PR #11, `vad/pipeline.py` in PR #10) |
| G5 | ShanghaiTech ships as frame FOLDERS; the sampler only reads video files | B4 (STRETCH only) | Open (stretch) |
| G6 | Baseline A0 needs OpenAI CLIP ViT-B/32 but only a SigLIP embedder is stubbed | B3 (make it model-generic) | **DONE** (`vad/embed/siglip.py` model-generic, PR #8) |
| G7 | Dashboard runs on fake data only | E1, E2 | **DONE** (`app/data.py` loads `RunResult` JSON, PR #10) |
| G8 | Baseline A0 needs CLIP B/32 embeddings of the SAME 290 videos: a second embedding pass. Decoding 30 fps video is likely the expensive part, not the model, so embed both models from ONE decode pass | B3 | **DONE** (one-decode dual-model pass in `scripts/precompute_embeddings.py`, 700 files cached) |

## 3. Per-person task lists

Format: ID | when | task | ACCEPTANCE (how we know it is done).

### A. Phước: tier-2 VLM (+ GPU)

| ID | When | Task | Acceptance | Status (synced D4 start) |
|---|---|---|---|---|
| A1 | D1 | Measure, do not assume: load 2-3 candidate generative VLMs in 4-bit (candidates to verify, e.g. Qwen2-VL-2B, Qwen2.5-VL-3B, SmolVLM2), record peak VRAM and seconds per window for 4 frames. Write `notes/vlm-selection.md` | One model chosen with measured peak VRAM < 5.5 GB. If none fits, decide Colab T4 for all of A5 right now | **`[x]` DONE (D2, merged PR #15).** `notes/vlm-selection.md`: Colab T4 GO for Qwen2.5-VL-3B (4.45 s/window, 2.93 GB peak, 16/16 parse). Handoff for A2 ready. |
| A2 | D1-D2 | Implement `VLMVerifier.verify` in `vad/verify/vlm.py`: frames + prompt from `configs/prompts.yaml` -> `parse_verifier_json`; one retry on parse failure, then raise (never fabricate) | Unit test with a mocked model runner; smoke run on 3 clips returns real `VerifyResult` | **`[x]` DONE (D2, merged PR #16, `53ee342`).** `VLMVerifier.verify` real: one retry then raise, safe bool parse, device wiring; `tests/vad_suite/test_vlm_verifier.py`. |
| A3 | D2-D3 | `scripts/run_verify.py`: reads `results/windows/<video>.json`, picks k evenly spaced frames per window, writes ONE json per window, resumable | Kill it mid-run, rerun, it skips finished windows | **`[x]` DONE (D3, PR #27 `d5514ad` + A3b PR #30 `7b3a174`).** `scripts/run_verify.py`: manifest-driven, resumable, config in fingerprint, `--continue-on-error`; A3b adds chunking <= 30 s (OR of chunks), `elapsed_s`, `--shard i/N`. 61 tests. |
| A4 | D2 (overnight) | Run embedding shard 1 once B3 lands (`--shard 1/4`, optionally `2/4`) | `cache/embeddings/*.npy` count matches shard size | **`[x]` SUPERSEDED / DONE.** All shards (350 videos × 2 models = 700 files) completed on Colab T4. |
| A5 | D5 | Run tier 2 on candidate windows (a SUBSET with measured cost is acceptable, see section 6). Colab T4 if A1 timing says the 3050 is too slow. Start the moment Thịnh hands over windows (D4 EOD) | Log: windows escalated (%), s/window, peak VRAM. Feeds the A6 ablation row | **`[x]` A5a + A5b DONE; A5c PR #43 OPEN.** A5a notebook merged (PR #29). A5b merged (PR #37, `bc90360`; clamp fix PR #35): `results/verify/` = 307/307 windows, 114 videos, `fake=false`, 129 confirmed / 178 rejected, 480 chunk-calls, median 3.71 s/window, p90 14.41 s, total 51.7 min. A5c = `notes/a5b-log.md` committed in PR #43 (review changes requested). |
| A6 | D5 | Hand-check 20 windows; tune the verifier prompt ONCE; compare yes/no-only vs caption+type | Short note with before/after precision on the 20 | **`[~]` A6a DONE (PR #36), A6b PR #44 OPEN, A6c SKIPPED.** A6a blind sample merged (PR #36). A6b hand check -> `notes/a6-handcheck.md` committed in PR #44 (confirmed precision 6/10). A6c prompt tuning SKIPPED per gate D4-2. |
| A7 | D5-D6 | Optional: false-positive analysis of the VLM, 10 annotated cases for the report. Only if A5/A6 finished | 10 annotated cases, or explicitly dropped | **`[~]` PR #45 OPEN.** `notes/a7-fp-analysis.md` committed: 5 lost TP + 1 surviving FA analyzed. |

Risk owner for: VLM does not fit 6 GB. Decision rule: A1 result on D1 EOD is final; do not discover this on D5.

### B. Huy: data + embedder wiring (critical path) + own clips

| ID | When | Task | Acceptance | Status (synced D4 start) |
|---|---|---|---|---|
| B3 | D1-D2 (**do first, others wait on it**) | Wire the real embedder in `vad/embed/siglip.py`, reusing `search.load_model`, `embed_images`, `embed_texts`. Make it model-generic (HF id) so the SAME class serves SigLIP 2 and OpenAI CLIP B/32 (needed for A0). Implement the real run in `scripts/precompute_embeddings.py` (sampler -> batches -> `save_embeddings`, resumable) | 3 videos embedded on his GPU; fp16 vs fp32 cosine > 0.999; img/s recorded; `--shard` run works; accepts 2 models in one decode pass (SigLIP 2 + CLIP B/32). **Must be merged by D2 midday** or the whole chain slips | **`[x]` DONE.** Implemented by Nam ahead of schedule (PR #8). Dual-model one-decode verified. Real video fp16-vs-fp32 cosine verified on Colab T4 by Thịnh (min cos 0.99997, `notes/embedder-benchmark.md`). |
| B1 | D1 (download runs while B3 is coded) | Check disk first. Download UCF-Crime test split + the temporal annotation file + ~60 train-normal videos (calibration). Build `manifests/ucf_test.csv`, `manifests/ucf_calib_normal.csv` using `probe_video` for `n_frames`/`fps` | 290 test rows (140 anomalous + 150 normal) and ~60 calib rows; no duplicate ids; `precompute --dry-run` works on both | **`[x]` DONE.** PR #13 merged (`0868fec`). 290 test + 60 calib rows; download & build manifest scripts added; dry-run green. |
| B2 | D2 | Verify the label convention against LAVAD / RTFM code: are `[start, end]` inclusive, 0-indexed, 30 fps? Fix `vad/datasets/ucf_crime.py` and its test if the answer differs. Write `notes/ucf-gt-convention.md` citing the source | Note cites the exact source lines; all 140 annotated video names exist in the manifest | **`[x]` DONE AHEAD OF TIME (D1).** PR #13 merged (`0868fec`). `notes/ucf-gt-convention.md` confirms inclusive, 0-indexed, 30 fps; matches LAVAD/RTFM; no parser change needed. |
| B5 | D2 (overnight) | Embedding shard 2 (35%) | Shard complete, cache files present | **`[x]` SUPERSEDED / DONE.** All shards completed via unified Colab T4 run. |
| E3 | D2-D4 | Own Vietnamese street clips: **about 5** clips of 10-30 s, at least 2 normal; any "incident" clips must be STAGED with consent, no real people without permission. Hand-label start/end seconds. `manifests/own_demo.csv` + labels file. Demo only, NOT benchmark | Manifest + labels committed | **`[~]` PARTIAL (D3).** Manifest + labels committed, but all 7 clips are iStock previews (`valid_for_e3=false`), not self-recorded/staged/consented. E3b settled provenance; E3 original unmet. |
| E4 | D5 | Run the own clips through the pipeline on Huy's GPU | Results JSON per clip | **`[ ]` PENDING (planned D4: E4a embed, E4b results into `results/run_demo/`).** No embeddings or results for the 7 demo clips are visible in the repo. |
| B7 | D5 | Spare capacity: help Phước with the tier-2 batch (split windows by hash, same as shards) | n/a | **`[ ]` NOT NEEDED SO FAR.** A5b finished on Phước's Colab; only becomes relevant if A6c reruns the 307 windows. |
| B4 | **STRETCH** | ShanghaiTech: only if tier 1 is frozen on D4 with time left. `vad/io/frames.py`, `vad/datasets/shanghaitech.py`. Otherwise dropped | See section 6 | **`[ ]` STRETCH / NO-GO.** Confirmed NO-GO in Day 4 gate (D4-1). |
| B6 | **STRETCH** | ShanghaiTech embeddings + normal-memory banks. Depends on B4 | See section 6 | **`[ ]` STRETCH / NO-GO.** Confirmed NO-GO in Day 4 gate (D4-1). |

### C. Nam: alerts, results schema, event typer

| ID | When | Task | Acceptance | Status (synced D4 start) |
|---|---|---|---|---|
| C1 | D1 (**do first, unblocks the dashboard**) | `vad/results.py`: versioned (`schema_version`) dataclass + JSON read/write for one video's run: sample_fps, score array, windows, verify results, alerts, meta (model, commit, timestamp). Pin the on-disk layout under `results/`. Commit a hand-written sample JSON | Round-trip test; Thịnh confirms he can build against it. **Merged by D1 EOD** | **`[x]` DONE.** PR #9 merged (`c1a0580`). `RunResult` dataclass, JSON round-trip tests, `results/sample_run.json`. Schema accepted by Thịnh. |
| C2 | D2 | Fix policy debts G1-G3: implement `cooldown_s` per (camera, event); a function that computes `co_events` from time-overlapping windows; make `responders_in(zone)` real or delete it | Tests: second alert inside cooldown suppressed and allowed after; two overlapping windows trigger escalation end to end | **`[x]` DONE AHEAD OF TIME (D1).** Merged in `9cf6561`. Cooldown, co_events escalation, and real coverage queries implemented in `vad/graph/policy.py` & `alerts.py`. |
| C4 | D2 (overnight) | Embedding shard 3 (25%) on the 1060. D1 check: `torch.cuda.get_arch_list()` contains `sm_61`. If not, pin an older torch or hand the shard to a Colab run | Shard complete, or handed off on D1 | **`[x]` DONE.** `sm_61` confirmed on GTX 1060; all shards computed via unified Colab T4 run. |
| C3 | D3 | Zero-shot event typer: window-mean of cached embeddings vs the 13 `event_typing` prompts -> `EventType`. Evaluate on UCF GT segments: 13-class top-1 accuracy | Accuracy next to the 1/13 chance level; kept SEPARATE from the detection AUC. Confusion matrix is optional | **`[x]` DONE AHEAD OF TIME (D1).** Merged in PR #12 (`9bf274d`). `vad/score/typing.py`, `scripts/eval_typer.py`. Measured on full UCF GT: strict 40.4%, lenient 50.0% vs chance 7.7% (`results/typing_eval.json`). |
| C5 | D4 | `vad/alerting.py`: windows + verify results + context (zone, hour) -> merged alerts (GraphPolicy + `merge_alerts` + cooldown) | Scenario test table of at least 10 cases, all green | **`[x]` DONE AHEAD OF TIME (D1).** Merged in PR #11 (`9751e31`). Scenario test table with 10+ cases all passing. |
| C7 | D5 | `scripts/build_results.py`: cached embeddings + windows + tier-2 outputs -> alerts -> `results/*.json` for all videos | Dashboard loads real results | **`[x]` ALL MERGED (C7 + C7a-e).** PR #24 (`d7da1ec`), #38 (`--verify-dir`), #39 (`results/run_A4c/` tier 1), #40 (C7d: `results/run_A4c/` rebuilt with tier 2, 129 confirmed windows -> 101 alerts), #41 (C7e: VLM typing eval -> `results/vlm_typing_eval.json`). C7f figures in PR #42 (open). |
| C6 | **OPTIONAL** | One real coverage query + DOT export of the policy subgraph. First thing to cut | See section 6 | **`[x]` PARTLY DONE.** Real coverage queries (`responders_in`, `coverage_gaps`, `events_requiring`) implemented; DOT export optional. |

### D. Thịnh (task IDs TH1-TH8, E1-E7): tier-1, benchmark, ablations + report + whole dashboard

| ID | When | Task | Acceptance | Status (synced D4 start) |
|---|---|---|---|---|
| TH1 | D1 | Read the LAVAD protocol (prompts, CLIP B/32, how AUC is pooled); prepare a Colab notebook that runs one shard; confirm CI is still green | One-page protocol note; Colab notebook runs `--dry-run` | **`[x]` DONE.** PR #10 merged (`4ac605b`). `notes/lavad-protocol.md`, `notebooks/colab_shard.ipynb`. Full Colab T4 run completed. |
| TH2 | D1-D2 | `vad/pipeline.py` (tier-1): cached embeddings -> text score -> optional memory -> temporal ops -> candidate windows, all ablation switches in one config dataclass | Tests with `FakeEmbedder`; same config reproduces the same scores | **`[x]` DONE.** PR #10 merged (`4ac605b`). `PipelineConfig`, `ABLATIONS` A0..A5, `run_tier1`, 23 unit tests passing. |
| TH3 | D2-D3 | `scripts/run_benchmark.py`: for each manifest video: load cache, score, `expand_to_frames`, GT, micro AUC + video-level bootstrap CI; writes `results/ablation_*.json` | Runs end to end on a 5-video subset first | **`[x]` DONE (D2).** PR #17 merged (`a67d68a`). Fast tie-aware AUC (0.67s/call), prompt cache, fail-loud guards, 30 tests (491 total passed). |
| TH4 | D3 (**gate**) | A0 reproduction: CLIP B/32, 2 baseline prompts. Also the cross-card check (same video embedded on two machines, cosine > 0.999) | A0 within about 3 points of 53. If not: STOP everything else, debug protocol with B2 | **`[x]` DONE AHEAD OF TIME (D2).** PR #17 merged. A0 evaluated on full 290 videos: 48.77% [38.30, 59.40] vs 53.16% target. Diagnosis table complete in `notes/a0-gate.md`: checks #1, #2, #4, #5, #6 clean, #3 (sampling phase) clean too at stride 16 on all 290 videos = 48.43% (PR #19), so sampling is not the cause. Verdict stays FAIL by the preset band; recommendation = accept within CI and report the deviation. Cross-card cosine check done earlier (min cos 0.99997). Bonus A1-A3 executed. |
| E1 | D2-D3 | Dashboard loader: `app/data.py` loads `results/*.json`; keep fake data behind an explicit `--demo` flag with a banner | Dashboard runs on Nam's sample `RunResult` JSON | **`[x]` DONE AHEAD OF TIME (D1).** PR #10 merged (`4ac605b`). `app/data.py` reads `results/*.json` and `results/sample_run.json`. |
| E2 | D3-D5 | Whole Streamlit app (`app/streamlit_app.py`): video player, score timeline with threshold line, alert feed with VLM caption, severity/responders/protocol, explanation path, replay mode. English UI | Works on sample JSON on D3, on real results on D5 | **`[~]` SKELETON READY.** Reads sample JSON; E2a loader fix merged (PR #31). E2b (real-data UI: tau default, bands, `st.video`, alert card), E2c (overview page), E2d (demo tab) ready to implement. |
| TH5 | D4 | Ablations A1-A5 + threshold calibration on `calib_normal` at a target FPR | Table with CIs, thresholds recorded | **`[x]` DONE AHEAD OF TIME (D2).** A1-A5 plus calib-normalised A3c-A5c (PR #17, #21). Threshold on `calib_normal`: A4c tau 1.1686, calib FPR 4.99% (PR #22). |
| TH6 | D4 EOD | Freeze tier 1: `notes/tier1-results.md`; export `results/windows/*.json` for A3/A5 | Phước has windows by D4 EOD | **`[x]` DONE AHEAD OF TIME (D2).** Frozen A4c, 307 windows in 114 videos, `notes/tier1-results.md`, `scripts/export_windows.py` (this PR). |
| TH7 | D5 | A6 (post-VLM) metrics and MINIMAL operational metrics: FA/hour and latency at the calibrated threshold. Per-class AUC only if time | Table; per-class is optional | **`[x]` ALL MERGED (TH7a-d).** TH7a/b merged (PR #32); TH7c merged (PR #46): `results/ablation_A6.json` computed on 307 windows (129 confirmed, 178 rejected), micro AUC 67.90% [58.39, 77.49] vs A4c 73.00%, FA/h 20.64 -> 0.17 (-99.2%), precision 28.0% -> 49.6%, paired delta CI [-11.42, -0.25]%. |
| TH8 | D1-D6 | **Report (Vietnamese)**: skeleton by D2 (related work + method), results chapters as numbers land, v1 on D5, slides + final on D6 | Report v1 reviewed by all 4 on D5 | **`[~]` IN PROGRESS.** Skeleton merged (PR #10); A0-A3 rows filled (PR #17); A4c/tau rows in (PR #22); A6 row, typer rows, tier-2 cost rows ready to fill from merged JSONs (`ablation_A6.json`, `vlm_typing_eval.json`). |
| E5 | **OPTIONAL** | Policy graph view (renders C6 DOT) in dashboard | Graph shown in UI | **`[ ]` OPTIONAL.** |
| E6 | D5 | Report figures: score timelines for 5 videos as PNG in `results/figures/`. Failure-case gallery is OPTIONAL. Add `matplotlib` to `requirements-app.txt` if used | 5 PNGs | **`[~]` PARTIAL.** Reassigned to Nam as C7f (`scripts/make_figures.py`, PR #42 open with 5 PNGs). |
| E7 | D6 | 2-3 minute screen recording as the GPU-free fallback; dashboard run instructions in `vad/README.md` | Video plays; README steps work on a clean clone | **`[ ]` PENDING (D6).** |

## 4. Critical path and handoffs

```
B3 embedder (D2 midday) ─┬─> shards A4/B5/C4 (D2 overnight) ─> TH3 benchmark ─> TH4 GATE (A0 ~53, D3) ─> TH5 ablations (D4) ─> TH6 windows (D4 EOD)
B2 convention (D2) ──────┘                                                                                  │
C1 schema (D1 EOD) ───────────────> dashboard E1/E2 (Thịnh)                                                 v
A1 VLM measured (D1) ─> A2/A3 (D1-D3) ──────────────────────────────────────────> A5 tier-2 (D5) ─> C7 build_results (D5) ─> dashboard live (D5)
                                                                                                              └─> freeze D5 EOD ─> D6 report + submit
```

There is ONE day of slack on this chain. If B3 is not merged by D2 midday, or TH4 fails on D3, apply the cut list now.

| Handoff | From -> To | Must be true before the next person starts | Status (synced D4 start) |
|---|---|---|---|
| H1 | Huy B3 -> Phước A4, Nam C4, Huy B5 | Real embedder + resumable precompute merged to `main` (D2 midday) | **DONE** (PR #8). Shards completed via unified Colab run (350 videos × 2 models = 700 `.npy` files). |
| H2 | Huy B1/B2 -> Thịnh TH3/TH4 | Manifests committed, convention note written | **DONE** (PR #13, `manifests/ucf_test.csv`, `notes/ucf-gt-convention.md`). |
| H3 | Nam C1 -> Thịnh E1/E2 | `vad/results.py` merged, sample JSON in `results/` (D1 EOD) | **DONE** (PR #9, accepted by Thịnh, `app/data.py` loader merged in PR #10). |
| H4 | Thịnh TH6 -> Phước A3/A5 | `results/windows/*.json` for all benchmark videos (D4 EOD) | **DONE AHEAD OF TIME (D2).** 307 windows in 114 videos, `results/windows/`. |
| H5 | Phước A5 -> Nam C7 -> Thịnh E2 | tier-2 JSON per window exists | **DONE.** A5b -> `results/verify/` merged (PR #37); C7d -> `results/run_A4c/` merged (PR #40); TH7c -> `results/ablation_A6.json` merged (PR #46). Dashboard E2b unblocked. |

## 5. Calendar (who is doing what each day)

| Day | Phước | Huy | Nam | Thịnh |
|---|---|---|---|---|
| D1 Oct 29 | A1 VLM measure, A2 start | B3 start, B1 download | **C1 schema (merge by EOD)**, C4 arch check | TH1, TH2 start, report skeleton |
| D2 Oct 30 | A2 finish, A3, A4 overnight | **B3 merge by midday**, B2, B5 overnight, E3 start | C2 policy debt, C4 overnight | TH2 finish, TH3, E1 dashboard loader |
| D3 Oct 31 | A3 finish | E3 clips | C3 typer | **TH4 gate**, E2 dashboard UI |
| D4 Nov 1 | prompt prep, wait for windows | E3 finish, B4 only if tier 1 is on track | C5 alerting | TH5 ablations, **TH6 freeze EOD**, E2 |
| D5 Nov 2 | **A5, A6** | B7 helps A5, E4 own clips | **C7 build_results** | TH7, E2 live, E6 figures, **report v1**. **FREEZE EOD** |
| D6 Nov 3 | A7 optional, packaging | packaging | packaging, README | report + slides final, E7 recording, **submit (cutoff 23:59)** |

## 6. Cuts (decided now because the deadline moved) and load balance

Decided up front, not "if we slip":
- **ShanghaiTech (B4, B6) is STRETCH.** UCF-Crime test is the only mandatory benchmark. Start B4 only if tier 1 is frozen on D4 with Huy free. Otherwise the report says "out of scope for time" and drops the "training-free + normal reference" labelling discussion.
- **A7 small-256 ablation dropped.** Required ablations are A0..A5 plus the A6 (post-VLM) row.
- **Own clips: about 5**, not 10-20.
- **Per-class analysis, confusion matrix, C6 graph rendering (E5), failure gallery: optional.** Do them only after everything else is done.
- **Tier 2 may run on a subset** of candidate windows, with measured cost reported. Never cut entirely (topic constraint).
- Report is written continuously from D1, not at the end (TH8).

Overload risks, named:
- Thịnh: TH1-TH8 plus whole dashboard (E1, E2, E6, E7). The cuts above keep this focused on essential features. Replay mode avoids writing runtime inference pipelines.
- Nam: C1-C7 backend pipeline and policy engine. If pressed, cut C6 graph rendering and C3 confusion matrix.
- Phước: single point of failure for VLM. Rule: A1 on D1 decides local vs Colab. Huy (B7) is the backup pair.

Cut order if we slip further (cut from the top, never touch the bottom):
1. Optional items above (C6/E5, failure gallery, per-class, A7 task)
2. ShanghaiTech entirely (already stretch)
3. Own clips beyond 3
4. Tier-2 coverage shrinks to fewer videos (still measured and reported honestly)
5. NEVER cut: tier 2 (topic constraint), the A0 gate (TH4), video-level CI, the fallback demo video (E7)

## 7. Working agreements

- One branch per task: `feat/<owner>-<id>` (for example `feat/huy-b3-embedder`). Merge to `main` via PR; CI must be green.
  With 6 days, PRs get reviewed the same day or the author merges after CI is green and tells the group.
- `vad/types.py` and `vad/interfaces.py` are frozen after D1 EOD: changing them needs a yes from every consumer.
- Commit style: `type(scope): message`, English (matches the repo log).
- 10-minute standup daily: yesterday, today, blocked. Blockers are posted the moment they happen, not at standup.
- Each task is "done" only when its acceptance line is true AND `python -m pytest -q` is green. Stubs stay stubs
  (`NotImplementedError`) until real; nobody ships fake output.

## 8. Day-1 checklist (everyone, 15 minutes)

| Who | Run | Report | Result / Status (synced D4 start) |
|---|---|---|---|
| everyone | `git pull`, `pip install -r requirements-dev.txt`, `python -m pytest -q` | 246 passed? | **461 passed** (up from 246 skeleton tests, all green, no GPU required). |
| Nam | `python -c "import torch; print(torch.cuda.get_arch_list())"` | contains `sm_61`? | **Confirmed `sm_61`**; all shards completed via unified Colab T4 run. |
| Huy, Phước, Nam | `nvidia-smi --query-gpu=name,memory.total --format=csv` and `df -h .` | free disk before any download | **Confirmed**; dataset 9.2 GB downloaded & processed on Colab T4. |
| Phước | VRAM test from A1 | go / no-go on local VLM | **DONE (D2, PR #15).** Colab T4 GO: Qwen2.5-VL-3B, 2.93 GB peak, 4.45 s/window (`notes/vlm-selection.md`). |
| Thịnh | CI green on `main`; Colab notebook | link posted | **DONE**; CI green, Colab T4 executed 700 embedding cache files + fp16/fp32 check. |
| Nam + Thịnh | agree sample `RunResult` JSON (C1 unblocks E1) | schema accepted | **DONE**; schema merged in PR #9, accepted and integrated in PR #10 (`app/data.py`). |

## 9. Open questions

1. Does everyone accept their track? Which tasks should swap?
2. Who has a Colab/Kaggle account with GPU quota? (spare shard, and the tier-2 fallback)
3. Does the lecturer require a report template or page count? (report skeleton waits on this)
4. Is the dashboard allowed to load results from disk at the defense, or must it run models live?
