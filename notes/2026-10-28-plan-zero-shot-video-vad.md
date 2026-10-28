# Plan: Zero-Shot Video Anomaly & Surveillance Alerting

Status: DRAFT v2 (user answers folded in; only Q1 graph choice still open, default = "policy graph, lite")
Hard deadline: submit BEFORE 2026-11-04 (hard cutoff: end of 2026-11-03 23:59) -> last working day 2026-11-03. Internal feature freeze = 2026-11-02 evening; Nov 3 = report + packaging + submit only.
Only 6 working days (Oct 29 .. Nov 3). Anything in this file that assumed 14 days is superseded by section 8.2/8.3 and by the cut list in `notes/2026-10-29-task-assignment.md` section 6.
Team: 4 people (Phước, Thịnh, Huy, Nam). Thịnh owns the Streamlit dashboard (see 8.1).
Language: report + defense in Vietnamese; dashboard UI in English; code/comments may stay bilingual.
Constraint from topic registration: the topic was signed up as VLM, so a GENERATIVE VLM is mandatory (tier 2 is in the MVP, not optional).
Hardware (GPU inventory, update as people confirm):
| Owner | GPU | Role in plan |
|---|---|---|
| Phước | RTX 3050 6 GB, 20 GB RAM (LOQ 2024) | Main GPU: VLM (T2) + embedding cache shard |
| Nam | GTX 1060 3 GB | Embedding cache shard only. Too small for the VLM. Pascal has very slow fp16 -> run fp32, vision tower only (~0.4 GB), small batch. Unverified: check the installed torch build still ships Pascal (sm_61) kernels. |
| Huy | RTX 3050 4 GB (confirmed) | Embedding cache shard (vision tower ~0.4 GB, fp16 is fine on Ampere). VLM only as a stretch: 4 GB is below the 6 GB target, so a 2B 4-bit model would be borderline and unmeasured. Not on the critical path. |
| Thịnh | none assumed (Thịnh's dev box measured: no usable GPU) | CPU dev + Colab/Kaggle T4 (~16 GB) |
Colab/Kaggle T4 is a first-class resource, not a backup, and is roomier than the 3050 for the VLM.
Disk on the dev box checked at ~42 GB free (re-check on each GPU machine before downloads).

---

## 0. One-paragraph pitch (what the defense says)

A training-free surveillance system. A video stream is sampled, each frame/clip is embedded with SigLIP 2,
scored against *generic* normal/suspicious text prompts and a *per-camera normal memory*, temporally
aggregated into an anomaly timeline, and, only for suspicious windows, verified by a small generative VLM that
describes what happens. A policy graph (camera/zone -> event type -> severity -> responders -> protocol)
turns verified events into alerts shown on a dashboard. Evaluated with frame-level AUC on UCF-Crime and
ShanghaiTech against published zero-shot baselines. No gradient step on the target datasets.

## 1. Honest framing (decide this up front, it protects the defense)

- Published zero-shot frame-level AUC on UCF-Crime (LAVAD, CVPR 2024): ZS-CLIP 53.16, ZS-ImageBind 53.65/55.78,
  LLaVA-1.5 72.84, LAVAD 80.28. Weakly-supervised SOTA ~ 85-87.
- Naive per-frame prompting is near chance. So the contribution is NOT "SigLIP 2 detects anomalies". It is
  the *system design that closes the gap*: temporal aggregation, normal-reference scoring, two-tier verification.
- Realistic target: beat ZS-CLIP clearly (>= 65 AUC with tier 1 only), approach LAVAD with tier 2
  (70-78 plausible, NOT guaranteed). Any number >= 80 on 6 GB VRAM would be a surprise. Report whatever comes out.
- ShanghaiTech is not truly zero-shot-friendly (anomalies = bikes, running, throwing; no crime semantics).
  Treat it as "training-free with normal-only reference frames", and label it that way in the report.
- Keep the 15-image repo as smoke fixtures only. It stops being evidence.

## 2. Scope

IN: offline video -> scores -> alerts -> dashboard; benchmark eval; ablations; policy graph; tests; CI.
OUT: training/fine-tuning, RTSP/realtime queue, audio, pixel-level localization, multi-camera tracking.
STRETCH (only if ahead): live webcam demo mode, Telegram/webhook push, Vietnamese-street self-recorded clips.

## 3. Data

| Dataset | Use | Notes |
|---|---|---|
| UCF-Crime **test split only** (140 anomalous + 150 normal videos, 13 classes, 30 fps) | main benchmark | Temporal annotation file gives start/end frames (up to 2 instances) -> frame-level GT. HF mirror `AllenXeon/ucf_crime` (CC0) and Kaggle mirror exist. |
| UCF-Crime train-normal **subset** (~50-100 videos) | threshold calibration + normal memory | Never touch test for calibration. |
| ShanghaiTech Campus: 107 test videos, 13 scenes, frame masks | secondary benchmark | Train part on Kaggle is 11.7 GB of frames; download test part + only a handful of train videos per scene for normal memory. Beware: not the "ShanghaiTech crowd counting" dataset. |
| Own clips (optional) | VN-flavored demo only, NOT benchmark | 10-20 clips, label by hand. |

Disk budget (42 GB free): do NOT pull the full 128 h UCF-Crime. Test split + calibration subset + ShanghaiTech test
only. Verify actual archive sizes before downloading (not measured yet). Store 1 fps / 384 px derivatives, delete raw if tight.
Embeddings cached to `.npy` per video, so re-experiments cost seconds not hours.

## 4. System architecture

```
video -> sampler (1-2 fps, uniform) -> SigLIP2 image encoder -> emb[t]  (cached .npy)
                                                         |
   Tier 1 scoring (cheap, every frame)                    v
     a) text score   : softmax over prompt ensembles (generic normal vs generic suspicious)
     b) memory score : 1 - max cos to normal-memory bank (per camera / per scene, optional)
     c) fuse -> s[t]
   Temporal layer: per-video robust normalisation (median/MAD) -> window pooling (top-k mean) -> EMA/median smoothing
   Candidate windows: s_smooth > tau  (tau from calibration set at fixed FPR budget)
                                                         |
   Tier 2 (expensive, only candidate windows) : small VLM captions k frames + asks yes/no + event class
                                                         v
   Event typer (zero-shot text prompts for 13 classes) -> Policy graph -> Alert(severity, responders, protocol)
                                                         v
                                   Dashboard (Streamlit): video + score timeline + alert feed + evidence frames
```

Modules (new layout, reuse existing `decision.py` metrics/bootstrap, `tests/`, CI):

```
vad/
  io/        video.py (sampler), datasets/{ucf_crime,shanghaitech}.py (GT loaders -> frame-level labels)
  embed/     siglip.py (fp16, batch, disk cache keyed by model+video+fps)
  score/     text.py, memory.py, fuse.py
  temporal/  normalize.py, pool.py, smooth.py
  verify/    vlm.py (tier 2, pluggable backend)
  graph/     ontology.py, policy.py, alerts.py
  eval/      metrics.py (reuses decision.py), run_benchmark.py, ablate.py, report.py
app/         streamlit_app.py
scripts/     download_*.sh, precompute_embeddings.py
tests/       unit tests, tiny synthetic video fixtures, mocked VLM
```

## 5. Technical design decisions (with reasons)

1. **Backbone**: SigLIP 2 base-patch16-384 in fp16 (~1 GB VRAM). Keep small-256 as ablation + low-spec fallback.
   Also run OpenAI CLIP ViT-B/32 as the *reproduction baseline* so our 53-ish number matches LAVAD's -> proves the pipeline is sane.
2. **Prompts must be generic.** Normal: "a normal everyday scene". Suspicious: "a scene with suspicious or criminal
   activity" (mirrors LAVAD's baseline so comparison is fair). Class-name prompts ("fighting", "road accident")
   are allowed ONLY in the event-typing stage, and typing is reported as a separate metric. This kills the label-leak
   problem the current repo admits to.
3. **Prompt ensembling**: 5-10 paraphrases per side, averaged in embedding space. Ablate 1 vs N.
4. **Per-video robust normalisation**: scores z-scored with median/MAD inside each video. Fixes the fact that raw
   cosine margins differ wildly between cameras/lighting. Biggest expected cheap gain.
5. **Temporal pooling**: window 5-10 s, top-k mean; then median filter + EMA. Ablate window and smoothing.
6. **Normal memory** (training-free): bank of normal frame embeddings per scene; score = 1 - max cos (existing idea, now real).
   Honest label: "few-shot normal reference", not zero-shot. Ablate with/without.
7. **Tier 2 VLM** (the part that makes it a real "VLM" system): candidate model Qwen2-VL-2B / Qwen2.5-VL-3B or
   SmolVLM2, 4-bit, must fit 6 GB. Run only on candidate windows (a few percent of frames) so cost stays bounded.
   Used to (a) confirm/reject, (b) produce a human-readable description for the alert. Model choice = open question,
   to be checked by a librarian lookup before Day 8. Fallback if it will not fit: Colab for tier-2 offline runs.
8. **Threshold**: chosen on calibration set to hit a target false-alarm budget (e.g. FPR 5%), never on test.
   Reported separately from threshold-free AUC.

## 6. Policy graph (my opinion, since you asked)

Recommendation: **build it, but small and honest, ~10-15% of effort. Call it an "Alert Policy Graph".**

Why not drop it: the pipeline output is an *event class + confidence + context*; mapping that to a response is
exactly where a typed graph earns its keep, if and only if the graph does something a dict cannot.
Why not oversell it: it is a policy layer, not "knowledge graph reasoning". Do not claim KG research.

What makes it non-trivial (each is a testable behaviour, none is a 1-to-1 lookup):
- Typed nodes: `Camera`, `Zone`, `EventType`, `Severity`, `Responder`, `Protocol`, `TimeBand`.
- Typed edges with attributes: `Camera -in-> Zone`, `EventType -escalates_to-> EventType`, `EventType -requires-> Responder`.
- **Context rules**: same `fighting` event -> different severity/responder at a school gate at 22:00 vs a market at noon.
- **Multi-hop**: `road_accident` + `explosion` in the same window -> escalation edge -> fire dept responder added.
- **Alert hygiene**: de-duplicate repeated events, cool-down per camera, merge adjacent windows into one incident.
- **Explainability**: alert carries the traversed path ("fighting @ Gate A, night -> high -> police+security -> protocol P3").
- Evaluation = scenario test table (input event+context -> expected alert) + property tests, NOT accuracy.

Side-by-side comparison with a plain rule table, and the recommended "lite" hybrid: see section 12a.

## 7. Evaluation protocol

Primary (both datasets): frame-level ROC-AUC, micro-averaged over concatenated frames (the protocol LAVAD/RTFM use).
Secondary: frame-level AP; per-class AUC (UCF 13 classes); anomalous-video detection rate.
Operational (what "alerting" really needs): incident-level recall, false alarms per hour on normal test videos at
the calibrated threshold, detection latency (seconds from GT start to first alert).
Typing: 13-way accuracy of the event typer on GT anomalous segments (separate from detection, labelled zero-shot classification).
Tier 2: AUC/precision before vs after VLM verification; fraction of windows sent to tier 2; seconds per window.
Statistics: bootstrap 95% CI over **videos** (not frames, frames are autocorrelated) - reuse `decision.bootstrap_ci`.
Compare against: ZS-CLIP (reproduced), ZS-SigLIP2 raw, published LAVAD / LLaVA-1.5 / weakly-supervised numbers (cited, not rerun).

Ablation table (the core of the report):
| # | Variant |
|---|---|
| A0 | CLIP B/32 per-frame, 2 prompts (reproduce ~53) |
| A1 | SigLIP2 per-frame, 2 prompts |
| A2 | + prompt ensemble |
| A3 | + per-video normalisation |
| A4 | + temporal pooling/smoothing |
| A5 | + normal memory (fuse) |
| A6 | + tier-2 VLM verification |
| A7 | small-256 backbone, speed/accuracy trade-off |

## 8. Team split and timeline (4 people, 2026-10-29 .. 2026-11-03)

### 8.1 Ownership (proposal, swap freely; each track has a clear interface so people do not block each other)

| Owner | Track | Interface it must expose |
|---|---|---|
| Thịnh | **T1 Core**: sampler, SigLIP2 embedding cache, tier-1 scoring, temporal layer, benchmark runner, ablations. Writes code CPU-testable on small clips; heavy runs go to Phước's GPU or Colab | `scores(video) -> np.ndarray[T]`, `embeddings/*.npy`, `run_benchmark.py` |
| Phước (owns the local GPU) | **T2 VLM verifier**: generative VLM backend (4-bit, <6 GB), prompts, structured JSON output, window cost stats. Also runs the big embedding-cache job for T1 (290 UCF videos; ShanghaiTech only if stretch is reached) | `verify(frames, context) -> {is_anomaly, event_type, caption, confidence}` |
| Huy | **T3 Data**: UCF-Crime test + calib-normal download, GT loaders, GT unit tests, own Vietnamese-street clips (about 5, hand-labelled, demo only). ShanghaiTech is STRETCH only (see 8.3) | `load_gt(dataset, video) -> bool[T]`, manifest CSV |
| Nam | **T4 Alerts**: event typer (13-class zero-shot), policy graph/rules, alert hygiene, scenario tests, build_results script | `to_alert(event, camera, time) -> Alert(severity, responders, protocol, path)` |
| Thịnh | **T5 Dashboard**: whole Streamlit app (English UI, replay mode from results, timeline, alert feed, explanation path, failure gallery, report figures, fallback recording) | consumes `scores`, `verify`, `to_alert` through the interfaces |
| Thịnh (report owner) | **Report + slides** (Vietnamese): written alongside the project, pulls numbers from `notes/` and `eval/report.py` output | reads results from `notes/` |

Rule: interfaces (dataclasses + type stubs + fake implementations) are written on Day 1-2 so T4/T5 can build against fakes before T1/T2 land.

### 8.2 Day plan

| Date | Day | Milestone | Exit check |
|---|---|---|---|
| Oct 29 | 1 | Repo restructure + stubs already merged; T2 measures 2-3 VLMs in 4-bit (VRAM, s/window) and decides local vs Colab; T3 starts UCF downloads; T1 embedder wiring starts; T4 results schema (C1) | VRAM numbers recorded; schema merged by EOD |
| Oct 30 | 2 | Embedder + resumable precompute done, shards start OVERNIGHT; T2 `VLMVerifier` real; T3 GT convention check + manifests; T4 cooldown/co_events fixes; dashboard core on the schema (fake sample JSON) | 3 videos embedded; shards launched |
| Oct 31 | 3 | Shards finish; tier-1 pipeline + `run_benchmark.py`; **Validity gate**: A0 (CLIP B/32, 2 prompts) near 53 AUC; T2 `run_verify.py`; event typer | A0 within ~3 of 53, else STOP and fix protocol |
| Nov 1 | 4 | Ablations A1-A5 (prompt ensemble, per-video norm, temporal, memory) + threshold on `calib_normal`; **tier 1 frozen EOD**, `results/windows/*.json` exported; alerting glue (C5); dashboard on real scores | ablation table with video-level CIs; windows handed to T2 |
| Nov 2 | 5 | Tier 2 on candidate windows (subset if timing says so; Colab if the 3050 is too slow); A6 row; `build_results.py`; dashboard live on real results; report v1 | A6 row + % windows escalated + s/window; **FEATURE FREEZE EOD** |
| Nov 3 | 6 | Report + slides final, fallback screen recording, README run steps, repo cleanup, CI green. NO new features | submitted before Nov 4 |

Critical path: T1 embedding cache (Oct 31) -> A0 gate -> ablations -> candidate windows (Nov 1 EOD) -> T2 batch (Nov 2) -> report numbers (Nov 3).
There is exactly ONE day of slack for the whole chain. If any step slips by a day, apply the cut list in `notes/2026-10-29-task-assignment.md` section 6 immediately; do not wait.
T2 starts Day 1 and runs in parallel, because the VLM is mandatory and is the single biggest technical risk.
GPU plan: shard the embedding-cache job instead of queueing it on one card.
- Split the video manifest (UCF test 290 + calib-normal subset + ShanghaiTech test 107) into N deterministic shards by sorted
  filename hash. Each machine runs `precompute_embeddings.py --shard i/N`, writes one `.npy` per video keyed by
  model + video + fps, resumable. Merging = copying files into one folder.
- Suggested split by speed: Phước ~40% (3050 6 GB, also busy with VLM), Huy ~35% (3050 4 GB, fp16, same chip class so
  roughly the same speed), Nam ~25% (1060, fp32, slowest). A Colab T4 run by Thịnh is the spare for any shard that
  stalls. No machine is a single point of failure. Use `--shard i/N` with N=4 and give Phước two shards if wanted.
- Consistency rule: every shard must use the same model id, preprocessing, fps and stored dtype (store fp16 on disk, compute
  in the dtype the card handles well). The Day 4 A0 reproduction gate also checks that a video embedded on two different
  cards gives cosine > 0.999 with itself, otherwise fix before merging.
- VLM tier 2 stays on Phước's 3050 or Colab T4 only. Nam's 3 GB cannot host it.
- Everything downstream of the cache (scoring, ablations, graph, dashboard) is CPU-only.
If T2 slips, the minimum acceptable tier 2 is: caption + yes/no verification on the top-N windows per video, on a
subset of videos, with measured cost. Do NOT cut tier 2 entirely (topic constraint).

### 8.3 Scope under the 6-day deadline (added 2026-10-29)

The original 14-day scope does not fit in 6 days. Decisions, applied now rather than when we slip:
- UCF-Crime test is the ONLY mandatory benchmark. ShanghaiTech is STRETCH: start it only if tier 1 is frozen on Nov 1 with time left. Otherwise state in the report that it was out of scope for time.
- Ablations: A0..A5 are required; A7 (small-256) is dropped.
- Own Vietnamese clips: about 5 (not 10-20), demo only.
- Dashboard: replay mode only. Required views: score timeline with threshold, alert feed with VLM caption, policy-path explanation. Graph rendering and failure gallery are optional.
- Tier 2 may run on a SUBSET of candidate windows with measured cost (this is the "minimum acceptable tier 2" above). It is never cut entirely.
- Nobody is double-booked on the same day without a named cut: Thịnh holds tier-1 + report + whole dashboard; to keep that feasible, the cuts above (stretch ShanghaiTech, dropped A7, fewer clips) are strictly applied.

## 9. Risks and mitigations

| Risk | P | Mitigation |
|---|---|---|
| Tier-1 AUC stays ~55-60 | high | Expected for naive prompts; ablation story (norm + temporal) is the contribution; tier 2 is the bump. Report honestly. |
| 6 GB VRAM too small for VLM | med-high | VLM is mandatory, so this is the #1 risk. T2 starts Day 1 and measures VRAM first; 4-bit small VLM (2-3B class, to be verified by Phước, not assumed); escalate only candidate windows; Colab/Kaggle T4 (16 GB) is the fallback and may be the primary for the batch run. |
| Mixed GPUs (3050 6 GB / 3050 4 GB / 1060 3 GB) give slightly different embeddings and speeds | med | Sharded resumable cache, same dtype/preprocessing everywhere, cross-card cosine check on Day 4, Colab T4 as tie-breaker. |
| Pascal (1060) not supported by the newest torch wheels | low-med | Check `torch.cuda.get_arch_list()` on Nam's box Day 1; pin an older torch build there, or give Nam's shard to Colab. |
| Only 6 working days, 4 people (deadline before Nov 4, hard cutoff end of Nov 3) | very high | Scope cuts in 8.3 applied up front; interface stubs + fakes already merged; feature freeze Nov 2 evening; Nov 3 is report + packaging only. One day of slack on the critical path. |
| Colab quota/session loss during tier-2 batch | med | Resumable per-window output files; run in chunks; keep a local subset result as fallback. |
| Disk (42 GB) | med | Test split only; 1 fps 384 px derivatives; embeddings only, drop raw after caching. |
| Dataset download friction (mirrors, ShanghaiTech link rot) | med | Kaggle/HF mirrors identified; confirm access Day 1, switch to Avenue if ShanghaiTech blocked. |
| Protocol bugs (off-by-one frames, fps) | med | A0 reproduction gate on Day 4. |
| RAM (6 GB free) | low | Stream frames, never hold whole videos; batch embeddings. |
| Examiner: "just API wrapper" | high | Contribution = ablated pipeline + calibration + operational metrics + policy graph + verification; present as system study. |
| Examiner: "ShanghaiTech not zero-shot" | high | Pre-empt: labelled "training-free + normal reference". |

## 10. What is reused from the current repo

Keep: `decision.py` (roc_auc ties, bootstrap_ci, binary_metrics, threshold sweep), test/CI scaffolding, README honesty
style, `utf-8-sig` conventions. Repurpose: `search.py` embedding code -> `vad/embed/siglip.py`; `graph.py` -> `vad/graph/`.
Retire: 15-image eval as evidence, `events.json` 1-image-per-event schema (replaced by ontology YAML).

## 11. Definition of done

- [ ] UCF-Crime test frame-level AUC with video-level bootstrap CI, A0..A6 table (ShanghaiTech = stretch, A7 dropped; see 8.3).
- [ ] A0 reproduces published ~53 +- 3 (pipeline validity).
- [ ] Operational metrics at a calibration-set threshold (FA/hour, recall, latency).
- [ ] Tier-2 verification measured (gain + cost), or explicitly cut with reason.
- [ ] Policy graph with scenario + property tests; alerts carry explanation path.
- [ ] Streamlit demo runs end-to-end offline from cached embeddings and live on a short clip.
- [ ] `pytest` green without model downloads; README states every limitation.

## 12. Decisions log

| # | Question | Answer |
|---|---|---|
| 1 | Graph vs rule table | OPEN. Default in plan = hybrid (see 12a). |
| 2 | Generative VLM mandatory? | YES (topic registered as VLM). Tier 2 stays in MVP. |
| 3 | ShanghaiTech as "training-free + normal reference" | OK |
| 4 | Team | 4 people (Phước, Thịnh, Huy, Nam), tracks in 8.1 |
| 5 | Language | Report + defense Vietnamese, dashboard English |
| 6 | Own Vietnamese demo clips | YES, demo only, not benchmark (owner: Huy) |
| 7 | Deadline | Submit before 2026-11-04 (hard cutoff: end of 2026-11-03 23:59). Feature freeze Nov 2 evening |
| 8 | Dashboard owner | Thịnh (whole dashboard) |

Demo policy (decided): live demo is NOT a priority. Dashboard runs in **replay mode** from precomputed embeddings, scores,
VLM captions and alerts, so it needs no GPU and no model downloads on any laptop. All heavy compute is done once on the
RTX 3050 or Colab and the cache ships with the repo. A recorded fallback video is kept as a second safety net.
Live inference on an arbitrary new clip is a stretch goal only (CPU: ~4 s/frame for SigLIP 2, VLM far slower).

Still unknown (ask lecturer, non-blocking): required report template and page count.

## 12a. Graph vs rule table, side by side

Both take the same input `(event_type, confidence, camera, zone, time)` and return the same `Alert`. The difference is
how the policy is stored and what you can ask of it.

| | Rule table (YAML + small engine) | Policy graph (NetworkX) |
|---|---|---|
| Storage | list of rows: `event=fighting, zone=school, night -> severity=high, responders=[police, security]` | nodes (Camera, Zone, EventType, Severity, Responder, Protocol) + typed edges |
| Simple lookup | trivial, fast to write | same result, more code |
| Context rules (zone/time) | one row per combination, can explode (event x zone x time band) | factored: `EventType -> Severity` edge modulated by `Zone` and `TimeBand` nodes, no combinatorial rows |
| Escalation chain (accident + explosion -> fire dept) | ad-hoc `if` in code | edge `escalates_to`, traversed generically; adding a new chain = adding an edge |
| "Who can respond in Zone X?" / "which events involve police?" | needs a hand-written query per question | graph queries, reuse the same structure |
| Explanation | you print the matching row | you print the traversed path (event -> severity -> responders -> protocol) |
| Visual for the defense | none | can render the subgraph in the dashboard |
| Effort | ~0.5 day | ~1.5-2 days |
| Defense risk | "this is just if-else" (true, nobody claims otherwise) | "why a graph?" (answerable only if the factored/escalation/query features exist) |

Blunt take: for a flat table of ~13 events x few zones, a rule table gives ~90% of the behaviour. The graph earns its
keep only through (a) factored context so rules do not explode, (b) escalation chains, (c) reusable queries, (d) the
explainable path and a visual. If you do not implement (a)-(d), the graph is decoration and a rule table is the honest choice.

Recommended hybrid ("policy graph, lite"): rules live in one YAML (source of truth, easy for anyone to edit), loaded into
a NetworkX graph at startup; ship ONLY (b) escalation, (c) one or two coverage queries, (d) explanation path + dashboard
rendering. Skip fancy reasoning. Cost ~1 day for Nam, and it gives a real artefact to show and a defensible answer to "why a graph".
Fallback if Nam is overloaded: ship the YAML + engine without the graph module and rename it "rule-based alert policy".
