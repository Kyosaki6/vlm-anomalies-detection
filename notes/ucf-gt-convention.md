# UCF-Crime ground-truth convention (B2)

Question: in `Temporal_Anomaly_Annotation_for_Testing_Videos.txt`, are `[start, end]` inclusive, 0-indexed, 30 fps?
Answer: **inclusive end, 30 fps, and the numbers are treated as 0-indexed by every Python baseline we compare to.**
`vad/datasets/ucf_crime.py` already matches that. **No code change.** Source lines below were read by a research
agent from the repos at the commit SHAs shown; re-open the permalinks before quoting them in the report.

## What each reference does

| Source | Commit | Rule | Evidence |
|---|---|---|---|
| Sultani CVPR 2018 (original, MATLAB) | `8aa957c` | **1-indexed, end inclusive**, clipped to video length | `Evaluate_Anomaly_Detector.m` L41-L82: `st_fr=max(Ann(ik,1),1); end_fr=min(Ann(ik,2),Actual_frames); GT(st_fr:end_fr)=1;` and `fps=30` |
| LAVAD CVPR 2024 (the paper we reproduce) | `1ad46c6` | numbers used as **0-indexed, end inclusive** | `src/eval.py` L25-L59: `int(start_idx) <= frame_index + video_record.start_frame <= int(end_idx)`; `start_frame = 0` (`extract_frames.py` L43); `video_fps = 30` (`slurm/ucf_crime/07_eval.sh` L21) |
| RTFM ICCV 2021 | `cf938f4` | **0-indexed, end inclusive** | `list/make_gt_ucf.py` L44-L97: `range(start_idx, end_idx + 1)` |
| VadCLIP AAAI 2024 | `c41067f` | end **exclusive** (drops last frame) | `list/make_gt_ucf.py` L23-L42: `gt_vec[frag[0]:frag[1]] = 1.0` |

Our `frame_labels` does `y[start:min(end, n_frames - 1) + 1] = True`, i.e. 0-indexed, end inclusive, clipped.
That equals LAVAD and RTFM.

## Decision

Keep the parser as is. It is identical to LAVAD, so the A0 gate (ZS-CLIP about 53 AUC) compares like with like.

## Known quirks (do not "fix" silently)

1. **Off-by-one inherited from MATLAB.** The file is 1-indexed (`Burglary035_x264.mp4 Burglary 1 1740` means frames
   1..1740 in MATLAB). LAVAD and RTFM read it as 0-indexed, so their labels are shifted one frame to the right.
   Subtracting 1 would be more correct but breaks like-for-like comparison. At 1 fps sampling the effect is below
   one sample. Report it as a known property, do not change it mid-sprint.
2. **LAVAD scores in 16-frame steps and repeats** (`np.repeat(scores, 16)`, then `[: num_frames]`). We sample at
   1 fps and expand with `expand_to_frames`. A real protocol difference, already flagged in `notes/lavad-protocol.md`.
3. **Pooling is micro** over all frames of all 290 videos in one `roc_curve` call (LAVAD `src/eval.py` L143-L177).
   No per-video AUC, no CI in the papers. Our video-level bootstrap CI is an addition, not a replication.
4. **At most 2 segments per video**, sentinel `-1` for absent. 150 normal videos are listed with `Normal -1 -1 -1 -1`.
   `parse_annotation_line` already drops sentinels.
5. **`end` can exceed the real frame count** (Sultani clips with `min(end, Actual_frames)`). `frame_labels` clips too.
6. RTFM and UR-DMU truncate videos to a multiple of 16 frames before labelling. We label the full video.

## Verified locally

- Annotation file: 290 lines, from `AllenXeon/ucf_crime` (same file as the Sultani repo).
- `Anomaly_Test.txt`: 290 entries, 150 normal.
- `manifests/ucf_test.csv`: 290 rows (140 anomalous + 150 normal), all 30 fps, 1,111,808 frames total, about 37k samples at 1 fps.
  Median video 1788 frames; two normal videos are 108k frames (60 min) and dominate decode time.
- All 290 annotation names are in the manifest. 14 labels = 13 crime classes + Normal.
- Anomalous frames = 84,331 of 1,111,808 (7.6%).
- Evidence for the 1-indexed origin: 7 segments end at or one past the real frame count, e.g. `Arson011` end 1267 vs
  1266 frames, `Assault006` end 8096 vs 8096. A 0-indexed file could not end past `n_frames - 1`. `frame_labels` clips them.
- `manifests/ucf_calib_normal.csv`: 60 train-normal videos, `--max-calib-mb 100`, seed 0. One is **25 fps**, not 30
  (`Normal_Videos586_x264`, 1588 frames). Checked: `sample_indices` uses the per-row fps and gives 64 samples (ceil(1588/25)), so the sampler is correct. Not part of any AUC.
- Manifest cross-check (every annotated name exists in `manifests/ucf_test.csv`) is enforced by
  `scripts/build_manifest.py --annotation`.
