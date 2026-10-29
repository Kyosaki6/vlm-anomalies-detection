# LAVAD Evaluation Protocol Audit: UCF-Crime Zero-Shot Benchmark

Audit of the zero-shot evaluation protocol in **LAVAD** (CVPR 2024: *"Harnessing Large Language Models for Training-free Video Anomaly Detection"*, [arXiv:2404.01014](https://arxiv.org/abs/2404.01014), code: [github.com/lucazanella/lavad](https://github.com/lucazanella/lavad)) on the UCF-Crime dataset. Conducted for baseline reproduction (A0 ~53.16% AUC) and benchmark harness alignment (`scripts/run_benchmark.py`).

---

## 1. ZS-CLIP Baseline Specification

- **Normal Prompt:** `"a standard scene"`
- **Anomaly Prompt:** `"a scene with suspicious or potentially criminal activities"`
- **Scoring Function:** Cosine similarity of visual frame embedding against text embeddings of both prompts, followed by 2-class softmax to obtain anomaly probability (paper states no temperature; our `vad/score/text.py` applies `logit_scale`, a deliberate difference to check at the A0 gate):
  $$s_{\text{anom}} = \frac{\exp(\cos(v, t_{\text{anom}}))}{\exp(\cos(v, t_{\text{norm}})) + \exp(\cos(v, t_{\text{anom}}))}$$
- **Reported Benchmark Metric:** **53.16% frame-level ROC-AUC** on UCF-Crime test set (140 anomalous + 150 normal = 290 videos).
- **Source Citation (Paper):**
  - Section 4.1 *"Comparison with state of the art"*, Paragraph 2 ([arXiv:2404.01014v1](https://arxiv.org/html/2404.01014v1#S4.SS1)):
    > *"Specifically, we introduce Zero-shot CLIP [22] (ZS CLIP) and Zero-shot ImageBind [6] (ZS ImageBind). For both baselines, we exploit their pre-trained encoders to compute the cosine similarities of each frame embedding against the textual embeddings of two prompts: a standard scene and a scene with suspicious or potentially criminal activities. We then apply a softmax function to the cosine similarities to obtain the anomaly score for each frame... We choose ViT-B/32 [3] as the visual encoder for ZS-CLIP... and both utilize CLIP’s text encoder [22]."*
  - Table 1 ([arXiv:2404.01014v1 Table 1](https://arxiv.org/html/2404.01014v1#S4.T1)): `ZS CLIP [22] | ViT | 53.16`.
- **Repository Code Status:** **UNVERIFIED in GitHub repository**. Standalone evaluation script for ZS-CLIP is not committed in `lucazanella/lavad` (the repo provides scripts for BLIP-2 + LLaMA-2 + ImageBind pipeline only). Baseline configuration and results are verified from the paper text.

---

## 2. CLIP Architecture and Visual Pooling

- **Model Variant:** OpenAI CLIP `ViT-B/32` (`openai/clip-vit-base-patch32`).
- **Visual Pooling:** Standard ViT pooling (class token `[CLS]` passed through linear visual projection layer `get_image_features` / `encode_image` to a 512-dimensional $\ell_2$-normalized vector).
- **Temporal Pooling:** None for frame scoring in ZS-CLIP; frames are embedded independently.
- **Source Citation:**
  - Paper Section 4.1 ([arXiv:2404.01014v1#S4.SS1](https://arxiv.org/html/2404.01014v1#S4.SS1)): ViT-B/32 citing Radford et al. (ICML 2021) and Dosovitskiy et al. (ICLR 2021).

---

## 3. Frame-Level AUC Computation: Micro-Pooled vs Macro-Averaged

- **Finding:** **MICRO-POOLED** (global concatenation of all test frames across all 290 videos).
- **Mechanism:**
  1. For each test video, per-frame scores and ground-truth binary labels are accumulated into global lists: `flat_scores.extend(video_scores)` and `flat_labels.extend(video_labels)`.
  2. Single global ROC curve and AUC computed over the aggregated arrays using `sklearn.metrics.roc_curve` and `sklearn.metrics.auc`.
  3. **NOT macro-averaged:** LAVAD does not compute AUC per video and average them.
  4. **No Bootstrap CI:** LAVAD does not compute bootstrap confidence intervals; our benchmark harness (`scripts/run_benchmark.py`) computes micro-AUC matching LAVAD plus video-level bootstrap CI.
- **Source Citation (Repository Code):**
  - File: [`src/eval.py`](https://github.com/lucazanella/lavad/blob/main/src/eval.py)
    - Lines 117–118: `flat_scores = []`, `flat_labels = []`
    - Lines 149–152 (grep-verified `flat_scores = []` at 117, `flat_labels = []` at 118):
      ```python
      flat_scores.extend(video_scores)
      if not without_labels:
          flat_labels.extend(video_labels)
      ```
    - Lines 171–176:
      ```python
      flat_scores = np.array(flat_scores)
      flat_labels = np.array(flat_labels)
      flat_binary_labels = flat_labels != normal_label
      fpr, tpr, threshold = roc_curve(flat_binary_labels, flat_scores)
      roc_auc = auc(fpr, tpr)
      ```

---

## 4. UCF-Crime Temporal Ground-Truth Annotation Convention

Resolves the open question in `AGENTS.md` (*"UCF-Crime annotation segments assumed inclusive [start, end], 0-indexed, 30 fps — UNVERIFIED against LAVAD/RTFM"*):

- **Index Base:** **0-indexed**. Extracted frame filenames start at `000000.jpg` (`frame_count = 0`), and video records register `start_frame = 0`.
- **Segment Boundary:** **INCLUSIVE** (`[start_idx, end_idx]`). Both `start_idx` and `end_idx` frames are labeled abnormal.
- **Video Framerate:** **30 fps**. Video evaluation and metadata assume standard 30 fps UCF-Crime streams.
- **Source Citations (Repository Code):**
  - [`src/eval.py:43`](https://github.com/lucazanella/lavad/blob/main/src/eval.py#L43):
    ```python
    if int(start_idx) <= frame_index + video_record.start_frame <= int(end_idx):
        frame_label = label
    ```
    (Uses `<=` on both ends; inclusive boundary check).
  - [`src/data/video_record.py:14,29`](https://github.com/lucazanella/lavad/blob/main/src/data/video_record.py#L14):
    Docstring states: `3) The third element is the inclusive ending frame id of the video`.
    Line 29: `num_frames = self.end_frame - self.start_frame + 1 # +1 because end frame is inclusive`.
  - [`src/preprocessing/extract_frames.py:16,43`](https://github.com/lucazanella/lavad/blob/main/src/preprocessing/extract_frames.py#L16):
    Line 16: `frame_count = 0` (0-indexed frames `000000.jpg`).
    Line 43: `f.write(f"{video_name} 0 {num_frames - 1} 0\n")` (defines `start_frame=0`, `end_frame=num_frames - 1`).
  - [`slurm/ucf_crime/07_eval.sh:18`](https://github.com/lucazanella/lavad/blob/main/slurm/ucf_crime/07_eval.sh#L18):
    `video_fps=30`.

---

## 5. Sampling Stride and Frame-Level Score Expansion

- **Sampling Stride:** `frame_interval = 16` frames (approx. 0.533 s at 30 fps).
- **Expansion to Frame Level:** Sampled anomaly scores are repeated across the 16-frame interval using `np.repeat(scores, frame_interval)` and truncated to `video.num_frames`.
- **Source Citations:**
  - Paper Section 4 *"Implementation Details"* ([arXiv:2404.01014v1#S4.SS1](https://arxiv.org/html/2404.01014v1#S4.SS1)):
    > *"We sample each video every 16 frames for computational efficiency."*
  - [`slurm/ucf_crime/07_eval.sh:16`](https://github.com/lucazanella/lavad/blob/main/slurm/ucf_crime/07_eval.sh#L16): `frame_interval=16`.
  - [`src/eval.py:78,146`](https://github.com/lucazanella/lavad/blob/main/src/eval.py#L78):
    Line 78: `scores = np.repeat(scores, frame_interval)`
    Line 146: `video_scores = video_scores[: video.num_frames]`

---

## 6. Summary Matrix for Implementation

| Aspect | LAVAD Protocol | Source Reference | Project Impact (`vad/` & `scripts/`) |
|---|---|---|---|
| **ZS-CLIP Prompts** | `"a standard scene"` vs `"a scene with suspicious or potentially criminal activities"` | Paper Sec. 4.1 | Matches `configs/prompts.yaml` (`detection.baseline`) |
| **CLIP Backbone** | `ViT-B/32` (`openai/clip-vit-base-patch32`) | Paper Sec. 4.1 | Used in `scripts/precompute_embeddings.py` |
| **Score Metric** | Softmax probability on cosine similarities | Paper Sec. 4.1 | Implemented in `vad/score/text.py` |
| **AUC Aggregation** | **Micro-pooled** across all frames | `src/eval.py:149-176` | `scripts/run_benchmark.py` MUST pool all frames |
| **Annotation Boundary** | **Inclusive** `[start, end]`, **0-indexed** | `src/eval.py:43`, `video_record.py:30` | Validates `vad/datasets/ucf_crime.py` |
| **Target AUC** | **53.16%** on UCF-Crime test set | Paper Table 1 | Gate A0 reproduction target (~53.2 AUC) |
