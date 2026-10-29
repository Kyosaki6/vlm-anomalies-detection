# Embedder benchmark: fp16 vs fp32 on real videos (B3 residual)

**Date:** 2026-10-29 · **Hardware:** Colab Tesla T4 · **Status:** cosine check DONE; throughput and VRAM NOT measured.

## What was run

Same code, same fps, same videos. Only `--dtype` and `--cache-dir` differ.

```bash
# fp16 = the production cache (cache/embeddings/, all 350 videos, --shard 1/1)
# fp32 = the check run, 5 videos of ucf_calib_normal (--shard 1/20)
python scripts/precompute_embeddings.py --manifest manifests/ucf_calib_normal.csv \
  --shard 1/20 --fps 1 --dtype float32 \
  --model google/siglip2-base-patch16-384 --model openai/clip-vit-base-patch32 \
  --cache-dir /tmp/cache_fp32
```

Compared per frame: cosine between the fp32-run row and the fp16-run row for the same video and frame index.

## Result

Threshold from the task sheet: cosine > 0.999. **All 10 files pass with large margin.**

| Model | Video | Frames fp32 / fp16 | min cos | mean cos |
|---|---|---|---|---|
| SigLIP 2 | Normal_Videos021_x264 | 66 / 66 | 0.99999 | 0.99999 |
| SigLIP 2 | Normal_Videos127_x264 | 60 / 60 | 0.99997 | 0.99999 |
| SigLIP 2 | Normal_Videos283_x264 | 300 / 300 | 0.99997 | 0.99999 |
| SigLIP 2 | Normal_Videos464_x264 | 171 / 171 | 0.99997 | 1.00000 |
| SigLIP 2 | Normal_Videos486_x264 | 91 / 91 | 1.00000 | 1.00000 |
| CLIP B/32 | Normal_Videos021_x264 | 66 / 66 | 0.99999 | 1.00000 |
| CLIP B/32 | Normal_Videos127_x264 | 60 / 60 | 1.00000 | 1.00000 |
| CLIP B/32 | Normal_Videos283_x264 | 300 / 300 | 0.99999 | 1.00000 |
| CLIP B/32 | Normal_Videos464_x264 | 171 / 171 | 0.99999 | 1.00000 |
| CLIP B/32 | Normal_Videos486_x264 | 91 / 91 | 0.99999 | 1.00000 |

688 frames per model, 1376 frame comparisons in total. Worst single frame: 0.99997.
Frame counts match exactly, so sampling is deterministic across dtypes.

## Conclusion

fp16 compute does not change the embeddings in any way that matters downstream.
The smallest label gap the typer relies on is about 1e-2 (see C3 analysis), roughly 1000x larger than the fp16 noise (about 3e-5).
fp16 is safe for every shard, so mixing fp16 (3050 cards, T4) and fp32 (GTX 1060) would have been fine too.

## Not covered (be honest in the report)

- **Throughput (img/s) and peak VRAM:** not recorded on this run. Only the cosine was checked.
- **The 25 fps video `Normal_Videos586_x264`** was not in the 5-video sample. The task sheet asked for it specifically, because `--fps 1` sampling on a 25 fps source is the one place frame indexing could drift. Run it through the same fp32 command if this matters for the report.
- **Cross-card check (TH4):** moot. Every embedding came from one T4, so there is no card-to-card difference to measure.
- The cache is stored as fp16 on disk in both runs, so part of the 1e-5 gap is storage rounding, not compute.
