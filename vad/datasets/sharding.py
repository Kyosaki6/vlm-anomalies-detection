# -*- coding: utf-8 -*-
"""Chia video thành N shard xác định (deterministic) để nhiều máy cùng tính embedding.

Quy tắc: shard của một video chỉ phụ thuộc video_id + N (hash ổn định, KHÔNG dùng hash() của Python vì
bị random hoá theo process). Máy nào chạy cũng ra cùng kết quả -> không trùng, không sót.
"""
from __future__ import annotations

import hashlib
from typing import Iterable, Sequence


def shard_of(video_id: str, n_shards: int) -> int:
    if n_shards < 1:
        raise ValueError("n_shards phải >= 1")
    digest = hashlib.sha256(video_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % n_shards


def select_shard(video_ids: Iterable[str], index: int, n_shards: int) -> list[str]:
    """Các video_id thuộc shard `index` (0-based), giữ thứ tự đầu vào."""
    if not 0 <= index < n_shards:
        raise ValueError(f"index phải trong [0, {n_shards}), nhận {index}")
    return [v for v in video_ids if shard_of(v, n_shards) == index]


def parse_shard_spec(spec: str) -> tuple[int, int]:
    """'1/4' -> (0, 4): người dùng đếm từ 1, code đếm từ 0."""
    try:
        i_str, n_str = spec.split("/")
        i, n = int(i_str), int(n_str)
    except ValueError as e:
        raise ValueError(f"Shard phải có dạng i/N (ví dụ 2/4), nhận {spec!r}") from e
    if n < 1 or not 1 <= i <= n:
        raise ValueError(f"Shard không hợp lệ: {spec!r} (cần 1 <= i <= N)")
    return i - 1, n


def check_partition(all_ids: Sequence[str], n_shards: int) -> None:
    """Kiểm tra mỗi video thuộc đúng 1 shard (dùng trong test và trước khi merge)."""
    seen: dict[str, int] = {}
    for idx in range(n_shards):
        for v in select_shard(all_ids, idx, n_shards):
            if v in seen:
                raise AssertionError(f"{v} nằm ở cả shard {seen[v]} và {idx}")
            seen[v] = idx
    missing = [v for v in all_ids if v not in seen]
    if missing:
        raise AssertionError(f"Video không thuộc shard nào: {missing[:5]}")
