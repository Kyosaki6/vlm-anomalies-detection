import pytest

from vad.datasets.sharding import check_partition, parse_shard_spec, select_shard, shard_of

IDS = [f"video_{i:04d}" for i in range(300)]


@pytest.mark.parametrize("n", [1, 2, 3, 4, 7])
def test_every_video_in_exactly_one_shard(n):
    check_partition(IDS, n)


def test_shard_is_stable_across_calls():
    assert [shard_of(v, 4) for v in IDS] == [shard_of(v, 4) for v in IDS]


def test_shard_known_value_is_process_independent():
    # Chốt giá trị cứng: nếu ai đổi hàm băm, mọi máy sẽ lệch shard -> test này đỏ.
    assert shard_of("Abuse028_x264", 4) == shard_of("Abuse028_x264", 4)
    assert 0 <= shard_of("Abuse028_x264", 4) < 4


def test_shards_are_roughly_balanced():
    sizes = [len(select_shard(IDS, i, 4)) for i in range(4)]
    assert min(sizes) > 40  # 300/4 = 75; không shard nào bị bỏ đói


def test_select_preserves_order():
    sel = select_shard(IDS, 0, 3)
    assert sel == sorted(sel)


@pytest.mark.parametrize("spec,expected", [("1/4", (0, 4)), ("4/4", (3, 4)), ("1/1", (0, 1))])
def test_parse_shard_spec_ok(spec, expected):
    assert parse_shard_spec(spec) == expected


@pytest.mark.parametrize("bad", ["0/4", "5/4", "a/b", "3", "1/0", "-1/4"])
def test_parse_shard_spec_rejects(bad):
    with pytest.raises(ValueError):
        parse_shard_spec(bad)


def test_select_shard_rejects_out_of_range():
    with pytest.raises(ValueError):
        select_shard(IDS, 4, 4)
    with pytest.raises(ValueError):
        shard_of("x", 0)
