import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import check_leakage as cl  # noqa: E402


def _img(path, seed, noise=0):
    rng = np.random.default_rng(seed)
    base = (rng.random((60, 80)) * 255).astype(np.uint8)
    if noise:
        base = np.clip(base.astype(int) + rng.integers(-noise, noise + 1, base.shape), 0, 255).astype(np.uint8)
    Image.fromarray(base).resize((320, 240)).save(path)


def test_finds_near_duplicate_across_splits(tmp_path):
    for s in ("train", "val"):
        (tmp_path / "images" / s).mkdir(parents=True)
    _img(tmp_path / "images/train/a.png", 1)
    _img(tmp_path / "images/train/b.png", 2)
    _img(tmp_path / "images/val/a_next_frame.png", 1, noise=3)   # same scene
    _img(tmp_path / "images/val/c.png", 3)

    rows = cl.collect(tmp_path)
    splits = [s for s, _ in rows]
    hashes = [cl.dhash(p) for _, p in rows]
    md5s = [p.name for _, p in rows]
    pairs = cl.find_cross_split_pairs(splits, hashes, md5s, max_dist=4)
    names = {(rows[i][1].name, rows[j][1].name) for i, j, _, _ in pairs}
    assert ("a.png", "a_next_frame.png") in names or ("a_next_frame.png", "a.png") in names
    assert all("c.png" not in n and "b.png" not in n for pair in names for n in pair)


def test_popcount():
    x = np.array([0, 1, 3, 0xFFFFFFFFFFFFFFFF], dtype=np.uint64)
    assert cl.popcount64(x).tolist() == [0, 1, 2, 64]


def test_scene_groups_chain_and_singleton():
    import inspect_leakage as il
    # 0-1 and 1-2 linked (a chain = one scene), 3 alone
    groups = il.scene_groups(4, [(0, 1, 2), (1, 2, 3)])
    assert groups[0] == groups[1] == groups[2] != groups[3]


def test_all_pairs_threshold():
    import inspect_leakage as il
    h = [np.uint64(0), np.uint64(0b111), np.uint64(0xFFFF)]
    assert il.all_pairs(h, 3) == [(0, 1, 3)]


def test_assign_groups_keeps_groups_whole_and_balances():
    from collections import Counter

    import make_group_split as mg
    groups = {g: {"n": n, "cls": Counter({3: n})} for g, n in enumerate([30, 20, 10, 10, 10, 5, 5, 5, 3, 2])}
    out = mg.assign_groups(groups, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=0)
    assert set(out) == set(groups)                      # every group gets exactly one split
    n = Counter()
    for g, s in out.items():
        n[s] += groups[g]["n"]
    assert n["train"] > n["val"] > 0 and n["test"] > 0
    assert out == mg.assign_groups(groups, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=0)  # seeded
