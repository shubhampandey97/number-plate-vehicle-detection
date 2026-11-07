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
