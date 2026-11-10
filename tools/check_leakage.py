"""Find exact and near-duplicate images ACROSS train / val / test splits.

Why: the merged dataset includes frames extracted from traffic videos
(e.g. 'Traffic Dataset'), and split_final_dataset.py shuffled images randomly
with no seed. Consecutive video frames are almost identical, so the same car
and plate can sit in train and in val. The reported mAP@0.5 = 0.985 is on that
val set (which also chose best.pt), so it may be optimistic.

Method: 64-bit difference hash (dHash) per image -> pairs whose Hamming
distance <= --max-dist are near-duplicates. CPU-light (PIL + NumPy only);
~2k images take well under a minute, safe to run while other training runs.

Usage:
    python tools/check_leakage.py --root data/final_merged_datasets --out docs/leakage
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

EXTS = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "val", "test")


def dhash(path: Path, size: int = 8) -> np.uint64:
    img = Image.open(path).convert("L").resize((size + 1, size), Image.BILINEAR)
    a = np.asarray(img, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return np.uint64(int("".join("1" if b else "0" for b in bits), 2))


def popcount64(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.uint64)
    b = x.view(np.uint8).reshape(*x.shape, 8)
    return np.unpackbits(b, axis=-1).sum(-1)


def collect(root: Path):
    rows = []
    for split in SPLITS:
        d = root / "images" / split
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in EXTS:
                rows.append((split, p))
    return rows


def find_cross_split_pairs(splits, hashes, md5s, max_dist):
    splits = np.asarray(splits)
    h = np.asarray(hashes, dtype=np.uint64)
    pairs = []
    for i in range(len(h)):
        j = np.arange(i + 1, len(h))
        j = j[splits[j] != splits[i]]
        if j.size == 0:
            continue
        d = popcount64(h[j] ^ h[i])
        for k in np.nonzero(d <= max_dist)[0]:
            jj = int(j[k])
            kind = "exact" if md5s[i] == md5s[jj] else "near"
            pairs.append((i, jj, int(d[k]), kind))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/final_merged_datasets"))
    ap.add_argument("--max-dist", type=int, default=4,
                    help="Hamming distance on 64-bit dHash (0=identical, <=4 ~ same scene)")
    ap.add_argument("--out", type=Path, default=Path("docs/leakage"))
    a = ap.parse_args()

    rows = collect(a.root)
    if not rows:
        raise SystemExit(f"No images under {a.root}/images/{{train,val,test}}")
    splits = [s for s, _ in rows]
    hashes = [dhash(p) for _, p in rows]
    md5s = [hashlib.md5(p.read_bytes()).hexdigest() for _, p in rows]
    pairs = find_cross_split_pairs(splits, hashes, md5s, a.max_dist)

    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out / "cross_split_pairs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split_a", "image_a", "split_b", "image_b", "hamming", "kind"])
        for i, j, d, kind in pairs:
            w.writerow([splits[i], rows[i][1].name, splits[j], rows[j][1].name, d, kind])

    counts = {s: splits.count(s) for s in SPLITS if s in splits}
    leaked = {s: set() for s in counts}
    for i, j, _, _ in pairs:
        for idx in (i, j):
            if splits[idx] != "train":
                leaked[splits[idx]].add(idx)
    lines = [f"# Cross-split leakage report (dHash, max Hamming {a.max_dist})", "",
             f"Images: {counts}", f"Cross-split pairs: {len(pairs)} "
             f"(exact {sum(p[3] == 'exact' for p in pairs)}, near {sum(p[3] == 'near' for p in pairs)})", ""]
    for s in ("val", "test"):
        if s in counts:
            n = len(leaked[s])
            lines.append(f"- {s}: {n}/{counts[s]} images ({100 * n / counts[s]:.1f}%) "
                         f"have a near-duplicate in another split")
    (a.out / "leakage_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nPairs written to {a.out / 'cross_split_pairs.csv'}")


if __name__ == "__main__":
    main()
