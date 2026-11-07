"""Look at what check_leakage.py found, and group images into "scenes".

1. Contact sheets: random cross-split pairs side by side, one sheet per
   Hamming bucket (0 = identical, 1-2, 3-4). Open them and judge: is it the
   same frame / next video frame (real leakage), or just the same fixed camera
   with different vehicles (milder, but still the same scene)?
2. Scene groups: union-find over ALL images (any split) whose dHash distance is
   <= --max-dist. Each connected component is one "scene". The table of group
   sizes says how much a group-aware re-split will have to keep together, and
   docs/leakage/groups.csv (image -> group id) is the input for that re-split.
   This recovers "which images belong together" without needing the lost
   source-dataset names.

CPU-light (PIL + NumPy). Contact sheets show real plates: they are written as
.jpg under docs/leakage/ which is git-ignored for images. Don't commit them.

Usage:
    python tools/check_leakage.py --root data/final_merged_datasets
    python tools/inspect_leakage.py --root data/final_merged_datasets
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_leakage as cl  # noqa: E402

BUCKETS = {"d0": (0, 0), "d1-2": (1, 2), "d3-4": (3, 4)}


class DSU:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def all_pairs(hashes, max_dist):
    """All pairs (any split) with Hamming distance <= max_dist."""
    h = np.asarray(hashes, dtype=np.uint64)
    out = []
    for i in range(len(h) - 1):
        d = cl.popcount64(h[i + 1:] ^ h[i])
        for k in np.nonzero(d <= max_dist)[0]:
            out.append((i, i + 1 + int(k), int(d[k])))
    return out


def scene_groups(n, pairs):
    dsu = DSU(n)
    for i, j, _ in pairs:
        dsu.union(i, j)
    roots = [dsu.find(i) for i in range(n)]
    ids = {r: k for k, r in enumerate(sorted(set(roots)))}
    return [ids[r] for r in roots]


def thumb(path: Path, h: int = 180) -> Image.Image:
    im = Image.open(path).convert("RGB")
    return im.resize((max(1, int(im.width * h / im.height)), h))


def contact_sheet(pairs, rows, out_path, n=12, seed=0):
    sample = random.Random(seed).sample(pairs, min(n, len(pairs)))
    tiles = []
    for i, j, d in sample:
        a, b = thumb(rows[i][1]), thumb(rows[j][1])
        t = Image.new("RGB", (a.width + b.width + 10, a.height + 22), "white")
        t.paste(a, (0, 22))
        t.paste(b, (a.width + 10, 22))
        ImageDraw.Draw(t).text(
            (4, 4), f"{rows[i][0]}/{rows[i][1].name}  vs  {rows[j][0]}/{rows[j][1].name}   d={d}", fill="black")
        tiles.append(t)
    if not tiles:
        return False
    w = max(t.width for t in tiles)
    sheet = Image.new("RGB", (w * 2 + 10, sum(t.height for t in tiles[::2]) + 10 * len(tiles)), "white")
    y = 0
    for k in range(0, len(tiles), 2):
        row = tiles[k:k + 2]
        for c, t in enumerate(row):
            sheet.paste(t, (c * (w + 10), y))
        y += max(t.height for t in row) + 10
    sheet.crop((0, 0, sheet.width, y)).save(out_path, quality=85)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/final_merged_datasets"))
    ap.add_argument("--max-dist", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("docs/leakage"))
    ap.add_argument("--per-sheet", type=int, default=12)
    a = ap.parse_args()

    rows = cl.collect(a.root)
    if not rows:
        raise SystemExit(f"No images under {a.root}/images/{{train,val,test}}")
    print(f"hashing {len(rows)} images ...")
    hashes = [cl.dhash(p) for _, p in rows]
    pairs = all_pairs(hashes, a.max_dist)
    cross = [(i, j, d) for i, j, d in pairs if rows[i][0] != rows[j][0]]

    a.out.mkdir(parents=True, exist_ok=True)
    made = []
    for name, (lo, hi) in BUCKETS.items():
        sel = [p for p in cross if lo <= p[2] <= hi]
        if contact_sheet(sel, rows, a.out / f"pairs_{name}.jpg", a.per_sheet):
            made.append(f"{name} ({len(sel)} pairs)")

    groups = scene_groups(len(rows), pairs)
    with open(a.out / "groups.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "image", "group"])
        for (split, p), g in zip(rows, groups):
            w.writerow([split, p.name, g])

    sizes = Counter(groups)
    n_groups = len(sizes)
    largest = sizes.most_common(5)
    singletons = sum(1 for s in sizes.values() if s == 1)
    buckets = Counter("1" if s == 1 else "2-5" if s <= 5 else "6-20" if s <= 20 else "21-100" if s <= 100 else ">100"
                      for s in sizes.values())
    splits_of = {}
    for (split, _), g in zip(rows, groups):
        splits_of.setdefault(g, set()).add(split)
    mixed = sum(1 for s in splits_of.values() if len(s) > 1)
    lines = [f"# Scene groups (dHash <= {a.max_dist}, any split)", "",
             f"Images: {len(rows)}; pairs within threshold: {len(pairs)} (cross-split {len(cross)})",
             f"Groups: {n_groups} (singletons {singletons}); groups spanning >1 split: {mixed}",
             "Group-size histogram: " + ", ".join(f"{k}: {buckets[k]}" for k in ["1", "2-5", "6-20", "21-100", ">100"]),
             "Largest groups (id: size): " + ", ".join(f"{g}: {s}" for g, s in largest),
             f"Largest group = {100 * largest[0][1] / len(rows):.1f}% of all images", "",
             "Contact sheets: " + (", ".join(made) or "none")]
    (a.out / "groups_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nOpen the pairs_*.jpg sheets in {a.out} and look at them. They are not committed (privacy).")


if __name__ == "__main__":
    main()
