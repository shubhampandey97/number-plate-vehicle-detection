"""Build a leak-free, seeded, group-aware train/val/test split (split v2).

Why: the original split put copies of the same frame (the merge included the
same data 2-3 times, once as .jpg and once as .png) and consecutive video
frames of the same fixed camera into train, val and test. See docs/leakage/.

Steps
1. Hash every image in data/final_merged_datasets (all old splits pooled).
2. Duplicate copies: images with dHash distance 0 form one "frame cluster";
   keep one (lowest file index), drop the rest.
3. Scenes: union-find over pairs with distance <= --max-dist (default 4).
   A scene (e.g. one fixed traffic camera) never spans two splits.
4. Assign whole scenes to train/val/test (default 70/15/15 of images),
   largest first, seeded, balancing image counts and per-class box counts.
5. Copy files to --out ({images,labels}/{train,val,test} + data.yaml) and
   write docs/leakage/split_v2.csv + split_v2_report.md.
6. Verify: no pair across the new splits is within --max-dist.

CPU-light. Run from the repo root:
    python tools/make_group_split.py
Then:
    python tools/check_leakage.py --root data/final_v2 --out docs/leakage/v2
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_leakage as cl  # noqa: E402
import inspect_leakage as il  # noqa: E402

SPLITS = ("train", "val", "test")


def label_path(root: Path, split: str, img: Path) -> Path:
    return root / "labels" / split / f"{img.stem}.txt"


def class_counts(lbl: Path) -> Counter:
    c = Counter()
    if lbl.exists():
        for line in lbl.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 5:
                c[int(float(parts[0]))] += 1
    return c


def assign_groups(groups: dict, fracs: dict, seed: int) -> dict:
    """groups: gid -> {"n": images, "cls": Counter}. Returns gid -> split.

    Greedy: biggest group first (seeded tie-break); each goes to the split
    with the largest relative shortfall in images plus the classes it holds.
    """
    rng = random.Random(seed)
    order = list(groups)
    rng.shuffle(order)
    order.sort(key=lambda g: -groups[g]["n"])
    tot_n = sum(g["n"] for g in groups.values())
    tot_c = Counter()
    for g in groups.values():
        tot_c.update(g["cls"])
    cur_n = {s: 0 for s in fracs}
    cur_c = {s: Counter() for s in fracs}
    out = {}
    for gid in order:
        g = groups[gid]
        best, best_score = None, None
        for s, f in fracs.items():
            tgt = f * tot_n
            score = (tgt - cur_n[s]) / max(tgt, 1e-9)
            if g["cls"]:
                score += sum((f * tot_c[c] - cur_c[s][c]) / max(f * tot_c[c], 1e-9)
                             for c in g["cls"]) / len(g["cls"])
            if best_score is None or score > best_score:
                best, best_score = s, score
        out[gid] = best
        cur_n[best] += g["n"]
        cur_c[best].update(g["cls"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/final_merged_datasets"))
    ap.add_argument("--out", type=Path, default=Path("data/final_v2"))
    ap.add_argument("--max-dist", type=int, default=4)
    ap.add_argument("--fracs", default="0.70,0.15,0.15")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--docs", type=Path, default=Path("docs/leakage"))
    a = ap.parse_args()

    if a.out.exists() and any(a.out.iterdir()):
        raise SystemExit(f"{a.out} is not empty - delete it first if you want to rebuild.")
    fracs = dict(zip(SPLITS, map(float, a.fracs.split(","))))
    cfg = yaml.safe_load(open(a.root / "data.yaml"))

    rows = cl.collect(a.root)
    print(f"hashing {len(rows)} images ...")
    hashes = [cl.dhash(p) for _, p in rows]
    pairs = il.all_pairs(hashes, a.max_dist)

    # 1) exact-frame clusters (distance 0): keep the lowest-numbered file
    frames = il.scene_groups(len(rows), [p for p in pairs if p[2] == 0])
    keep_of = {}
    for i, f in enumerate(frames):
        if f not in keep_of or rows[i][1].name < rows[keep_of[f]][1].name:
            keep_of[f] = i
    kept = sorted(set(keep_of.values()))
    kept_set = set(kept)
    dropped = [i for i in range(len(rows)) if i not in kept_set]

    # 2) scenes over kept images
    idx = {old: new for new, old in enumerate(kept)}
    kpairs = [(idx[i], idx[j], d) for i, j, d in pairs if i in idx and j in idx]
    scenes = il.scene_groups(len(kept), kpairs)

    groups = defaultdict(lambda: {"n": 0, "cls": Counter()})
    cls_of = {}
    for k, i in enumerate(kept):
        split, img = rows[i]
        cls_of[k] = class_counts(label_path(a.root, split, img))
        groups[scenes[k]]["n"] += 1
        groups[scenes[k]]["cls"].update(cls_of[k])
    split_of_group = assign_groups(dict(groups), fracs, a.seed)

    # 3) verify: no near-duplicate across new splits
    new_split = [split_of_group[scenes[k]] for k in range(len(kept))]
    leaks = cl.find_cross_split_pairs(new_split, [hashes[i] for i in kept], [""] * len(kept), a.max_dist)
    if leaks:
        raise SystemExit(f"BUG: {len(leaks)} cross-split pairs remain - not writing anything.")

    # 4) write files
    for s in SPLITS:
        (a.out / "images" / s).mkdir(parents=True, exist_ok=True)
        (a.out / "labels" / s).mkdir(parents=True, exist_ok=True)
    a.docs.mkdir(parents=True, exist_ok=True)
    with open(a.docs / "split_v2.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "old_split", "status", "frame_cluster", "scene", "new_split"])
        for k, i in enumerate(kept):
            split, img = rows[i]
            s = new_split[k]
            shutil.copy2(img, a.out / "images" / s / img.name)
            lbl = label_path(a.root, split, img)
            if lbl.exists():
                shutil.copy2(lbl, a.out / "labels" / s / lbl.name)
            w.writerow([img.name, split, "kept", frames[i], scenes[k], s])
        for i in dropped:
            split, img = rows[i]
            w.writerow([img.name, split, f"duplicate_of:{rows[keep_of[frames[i]]][1].name}", frames[i], "", ""])
    cfg2 = {"path": str(a.out.as_posix()), "train": "images/train", "val": "images/val",
            "test": "images/test", "names": cfg["names"]}
    (a.out / "data.yaml").write_text(yaml.safe_dump(cfg2, sort_keys=False))

    # 5) report
    names = cfg["names"]
    names = names if isinstance(names, dict) else dict(enumerate(names))
    per = {s: {"img": 0, "scenes": set(), "cls": Counter()} for s in SPLITS}
    for k in range(len(kept)):
        s = new_split[k]
        per[s]["img"] += 1
        per[s]["scenes"].add(scenes[k])
        per[s]["cls"].update(cls_of[k])
    old = Counter(r[0] for r in rows)
    lines = ["# Split v2: deduplicated, group-aware, seeded", "",
             f"Source: `{a.root}` ({len(rows)} images: {dict(old)}); seed {a.seed}; "
             f"scene threshold dHash <= {a.max_dist}; target {a.fracs}", "",
             f"- Duplicate copies removed (dHash 0, e.g. same frame as .jpg and .png): **{len(dropped)}**",
             f"- Unique images kept: **{len(kept)}**, in {len(set(scenes))} scenes",
             "- Cross-split pairs within threshold after the split: **0** (checked)", "",
             "| split | images | share | scenes | " + " | ".join(names[c] for c in sorted(names)) + " |",
             "|---|---|---|---|" + "---|" * len(names)]
    for s in SPLITS:
        p = per[s]
        lines.append(f"| {s} | {p['img']} | {100 * p['img'] / len(kept):.1f}% | {len(p['scenes'])} | "
                     + " | ".join(str(p["cls"][c]) for c in sorted(names)) + " |")
    (a.docs / "split_v2_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {a.out} and {a.docs / 'split_v2.csv'}")


if __name__ == "__main__":
    main()
