"""Açıq datasetləri Model 1 (boşluq, 4 künc) formatına çevirir — ilkin öyrətmə (pretrain) üçün.

Dəstəklənən:
  doordetect : https://github.com/MiguelARD/DoorDetect-Dataset (1213 şəkil, Open Images + MCIndoor)
               class 0 = door. Qutu -> divar xətlərinə yapışdırılmış 4 künc (alpino.zeroshot.refine_quad).

  git clone --depth 1 https://github.com/MiguelARD/DoorDetect-Dataset /tmp/DoorDetect
  python scripts/open_datasets.py doordetect --src /tmp/DoorDetect --out data/opening_open

Sonra: data/opening_open-u configs/opening.yaml-dakı yola qoyub `python scripts/train.py opening`,
yaxud öz datanızla birləşdirin (images/ və labels/ qovluqlarını kopyalamaq kifayətdir).
"""
import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from alpino.zeroshot import _edges, refine_quad  # noqa: E402


def pose_line(q, w, h):
    q = np.clip(np.asarray(q, np.float32) / [w, h], 0, 1)
    x0, y0 = q.min(0)
    x1, y1 = q.max(0)
    kps = " ".join(f"{x:.6f} {y:.6f} 2" for x, y in q)
    return f"0 {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {x1 - x0:.6f} {y1 - y0:.6f} {kps}"


def doordetect(src: Path, out: Path, val: float, neg: int, preview: int):
    files = sorted((src / "images").glob("*.jpg"))
    random.Random(0).shuffle(files)
    n_pos = n_neg = 0
    prev = []
    for i, f in enumerate(files):
        lbl = src / "labels" / (f.stem + ".txt")
        rows = [ln.split() for ln in lbl.read_text().splitlines() if ln.strip()] if lbl.exists() else []
        doors = [list(map(float, r[1:5])) for r in rows if r[0] == "0"]
        img = cv2.imread(str(f))
        if img is None:
            continue
        h, w = img.shape[:2]
        lines = []
        if doors:
            edges = _edges(img)
            for cx, cy, bw, bh in doors:
                if bh < 0.2:  # çox kiçik / uzaqdakı qapılar
                    continue
                box = np.float32([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
                q = refine_quad(edges, box)
                lines.append(pose_line(q, w, h))
                if len(prev) < preview:
                    prev.append((img, q))
        if not lines:
            if n_neg >= neg:
                continue
            n_neg += 1   # qapısız şəkillər: yalançı tapıntıları azaldır
        else:
            n_pos += 1
        split = "val" if (i % int(1 / val)) == 0 else "train"
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / "images" / split / f.name), img)
        (out / "labels" / split / (f.stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
    print(f"doordetect: {n_pos} qapılı + {n_neg} qapısız şəkil -> {out}")
    return prev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=["doordetect"])
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val", type=float, default=0.12)
    ap.add_argument("--neg", type=int, default=150, help="neçə qapısız (fon) şəkil əlavə olunsun")
    ap.add_argument("--preview", default="", help="yoxlama üçün nümunə vərəqi (jpg)")
    a = ap.parse_args()
    prev = doordetect(Path(a.src), Path(a.out), a.val, a.neg, 12 if a.preview else 0)
    if a.preview and prev:
        tiles = []
        for img, q in prev:
            t = img.copy()
            cv2.polylines(t, [np.int32(q)], True, (0, 220, 255), max(2, t.shape[1] // 200))
            tiles.append(cv2.resize(t, (300, 400)))
        rows = [np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles) - len(tiles) % 6, 6)]
        cv2.imwrite(a.preview, np.vstack(rows))


if __name__ == "__main__":
    main()
