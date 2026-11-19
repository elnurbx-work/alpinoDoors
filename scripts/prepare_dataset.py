"""Etiketlənmiş şəkilləri YOLO formatına hazırlayır və train/val bölür.

Etiketləmə: CVAT / Label Studio / Roboflow-da hər boşluğu 4 nöqtəli POLİQON kimi çəkin
(TL -> TR -> BR -> BL, istənilən sıra da olar — skript özü sıralayır) və "YOLO segmentation"
formatında export edin. Bu skript:
  * --task opening : poliqonları 4 künclü POSE etiketlərinə çevirir (Model 1)
  * --task door    : poliqonları olduğu kimi SEG etiketi saxlayır (Model 2)

İstifadə:
  python scripts/prepare_dataset.py --task opening --src export_dir --out data/opening
  (export_dir içində images/ və labels/ qovluqları olmalıdır)
"""
import argparse
import random
import shutil
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from alpino.geometry import mask_to_quad, order_corners  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def poly_to_pose(line: str) -> str | None:
    v = line.split()
    cls, pts = int(v[0]), np.array(v[1:], np.float32).reshape(-1, 2)
    if len(pts) < 4:
        return None
    if len(pts) == 4:
        q = order_corners(pts)
    else:  # çox nöqtəli poliqon (məs. tağlı boşluq) -> 4 künc
        import cv2
        S = 2000
        m = np.zeros((S, S), np.uint8)
        cv2.fillPoly(m, [(pts * S).astype(np.int32)], 1)
        q = mask_to_quad(m) / S
    x0, y0 = q.min(0)
    x1, y1 = q.max(0)
    box = [(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]
    kps = " ".join(f"{x:.6f} {y:.6f} 2" for x, y in np.clip(q, 0, 1))
    return f"{cls} " + " ".join(f"{b:.6f}" for b in box) + " " + kps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["opening", "door"], required=True)
    ap.add_argument("--src", required=True, help="images/ və labels/ olan export qovluğu")
    ap.add_argument("--out", required=True)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    src, out = Path(a.src), Path(a.out)
    imgs = sorted(p for p in (src / "images").rglob("*") if p.suffix.lower() in IMG_EXT)
    random.Random(a.seed).shuffle(imgs)
    n_val = max(1, int(len(imgs) * a.val))
    n_ok = 0
    for i, img in enumerate(imgs):
        split = "val" if i < n_val else "train"
        lbl = src / "labels" / (img.stem + ".txt")
        if not lbl.exists():
            continue
        lines = [ln.strip() for ln in lbl.read_text().splitlines() if ln.strip()]
        if a.task == "opening":
            lines = [r for r in (poly_to_pose(ln) for ln in lines) if r]
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, out / "images" / split / img.name)
        (out / "labels" / split / (img.stem + ".txt")).write_text("\n".join(lines) + "\n")
        n_ok += 1
    print(f"{n_ok} şəkil hazırlandı -> {out}  (val: {n_val})")


if __name__ == "__main__":
    main()
