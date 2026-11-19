"""Avtomatik etiketləmə (Model 1 üçün): əl ilə etiket yazmadan boşluq datasetinin qurulması.

1. YOLOE (açıq lüğətli detektor) mətn ilə boşluq namizədlərini tapır: "doorway", "door opening"...
2. SAM 2 hər namizədin dəqiq maskasını çıxarır -> 4 künc (quad).
3. Nəticələr YOLO pose etiketi kimi yazılır + yoxlama üçün "contact sheet" (12 şəkil bir vərəqdə).
4. Yoxlamadan sonra pis olanlar --reject faylı ilə çıxarılır.

  python scripts/auto_label.py --src fotolar/ --out data/opening_auto
  python scripts/auto_label.py --src fotolar/ --out data/opening_auto --reject rejected.txt   # təkrar, pisləri atır

rejected.txt formatı: hər sətirdə şəkil adı (bütün şəkli at) və ya "ad:indeks" (yalnız o namizədi at).
"""
import argparse
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from alpino.geometry import mask_to_quad  # noqa: E402

PROMPTS = ["doorway", "door opening", "door", "archway", "door frame"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}
COLORS = [(0, 200, 255), (60, 220, 60), (255, 120, 0)]


def load_models(device=None):
    from ultralytics import SAM, YOLOE
    det = YOLOE("yoloe-11l-seg.pt")
    det.set_classes(PROMPTS, det.get_text_pe(PROMPTS))
    sam = SAM("sam2.1_b.pt")
    return det, sam


def _iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1)


def candidates(det, img, conf=0.05, max_n=3):
    h, w = img.shape[:2]
    r = det.predict(img, conf=conf, verbose=False)[0]
    out = []
    for b, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
        bw, bh = b[2] - b[0], b[3] - b[1]
        if bh < 0.25 * h or bh / max(bw, 1) < 0.8:      # boşluq hündür olur
            continue
        if bw * bh > 0.6 * w * h or bw > 0.85 * w or bw < 0.08 * w:  # bütün kadr və ya nazik zolaq
            continue
        cx = (b[0] + b[2]) / 2 / w
        area = bw * bh / (w * h)
        # ölçü şəkillərində əsas boşluq adətən böyük və mərkəzdədir
        score = float(s) * (0.3 + area) * (1.2 - abs(cx - 0.5))
        out.append((score, b))
    out.sort(key=lambda t: -t[0])
    keep = []
    for sc, b in out:
        if all(_iou(b, k[1]) < 0.5 for k in keep):
            keep.append((sc, b))
    return keep[:max_n]


def refine(sam, img, box):
    r = sam.predict(img, bboxes=[box.tolist()], verbose=False)[0]
    if r.masks is None or not len(r.masks):
        return None
    m = r.masks.data[0].cpu().numpy().astype(np.uint8)
    q = mask_to_quad(m)
    if q is None:
        return None
    # SAM maskası qutudan çox kənara çıxırsa, qutunun özündən istifadə et
    bx = np.float32([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]])
    if np.abs(q - bx).max() > 0.25 * (box[3] - box[1]):
        return bx
    return q


def pose_line(q, w, h):
    q = np.clip(q / [w, h], 0, 1)
    x0, y0 = q.min(0)
    x1, y1 = q.max(0)
    kps = " ".join(f"{x:.6f} {y:.6f} 2" for x, y in q)
    return f"0 {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {x1 - x0:.6f} {y1 - y0:.6f} {kps}"


def contact_sheets(items, out_dir, per=12, cols=4, tw=300):
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in range(0, len(items), per):
        tiles = []
        for name, img, quads in items[p:p + per]:
            s = tw / img.shape[1]
            t = cv2.resize(img, None, fx=s, fy=s)
            for i, q in enumerate(quads):
                c = COLORS[i % 3]
                cv2.polylines(t, [(q * s).astype(np.int32)], True, c, 2)
                cv2.putText(t, str(i), tuple(int(v) for v in (q[0] * s + 6)), cv2.FONT_HERSHEY_SIMPLEX, .7, c, 2)
            cv2.rectangle(t, (0, 0), (tw, 24), (0, 0, 0), -1)
            cv2.putText(t, name[:34], (4, 17), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
            tiles.append(t)
        th = max(t.shape[0] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, th - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles]
        while len(tiles) % cols:
            tiles.append(np.full_like(tiles[0], 255))
        rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
        cv2.imwrite(str(out_dir / f"sheet_{p // per:03d}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reject", help="atılacaq şəkillər/namizədlər siyahısı")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--max-side", type=int, default=1600)
    a = ap.parse_args()

    rej_img, rej_cand = set(), set()
    if a.reject and Path(a.reject).exists():
        for ln in Path(a.reject).read_text().split():
            (rej_cand if ":" in ln else rej_img).add(ln)

    det, sam = load_models()
    out = Path(a.out)
    files = sorted(p for p in Path(a.src).rglob("*") if p.suffix.lower() in IMG_EXT)
    random.Random(0).shuffle(files)
    review, rows = [], []
    for k, f in enumerate(files):
        if f.name in rej_img:
            continue
        img = cv2.imread(str(f))
        if img is None:
            continue
        s = a.max_side / max(img.shape[:2])
        if s < 1:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        quads = []
        for i, (sc, box) in enumerate(candidates(det, img)):
            q = refine(sam, img, box)
            if q is not None and f"{f.name}:{i}" not in rej_cand:
                quads.append(q)
                rows.append([f.name, i, round(sc, 3)])
        review.append((f.name, img, quads))
        if not quads:
            continue
        split = "val" if k < len(files) * a.val else "train"
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / "images" / split / (f.stem + ".jpg")), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (out / "labels" / split / (f.stem + ".txt")).write_text("\n".join(pose_line(q, w, h) for q in quads) + "\n")
        print(f"[{k + 1}/{len(files)}] {f.name}: {len(quads)} boşluq")

    contact_sheets(review, out / "review")
    with open(out / "review" / "candidates.csv", "w", newline="") as fh:
        csv.writer(fh).writerows([["image", "idx", "score"], *rows])
    print(f"Hazırdır. Yoxlama vərəqləri: {out / 'review'}")


if __name__ == "__main__":
    main()
