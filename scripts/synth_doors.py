"""Model 2 üçün avtomatik (sintetik) dataset: parametrik qapıları real interyer şəkillərinin
üzərinə təsadüfi perspektivlə qoyur və seq etiketlərini özü yazır. Əl ilə etiketlənmiş
şəkillərlə birlikdə istifadə edin — modeli sürətlə "isitmək" üçündür.

  python scripts/synth_doors.py --bg fotolar/ --out data/door_synth --n 500
"""
import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from alpino import PRESETS, DoorStyle, place_door, render_door  # noqa: E402


def rand_quad(w, h, rng):
    dw = rng.uniform(0.25, 0.6) * w
    dh = rng.uniform(0.5, 0.85) * h
    x0 = rng.uniform(0.05 * w, w - dw - 0.05 * w)
    y0 = rng.uniform(0.05 * h, h - dh - 0.02 * h)
    j = lambda s: rng.uniform(-s, s)  # noqa: E731
    k = 0.06
    return np.float32([[x0 + j(k * dw), y0 + j(k * dh)], [x0 + dw + j(k * dw), y0 + j(k * dh)],
                       [x0 + dw + j(k * dw), y0 + dh + j(k * dh)], [x0 + j(k * dw), y0 + dh + j(k * dh)]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--val", type=float, default=0.15)
    a = ap.parse_args()
    rng = random.Random(0)
    bgs = [p for p in Path(a.bg).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    presets = list(PRESETS.values())
    out = Path(a.out)
    for i in range(a.n):
        split = "val" if i < a.n * a.val else "train"
        img = cv2.imread(str(rng.choice(bgs)))
        h, w = img.shape[:2]
        st = DoorStyle(**{**rng.choice(presets).to_dict(),
                          "glass": rng.choice(["clear", "clear", "bronze", "fluted", "frosted"]),
                          "h_bars": sorted(rng.sample([0.2, 0.33, 0.42, 0.5, 0.66, 0.75], rng.randint(0, 3)))})
        wmm, hmm = rng.uniform(700, 2000), rng.uniform(2000, 2800)
        door = render_door(wmm, hmm, st, px_per_mm=0.4)
        q = rand_quad(w, h, rng)
        comp = place_door(img, q, door, wmm, hmm, annotate=False)
        # maska: qapının alfa kanalı perspektivlə
        M = cv2.getPerspectiveTransform(np.float32([[0, 0], [door.shape[1], 0], [door.shape[1], door.shape[0]],
                                                    [0, door.shape[0]]]), q)
        m = cv2.warpPerspective((door[..., 3] > 0).astype(np.uint8), M, (w, h))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = cv2.approxPolyDP(max(cnts, key=cv2.contourArea), 2, True).reshape(-1, 2)
        poly = " ".join(f"{x / w:.5f} {y / h:.5f}" for x, y in c)
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / "images" / split / f"s{i:05d}.jpg"), comp, [cv2.IMWRITE_JPEG_QUALITY, 90])
        (out / "labels" / split / f"s{i:05d}.txt").write_text(f"0 {poly}\n")
    print(f"{a.n} sintetik şəkil -> {out}")


if __name__ == "__main__":
    main()
