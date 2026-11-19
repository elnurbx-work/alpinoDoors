"""Komanda sətrindən bir vizual hazırlamaq.

  python scripts/render.py --photo bosluq.jpg --width 1588 --height 2518 --door double_bar
  python scripts/render.py --photo bosluq.jpg --width 1588 --height 2518 --door french_arch \
         --quad "226,292 780,318 770,1140 245,1213" --annotate --out netice.jpg

--quad verilməsə, Model 1 (weights/opening.pt) boşluğu özü tapır.
"""
import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from alpino import OpeningDetector, get_door, place_door  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo", required=True)
    ap.add_argument("--width", type=float, required=True, help="boşluğun eni, mm")
    ap.add_argument("--height", type=float, required=True, help="boşluğun hündürlüyü, mm")
    ap.add_argument("--door", default="double_bar")
    ap.add_argument("--quad", help='"x,y x,y x,y x,y" (TL TR BR BL)')
    ap.add_argument("--inset", type=float, default=0)
    ap.add_argument("--annotate", action="store_true")
    ap.add_argument("--out", default="netice.jpg")
    a = ap.parse_args()

    img = cv2.imread(a.photo)
    if a.quad:
        quad = [[float(v) for v in p.split(",")] for p in a.quad.split()]
    else:
        det = OpeningDetector().detect(img)
        if not det:
            sys.exit("Boşluq tapılmadı. --quad ilə küncləri əllə verin və ya Model 1-i öyrədin.")
        quad = det[0]["quad"]
    door = get_door(a.door, a.width, a.height)
    out = place_door(img, quad, door, a.width, a.height, inset_mm=a.inset, annotate=a.annotate)
    cv2.imwrite(a.out, out, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("Yazıldı:", a.out)


if __name__ == "__main__":
    main()
