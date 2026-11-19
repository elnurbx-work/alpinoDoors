"""Instagram/kataloq şəklindən qapı şablonu çıxarıb kataloqa əlavə edir (Model 2 ilə).

  python scripts/make_template.py --photo post.jpg --id ms_cut_01 --name "Cüt tay, qara" \
         --width 1600 --height 2400
  # model yoxdursa, qapının 4 küncünü əllə verin:
  python scripts/make_template.py --photo post.jpg --id x --name X --quad "100,80 700,90 690,1300 110,1290"

Ən yaxşı nəticə üçün: qapı BAĞLI, düz qarşıdan çəkilmiş olsun.
"""
import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from alpino import DoorCutter, save_template  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo", required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--width", type=float)
    ap.add_argument("--height", type=float)
    ap.add_argument("--quad")
    ap.add_argument("--glass-alpha", type=float, default=0.25,
                    help="şüşənin örtücülüyü 0..1 (1 = şüşə şəffaf olmasın)")
    a = ap.parse_args()

    img = cv2.imread(a.photo)
    quad = [[float(v) for v in p.split(",")] for p in a.quad.split()] if a.quad else None
    rgba = DoorCutter().cut(img, quad=quad, width_mm=a.width, height_mm=a.height,
                            glass_alpha=None if a.glass_alpha >= 1 else a.glass_alpha)
    save_template(a.id, rgba, a.name, width_mm=a.width, height_mm=a.height, source=Path(a.photo).name)
    print(f"Kataloqa əlavə olundu: catalog/{a.id}.png")


if __name__ == "__main__":
    main()
