"""Hər iki modeli sadə şəkildə öyrədir (transfer learning, kiçik YOLO).

  python scripts/train.py opening            # Model 1: boşluq (pose, 4 künc)
  python scripts/train.py door               # Model 2: qapı (seg)
  python scripts/train.py opening --epochs 150 --device 0 --size s

Nəticə: weights/opening.pt və weights/door.pt
"""
import argparse
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
TASK = {
    "opening": ("pose", ROOT / "configs/opening.yaml"),
    "door": ("seg", ROOT / "configs/door.yaml"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=TASK)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--size", default="n", help="n / s / m — böyüdükcə dəqiq, amma yavaş")
    ap.add_argument("--device", default=None, help="0 (GPU), cpu və s.")
    a = ap.parse_args()

    kind, cfg = TASK[a.model]
    # ultralytics nisbi yolları öz qovluğuna bağlayır — mütləq yol veririk
    d = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    d["path"] = str((cfg.parent / d["path"]).resolve())
    data = ROOT / "runs" / f"{a.model}_data.yaml"
    data.parent.mkdir(exist_ok=True)
    data.write_text(yaml.safe_dump(d, allow_unicode=True), encoding="utf-8")
    base = YOLO(f"yolo11{a.size}-{kind}.pt")  # COCO-da öncədən öyrədilmiş baza
    res = base.train(
        data=str(data), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
        project=str(ROOT / "runs"), name=a.model, exist_ok=True, patience=30,
        # sadə augmentasiyalar: fərqli işıq, bucaq, kəsik
        hsv_v=0.4, degrees=3, perspective=0.0005, fliplr=0.5, mosaic=0.5,
    )
    best = Path(res.save_dir) / "weights" / "best.pt"
    (ROOT / "weights").mkdir(exist_ok=True)
    shutil.copy2(best, ROOT / "weights" / f"{a.model}.pt")
    print(f"Hazırdır: weights/{a.model}.pt")


if __name__ == "__main__":
    main()
