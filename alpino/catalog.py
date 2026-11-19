"""Qapı kataloqu: parametrik modellər + şəkildən kəsilmiş şablonlar (catalog/*.png + *.json)."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .parametric import PRESETS, DoorStyle, render_door

CATALOG_DIR = Path(__file__).resolve().parent.parent / "catalog"


def list_doors(catalog_dir: Path = CATALOG_DIR) -> list[dict]:
    items = [{"id": k, "name": v.name, "kind": "parametric"} for k, v in PRESETS.items()]
    for meta in sorted(catalog_dir.glob("*.json")):
        d = json.loads(meta.read_text(encoding="utf-8"))
        items.append({"id": meta.stem, "name": d.get("name", meta.stem), "kind": "photo"})
    return items


def get_door(door_id: str, width_mm: float, height_mm: float, catalog_dir: Path = CATALOG_DIR,
             overrides: dict | None = None) -> np.ndarray:
    """Seçilmiş qapını BGRA şablon kimi qaytarır."""
    if door_id in PRESETS:
        st = DoorStyle(**{**PRESETS[door_id].to_dict(), **(overrides or {})})
        return render_door(width_mm, height_mm, st)
    png = catalog_dir / f"{door_id}.png"
    if not png.exists():
        raise KeyError(f"Kataloqda belə qapı yoxdur: {door_id}")
    img = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
    if img.shape[2] == 3:
        img = np.dstack([img, np.full(img.shape[:2], 255, np.uint8)])
    return img


def save_template(door_id: str, rgba: np.ndarray, name: str, catalog_dir: Path = CATALOG_DIR, **meta):
    catalog_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(catalog_dir / f"{door_id}.png"), rgba)
    (catalog_dir / f"{door_id}.json").write_text(
        json.dumps({"name": name, **meta}, ensure_ascii=False, indent=2), encoding="utf-8")
