"""Veb servis: boşluq şəkli + ölçü + qapı modeli -> vizual.
Hər işlənmiş şəkil (künclərlə birlikdə) öyrətmə datasetinə yazılır.

  uvicorn app.server:app --host 0.0.0.0 --port 8000

Mühit dəyişənləri:
  ALPINO_DATASET  dataset qovluğu (default: ./dataset) — serverdə volume kimi qoşun
  ADMIN_TOKEN     /api/dataset/export üçün açar (verilməyibsə export bağlıdır)
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import sys
import time
import zipfile
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from alpino import DoorCutter, OpeningDetector, get_door, list_doors, place_door, save_template  # noqa: E402
from alpino.geometry import order_corners  # noqa: E402

app = FastAPI(title="Alpino Doors – vizualizasiya")
opening = OpeningDetector(ROOT / "weights" / "opening.pt")
cutter = DoorCutter(ROOT / "weights" / "door.pt")
MAX_SIDE = 2400
DATASET = Path(os.environ.get("ALPINO_DATASET", ROOT / "dataset"))
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")


async def _read(file: UploadFile) -> tuple[np.ndarray, str]:
    data = await file.read()
    try:
        im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    except Exception:
        raise HTTPException(400, "Şəkil oxunmadı")
    im.thumbnail((MAX_SIDE, MAX_SIDE))
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR), hashlib.sha1(data).hexdigest()[:16]


def _quad(quad: str) -> np.ndarray:
    try:
        q = np.float32(json.loads(quad))
    except Exception:
        raise HTTPException(400, "quad oxunmadı")
    if q.shape != (4, 2):
        raise HTTPException(400, "quad 4 nöqtə olmalıdır")
    return order_corners(q)


def _save_example(img, key, q, **meta):
    """Şəkil + künclər -> YOLO pose etiketi (Model 1 üçün öyrətmə nümunəsi)."""
    h, w = img.shape[:2]
    (DATASET / "images").mkdir(parents=True, exist_ok=True)
    (DATASET / "labels").mkdir(parents=True, exist_ok=True)
    (DATASET / "meta").mkdir(parents=True, exist_ok=True)
    ip = DATASET / "images" / f"{key}.jpg"
    if not ip.exists():
        cv2.imwrite(str(ip), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    n = np.clip(q / [w, h], 0, 1)
    x0, y0 = n.min(0)
    x1, y1 = n.max(0)
    kps = " ".join(f"{x:.6f} {y:.6f} 2" for x, y in n)
    (DATASET / "labels" / f"{key}.txt").write_text(
        f"0 {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {x1 - x0:.6f} {y1 - y0:.6f} {kps}\n")
    (DATASET / "meta" / f"{key}.json").write_text(json.dumps(
        {"quad": q.tolist(), "size": [w, h], "time": time.time(), **meta}, ensure_ascii=False))


def _count():
    d = DATASET / "labels"
    return len(list(d.glob("*.txt"))) if d.exists() else 0


@app.get("/")
def index():
    return FileResponse(ROOT / "app" / "static" / "index.html")


@app.get("/api/doors")
def doors():
    return {"doors": list_doors(), "opening_model": opening.ready, "door_model": cutter.ready,
            "dataset": _count()}


@lru_cache(maxsize=64)
def _preview(door_id: str) -> bytes:
    d = get_door(door_id, 1600, 2400)
    h = 360
    d = cv2.resize(d, (int(d.shape[1] * h / d.shape[0]), h), interpolation=cv2.INTER_AREA)
    bg = np.full(d.shape[:2] + (3,), 236, np.float32)
    a = d[..., 3:] / 255.0
    out = (d[..., :3] * a + bg * (1 - a)).astype(np.uint8)
    out = cv2.copyMakeBorder(out, 16, 16, 24, 24, cv2.BORDER_CONSTANT, value=(236, 236, 236))
    return cv2.imencode(".png", out)[1].tobytes()


@app.get("/api/door/{door_id}.png")
def door_preview(door_id: str):
    try:
        return Response(_preview(door_id), media_type="image/png",
                        headers={"Cache-Control": "public, max-age=86400"})
    except KeyError:
        raise HTTPException(404)


@app.post("/api/detect")
async def detect(photo: UploadFile = File(...)):
    img, _ = await _read(photo)
    h, w = img.shape[:2]
    found = opening.detect(img)
    if found:
        return {"quad": found[0]["quad"], "auto": True, "score": found[0]["score"], "size": [w, h]}
    return {"quad": [[w * .3, h * .2], [w * .7, h * .2], [w * .7, h * .9], [w * .3, h * .9]],
            "auto": False, "size": [w, h]}


@app.post("/api/render")
async def render(photo: UploadFile = File(...), quad: str = Form(...), door_id: str = Form("double_bar"),
                 width_mm: float = Form(...), height_mm: float = Form(...), inset_mm: float = Form(0),
                 glass: str = Form(""), annotate: bool = Form(False), save: bool = Form(True)):
    img, key = await _read(photo)
    q = _quad(quad)
    if not (200 <= width_mm <= 6000 and 500 <= height_mm <= 4000):
        raise HTTPException(400, "Ölçülər mm ilə olmalıdır (məs. 1588 x 2518)")
    try:
        door = get_door(door_id, width_mm, height_mm, overrides={"glass": glass} if glass else None)
    except KeyError as e:
        raise HTTPException(404, str(e))
    if save:
        _save_example(img, key, q, width_mm=width_mm, height_mm=height_mm, door_id=door_id)
    out = place_door(img, q, door, width_mm, height_mm, inset_mm=inset_mm, annotate=annotate)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return Response(buf.tobytes(), media_type="image/jpeg", headers={"X-Dataset-Count": str(_count())})


@app.post("/api/label")
async def label(photo: UploadFile = File(...), quad: str = Form(...),
                width_mm: float | None = Form(None), height_mm: float | None = Form(None)):
    """Yalnız etiketi saxla (vizual hazırlamadan) — sürətli dataset yığmaq üçün."""
    img, key = await _read(photo)
    _save_example(img, key, _quad(quad), width_mm=width_mm, height_mm=height_mm)
    return {"saved": key, "dataset": _count()}


@app.get("/api/dataset/export")
def export(token: str = ""):
    if not ADMIN_TOKEN or token != ADMIN_TOKEN:
        raise HTTPException(403, "İcazə yoxdur")

    def gen():
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
            for p in sorted(DATASET.rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(DATASET))
        buf.seek(0)
        yield from iter(lambda: buf.read(1 << 20), b"")

    name = time.strftime("alpino_dataset_%Y%m%d_%H%M.zip")
    return StreamingResponse(gen(), media_type="application/zip",
                             headers={"Content-Disposition": f'attachment; filename="{name}"'})


@app.post("/api/template")
async def template(photo: UploadFile = File(...), name: str = Form(...), quad: str = Form(""),
                   width_mm: float | None = Form(None), height_mm: float | None = Form(None)):
    """Instagram/kataloq şəklindən yeni qapı şablonu (Model 2)."""
    img, _ = await _read(photo)
    q = json.loads(quad) if quad else None
    try:
        rgba = cutter.cut(img, quad=q, width_mm=width_mm, height_mm=height_mm)
    except ValueError as e:
        raise HTTPException(422, str(e))
    door_id = re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_") or "door"
    save_template(door_id, rgba, name, width_mm=width_mm, height_mm=height_mm)
    return {"id": door_id, "name": name}
