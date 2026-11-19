"""Veb servis: boşluq şəkli + ölçü + qapı modeli -> vizual.

  uvicorn app.server:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from alpino import DoorCutter, OpeningDetector, get_door, list_doors, place_door, save_template  # noqa: E402

app = FastAPI(title="Alpino Doors – vizualizasiya")
opening = OpeningDetector(ROOT / "weights" / "opening.pt")
cutter = DoorCutter(ROOT / "weights" / "door.pt")
MAX_SIDE = 2400


async def _read(file: UploadFile) -> np.ndarray:
    data = await file.read()
    try:
        im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    except Exception:
        raise HTTPException(400, "Şəkil oxunmadı")
    im.thumbnail((MAX_SIDE, MAX_SIDE))
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def _default_quad(w, h):
    return [[w * .3, h * .2], [w * .7, h * .2], [w * .7, h * .9], [w * .3, h * .9]]


@app.get("/")
def index():
    return FileResponse(ROOT / "app" / "static" / "index.html")


@app.get("/api/doors")
def doors():
    return {"doors": list_doors(), "opening_model": opening.ready, "door_model": cutter.ready}


@app.post("/api/detect")
async def detect(photo: UploadFile = File(...)):
    img = await _read(photo)
    h, w = img.shape[:2]
    found = opening.detect(img)
    if found:
        return {"quad": found[0]["quad"], "auto": True, "score": found[0]["score"], "size": [w, h]}
    return {"quad": _default_quad(w, h), "auto": False, "size": [w, h]}


@app.post("/api/render")
async def render(photo: UploadFile = File(...), quad: str = Form(...), door_id: str = Form("double_bar"),
                 width_mm: float = Form(...), height_mm: float = Form(...), inset_mm: float = Form(0),
                 glass: str = Form(""), annotate: bool = Form(False)):
    img = await _read(photo)
    q = np.float32(json.loads(quad))
    if q.shape != (4, 2):
        raise HTTPException(400, "quad 4 nöqtə olmalıdır")
    if not (200 <= width_mm <= 6000 and 500 <= height_mm <= 4000):
        raise HTTPException(400, "Ölçülər mm ilə olmalıdır (məs. 1588 x 2518)")
    try:
        door = get_door(door_id, width_mm, height_mm, overrides={"glass": glass} if glass else None)
    except KeyError as e:
        raise HTTPException(404, str(e))
    out = place_door(img, q, door, width_mm, height_mm, inset_mm=inset_mm, annotate=annotate)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return Response(buf.tobytes(), media_type="image/jpeg")


@app.post("/api/template")
async def template(photo: UploadFile = File(...), name: str = Form(...), quad: str = Form(""),
                   width_mm: float | None = Form(None), height_mm: float | None = Form(None)):
    """Instagram/kataloq şəklindən yeni qapı şablonu (Model 2)."""
    img = await _read(photo)
    q = json.loads(quad) if quad else None
    try:
        rgba = cutter.cut(img, quad=q, width_mm=width_mm, height_mm=height_mm)
    except ValueError as e:
        raise HTTPException(422, str(e))
    door_id = re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_") or "door"
    save_template(door_id, rgba, name, width_mm=width_mm, height_mm=height_mm)
    return {"id": door_id, "name": name}
