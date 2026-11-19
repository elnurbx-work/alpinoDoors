"""Öyrətmədən boşluq tapmaq (Model 1 hazır olana qədər).

1. YOLOE (açıq lüğətli detektor, lüğət əvvəlcədən yazılıb: doorway / door opening / archway / door)
   bütün namizəd qutuları verir.
2. Ölçü şəkli üçün sıralama: böyük, hündür və mərkəzə yaxın boşluq önə çıxır.
3. Qutunun kənarları şəkildəki real xətlərə (divarın kənarı, boşluğun üstü) "yapışdırılır" —
   nəticədə perspektivli 4 künc alınır.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .geometry import order_corners


class ZeroShotOpening:
    def __init__(self, weights: str | Path = "weights/yoloe_doors.pt"):
        self.model = None
        if Path(weights).exists():
            from ultralytics import YOLO
            self.model = YOLO(str(weights))

    @property
    def ready(self):
        return self.model is not None

    def detect(self, img: np.ndarray, max_n: int = 3) -> list[dict]:
        if not self.ready:
            return []
        h, w = img.shape[:2]
        r = self.model.predict(img, conf=0.03, verbose=False, imgsz=640)[0]
        cands = []
        for b, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            bw, bh = b[2] - b[0], b[3] - b[1]
            if bh < 0.3 * h or bh / max(bw, 1) < 0.9 or bw * bh > 0.7 * w * h or bw < 0.1 * w:
                continue
            area = bw * bh / (w * h)
            cx = (b[0] + b[2]) / 2 / w
            score = float(np.sqrt(s)) * area ** 0.7 * max(0.05, 1.3 - abs(cx - 0.5) * 1.5)
            cands.append((score, b))
        cands.sort(key=lambda t: -t[0])
        keep = []
        for sc, b in cands:
            if all(_iou(b, k[1]) < 0.6 for k in keep):
                keep.append((sc, b))
        edges = _edges(img)
        out = []
        for sc, b in keep[:max_n]:
            out.append({"quad": refine_quad(edges, b).tolist(), "score": round(float(sc), 3)})
        return out


def _iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    return inter / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter, 1)


def _edges(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    v = np.median(g)
    return cv2.Canny(g, int(max(0, 0.66 * v)), int(min(255, 1.33 * v)))


def _best_line(edges, x0, y0, x1, y1, vertical: bool, min_len: float, with_end: bool = False):
    """Zolaq daxilində ən uzun (təxminən) dik/üfüqi xətti tapır: (a, b) — x=a*y+b və ya y=a*x+b."""
    H, W = edges.shape
    x0, x1 = int(max(0, x0)), int(min(W, x1))
    y0, y1 = int(max(0, y0)), int(min(H, y1))
    if x1 - x0 < 5 or y1 - y0 < 5:
        return None
    crop = edges[y0:y1, x0:x1]
    lines = cv2.HoughLinesP(crop, 1, np.pi / 360, threshold=40, minLineLength=int(min_len),
                            maxLineGap=25 if vertical else 12)
    if lines is None:
        return None
    best, bl = None, 0
    for xa, ya, xb, yb in lines.reshape(-1, 4).astype(float):
        dx, dy = xb - xa, yb - ya
        L = np.hypot(dx, dy)
        if vertical and abs(dx) > 0.15 * abs(dy):
            continue
        if not vertical and abs(dy) > 0.15 * abs(dx):
            continue
        if L > bl:
            bl, best = L, (xa + x0, ya + y0, xb + x0, yb + y0)
    if best is None:
        return None
    xa, ya, xb, yb = best
    if vertical:
        a = (xb - xa) / (yb - ya)
        return ((a, xa - a * ya), max(ya, yb)) if with_end else (a, xa - a * ya)
    a = (yb - ya) / (xb - xa)
    return a, ya - a * xa


def refine_quad(edges, box) -> np.ndarray:
    x0, y0, x1, y1 = [float(v) for v in box]
    bw, bh = x1 - x0, y1 - y0
    band_x, band_y = 0.12 * bw, 0.08 * bh
    H = edges.shape[0]
    ylo = min(H, y1 + 0.3 * bh)  # yan xətlər döşəməyə qədər davam edir — qutu çox vaxt qısa olur
    Le = _best_line(edges, x0 - band_x, y0, x0 + band_x, ylo, True, 0.35 * bh, with_end=True)
    Re = _best_line(edges, x1 - band_x, y0, x1 + band_x, ylo, True, 0.35 * bh, with_end=True)
    L, yl = Le if Le else (None, y1)
    R, yr = Re if Re else (None, y1)
    T = _best_line(edges, x0, y0 - band_y, x1, y0 + band_y, False, 0.35 * bw)
    L = L or (0.0, x0)
    R = R or (0.0, x1)
    T = T or (0.0, y0)
    yl, yr = max(yl, y1), max(yr, y1)
    if abs(yl - yr) > 0.08 * bh:  # bir tərəf etibarsızdırsa, digərinə uyğunlaşdır
        yl = yr = max(yl, yr)
    xl, xr = L[0] * yl + L[1], R[0] * yr + R[1]
    a = (yr - yl) / max(xr - xl, 1)
    B = (a, yl - a * xl)  # döşəmə xətti: yan xətlərin bitdiyi nöqtələrdən
    y1 = max(yl, yr)

    def cross(v, hz):  # x = a1*y + b1 ; y = a2*x + b2
        a1, b1 = v
        a2, b2 = hz
        y = (a2 * b1 + b2) / (1 - a1 * a2)
        return [a1 * y + b1, y]

    q = np.float32([cross(L, T), cross(R, T), cross(R, B), cross(L, B)])
    # ağlabatanlıq: qutudan çox uzaqlaşıbsa, qutunu qaytar
    bx = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    if np.abs(q - bx).max() > 0.2 * max(bw, bh):
        return bx
    return order_corners(q)


def snap_points(img: np.ndarray, n: int = 400) -> list[list[float]]:
    """Künc "maqniti" üçün güclü künc nöqtələri (brauzer nöqtəni buraxanda ən yaxınına yapışır)."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pts = cv2.goodFeaturesToTrack(g, maxCorners=n, qualityLevel=0.01, minDistance=max(6, img.shape[1] // 120),
                                  blockSize=7, useHarrisDetector=False)
    return [] if pts is None else pts.reshape(-1, 2).round(1).tolist()
