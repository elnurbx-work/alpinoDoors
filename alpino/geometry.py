"""Həndəsə köməkçiləri: künclərin sıralanması, maskadan 4 küncün çıxarılması, mm <-> piksel."""
from __future__ import annotations

import cv2
import numpy as np


def order_corners(pts) -> np.ndarray:
    """4 nöqtəni TL, TR, BR, BL sırasına düzür."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = pts[:, 1] - pts[:, 0]
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    out = np.stack([tl, tr, br, bl])
    if len({tuple(p) for p in out.tolist()}) < 4:  # degenerat hal: bucağa görə sırala
        c = pts.mean(axis=0)
        ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
        out = pts[np.argsort(ang)]
        start = np.argmin(out.sum(axis=1))
        out = np.roll(out, -start, axis=0)
    return out.astype(np.float32)


def mask_to_quad(mask: np.ndarray) -> np.ndarray | None:
    """Binar maskadan (boşluq və ya qapı) ən yaxşı 4 künclü poliqonu tapır."""
    m = (mask > 0).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    peri = cv2.arcLength(hull, True)
    for eps in np.linspace(0.01, 0.1, 19):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            return order_corners(approx.reshape(4, 2))
    # Tağlı boşluqlar 4 küncə enmir: kənar nöqtələrdən dördbucaqlı qururuq
    pts = hull.reshape(-1, 2).astype(np.float32)
    return order_corners(_extreme_quad(pts))


def _extreme_quad(pts: np.ndarray) -> np.ndarray:
    s = pts.sum(axis=1)
    d = pts[:, 1] - pts[:, 0]
    tl, br = pts[np.argmin(s)], pts[np.argmax(s)]
    tr, bl = pts[np.argmin(d)], pts[np.argmax(d)]
    # tağ üçün: yuxarı künclər üst nöqtənin hündürlüyünə qaldırılır
    top = pts[:, 1].min()
    tl = np.array([tl[0], min(tl[1], top)])
    tr = np.array([tr[0], min(tr[1], top)])
    return np.stack([tl, tr, br, bl])


def mm_homography(quad: np.ndarray, width_mm: float, height_mm: float) -> np.ndarray:
    """mm müstəvisindən (0..W, 0..H) şəkil piksellərinə homoqrafiya."""
    src = np.float32([[0, 0], [width_mm, 0], [width_mm, height_mm], [0, height_mm]])
    return cv2.getPerspectiveTransform(src, order_corners(quad))


def project(H: np.ndarray, pts_mm) -> np.ndarray:
    pts = np.asarray(pts_mm, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def quad_aspect(quad: np.ndarray) -> float:
    """Dördbucaqlının təxmini en/hündürlük nisbəti (perspektiv nəzərə alınmadan)."""
    q = order_corners(quad)
    w = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
    h = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
    return float(w / max(h, 1e-6))
