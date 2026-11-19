"""Qapını boşluğun üzərinə perspektivlə oturtmaq + sadə işıq/kölgə uyğunlaşdırması."""
from __future__ import annotations

import cv2
import numpy as np

from .geometry import mm_homography, order_corners, project


def place_door(photo: np.ndarray, quad, door_rgba: np.ndarray, width_mm: float, height_mm: float,
               inset_mm: float = 0.0, harmonize: bool = True, shadow: bool = True,
               annotate: bool = False) -> np.ndarray:
    """photo: BGR; quad: boşluğun 4 küncü (TL, TR, BR, BL) piksel ilə; door_rgba: BGRA şablon."""
    quad = order_corners(quad)
    h, w = photo.shape[:2]
    Hm = mm_homography(quad, width_mm, height_mm)

    # qapı mm müstəvisində: yanlardan və yuxarıdan inset, aşağıda döşəməyə qədər
    rect_mm = [[inset_mm, inset_mm], [width_mm - inset_mm, inset_mm],
               [width_mm - inset_mm, height_mm], [inset_mm, height_mm]]
    dst = project(Hm, rect_mm).astype(np.float32)

    door = _fit_template(door_rgba, dst)
    th, tw = door.shape[:2]
    src = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(door, M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    rgb = warped[..., :3].astype(np.float32)
    a = warped[..., 3:].astype(np.float32) / 255.0
    bg = photo.astype(np.float32)

    if harmonize:
        rgb = _match_light(rgb, a, bg, dst)

    if shadow:
        bg = _cast_shadow(bg, a, dst)

    out = rgb * a + bg * (1 - a)
    # kamera dənəvərliyi: qapı "yapışdırılmış" görünməsin
    noise = np.random.default_rng(0).normal(0, 1.6, out.shape[:2])[..., None]
    out = np.where(a > 0.05, out + noise, out)
    out = np.clip(out, 0, 255).astype(np.uint8)

    if annotate:
        out = draw_dimensions(out, quad, width_mm, height_mm)
    return out


def _fit_template(door, dst):
    """Şablonu təyinat ölçüsünə yaxın kiçildir (aliasing olmasın)."""
    tw = max(np.linalg.norm(dst[1] - dst[0]), np.linalg.norm(dst[2] - dst[3]))
    th = max(np.linalg.norm(dst[3] - dst[0]), np.linalg.norm(dst[2] - dst[1]))
    scale = min(1.0, 1.5 * max(tw / door.shape[1], th / door.shape[0]))
    if scale < 1.0:
        door = cv2.resize(door, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return door


def _match_light(rgb, a, bg, dst):
    """Boşluğun ətrafındakı divarın parlaqlığına görə qapının tonunu bir az düzəldir."""
    h, w = bg.shape[:2]
    poly = dst.astype(np.int32)
    inner = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(inner, poly, 1)
    k = max(15, int(0.08 * np.linalg.norm(dst[2] - dst[0])))
    ring = cv2.dilate(inner, np.ones((k, k), np.uint8)) - inner
    if ring.sum() < 50:
        return rgb
    scene = cv2.cvtColor(bg.astype(np.uint8), cv2.COLOR_BGR2LAB)
    L = scene[..., 0][ring > 0].mean()
    warm = scene[..., 2][ring > 0].mean() - 128  # sarı/göy çalarlığı
    gain = np.clip(L / 190.0, 0.75, 1.15)
    out = rgb * gain
    out[..., 2] += warm * 0.25  # R
    out[..., 0] -= warm * 0.25  # B
    return out


def _cast_shadow(bg, a, dst):
    """Profilin arxa fona düşən yumşaq kölgəsi + döşəmədə kontakt kölgəsi."""
    diag = np.linalg.norm(dst[2] - dst[0])
    solid = (a[..., 0] > 0.9).astype(np.float32)
    k = int(max(3, diag * 0.012)) | 1
    off = int(max(1, diag * 0.004))
    sh = cv2.GaussianBlur(solid, (k, k), 0)
    sh = np.roll(sh, (off, off), axis=(0, 1))
    shade = 1 - 0.28 * sh * (1 - solid)

    # döşəmə xətti boyunca kontakt kölgəsi
    h, w = bg.shape[:2]
    floor = np.zeros((h, w), np.float32)
    p0, p1 = tuple(dst[3].astype(int)), tuple(dst[2].astype(int))
    cv2.line(floor, p0, p1, 1.0, max(2, int(diag * 0.01)))
    floor = cv2.GaussianBlur(floor, (k * 2 + 1, k * 2 + 1), 0)
    shade *= 1 - 0.35 * floor
    return bg * shade[..., None]


def draw_dimensions(img, quad, width_mm, height_mm, color=(40, 40, 200)):
    out = img.copy()
    q = order_corners(quad).astype(int)
    t = max(2, img.shape[1] // 400)
    fs = img.shape[1] / 1300
    off = int(img.shape[1] * 0.025)
    # en
    a, b = q[0] - [0, off], q[1] - [0, off]
    cv2.arrowedLine(out, tuple(a), tuple(b), color, t, tipLength=0.03)
    cv2.arrowedLine(out, tuple(b), tuple(a), color, t, tipLength=0.03)
    mid = (a + b) // 2
    _label(out, f"{int(width_mm)}", (mid[0], mid[1] - 8), fs, color)
    # hündürlük
    a, b = q[0] - [off, 0], q[3] - [off, 0]
    cv2.arrowedLine(out, tuple(a), tuple(b), color, t, tipLength=0.02)
    cv2.arrowedLine(out, tuple(b), tuple(a), color, t, tipLength=0.02)
    mid = (a + b) // 2
    _label(out, f"{int(height_mm)}", (mid[0] - int(70 * fs), mid[1]), fs, color)
    return out


def _label(img, text, org, fs, color):
    th = max(1, int(fs * 2))
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, fs, color, th, cv2.LINE_AA)
