"""Parametrik qapı generatoru: qara metal profil + şüşə, dəqiq mm ölçüsü ilə.

Alpino-nun məhsulları (metal-şüşə arakəsmə qapıları) parametrikdir, ona görə
şəkildən kəsilmiş şablon olmasa da qapını ölçüyə tam uyğun çəkmək olur.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import cv2
import numpy as np


@dataclass
class DoorStyle:
    name: str = "Klassik tək"
    leaves: int = 0                 # 0 = avtomatik (en > 1200 mm olarsa 2 tay)
    arch: str = "none"              # none | full | leaf
    frame_mm: float = 40            # xarici çərçivə profili
    stile_mm: float = 30            # tayların görüşdüyü dik profil
    bar_mm: float = 25              # üfüqi/dik arakəsmə profilləri
    h_bars: list[float] = field(default_factory=lambda: [0.42])   # hündürlüyə nisbətən (yuxarıdan)
    v_bars: int = 0                 # hər tayda əlavə dik arakəsmə sayı
    color: tuple[int, int, int] = (28, 28, 30)        # RGB
    glass: str = "clear"            # clear | fluted | frosted | bronze
    handle: bool = True

    def to_dict(self):
        return asdict(self)


GLASS = {
    #           RGB tint          alpha  (alpha = nə qədər örtür; 0 = tam şəffaf)
    "clear":   ((200, 210, 210), 0.18),
    "bronze":  ((120, 95, 70),   0.45),
    "fluted":  ((225, 230, 230), 0.70),
    "frosted": ((235, 238, 238), 0.85),
}

PRESETS = {
    "single_bar": DoorStyle(name="Tək tay, 1 arakəsmə", leaves=1),
    "single_clean": DoorStyle(name="Tək tay, təmiz şüşə", leaves=1, h_bars=[]),
    "double_bar": DoorStyle(name="Cüt tay, 1 arakəsmə", leaves=2),
    "double_grid": DoorStyle(name="Cüt tay, şəbəkə", leaves=2, h_bars=[0.25, 0.5, 0.75], v_bars=1),
    "french_arch": DoorStyle(name="Fransız tağlı, cüt", leaves=2, arch="full", h_bars=[0.45], v_bars=1),
    "arch_single": DoorStyle(name="Tağlı tək", leaves=1, arch="leaf", h_bars=[0.5]),
    "fluted_double": DoorStyle(name="Cüt tay, rifli şüşə", leaves=2, h_bars=[], glass="fluted"),
}


def _shape_mask(W, H, leaves, arch):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    m = np.ones((H, W), bool)
    if arch == "full":
        r = W / 2
        cy = r
        top = yy < cy
        m &= ~top | ((xx - W / 2) ** 2 + (yy - cy) ** 2 <= r * r)
    elif arch == "leaf":
        lw = W / leaves
        for i in range(leaves):
            r = lw / 2
            cx = lw * i + r
            sel = (xx >= lw * i) & (xx < lw * (i + 1)) & (yy < r)
            m &= ~sel | ((xx - cx) ** 2 + (yy - r) ** 2 <= r * r)
    return m


def render_door(width_mm: float, height_mm: float, style: DoorStyle | None = None,
                px_per_mm: float = 0.6) -> np.ndarray:
    """Qapını RGBA (H, W, 4) uint8 kimi çəkir. Şəkil ölçüsü = mm * px_per_mm."""
    st = style or DoorStyle()
    s = px_per_mm
    W, H = max(int(round(width_mm * s)), 8), max(int(round(height_mm * s)), 8)
    leaves = st.leaves or (2 if width_mm > 1200 else 1)

    shape = _shape_mask(W, H, leaves, st.arch)
    dist = cv2.distanceTransform(np.pad(shape, 1).astype(np.uint8), cv2.DIST_L2, 5)[1:-1, 1:-1]
    glass = dist > st.frame_mm * s

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    lw = W / leaves
    # tayların görüşdüyü profil
    for i in range(1, leaves):
        glass &= np.abs(xx - lw * i) > st.stile_mm * s / 2
    # leaf arch: hər tayın öz çərçivəsi
    if st.arch == "leaf" and leaves > 1:
        for i in range(leaves):
            sub = (xx >= lw * i) & (xx < lw * (i + 1))
            lm = _shape_mask(int(round(lw)), H, 1, "leaf")
            ld = cv2.distanceTransform(np.pad(lm, 1).astype(np.uint8), cv2.DIST_L2, 5)[1:-1, 1:-1]
            full = np.zeros_like(glass)
            x0 = int(round(lw * i))
            wcut = min(ld.shape[1], W - x0)
            full[:, x0:x0 + wcut] = ld[:, :wcut] > st.stile_mm * s / 2
            glass &= ~sub | full
    # üfüqi arakəsmələr
    for f in st.h_bars:
        y = f * H
        glass &= np.abs(yy - y) > st.bar_mm * s / 2
    # tağın dibində üfüqi xətt
    if st.arch in ("full", "leaf"):
        r = (W / 2) if st.arch == "full" else lw / 2
        glass &= np.abs(yy - r) > st.bar_mm * s / 2
    # dik arakəsmələr
    if st.v_bars:
        for i in range(leaves):
            for k in range(1, st.v_bars + 1):
                x = lw * i + lw * k / (st.v_bars + 1)
                glass &= np.abs(xx - x) > st.bar_mm * s / 2

    profile = shape & ~glass

    rgb = np.zeros((H, W, 3), np.float32)
    alpha = np.zeros((H, W), np.float32)

    # profil: metal, kənarlarda yüngül işıq
    col = np.array(st.color, np.float32)
    pd = cv2.distanceTransform(np.pad(profile, 1).astype(np.uint8), cv2.DIST_L2, 3)[1:-1, 1:-1]
    bevel = np.clip(1.0 - pd / max(3.0, 6 * s), 0, 1) * 18
    rgb[profile] = col + bevel[profile, None]
    alpha[profile] = 1.0

    # şüşə: tint + diaqonal əks
    tint, a = GLASS.get(st.glass, GLASS["clear"])
    refl = ((xx / W) * 0.6 + (1 - yy / H) * 0.4)
    refl = np.clip(np.sin(refl * np.pi * 1.3) * 0.5 + 0.5, 0, 1) * 0.10
    rgb[glass] = np.array(tint, np.float32)
    alpha[glass] = a + refl[glass]
    if st.glass == "fluted":
        flutes = (np.sin(xx * 2 * np.pi / max(2.0, 25 * s)) * 0.5 + 0.5)
        rgb[glass] *= (0.9 + 0.1 * flutes[glass, None])

    # tutacaq
    if st.handle:
        hx_list = [lw * i for i in range(1, leaves)] if leaves > 1 else [W - (st.frame_mm + 60) * s]
        for hx in hx_list:
            for side in ((-1, 1) if leaves > 1 else (1,)):
                x0 = int(hx + side * (st.stile_mm * s / 2 + 18 * s)) if leaves > 1 else int(hx)
                y0, y1 = int(H * 0.40), int(H * 0.62)
                w = max(2, int(14 * s))
                xa, xb = sorted((x0, x0 + side * w))
                cv2.rectangle(rgb, (xa, y0), (xb, y1), (15, 15, 15), -1)
                cv2.rectangle(alpha, (xa, y0), (xb, y1), 1.0, -1)

    bgr = np.clip(rgb[..., ::-1], 0, 255)  # RGB -> BGR (OpenCV)
    return np.dstack([bgr, np.clip(alpha, 0, 1) * 255]).astype(np.uint8)
