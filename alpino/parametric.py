"""Parametrik qapı generatoru: qara metal profil + şüşə, dəqiq mm ölçüsü ilə.

Alpino-nun məhsulları (metal-şüşə arakəsmə qapıları) parametrikdir, ona görə
şəkildən kəsilmiş şablon olmasa da qapını ölçüyə tam uyğun çəkmək olur.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np


@dataclass
class DoorStyle:
    name: str = "Klassik tək"
    leaves: int = 0                 # 0 = avtomatik (en > 1200 mm olarsa 2 tay)
    leaf_ratios: list[float] = field(default_factory=list)  # qeyri-bərabər taylar, məs. [0.72, 0.28] = qapı + yan şüşə
    active: list[int] = field(default_factory=list)         # tutacaqlı (açılan) taylar; boş = hamısı
    arch: str = "none"              # none | full | leaf
    frame_mm: float = 40            # xarici çərçivə profili
    stile_mm: float = 30            # tayların görüşdüyü dik profil
    bar_mm: float = 25              # üfüqi/dik arakəsmə profilləri
    h_bars: list[float] = field(default_factory=lambda: [0.42])   # hündürlüyə nisbətən (yuxarıdan)
    v_bars: int = 0                 # hər tayda əlavə dik arakəsmə sayı
    bottom_panel: float = 0.0       # aşağıdan neçə hissə metal panel (0..0.6)
    color: tuple[int, int, int] = (28, 28, 30)        # RGB
    glass: str = "clear"            # clear | fluted | frosted | bronze
    handle: bool = True
    handle_style: str = "bar"       # bar (uzun) | short (qısa)

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
    # böyük / cüt
    "double_bar": DoorStyle(name="Cüt tay, 1 arakəsmə", leaves=2),
    "double_grid": DoorStyle(name="Cüt tay, şəbəkə", leaves=2, h_bars=[0.25, 0.5, 0.75], v_bars=1),
    "french_arch": DoorStyle(name="Fransız tağlı, cüt", leaves=2, arch="full", h_bars=[0.45], v_bars=1),
    "fluted_double": DoorStyle(name="Cüt tay, rifli şüşə", leaves=2, h_bars=[], glass="fluted"),
    "bronze_double": DoorStyle(name="Cüt tay, bronz şüşə", leaves=2, h_bars=[0.33, 0.66], glass="bronze"),
    "triple_bar": DoorStyle(name="Üç tay, ortası açılan", leaves=3, active=[1], h_bars=[0.42]),
    # tək / kiçik qapılar (700–1000 mm)
    "single_bar": DoorStyle(name="Tək tay, 1 arakəsmə", leaves=1),
    "single_clean": DoorStyle(name="Tək tay, təmiz şüşə", leaves=1, h_bars=[]),
    "single_grid": DoorStyle(name="Tək tay, şəbəkə", leaves=1, h_bars=[0.25, 0.5, 0.75], v_bars=1),
    "arch_single": DoorStyle(name="Tağlı tək", leaves=1, arch="leaf", h_bars=[0.5]),
    "single_slim": DoorStyle(name="Dar qapı, qısa tutacaq", leaves=1, h_bars=[], frame_mm=30, handle_style="short"),
    "single_transom": DoorStyle(name="Tək tay, üstü pəncərəli", leaves=1, h_bars=[0.2]),
    "single_sidelight": DoorStyle(name="Qapı + yan şüşə", leaves=2, leaf_ratios=[0.72, 0.28], active=[0], h_bars=[0.42]),
    "half_panel": DoorStyle(name="Tək tay, aşağısı metal", leaves=1, h_bars=[], bottom_panel=0.38),
    "bath_frosted": DoorStyle(name="Hamam qapısı, mat şüşə", leaves=1, h_bars=[], glass="frosted", handle_style="short"),
}


def _shape_mask(W, H, xs, arch):
    """xs: tayların sərhədləri (piksel), [0, ..., W]."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    m = np.ones((H, W), bool)
    if arch == "full":
        r = W / 2
        m &= (yy >= r) | ((xx - W / 2) ** 2 + (yy - r) ** 2 <= r * r)
    elif arch == "leaf":
        for a, b in zip(xs[:-1], xs[1:]):
            r = (b - a) / 2
            sel = (xx >= a) & (xx < b) & (yy < r)
            m &= ~sel | ((xx - (a + r)) ** 2 + (yy - r) ** 2 <= r * r)
    return m


def _dist(mask):
    return cv2.distanceTransform(np.pad(mask, 1).astype(np.uint8), cv2.DIST_L2, 5)[1:-1, 1:-1]


def render_door(width_mm: float, height_mm: float, style: DoorStyle | None = None,
                px_per_mm: float = 0.6) -> np.ndarray:
    """Qapını BGRA (H, W, 4) uint8 kimi çəkir. Şəkil ölçüsü = mm * px_per_mm."""
    st = style or DoorStyle()
    s = px_per_mm
    W, H = max(int(round(width_mm * s)), 8), max(int(round(height_mm * s)), 8)
    leaves = st.leaves or (2 if width_mm > 1200 else 1)
    ratios = st.leaf_ratios if len(st.leaf_ratios) == leaves else [1 / leaves] * leaves
    ratios = np.asarray(ratios, np.float64) / np.sum(ratios)
    xs = np.concatenate([[0], np.cumsum(ratios) * W])
    xs[-1] = W

    shape = _shape_mask(W, H, xs, st.arch)
    glass = _dist(shape) > st.frame_mm * s
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    for x in xs[1:-1]:  # tayların görüşdüyü profil
        glass &= np.abs(xx - x) > st.stile_mm * s / 2
    if st.arch == "leaf" and leaves > 1:  # hər tayın öz tağ çərçivəsi
        for a, b in zip(xs[:-1], xs[1:]):
            sub = (xx >= a) & (xx < b)
            lm = _shape_mask(W, H, np.array([a, b]), "leaf") & sub
            glass &= ~sub | (_dist(lm) > st.stile_mm * s / 2)
    for f in st.h_bars:
        glass &= np.abs(yy - f * H) > st.bar_mm * s / 2
    if st.arch in ("full", "leaf"):  # tağın dibində üfüqi xətt
        r = W / 2 if st.arch == "full" else (xs[1] - xs[0]) / 2
        glass &= np.abs(yy - r) > st.bar_mm * s / 2
    if st.v_bars:
        for a, b in zip(xs[:-1], xs[1:]):
            for k in range(1, st.v_bars + 1):
                glass &= np.abs(xx - (a + (b - a) * k / (st.v_bars + 1))) > st.bar_mm * s / 2
    if st.bottom_panel > 0:  # aşağı metal panel + üstündə ayırıcı xətt
        yb = H * (1 - min(st.bottom_panel, 0.6))
        glass &= yy < yb - st.bar_mm * s / 2

    profile = shape & ~glass
    rgb = np.zeros((H, W, 3), np.float32)
    alpha = np.zeros((H, W), np.float32)

    col = np.array(st.color, np.float32)
    bevel = np.clip(1.0 - _dist(profile) / max(3.0, 6 * s), 0, 1) * 18
    rgb[profile] = col + bevel[profile, None]
    alpha[profile] = 1.0
    if st.bottom_panel > 0:  # panel üzərində yüngül metal parıltısı
        panel = profile & (yy > H * (1 - st.bottom_panel))
        rgb[panel] += (np.sin(xx[panel] / W * np.pi) * 6)[:, None]

    tint, a = GLASS.get(st.glass, GLASS["clear"])
    refl = np.clip(np.sin(((xx / W) * 0.6 + (1 - yy / H) * 0.4) * np.pi * 1.3) * 0.5 + 0.5, 0, 1) * 0.10
    rgb[glass] = np.array(tint, np.float32)
    alpha[glass] = a + refl[glass]
    if st.glass == "fluted":
        flutes = np.sin(xx * 2 * np.pi / max(2.0, 25 * s)) * 0.5 + 0.5
        rgb[glass] *= (0.9 + 0.1 * flutes[glass, None])

    if st.handle:
        _handles(rgb, alpha, xs, H, s, st, leaves)

    bgr = np.clip(rgb[..., ::-1], 0, 255)  # RGB -> BGR (OpenCV)
    return np.dstack([bgr, np.clip(alpha, 0, 1) * 255]).astype(np.uint8)


def _handles(rgb, alpha, xs, H, s, st, leaves):
    active = st.active or list(range(leaves))
    w = max(2, int(14 * s))
    y0, y1 = (int(H * 0.40), int(H * 0.62)) if st.handle_style == "bar" else (int(H * 0.47), int(H * 0.53))
    for i in active:
        a, b = xs[i], xs[i + 1]
        # tutacaq qonşu tayın tərəfində; tək tayda — sağda
        if leaves == 1:
            x0 = int(b - (st.frame_mm + 95) * s)
        elif i < leaves - 1 and (i + 1 in active or i == 0):
            x0 = int(b - (st.stile_mm / 2 + 18) * s - w)
        else:
            x0 = int(a + (st.stile_mm / 2 + 18) * s)
        cv2.rectangle(rgb, (x0, y0), (x0 + w, y1), (15, 15, 15), -1)
        cv2.rectangle(alpha, (x0, y0), (x0 + w, y1), 1.0, -1)
        if st.handle_style == "short":  # qısa tutacaq: üfüqi qol
            L = int(110 * s)
            xa = x0 - L if x0 > (a + b) / 2 else x0
            cv2.rectangle(rgb, (xa, y0), (xa + L + w, y0 + w), (15, 15, 15), -1)
            cv2.rectangle(alpha, (xa, y0), (xa + L + w, y0 + w), 1.0, -1)
