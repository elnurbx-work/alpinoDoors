"""İki öyrədilən model:

1. OpeningDetector  – şəkildə qapı boşluğunu tapır, 4 küncünü qaytarır (YOLO pose, 4 keypoint).
2. DoorCutter       – məhsul şəklindən qapını kəsir, düz (frontal) şəffaf fonlu şablon çıxarır (YOLO seg).

Hər iki model `ultralytics` ilə öyrədilir (bax: scripts/train_*.py).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .geometry import mask_to_quad, order_corners, quad_aspect


def _load_yolo(weights):
    if not weights or not Path(weights).exists():
        return None
    from ultralytics import YOLO
    return YOLO(str(weights))


class OpeningDetector:
    """Model 1: qapı boşluğu. Pose modeli (4 künc) və ya seg modeli (maska) ilə işləyir."""

    def __init__(self, weights: str | Path | None = "weights/opening.pt", conf: float = 0.25):
        self.model = _load_yolo(weights)
        self.conf = conf

    @property
    def ready(self) -> bool:
        return self.model is not None

    def detect(self, img_bgr: np.ndarray) -> list[dict]:
        """[{quad: [[x,y]x4], score}] – ən inamlıdan başlayaraq."""
        if not self.ready:
            return []
        r = self.model.predict(img_bgr, conf=self.conf, verbose=False)[0]
        out = []
        if r.keypoints is not None and len(r.keypoints):
            kps = r.keypoints.xy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            for k, s in zip(kps, scores):
                out.append({"quad": order_corners(k[:4]).tolist(), "score": float(s)})
        elif r.masks is not None and len(r.masks):
            h, w = img_bgr.shape[:2]
            for m, s in zip(r.masks.data.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                q = mask_to_quad(m)
                if q is not None:
                    out.append({"quad": q.tolist(), "score": float(s)})
        return sorted(out, key=lambda d: -d["score"])


class DoorCutter:
    """Model 2: məhsul şəklindən qapını kəsib düzləndirilmiş BGRA şablon qaytarır."""

    def __init__(self, weights: str | Path | None = "weights/door.pt", conf: float = 0.25):
        self.model = _load_yolo(weights)
        self.conf = conf

    @property
    def ready(self) -> bool:
        return self.model is not None

    def mask(self, img_bgr: np.ndarray) -> np.ndarray | None:
        if not self.ready:
            return None
        r = self.model.predict(img_bgr, conf=self.conf, verbose=False)[0]
        if r.masks is None or not len(r.masks):
            return None
        i = int(r.boxes.conf.argmax())
        h, w = img_bgr.shape[:2]
        m = r.masks.data[i].cpu().numpy()
        return (cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR) > 0.5).astype(np.uint8)

    def cut(self, img_bgr, quad=None, width_mm=None, height_mm=None, glass_alpha: float | None = 0.25,
            height_px: int = 1400) -> np.ndarray:
        """Qapı şablonu (BGRA). quad verilməsə, model maskasından tapılır.

        glass_alpha: qara profil tam saxlanılır, açıq rəngli (şüşə) piksellər yarı-şəffaf olur ki,
        yeni otağın arxa planı şüşədən görünsün. None = hamısı qeyri-şəffaf.
        """
        mask = self.mask(img_bgr)
        if quad is None:
            if mask is None:
                raise ValueError("Qapı tapılmadı: ya model öyrədilməyib, ya da künclər əllə verilməlidir.")
            quad = mask_to_quad(mask)
        quad = order_corners(quad)
        aspect = (width_mm / height_mm) if (width_mm and height_mm) else quad_aspect(quad)
        H = height_px
        W = int(round(H * aspect))
        dst = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
        M = cv2.getPerspectiveTransform(quad, dst)
        flat = cv2.warpPerspective(img_bgr, M, (W, H), flags=cv2.INTER_CUBIC)
        if mask is not None:
            a = cv2.warpPerspective(mask * 255, M, (W, H), flags=cv2.INTER_LINEAR)
        else:
            a = np.full((H, W), 255, np.uint8)
        if glass_alpha is not None:
            a = (a.astype(np.float32) * glass_transparency(flat, glass_alpha)).astype(np.uint8)
        return np.dstack([flat, a])


def glass_transparency(flat_bgr: np.ndarray, glass_alpha: float = 0.25) -> np.ndarray:
    """Qara metal profil -> 1.0, şüşə (açıq) sahələr -> glass_alpha. Yumşaq keçidlə."""
    v = cv2.cvtColor(flat_bgr, cv2.COLOR_BGR2HSV)[..., 2].astype(np.float32)
    dark = np.clip((110 - v) / 50.0, 0, 1)          # V<60 tam profil, V>110 şüşə
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return glass_alpha + (1 - glass_alpha) * dark
