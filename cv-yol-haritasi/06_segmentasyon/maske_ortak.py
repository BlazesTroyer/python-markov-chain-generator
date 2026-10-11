"""
maske_ortak.py — Segmentasyonun ortak araçları: SAM ile istemli (prompt) maske, el maskesi, maske metrikleri.

SEGMENTASYON TÜRLERİ:
    Semantik  : Her piksele bir SINIF ("insan", "masa"). Aynı sınıftan iki nesne ayırt edilmez.
    Instance  : Her NESNEYE ayrı maske (insan#1, insan#2). YOLO-seg bunu yapar.
    İstemli (promptable): "Şu kutudaki / şu noktadaki şeyi" maskele. SAM bunu yapar — sınıf bilmez,
                "nesne" kavramını bilir. One-shot tarayıcı için ideal: nesnenin ne olduğunu bilmesine gerek yok.

SAM (Segment Anything) NASIL ÇALIŞIR? (kabaca)
    görüntü kodlayıcı (ağır, bir kez) -> görüntü embedding'i
    istem kodlayıcı (hafif)          -> kutu/nokta embedding'i
    maske çözücü                     -> istemdeki nesnenin maskesi
    Bu dosyada Ultralytics'in SAM sarmalayıcısı kullanılır (sam2.1_t.pt = en küçük SAM 2.1, ~78 MB).

MASKE METRİKLERİ (sıfırdan):
    IoU  = |A ∩ B| / |A ∪ B|
    Dice = 2|A ∩ B| / (|A| + |B|)       (IoU'ya göre küçük nesnelerde daha "nazik"; tıbbi görüntülemede yaygın)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ortak.yukle import gerekli  # noqa: E402

DOSYA = "06_segmentasyon/maske_ortak.py"
_SAM: dict[str, object] = {}


def sam_maske(kare: np.ndarray, kutu=None, nokta=None, model: str = "sam2.1_t.pt") -> np.ndarray:
    """Kutu ([x1,y1,x2,y2]) ya da nokta ([x,y]) istemiyle SAM maskesi -> bool (Y, G)."""
    ultra = gerekli("ultralytics", "pip install ultralytics")
    if model not in _SAM:
        _SAM[model] = ultra.SAM(model)
    arg = {"bboxes": [list(map(float, kutu))]} if kutu is not None else {"points": [list(map(float, nokta))], "labels": [1]}
    r = _SAM[model](kare, verbose=False, **arg)[0]
    if r.masks is None or len(r.masks.data) == 0:
        return np.zeros(kare.shape[:2], bool)
    m = r.masks.data[0].cpu().numpy().astype(bool)
    if m.shape != kare.shape[:2]:
        m = cv2.resize(m.astype(np.uint8), (kare.shape[1], kare.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    return m


def el_maskesi(el_noktalari: list[np.ndarray], boyut: tuple[int, int], genislet: int = 25) -> np.ndarray:
    """MediaPipe el noktalarından (normalize) elin kaba maskesi: dışbükey örtü (convex hull) + genişletme.

    Parmaklar arasındaki boşluklar da dahil olur — "eli görmezden gel" için güvenli taraf.
    """
    y, g = boyut
    m = np.zeros((y, g), np.uint8)
    for n in el_noktalari:
        p = (n[:, :2] * [g, y]).astype(np.int32)
        cv2.fillConvexPoly(m, cv2.convexHull(p), 1)
    if genislet > 0:
        m = cv2.dilate(m, np.ones((genislet, genislet), np.uint8))
    return m.astype(bool)


def kutu_from_maske(m: np.ndarray) -> np.ndarray | None:
    """Maskenin kapsayan kutusu [x1,y1,x2,y2] (copy-paste etiketleri böylece PİKSEL KESİN olur)."""
    ys, xs = np.where(m)
    if len(xs) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float32)


def en_buyuk_parca(m: np.ndarray) -> np.ndarray:
    """Maskedeki en büyük bağlı bileşeni bırak (el çıkarılınca kalan kırıntıları temizler)."""
    n, etiket, ist, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    if n <= 1:
        return m
    en = 1 + int(np.argmax(ist[1:, cv2.CC_STAT_AREA]))
    return etiket == en


def maske_iou(a: np.ndarray, b: np.ndarray) -> float:
    birlesim = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / birlesim) if birlesim else 0.0


def dice(a: np.ndarray, b: np.ndarray) -> float:
    toplam = a.sum() + b.sum()
    return float(2 * np.logical_and(a, b).sum() / toplam) if toplam else 0.0


def kesit_rgba(kare: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Maskelenmiş nesne -> kutuya kırpılmış BGRA (alfa = maske). copy_paste_aug bunu yapıştırır."""
    k = kutu_from_maske(m)
    if k is None:
        return np.zeros((1, 1, 4), np.uint8)
    x1, y1, x2, y2 = k.astype(int)
    bgra = cv2.cvtColor(kare[y1:y2, x1:x2], cv2.COLOR_BGR2BGRA)
    bgra[..., 3] = m[y1:y2, x1:x2].astype(np.uint8) * 255
    return bgra


def dama_uzerinde(bgra: np.ndarray, kare: int = 12) -> np.ndarray:
    """Saydam görüntüyü dama tahtası zemin üzerinde göster (saydamlık gözle görünsün)."""
    y, g = bgra.shape[:2]
    zemin = ((np.indices((y, g)) // kare).sum(0) % 2 * 60 + 150).astype(np.uint8)
    zemin = cv2.cvtColor(zemin, cv2.COLOR_GRAY2BGR)
    alfa = bgra[..., 3:4] / 255.0
    return (bgra[..., :3] * alfa + zemin * (1 - alfa)).astype(np.uint8)
