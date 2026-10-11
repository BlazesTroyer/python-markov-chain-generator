"""
yolo_yardimci.py — Ultralytics YOLO'yu kullanan bütün dosyaların paylaştığı küçük yardımcılar.

    model_yukle(yol)        : Modeli BİR KEZ yükler, sonra önbellekten verir (her karede yüklemek çok yavaş).
    sonucu_ayikla(r)        : Ultralytics'in Results nesnesini sade numpy dizilerine çevirir:
                              kutular (N,4) x1y1x2y2 piksel, siniflar (N,), guvenler (N,), isimler {no: ad}
    AUG_ON_AYARLARI         : "kapali / varsayilan / agir" augmentation profillerinin Ultralytics karşılığı.

Ultralytics'in çevrimiçi augmentation parametreleri (eğitimde her görüntüye anında uygulanır):
    hsv_h, hsv_s, hsv_v : renk tonu / doygunluk / parlaklık oynatma oranı
    degrees             : ± döndürme açısı
    translate           : ± kaydırma (görüntü boyuna oran)
    scale               : ± ölçekleme oranı
    fliplr              : yatay çevirme olasılığı
    mosaic              : 4 görüntüyü tek görüntüde birleştirme olasılığı (küçük nesne ve bağlam çeşitliliği)
    mixup               : iki görüntüyü saydam üst üste bindirme olasılığı
    close_mosaic        : son N epoch'ta mosaic'i kapat (model gerçek görüntü dağılımına alışsın)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from ortak.yukle import gerekli, yol_coz  # noqa: E402

AUG_ON_AYARLARI = {
    "kapali":    dict(hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=0.0, translate=0.0, scale=0.0,
                      fliplr=0.0, mosaic=0.0, mixup=0.0),
    "varsayilan": dict(hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5,
                       fliplr=0.5, mosaic=1.0, mixup=0.0),
    "agir":      dict(hsv_h=0.03, hsv_s=0.8, hsv_v=0.6, degrees=15.0, translate=0.2, scale=0.7,
                      fliplr=0.5, mosaic=1.0, mixup=0.2),
}

_ONBELLEK: dict[str, object] = {}


def model_yukle(yol: str):
    """YOLO modelini yükle (önbellekli).

    yol: 'yolo11n.pt' gibi hazır bir ad (ilk seferde internetten iner) ya da kendi best.pt yolun.
    """
    ultra = gerekli("ultralytics", "pip install ultralytics")
    p = yol_coz(yol) if ("/" in yol or "\\" in yol) else Path(yol)
    anahtar = str(p)
    if anahtar not in _ONBELLEK:
        _ONBELLEK[anahtar] = ultra.YOLO(anahtar)
    return _ONBELLEK[anahtar]


def sonucu_ayikla(r):
    """Ultralytics Results -> (kutular (N,4), siniflar (N,), guvenler (N,), isimler dict)."""
    if r.boxes is None or len(r.boxes) == 0:
        return np.zeros((0, 4), np.float32), np.zeros(0, int), np.zeros(0, np.float32), r.names
    return (r.boxes.xyxy.cpu().numpy().astype(np.float32),
            r.boxes.cls.cpu().numpy().astype(int),
            r.boxes.conf.cpu().numpy().astype(np.float32),
            r.names)
