"""
yuz_ortak.py — Bölüm 04'ün temel taşları: yüz TESPİTİ (YuNet), HİZALAMA ve EMBEDDING (SFace).

YÜZ TANIMA HATTI (her yüz tanıma sistemi bu 4 adımdır — DeepFace de, InsightFace de):
    1. TESPİT   : Görüntüde yüz nerede?  YuNet -> kutu + 5 nokta (sağ göz, sol göz, burun, ağız köşeleri)
    2. HİZALAMA : Yüzü, gözler hep aynı yerde olacak şekilde döndür/ölçekle (112x112).
                  Neden? Model "eğik kafa" ile "başka kişi"yi karıştırmasın.
    3. EMBEDDING: Hizalanmış yüz -> 128 sayılık vektör (SFace). Aynı kişinin vektörleri birbirine YAKIN,
                  farklı kişilerinki UZAK olacak şekilde eğitilmiş (ArcFace tarzı kayıp fonksiyonu).
    4. KARŞILAŞTIRMA: İki vektör arası kosinüs benzerliği. > eşik ise "aynı kişi".
                  (SFace için OpenCV'nin önerdiği eşik ≈ 0.363; kendi verinle esik_analizi ile ayarla.)

NEDEN OpenCV YuNet + SFace?
    Ek pip paketi gerektirmez (sadece opencv) ve iki küçük ONNX dosyasıdır -> kurulumu ve içi şeffaf.
    Daha güçlü alternatif: InsightFace (buffalo_l / ArcFace). README'de nasıl değiştirileceği var.

KULLANAN: emotion_v2/ (yüz kırpma), yuz_tanima/ (kayıt + tanıma)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ortak.indir import model_dosyasi  # noqa: E402

DOSYA = "04_yuz/yuz_ortak.py"
_DEDEKTOR = None
_TANIYICI = None


def yuzleri_bul(kare: np.ndarray, guven: float = 0.7) -> np.ndarray:
    """YuNet ile yüzleri bul.

    Döndürür: (N, 15) dizi — her satır:
        [x, y, w, h,  sag_goz_x, sag_goz_y, sol_goz_x, sol_goz_y, burun_x, burun_y,
         agiz_sag_x, agiz_sag_y, agiz_sol_x, agiz_sol_y,  skor]
    (Bu ham format SFace'in alignCrop fonksiyonunun beklediği formattır.)
    """
    global _DEDEKTOR
    y, g = kare.shape[:2]
    if _DEDEKTOR is None:
        _DEDEKTOR = cv2.FaceDetectorYN.create(str(model_dosyasi("yunet")), "", (g, y), guven, 0.3, 5000)
    _DEDEKTOR.setInputSize((g, y))          # her karede boyut değişebilir (panelde farklı fotoğraflar)
    _DEDEKTOR.setScoreThreshold(guven)
    _, yuzler = _DEDEKTOR.detect(kare)
    return np.zeros((0, 15), np.float32) if yuzler is None else yuzler


def kutu_xyxy(yuz: np.ndarray) -> np.ndarray:
    """YuNet satırı -> [x1, y1, x2, y2]"""
    x, y, w, h = yuz[:4]
    return np.array([x, y, x + w, y + h])


def _taniyici():
    global _TANIYICI
    if _TANIYICI is None:
        _TANIYICI = cv2.FaceRecognizerSF.create(str(model_dosyasi("sface")), "")
    return _TANIYICI


def hizala(kare: np.ndarray, yuz: np.ndarray) -> np.ndarray:
    """5 noktayı kullanarak yüzü 112x112 standart pozisyona getir (SFace'in istediği girdi)."""
    return _taniyici().alignCrop(kare, yuz)


def embedding(hizali_yuz: np.ndarray) -> np.ndarray:
    """112x112 hizalı yüz -> L2 normalize edilmiş 128 boyutlu vektör."""
    v = _taniyici().feature(hizali_yuz).flatten().astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def kosinus(a: np.ndarray, b: np.ndarray) -> float:
    """İki normalize vektörün kosinüs benzerliği: 1 = aynı yön, 0 = ilgisiz, -1 = zıt."""
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))


def noktalari_ciz(img, yuz: np.ndarray):
    """5 yüz noktasını çiz (hizalamanın neye dayandığını görmek için)."""
    renkler = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 255), (0, 255, 255)]
    for i in range(5):
        cv2.circle(img, (int(yuz[4 + 2 * i]), int(yuz[5 + 2 * i])), 3, renkler[i], -1)
    return img


def yuz_kirp(kare: np.ndarray, yuz: np.ndarray, pay: float = 0.15, boyut: int = 224) -> np.ndarray:
    """Duygu modeli için: kutuyu biraz genişletip kare olarak kırp ve yeniden boyutlandır."""
    x1, y1, x2, y2 = kutu_xyxy(yuz)
    w, h = x2 - x1, y2 - y1
    kenar = max(w, h) * (1 + 2 * pay)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    a1, b1 = int(max(0, cx - kenar / 2)), int(max(0, cy - kenar / 2))
    a2, b2 = int(min(kare.shape[1], cx + kenar / 2)), int(min(kare.shape[0], cy + kenar / 2))
    kirpinti = kare[b1:b2, a1:a2]
    return cv2.resize(kirpinti, (boyut, boyut)) if kirpinti.size else np.zeros((boyut, boyut, 3), np.uint8)
