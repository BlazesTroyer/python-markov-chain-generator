"""
noktalar.py — MediaPipe ile el (21 nokta) ve vücut (33 nokta) anahtar noktaları + bunlarla yapılan HESAPLAR.

MEDIAPIPE NOTU:
    Eski API (mp.solutions.hands) yeni MediaPipe sürümlerinde KALDIRILDI. Burada her sürümde çalışan
    "Tasks" API'si kullanılır: model dosyası (.task) + HandLandmarker / PoseLandmarker.
    Model dosyaları ilk kullanımda veri/modeller/ altına iner (ortak/indir.py).

EL NOKTALARI (21):                         POZ NOKTALARI (kullandıklarımız):
    0 bilek                                    0 burun,  7/8 sol/sağ kulak
    1-4 başparmak (4 = uç)                     11/12 sol/sağ omuz, 13/14 dirsek, 15/16 bilek
    5-8 işaret (5 = kök, 8 = uç)               23/24 sol/sağ kalça, 25/26 diz, 27/28 ayak bileği
    9-12 orta, 13-16 yüzük, 17-20 serçe
    Koordinatlar 0..1 normalize (x: genişliğe, y: yüksekliğe oran). z: derinlik (göreli).

BU DOSYADAKİ SAF HESAPLAR (hepsi test edilir):
    el_normalize(n)   : Jest tanıma için konum/ölçek/el (sağ-sol) bağımsız özellik vektörü
    nokta_augment(d)  : Jest dizileri için augmentation (döndürme, ölçek, gürültü, zaman kaydırma)
    aci(a, b, c)      : b köşesindeki açı (derece) — tekrar sayacı ve ergonomi bunu kullanır
    dikeyle_aci(a, b) : a->b doğrusunun dikeyle yaptığı açı — duruş (eğilme) ölçümü
"""
from __future__ import annotations

import atexit
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ortak.indir import model_dosyasi  # noqa: E402
from ortak.yukle import gerekli  # noqa: E402

DOSYA = "05_el_poz/noktalar.py"

EL_BAGLANTILARI = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
                   (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
POZ_BAGLANTILARI = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
                    (24, 26), (26, 28), (0, 7), (0, 8)]

_EL, _POZ = {}, None


@atexit.register
def _kapat():
    """Program biterken MediaPipe modellerini düzgün kapat (yoksa çıkışta gürültülü hata basar)."""
    for m in list(_EL.values()) + ([_POZ] if _POZ is not None else []):
        try:
            m.close()
        except Exception:  # noqa: BLE001
            pass


# ============================================================================ MediaPipe sarmalayıcıları
def el_noktalari(kare: np.ndarray, en_fazla_el: int = 1) -> list[dict]:
    """Döndürür: [{'noktalar': (21,3) normalize, 'taraf': 'Left'|'Right', 'skor': float}, ...]

    Not: MediaPipe 'taraf'ı kameradaki görüntüye göre verir; aynalanmış (selfie) görüntüde ters olabilir.
    """
    mp = gerekli("mediapipe", "pip install mediapipe")
    from mediapipe.tasks import python as mp_gorev
    from mediapipe.tasks.python import vision
    if en_fazla_el not in _EL:
        secenek = vision.HandLandmarkerOptions(base_options=mp_gorev.BaseOptions(model_asset_path=str(model_dosyasi("el"))),
                                               num_hands=en_fazla_el, running_mode=vision.RunningMode.IMAGE)
        _EL[en_fazla_el] = vision.HandLandmarker.create_from_options(secenek)
    sonuc = _EL[en_fazla_el].detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(kare, cv2.COLOR_BGR2RGB)))
    eller = []
    for noktalar, taraf in zip(sonuc.hand_landmarks, sonuc.handedness):
        eller.append({"noktalar": np.array([[p.x, p.y, p.z] for p in noktalar], np.float32),
                      "taraf": taraf[0].category_name, "skor": float(taraf[0].score)})
    return eller


def poz_noktalari(kare: np.ndarray) -> np.ndarray | None:
    """İlk kişinin 33 noktası: (33, 4) -> x, y, z, gorunurluk. Kişi yoksa None."""
    global _POZ
    mp = gerekli("mediapipe", "pip install mediapipe")
    from mediapipe.tasks import python as mp_gorev
    from mediapipe.tasks.python import vision
    if _POZ is None:
        secenek = vision.PoseLandmarkerOptions(base_options=mp_gorev.BaseOptions(model_asset_path=str(model_dosyasi("poz"))),
                                               running_mode=vision.RunningMode.IMAGE)
        _POZ = vision.PoseLandmarker.create_from_options(secenek)
    sonuc = _POZ.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(kare, cv2.COLOR_BGR2RGB)))
    if not sonuc.pose_landmarks:
        return None
    return np.array([[p.x, p.y, p.z, p.visibility] for p in sonuc.pose_landmarks[0]], np.float32)


def iskelet_ciz(img, noktalar: np.ndarray, baglantilar, renk=(0, 255, 0), vurgula: tuple = ()):
    """Normalize noktaları piksele çevirip iskeleti çiz. 'vurgula' indeksleri büyük kırmızı çizilir."""
    y, g = img.shape[:2]
    p = (noktalar[:, :2] * [g, y]).astype(int)
    for a, b in baglantilar:
        cv2.line(img, tuple(p[a]), tuple(p[b]), renk, 2)
    for i, q in enumerate(p):
        cv2.circle(img, tuple(q), 6 if i in vurgula else 3, (0, 0, 255) if i in vurgula else (255, 255, 255), -1)
    return img


# ============================================================================ saf hesaplar
def el_normalize(n: np.ndarray, taraf: str = "Right") -> np.ndarray:
    """(21,3) el noktası -> (63,) konum/ölçek/taraf bağımsız özellik.

    1. Bileği (0) orijine taşı          -> elin ekrandaki YERİ önemsizleşir
    2. Bilek–orta parmak kökü (9) mesafesine böl -> elin kameraya UZAKLIĞI önemsizleşir
    3. Sol eli x ekseninde aynala        -> sağ ve sol el aynı jesti aynı vektörle verir
    Not: Dinamik jestlerde (kaydırma) hareketin kendisi önemli olduğundan jest_kaydet.py bilek
    konumunu ayrıca (normalize edilmemiş) ek özellik olarak saklar.
    """
    v = n - n[0]
    olcek = np.linalg.norm(v[9, :2]) + 1e-6
    v = v / olcek
    if taraf == "Left":
        v[:, 0] = -v[:, 0]
    return v.flatten().astype(np.float32)


def nokta_augment(dizi: np.ndarray, rng, guc: float = 1.0) -> np.ndarray:
    """Jest dizisi (T, F) için augmentation. F = 63 el özelliği + 2 bilek konumu (x, y).

    - Döndürme: el noktalarını xy düzleminde ±15° döndür (eli biraz eğik tutmak)
    - Ölçek   : ±10% (el boyu farkı)
    - Gürültü : küçük Gauss gürültüsü (MediaPipe titremesi)
    - Zaman   : diziyi birkaç kare kaydır (jesti biraz erken/geç başlatmak)
    """
    d = dizi.copy()
    T = d.shape[0]
    el = d[:, :63].reshape(T, 21, 3)
    aci_r = np.deg2rad(rng.uniform(-15, 15) * guc)
    R = np.array([[np.cos(aci_r), -np.sin(aci_r)], [np.sin(aci_r), np.cos(aci_r)]], np.float32)
    el[:, :, :2] = el[:, :, :2] @ R.T
    el *= 1 + rng.uniform(-0.1, 0.1) * guc
    el += rng.normal(0, 0.02 * guc, el.shape)
    d[:, :63] = el.reshape(T, 63)
    kay = int(rng.integers(-3, 4) * guc)
    if kay:
        d = np.roll(d, kay, axis=0)
        if kay > 0:
            d[:kay] = d[kay]
        else:
            d[kay:] = d[kay - 1]
    return d


def aci(a, b, c) -> float:
    """b köşesindeki açı (derece): a-b-c. Ör. aci(kalça, diz, ayak bileği) = diz açısı (düz bacak ≈ 180°)."""
    ba, bc = np.asarray(a[:2], float) - b[:2], np.asarray(c[:2], float) - b[:2]
    kos = np.dot(ba, bc) / max(np.linalg.norm(ba) * np.linalg.norm(bc), 1e-12)   # sıfıra bölmeyi önle, sonucu bozma
    return float(np.degrees(np.arccos(np.clip(kos, -1, 1))))


def dikeyle_aci(alt, ust) -> float:
    """alt->ust doğrusunun DİKEYLE yaptığı açı (derece). Dik duruş ≈ 0°, öne eğilme büyür.
    (Görüntüde y aşağı doğru artar; o yüzden 'yukarı' = -y yönü.)"""
    dx, dy = ust[0] - alt[0], ust[1] - alt[1]
    return float(np.degrees(np.arctan2(abs(dx), -dy))) if dy < 0 else 90.0 + float(np.degrees(np.arctan2(dy, abs(dx) + 1e-9)))
