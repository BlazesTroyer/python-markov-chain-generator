"""
augment_sifirdan.py — Augmentation (veri çoğaltma) dönüşümlerini SIFIRDAN, numpy/OpenCV ile.

AUGMENTATION NEDİR?
    Elindeki eğitim görüntülerini rastgele ama GERÇEKÇİ şekilde değiştirip modelin
    "aynı nesnenin farklı hallerini" görmesini sağlamak. Model böylece:
        - ezberlemez (overfitting azalır),
        - farklı ışığa, açıya, bulanıklığa, kısmi kapanmaya dayanıklı olur.

    EN ÖNEMLİ KURAL: Görüntü geometrik olarak değişiyorsa (çevirme, döndürme, kırpma)
    ETİKETLER DE (kutular, keypoint'ler) AYNI ŞEKİLDE DEĞİŞMELİDİR. Aksi halde modele
    "nesne burada" diye yanlış yer öğretirsin. Bu dosyadaki her geometrik fonksiyon kutuyu
    da dönüştürür — nasıl yapıldığını satır satır görebilirsin.

İKİ TÜR DÖNÜŞÜM:
    Fotometrik (piksel değerleri değişir, konum aynı) -> kutu DEĞİŞMEZ
        parlaklik_kontrast, hsv_kaydir, bulaniklik, gurultu
    Geometrik (pikseller yer değiştirir)               -> kutu DA dönüştürülür
        yatay_cevir, dondur, olcekle_kaydir
    Kapatma (occlusion)                                -> kutu aynı, içi kısmen kapanır
        kesip_cikar (cutout)

KUTU FORMATI: numpy dizi (N, 4) -> her satır x1, y1, x2, y2 (piksel)
HER FONKSİYON: (img, kutular, rng, guc) -> (yeni_img, yeni_kutular, "ne yapıldı" metni)
    guc: 0..1 arası "ne kadar sert" (hafif ~0.3, ağır ~1.0)

KİM KULLANIR?
    00_temeller/augmentation_gorsellestir.py  (görselleştirme)
    01_veri_fabrikasi/augment_uret.py         (dataset çoğaltma — sıfırdan modu)
"""
from __future__ import annotations

import cv2
import numpy as np


# ============================================================================ fotometrik
def parlaklik_kontrast(img, kutular, rng, guc=0.5):
    """piksel' = alfa * piksel + beta.  alfa: kontrast, beta: parlaklık. Kutu değişmez."""
    alfa = 1.0 + rng.uniform(-0.4, 0.4) * guc       # 0.6..1.4 (guc=1)
    beta = rng.uniform(-50, 50) * guc               # -50..+50
    yeni = np.clip(img.astype(np.float32) * alfa + beta, 0, 255).astype(np.uint8)
    return yeni, kutular.copy(), f"parlaklik_kontrast(alfa={alfa:.2f}, beta={beta:+.0f})"


def hsv_kaydir(img, kutular, rng, guc=0.5):
    """Renk tonu (H), doygunluk (S), parlaklık (V) kaydırma. Farklı ışık rengini taklit eder."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    dh = int(rng.uniform(-10, 10) * guc)            # OpenCV'de H: 0..179
    ds = int(rng.uniform(-60, 60) * guc)
    dv = int(rng.uniform(-40, 40) * guc)
    hsv[..., 0] = (hsv[..., 0] + dh) % 180          # ton döngüseldir
    hsv[..., 1] = np.clip(hsv[..., 1] + ds, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] + dv, 0, 255)
    yeni = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return yeni, kutular.copy(), f"hsv_kaydir(dH={dh:+d}, dS={ds:+d}, dV={dv:+d})"


def bulaniklik(img, kutular, rng, guc=0.5):
    """Hareket bulanıklığı: tek yönlü çizgi çekirdeği ile evrişim (kamera sarsıntısı taklidi)."""
    k = max(3, int(3 + rng.integers(0, 12) * guc)) | 1   # tek sayı çekirdek boyu
    cekirdek = np.zeros((k, k), np.float32)
    if rng.random() < 0.5:
        cekirdek[k // 2, :] = 1.0                         # yatay
    else:
        cekirdek[:, k // 2] = 1.0                         # dikey
    cekirdek /= cekirdek.sum()
    return cv2.filter2D(img, -1, cekirdek), kutular.copy(), f"hareket_bulanikligi(k={k})"


def gurultu(img, kutular, rng, guc=0.5):
    """Gauss gürültüsü: karanlık ortamdaki sensör karıncalanması."""
    sigma = 25 * guc * rng.uniform(0.3, 1.0)
    yeni = np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)
    return yeni, kutular.copy(), f"gauss_gurultu(sigma={sigma:.1f})"


# ============================================================================ geometrik
def yatay_cevir(img, kutular, rng, guc=0.5):
    """Ayna görüntüsü. Kutu: x1' = G - x2,  x2' = G - x1  (sol-sağ yer değiştirir)."""
    g = img.shape[1]
    yeni_k = kutular.copy()
    yeni_k[:, 0] = g - kutular[:, 2]
    yeni_k[:, 2] = g - kutular[:, 0]
    return img[:, ::-1].copy(), yeni_k, "yatay_cevir()"


def _kutulari_matrisle_donustur(kutular, M, g, y):
    """Afin matris M (2x3) ile kutuları dönüştür.

    Dönmüş bir dikdörtgen artık eksen hizalı değildir. Bu yüzden kutunun 4 köşesini
    dönüştürüp, onları içine alan en küçük eksen hizalı kutuyu alırız. (Bu yüzden döndürülen
    kutular biraz 'şişer' — bilinen bir yan etkidir.)
    """
    yeni = []
    for x1, y1, x2, y2 in kutular:
        koseler = np.array([[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]], np.float32)  # (4,3)
        donmus = koseler @ M.T                                                                  # (4,2)
        nx1, ny1 = donmus.min(axis=0)
        nx2, ny2 = donmus.max(axis=0)
        yeni.append([np.clip(nx1, 0, g), np.clip(ny1, 0, y), np.clip(nx2, 0, g), np.clip(ny2, 0, y)])
    return np.array(yeni, np.float32).reshape(-1, 4)


def dondur(img, kutular, rng, guc=0.5):
    """Merkez etrafında açı kadar döndür; kutuları köşe dönüşümüyle güncelle."""
    y, g = img.shape[:2]
    aci = rng.uniform(-20, 20) * guc
    M = cv2.getRotationMatrix2D((g / 2, y / 2), aci, 1.0)       # 2x3 afin matris
    yeni = cv2.warpAffine(img, M, (g, y), borderMode=cv2.BORDER_REFLECT_101)
    return yeni, _kutulari_matrisle_donustur(kutular, M, g, y), f"dondur(aci={aci:+.1f}°)"


def olcekle_kaydir(img, kutular, rng, guc=0.5):
    """Yakınlaştır/uzaklaştır + kaydır (kameranın farklı mesafeden bakması)."""
    y, g = img.shape[:2]
    olcek = 1.0 + rng.uniform(-0.25, 0.25) * guc
    tx, ty = rng.uniform(-0.1, 0.1, size=2) * guc * np.array([g, y])
    M = np.array([[olcek, 0, (1 - olcek) * g / 2 + tx],
                  [0, olcek, (1 - olcek) * y / 2 + ty]], np.float32)
    yeni = cv2.warpAffine(img, M, (g, y), borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))
    return yeni, _kutulari_matrisle_donustur(kutular, M, g, y), f"olcekle_kaydir(olcek={olcek:.2f}, tx={tx:+.0f}, ty={ty:+.0f})"


# ============================================================================ kapatma
def kesip_cikar(img, kutular, rng, guc=0.5):
    """Cutout: rastgele gri kare(ler) yapıştır. Nesne kısmen kapalıyken de tanınmayı öğretir.

    Senin one-shot projendeki 'elin nesneyi kısmen kapatması' durumunun yapay hali.
    """
    y, g = img.shape[:2]
    yeni = img.copy()
    adet = 1 + int(rng.integers(0, 3) * guc)
    for _ in range(adet):
        boy = int(min(g, y) * rng.uniform(0.08, 0.25) * max(guc, 0.3))
        cx, cy = rng.integers(0, g), rng.integers(0, y)
        yeni[max(0, cy - boy // 2):cy + boy // 2, max(0, cx - boy // 2):cx + boy // 2] = 114
    return yeni, kutular.copy(), f"kesip_cikar(adet={adet})"


# ============================================================================ politika
DONUSUMLER = {
    "parlaklik_kontrast": parlaklik_kontrast,
    "hsv_kaydir": hsv_kaydir,
    "bulaniklik": bulaniklik,
    "gurultu": gurultu,
    "yatay_cevir": yatay_cevir,
    "dondur": dondur,
    "olcekle_kaydir": olcekle_kaydir,
    "kesip_cikar": kesip_cikar,
}

# Her seviyede her dönüşümün uygulanma olasılığı ve gücü.
#   "hafif": günlük kullanımda iyi başlangıç   "agir": az veride (ör. 6 fotoğraf) daha çok çeşitlilik
POLITIKALAR = {
    "yok":   {"olasilik": {}, "guc": 0.0},
    "hafif": {"olasilik": {"parlaklik_kontrast": 0.5, "hsv_kaydir": 0.3, "yatay_cevir": 0.5,
                           "dondur": 0.2, "olcekle_kaydir": 0.3}, "guc": 0.4},
    "agir":  {"olasilik": {"parlaklik_kontrast": 0.8, "hsv_kaydir": 0.6, "bulaniklik": 0.3,
                           "gurultu": 0.3, "yatay_cevir": 0.5, "dondur": 0.5,
                           "olcekle_kaydir": 0.6, "kesip_cikar": 0.4}, "guc": 1.0},
}


def kutulari_temizle(kutular, etiketler, g, y, min_alan_orani=0.2, orijinal=None):
    """Görüntü dışına taşan/çok küçülen kutuları at. (Kırpma/kaydırmadan sonra gerekli.)"""
    kutular = kutular.copy()
    kutular[:, [0, 2]] = kutular[:, [0, 2]].clip(0, g)
    kutular[:, [1, 3]] = kutular[:, [1, 3]].clip(0, y)
    alan = (kutular[:, 2] - kutular[:, 0]) * (kutular[:, 3] - kutular[:, 1])
    tut = alan > 4
    if orijinal is not None:
        ilk_alan = (orijinal[:, 2] - orijinal[:, 0]) * (orijinal[:, 3] - orijinal[:, 1])
        tut &= alan >= min_alan_orani * ilk_alan
    return kutular[tut], [e for e, t in zip(etiketler, tut) if t]


def uygula(img, kutular, etiketler, politika="hafif", rng=None):
    """Bir politikayı uygula: her dönüşüm kendi olasılığıyla çalışır.

    Döndürür: (yeni_img, yeni_kutular, yeni_etiketler, uygulanan_adimlar_listesi)
    """
    rng = rng or np.random.default_rng()
    p = POLITIKALAR[politika]
    kutular = np.asarray(kutular, np.float32).reshape(-1, 4)
    orijinal = kutular.copy()
    adimlar = []
    for isim, olasilik in p["olasilik"].items():
        if rng.random() < olasilik:
            img, kutular, aciklama = DONUSUMLER[isim](img, kutular, rng, p["guc"])
            adimlar.append(aciklama)
    kutular, etiketler = kutulari_temizle(kutular, list(etiketler), img.shape[1], img.shape[0], orijinal=orijinal)
    return img, kutular, etiketler, adimlar
