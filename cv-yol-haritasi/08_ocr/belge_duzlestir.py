"""
belge_duzlestir.py — Eğik çekilmiş bir fişi/belgeyi bulup "tarayıcıdan çıkmış gibi" düzleştirir.

NEDEN?
    OCR, düz ve karşıdan çekilmiş metinde çok daha iyi çalışır. Telefonla çekilen fiş eğiktir,
    perspektifi bozuktur, arka planda masa vardır. Bu dosya klasik görüntü işlemeyle düzeltir:

ADIMLAR (belge_bul + duzlestir):
    1. Gri -> Gauss bulanıklık -> Canny kenar    (00_temeller/klasik_islemler.py'deki aynı hat)
    2. Kenarları genişlet (dilate): kopuk kenar çizgileri birleşsin
    3. Dış konturları bul, ALANA göre sırala
    4. Büyükten küçüğe: konturu çokgene yaklaştır (approxPolyDP). 4 köşeli ilk büyük çokgen = belge.
       Bulunamazsa: en büyük konturun minimum alanlı döndürülmüş dikdörtgeni (yedek yöntem).
    5. Köşeleri SIRALA: sol-üst, sağ-üst, sağ-alt, sol-alt (kose_sirala — toplam/fark hilesi)
    6. HOMOGRAFİ: 4 köşeyi hedef dikdörtgenin 4 köşesine eşleyen 3x3 matris (getPerspectiveTransform)
       ve warpPerspective ile tüm görüntüyü bu matrisle dönüştür.
    7. (opsiyonel) Adaptif eşik: "taranmış belge" görünümü (gölgeli fişlerde OCR'ı iyileştirir)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ortak.arayuz import KareIsleyici, Sonuc, ayar  # noqa: E402
from ortak.cizim import yazi_yaz  # noqa: E402

DOSYA = "08_ocr/belge_duzlestir.py"


def kose_sirala(p: np.ndarray) -> np.ndarray:
    """4 noktayı (sol-üst, sağ-üst, sağ-alt, sol-alt) sırasına koy.

    Hile: x+y en küçük = sol-üst, en büyük = sağ-alt;  y-x en küçük = sağ-üst, en büyük = sol-alt.
    """
    p = np.asarray(p, np.float32).reshape(4, 2)
    s, d = p.sum(1), np.diff(p, axis=1).ravel()
    return np.array([p[s.argmin()], p[d.argmin()], p[s.argmax()], p[d.argmax()]], np.float32)


def belge_bul(kare: np.ndarray, canny_alt=50, canny_ust=150, min_alan_orani=0.1):
    """Döndürür: (köşeler (4,2) ya da None, ara görseller dict, yöntem adı)"""
    gri = cv2.cvtColor(kare, cv2.COLOR_BGR2GRAY)
    bul = cv2.GaussianBlur(gri, (5, 5), 0)
    kenar = cv2.Canny(bul, canny_alt, canny_ust)
    kenar = cv2.dilate(kenar, np.ones((5, 5), np.uint8), iterations=1)
    konturlar, _ = cv2.findContours(kenar, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    konturlar = sorted(konturlar, key=cv2.contourArea, reverse=True)[:5]
    ara = {"1_kenarlar": kenar}
    min_alan = min_alan_orani * kare.shape[0] * kare.shape[1]
    for c in konturlar:
        if cv2.contourArea(c) < min_alan:
            break
        cevre = cv2.arcLength(c, True)
        yaklasik = cv2.approxPolyDP(c, 0.02 * cevre, True)       # çevrenin %2'si toleransla basitleştir
        if len(yaklasik) == 4:
            return kose_sirala(yaklasik), ara, "4 köşeli kontur"
    if konturlar and cv2.contourArea(konturlar[0]) >= min_alan:
        return kose_sirala(cv2.boxPoints(cv2.minAreaRect(konturlar[0]))), ara, "minAreaRect (yedek)"
    return None, ara, "bulunamadı"


def duzlestir(kare: np.ndarray, koseler: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Homografi ile düzleştir. Döndürür: (düz görüntü, 3x3 homografi matrisi)."""
    sol_ust, sag_ust, sag_alt, sol_alt = koseler
    genislik = int(max(np.linalg.norm(sag_ust - sol_ust), np.linalg.norm(sag_alt - sol_alt)))
    yukseklik = int(max(np.linalg.norm(sol_alt - sol_ust), np.linalg.norm(sag_alt - sag_ust)))
    hedef = np.array([[0, 0], [genislik - 1, 0], [genislik - 1, yukseklik - 1], [0, yukseklik - 1]], np.float32)
    H = cv2.getPerspectiveTransform(koseler, hedef)
    return cv2.warpPerspective(kare, H, (genislik, yukseklik)), H


@dataclass
class Ayarlar:
    canny_alt: int = ayar(50, "Canny alt eşik", min=0, max=300)
    canny_ust: int = ayar(150, "Canny üst eşik", min=0, max=500)
    min_alan_orani: float = ayar(0.1, "Belge, görüntünün en az bu oranını kaplamalı", min=0.01, max=0.9, adim=0.01)
    tarama_gorunumu: bool = ayar(False, "Düzleştirilmiş görüntüye adaptif eşik uygula (siyah-beyaz tarama)",
                                 etki="Gölgeli/buruşuk fişlerde OCR'ı iyileştirebilir; renkli logoları bozar")


class BelgeDuzlestir(KareIsleyici):
    ISIM = "Belge/fiş düzleştir (kontur + homografi)"
    BOLUM = "08"
    DOSYA = DOSYA
    ACIKLAMA = "Fişin/belgenin 4 köşesini klasik yöntemlerle bulur ve homografiyle karşıdan çekilmiş gibi düzleştirir. Ara adımlar görünür."
    Ayarlar = Ayarlar

    def isle(self, kare, a: Ayarlar, anlat):
        koseler, ara, yontem = belge_bul(kare, a.canny_alt, a.canny_ust, a.min_alan_orani)
        anlat.adim(f"Belge arama: {yontem}", kod=f"{DOSYA}:belge_bul", veri={"koseler": None if koseler is None else koseler.round().tolist()})
        isaretli = kare.copy()
        if koseler is None:
            yazi_yaz(isaretli, "belge bulunamadi")
            return Sonuc(goruntu=isaretli, ara_gorseller=ara, metin="Belge bulunamadı (arka planla kontrastı artır, belgeyi tamamen kadraja al).")
        cv2.polylines(isaretli, [koseler.astype(np.int32)], True, (0, 255, 0), 3)
        for ad, p in zip(["SU", "SgU", "SgA", "SA"], koseler):
            cv2.putText(isaretli, ad, tuple(p.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        duz, H = duzlestir(kare, koseler)
        anlat.adim(f"Homografi uygulandı → {duz.shape[1]}x{duz.shape[0]}", kod=f"{DOSYA}:duzlestir", veri={"H": H.round(3).tolist()})
        if a.tarama_gorunumu:
            duz = cv2.cvtColor(cv2.adaptiveThreshold(cv2.cvtColor(duz, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                     cv2.THRESH_BINARY, 31, 15), cv2.COLOR_GRAY2BGR)
            anlat.adim("Adaptif eşik (tarama görünümü)", kod="cv2.adaptiveThreshold")
        ara["2_bulunan köşeler"] = isaretli
        return Sonuc(goruntu=duz, ara_gorseller=ara, veri={"yontem": yontem, "boyut": [duz.shape[1], duz.shape[0]]},
                     metin=f"Belge bulundu ({yontem}) ve düzleştirildi: {duz.shape[1]}×{duz.shape[0]}")


if __name__ == "__main__":
    from ortak.cli import calistir
    calistir(BelgeDuzlestir)
