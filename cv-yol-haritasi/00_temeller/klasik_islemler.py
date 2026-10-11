"""
klasik_islemler.py — Derin öğrenmeden önceki temel görüntü işleme adımları, ara çıktılarıyla.

NEDEN?
    Birçok kişisel proje sorunu model gerektirmez: "masadaki kırmızı kupayı bul", "kâğıdın
    kenarını bul", "ekrandaki parlak noktayı takip et"... Bunlar renk eşiği + kontur ile
    milisaniyede çözülür. Ayrıca YOLO'nun ön/son işlemesinde de hep bunlar kullanılır.

İŞLEMLER (islem ayarı):
    hsv_renk   : BGR -> HSV; belirli renk aralığını maskele (ışığa RGB'den daha dayanıklı)
    canny      : Gri -> Gauss -> Canny kenar (iki eşikli histerezis)
    kontur     : Otsu eşiği -> morfolojik açma/kapama -> kontur -> alan/çevre/dikdörtgen
    adaptif    : Bölgesel (adaptif) eşik — eşit olmayan ışıkta belge/yazı için (08_ocr kullanır)

Her işlemde ARA GÖRSELLER sırayla verilir: hangi adım görüntüyü nasıl değiştirdi?
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ortak.arayuz import KareIsleyici, Sonuc, ayar  # noqa: E402
from ortak.cizim import kutu_ciz, yazi_yaz  # noqa: E402

DOSYA = "00_temeller/klasik_islemler.py"


@dataclass
class Ayarlar:
    islem: str = ayar("hsv_renk", "Hangi klasik işlem hattı?", secenekler=["hsv_renk", "canny", "kontur", "adaptif"])
    h_min: int = ayar(0, "HSV renk tonu alt sınır (0-179; kırmızı≈0/170, yeşil≈60, mavi≈110)", min=0, max=179)
    h_max: int = ayar(10, "HSV renk tonu üst sınır", min=0, max=179)
    s_min: int = ayar(100, "Doygunluk alt sınır (düşükse gri/beyazlar da girer)", min=0, max=255)
    v_min: int = ayar(60, "Parlaklık alt sınır (düşükse karanlık pikseller de girer)", min=0, max=255)
    canny_alt: int = ayar(50, "Canny alt eşik", min=0, max=500, etki="Düşürürsen zayıf kenarlar da bağlanır")
    canny_ust: int = ayar(150, "Canny üst eşik", min=0, max=500, etki="Bunun üstü kesin kenar sayılır")
    min_alan: int = ayar(500, "Kontur için en küçük alan (piksel²)", min=10, max=50000, adim=10)


class KlasikIslemler(KareIsleyici):
    ISIM = "Klasik işlemler (HSV, Canny, kontur, adaptif eşik)"
    BOLUM = "00"
    DOSYA = DOSYA
    ACIKLAMA = "Renk eşiği, kenar bulma, kontur ve adaptif eşik — her ara adım ayrı görselde. Ayarları kaydırıp anında etkisini gör."
    Ayarlar = Ayarlar

    def isle(self, kare, ayarlar: Ayarlar, anlat):
        cikti, ara, veri = kare.copy(), {}, {}
        if ayarlar.islem == "hsv_renk":
            hsv = cv2.cvtColor(kare, cv2.COLOR_BGR2HSV)
            alt = np.array([ayarlar.h_min, ayarlar.s_min, ayarlar.v_min])
            ust = np.array([ayarlar.h_max, 255, 255])
            maske = cv2.inRange(hsv, alt, ust)
            anlat.adim(f"HSV aralığı {alt.tolist()}..{ust.tolist()} maskelendi", kod=f"{DOSYA}:cv2.inRange",
                       veri={"maske_%": float((maske > 0).mean() * 100)})
            maske = cv2.morphologyEx(maske, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))   # tek tük noktaları sil
            ara = {"1_H_kanali": hsv[..., 0], "2_maske_ham": cv2.inRange(hsv, alt, ust), "3_maske_acilmis": maske,
                   "4_sadece_renk": cv2.bitwise_and(kare, kare, mask=maske)}
            cikti = self._konturlari_ciz(cikti, maske, ayarlar.min_alan, anlat, veri)
        elif ayarlar.islem == "canny":
            gri = cv2.cvtColor(kare, cv2.COLOR_BGR2GRAY)
            bul = cv2.GaussianBlur(gri, (5, 5), 0)
            kenar = cv2.Canny(bul, ayarlar.canny_alt, ayarlar.canny_ust)
            anlat.adim(f"Canny({ayarlar.canny_alt}, {ayarlar.canny_ust})", kod=f"{DOSYA}:cv2.Canny",
                       veri={"kenar_piksel_%": float((kenar > 0).mean() * 100)})
            ara = {"1_gri": gri, "2_bulanik": bul, "3_kenarlar": kenar}
            cikti[kenar > 0] = (0, 255, 255)
        elif ayarlar.islem == "kontur":
            gri = cv2.cvtColor(kare, cv2.COLOR_BGR2GRAY)
            esik_degeri, ikili = cv2.threshold(cv2.GaussianBlur(gri, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            anlat.adim(f"Otsu eşiği otomatik seçti: {esik_degeri:.0f}", kod=f"{DOSYA}:cv2.threshold(OTSU)")
            temiz = cv2.morphologyEx(ikili, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
            ara = {"1_gri": gri, "2_otsu": ikili, "3_kapama": temiz}
            cikti = self._konturlari_ciz(cikti, temiz, ayarlar.min_alan, anlat, veri)
        else:  # adaptif
            gri = cv2.cvtColor(kare, cv2.COLOR_BGR2GRAY)
            global_e = cv2.threshold(gri, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            adaptif = cv2.adaptiveThreshold(gri, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
            anlat.adim("Global (Otsu) ve adaptif eşik karşılaştırması", kod=f"{DOSYA}:cv2.adaptiveThreshold")
            ara = {"1_gri": gri, "2_global_otsu": global_e, "3_adaptif": adaptif}
            cikti = cv2.cvtColor(adaptif, cv2.COLOR_GRAY2BGR)
        yazi_yaz(cikti, ayarlar.islem)
        return Sonuc(goruntu=cikti, ara_gorseller=ara, veri=veri, metin=f"İşlem: **{ayarlar.islem}**")

    def _konturlari_ciz(self, cikti, maske, min_alan, anlat, veri):
        konturlar, _ = cv2.findContours(maske, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        buyukler = [c for c in konturlar if cv2.contourArea(c) >= min_alan]
        anlat.adim(f"{len(konturlar)} kontur, {len(buyukler)} tanesi ≥ {min_alan}px²", kod=f"{DOSYA}:cv2.findContours")
        for c in buyukler:
            x, y, w, h = cv2.boundingRect(c)
            cv2.drawContours(cikti, [c], -1, (255, 0, 255), 2)
            kutu_ciz(cikti, (x, y, x + w, y + h), f"alan={cv2.contourArea(c):.0f}")
        veri["kontur"] = len(buyukler)
        return cikti


if __name__ == "__main__":
    from ortak.cli import calistir
    calistir(KlasikIslemler)
