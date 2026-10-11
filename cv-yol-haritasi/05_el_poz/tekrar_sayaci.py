"""
tekrar_sayaci.py — Egzersiz tekrar sayacı: eklem AÇISI + HİSTEREZİSLİ durum makinesi.

AÇI (noktalar.aci): Üç noktanın ortadakindeki açı.
    squat  : kalça – DİZ – ayak bileği     (ayakta ≈ 170°, çömelince ≈ 80-100°)
    sinav  : omuz – DİRSEK – bilek         (kollar düz ≈ 160°, aşağıda ≈ 70-90°)
    biceps : omuz – DİRSEK – bilek         (kol açık ≈ 160°, bükülü ≈ 40°)

NEDEN İKİ EŞİK (HİSTEREZİS)?
    Tek eşik (ör. 120°) kullanırsan, açı 119-121 arasında titrerken her titreme bir "tekrar" sayılır.
    İki eşikle:
        açı < alt_esik           -> faz = "asagi"
        faz "asagi" iken açı > ust_esik -> faz = "yukari" ve TEKRAR += 1
        aradaki bölgede (alt..üst) hiçbir şey değişmez -> titreşim sayılmaz.
    Bu, termostatların da kullandığı klasik bir mühendislik hilesidir.

TARAF SEÇİMİ: Sol ve sağ eklem noktalarından GÖRÜNÜRLÜĞÜ yüksek olan taraf kullanılır
(kameraya yan durunca uzak taraf kapanır).

ARA GÖRSEL: Son 150 karenin açı grafiği + iki eşik çizgisi -> histerezisi gözle görürsün.
OLAY: her tekrar tamamlanınca {"tur": "tekrar", "nesne": "squat", "ek": {"sayi": 7}}
"""
from __future__ import annotations

import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ortak.arayuz import KareIsleyici, Sonuc, ayar  # noqa: E402
from ortak.cizim import egri_ciz, yazi_yaz  # noqa: E402
from ortak.yukle import modul_yukle  # noqa: E402

nk = modul_yukle("05_el_poz/noktalar.py")
DOSYA = "05_el_poz/tekrar_sayaci.py"

# egzersiz: ((sol a, b, c), (sağ a, b, c), varsayılan alt, varsayılan üst)
EGZERSIZLER = {
    "squat": ((23, 25, 27), (24, 26, 28), 100, 160),
    "sinav": ((11, 13, 15), (12, 14, 16), 90, 155),
    "biceps": ((11, 13, 15), (12, 14, 16), 50, 150),
}


class TekrarSayaci:
    """Histerezisli faz takibi (saf — test edilir)."""

    def __init__(self, alt: float, ust: float):
        self.alt, self.ust = alt, ust
        self.faz, self.sayi = "?", 0

    def guncelle(self, aci: float) -> bool:
        """Açıyı işle; bir tekrar TAMAMLANDIYSA True döndür."""
        if aci < self.alt:
            self.faz = "asagi"
        elif aci > self.ust:
            if self.faz == "asagi":
                self.faz = "yukari"
                self.sayi += 1
                return True
            self.faz = "yukari"
        return False


@dataclass
class Ayarlar:
    egzersiz: str = ayar("squat", "Egzersiz", secenekler=list(EGZERSIZLER))
    alt_esik: float = ayar(0.0, "Alt eşik (derece). 0 = egzersizin varsayılanı", min=0, max=180, adim=1,
                           etki="Düşürürsen sadece tam/derin tekrarlar sayılır")
    ust_esik: float = ayar(0.0, "Üst eşik (derece). 0 = egzersizin varsayılanı", min=0, max=180, adim=1)
    min_gorunurluk: float = ayar(0.5, "Eklem noktalarının en düşük görünürlüğü", min=0.0, max=1.0, adim=0.05)


class TekrarSayaciDemo(KareIsleyici):
    ISIM = "Egzersiz tekrar sayacı (açı + histerezis)"
    BOLUM = "05"
    DOSYA = DOSYA
    ACIKLAMA = ("MediaPipe poz noktalarından eklem açısını hesaplar; iki eşikli (histerezisli) durum makinesiyle tekrar sayar. "
                "Ara görselde açı grafiği ve eşikler. Kameraya YANDAN dur.")
    GEREKSINIMLER = ("mediapipe",)
    Ayarlar = Ayarlar

    def __init__(self):
        self.sifirla()

    def sifirla(self):
        self.sayac, self.aci_gecmisi, self._egz = None, deque(maxlen=150), None

    def isle(self, kare, a: Ayarlar, anlat):
        sol, sag, v_alt, v_ust = EGZERSIZLER[a.egzersiz]
        alt, ust = a.alt_esik or v_alt, a.ust_esik or v_ust
        if self.sayac is None or self._egz != a.egzersiz:
            self.sayac, self._egz = TekrarSayaci(alt, ust), a.egzersiz
        self.sayac.alt, self.sayac.ust = alt, ust
        p = nk.poz_noktalari(kare)
        cikti, olaylar = kare.copy(), []
        if p is None:
            yazi_yaz(cikti, "kisi yok")
            return Sonuc(goruntu=cikti, metin="Kişi yok.")
        # Görünürlüğü yüksek tarafı seç
        taraf = sol if p[list(sol), 3].mean() >= p[list(sag), 3].mean() else sag
        gorunurluk = float(p[list(taraf), 3].min())
        nk.iskelet_ciz(cikti, p, nk.POZ_BAGLANTILARI, vurgula=taraf)
        if gorunurluk < a.min_gorunurluk:
            anlat.uyari(f"Eklemler yeterince görünmüyor ({gorunurluk:.2f})")
            yazi_yaz(cikti, [f"{a.egzersiz}: {self.sayac.sayi}", "eklemler gorunmuyor"])
            return Sonuc(goruntu=cikti, veri={"sayi": self.sayac.sayi})
        a_, b_, c_ = (p[i] for i in taraf)
        derece = nk.aci(a_, b_, c_)
        self.aci_gecmisi.append(derece)
        tamamlandi = self.sayac.guncelle(derece)
        anlat.adim(f"açı={derece:.0f}° (alt {alt}, üst {ust}) faz={self.sayac.faz} sayı={self.sayac.sayi}",
                   kod=f"05_el_poz/noktalar.py:aci → {DOSYA}:TekrarSayaci.guncelle")
        if tamamlandi:
            olaylar.append({"tur": "tekrar", "kaynak": DOSYA, "nesne": a.egzersiz, "ek": {"sayi": self.sayac.sayi}})
        y, g = kare.shape[:2]
        cv2.putText(cikti, f"{derece:.0f}", (int(b_[0] * g) + 10, int(b_[1] * y)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        yazi_yaz(cikti, [f"{a.egzersiz.upper()}: {self.sayac.sayi}", f"faz: {self.sayac.faz}"])
        n = len(self.aci_gecmisi)
        grafik = egri_ciz({"aci": list(self.aci_gecmisi), "alt_esik": [alt] * n, "ust_esik": [ust] * n}, "eklem acisi (derece)")
        return Sonuc(goruntu=cikti, ara_gorseller={"açı geçmişi": grafik}, olaylar=olaylar,
                     veri={"sayi": self.sayac.sayi, "aci": round(derece, 1), "faz": self.sayac.faz},
                     metin=f"**{a.egzersiz}: {self.sayac.sayi} tekrar** (açı {derece:.0f}°, faz {self.sayac.faz})")


if __name__ == "__main__":
    from ortak.cli import calistir
    calistir(TekrarSayaciDemo)
