"""
hareket_kapisi.py — "Hareket var mı?" ön filtresi (klasik görüntü işleme, derin öğrenme YOK).

NEDEN?
    7/24 çalışan bir kamerada YOLO'yu her karede çalıştırmak CPU/GPU ve elektrik israfıdır.
    Önce çok ucuz bir yöntemle "sahnede bir şey değişti mi?" diye bakarız; değiştiyse
    pahalı modeli çalıştırırız. (Bölüm 10'daki 7/24 kamera bunu kullanır.)

İKİ YÖNTEM:
    yontem="fark"   : Kare farkı (frame differencing)
        1. Gri tona çevir, hafif bulanıklaştır (gürültüyü bastır)
        2. |şimdiki - önceki| mutlak farkı al
        3. Eşikle (fark > esik -> beyaz)
        4. Morfolojik genişletme (dilate) ile kopuk parçaları birleştir
        5. Konturları bul; alanı min_alan'dan büyük olanlar = hareket bölgesi
    yontem="akis"   : Yoğun optik akış (Farneback)
        Her piksel için "bir önceki kareden bu kareye nereye kaydı?" vektörünü hesaplar.
        Büyüklüğü (hız) eşikten büyük pikseller hareketli sayılır. Yön bilgisi de verir
        (renk = yön, parlaklık = hız). Daha pahalı ama kamera gürültüsüne daha dayanıklı.

DURUM: Önceki kareyi hatırlar (sifirla() ile temizlenir).
OLAYLAR: "hareket_basladi" / "hareket_bitti" (durum değişiminde bir kez)
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

DOSYA = "00_temeller/hareket_kapisi.py"


@dataclass
class Ayarlar:
    yontem: str = ayar("fark", "fark: kare farkı (çok hızlı) | akis: Farneback optik akış (yönü de verir)",
                       secenekler=["fark", "akis"])
    esik: int = ayar(25, "Fark yönteminde piksel farkı eşiği (0-255)", min=5, max=100, adim=1,
                     etki="Düşürürsen ışık titremesi bile hareket sayılır; artırırsan yavaş hareketleri kaçırırsın")
    akis_esik: float = ayar(1.5, "Akış yönteminde piksel/kare hız eşiği", min=0.2, max=10.0, adim=0.1)
    min_alan: int = ayar(800, "Hareket sayılacak en küçük bölge alanı (piksel²)", min=50, max=20000, adim=50,
                         etki="Artırırsan küçük kıpırtılar (yaprak, perde) yok sayılır")
    bulanik: int = ayar(5, "Ön bulanıklaştırma çekirdeği (tek sayı)", min=1, max=21, adim=2)
    olcek: float = ayar(0.5, "İşlemeden önce küçültme oranı (hız için)", min=0.2, max=1.0, adim=0.1)


class HareketKapisi(KareIsleyici):
    ISIM = "Hareket kapısı (kare farkı / optik akış)"
    BOLUM = "00"
    DOSYA = DOSYA
    ACIKLAMA = ("Klasik görüntü işleme ile **'sahnede bir şey değişti mi?'** sorusunu çok ucuza cevaplar. "
                "Pahalı modelleri sadece hareket varken çalıştırmak için ön filtre. Ara adımlarda fark, eşik, "
                "morfoloji ve akış görselleri görünür. Video ya da canlı webcam ile kullan (tek fotoğrafta önceki kare yoktur).")
    Ayarlar = Ayarlar

    def __init__(self):
        self.onceki: np.ndarray | None = None
        self.hareket_vardi = False

    def sifirla(self):
        self.onceki, self.hareket_vardi = None, False

    def isle(self, kare, ayarlar: Ayarlar, anlat):
        kucuk = cv2.resize(kare, None, fx=ayarlar.olcek, fy=ayarlar.olcek)
        gri = cv2.cvtColor(kucuk, cv2.COLOR_BGR2GRAY)
        k = ayarlar.bulanik | 1
        gri = cv2.GaussianBlur(gri, (k, k), 0)
        anlat.adim(f"Küçült ({ayarlar.olcek}x) → gri → Gauss bulanıklık (k={k})", kod=f"{DOSYA}:HareketKapisi.isle")

        if self.onceki is None or self.onceki.shape != gri.shape:
            self.onceki = gri
            anlat.bilgi("İlk kare: karşılaştıracak önceki kare yok, sadece hafızaya alındı.")
            return Sonuc(goruntu=yazi_yaz(kare.copy(), "Ilk kare - bekleniyor"), metin="İlk kare hafızaya alındı.")

        ara = {}
        if ayarlar.yontem == "fark":
            fark = cv2.absdiff(gri, self.onceki)
            _, esikli = cv2.threshold(fark, ayarlar.esik, 255, cv2.THRESH_BINARY)
            maske = cv2.dilate(esikli, None, iterations=2)
            anlat.adim("|şimdiki − önceki| → eşik → genişletme", kod=f"{DOSYA}:cv2.absdiff/threshold/dilate",
                       veri={"degisen_piksel_%": float((esikli > 0).mean() * 100)})
            ara = {"1_fark": fark, "2_esik": esikli, "3_genisletilmis": maske}
        else:
            akis = cv2.calcOpticalFlowFarneback(self.onceki, gri, None, 0.5, 3, 15, 3, 5, 1.2, 0)  # (Y,G,2)
            buyukluk, aci = cv2.cartToPolar(akis[..., 0], akis[..., 1])
            maske = ((buyukluk > ayarlar.akis_esik) * 255).astype(np.uint8)
            maske = cv2.dilate(maske, None, iterations=2)
            hsv = np.zeros((*gri.shape, 3), np.uint8)
            hsv[..., 0] = (aci * 180 / np.pi / 2).astype(np.uint8)     # renk = yön
            hsv[..., 1] = 255
            hsv[..., 2] = cv2.normalize(buyukluk, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  # parlaklık = hız
            anlat.adim("Farneback optik akış → hız/yön → hız eşiği", kod=f"{DOSYA}:cv2.calcOpticalFlowFarneback",
                       veri={"ort_hiz": float(buyukluk.mean()), "max_hiz": float(buyukluk.max())})
            ara = {"1_akis_renk=yon_parlaklik=hiz": cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), "2_hiz_maskesi": maske}

        konturlar, _ = cv2.findContours(maske, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_alan_kucuk = ayarlar.min_alan * ayarlar.olcek ** 2   # alan, küçültmeyle karesel küçülür
        bolgeler = [cv2.boundingRect(c) for c in konturlar if cv2.contourArea(c) >= min_alan_kucuk]
        anlat.adim(f"{len(konturlar)} kontur, {len(bolgeler)} tanesi min_alan üstünde", kod=f"{DOSYA}:cv2.findContours")

        cikti = kare.copy()
        for (x, y, w, h) in bolgeler:
            kutu_ciz(cikti, np.array([x, y, x + w, y + h]) / ayarlar.olcek, "hareket", (0, 165, 255))
        hareket = len(bolgeler) > 0
        yazi_yaz(cikti, f"HAREKET: {'VAR' if hareket else 'yok'}  ({ayarlar.yontem})", renk=(0, 255, 255) if hareket else (200, 200, 200))

        olaylar = []
        if hareket != self.hareket_vardi:  # sadece DURUM DEĞİŞİMİNDE olay üret (her karede değil)
            tur = "hareket_basladi" if hareket else "hareket_bitti"
            olaylar.append({"tur": tur, "kaynak": DOSYA, "ek": {"bolge_sayisi": len(bolgeler)}})
            anlat.adim(f"Durum değişti → olay: {tur}", kod=f"{DOSYA}:HareketKapisi.isle")
        self.hareket_vardi = hareket
        self.onceki = gri

        return Sonuc(goruntu=cikti, ara_gorseller=ara, olaylar=olaylar,
                     metin=f"Hareket: **{'VAR' if hareket else 'yok'}**, bölge sayısı: {len(bolgeler)}",
                     veri={"hareket": hareket, "bolge": len(bolgeler)})


if __name__ == "__main__":
    from ortak.cli import calistir
    calistir(HareketKapisi)
