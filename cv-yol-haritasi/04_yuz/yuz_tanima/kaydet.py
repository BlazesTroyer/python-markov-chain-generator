"""
kaydet.py — Yüz tanıma için kişi KAYDI, adım 1/3: hizalanmış yüz fotoğrafı topla.

    (Adım 2: veritabani.py — embedding veritabanı + eşik analizi. Adım 3: tani.py — tanıma.)

    YuzFotografiTopla (Kare işleyici)
        Webcam/video karesinde TEK yüz varsa, hizalanmış (112x112) yüzü
        veri/yuzler/<isim>/ klasörüne kaydeder. Kafanı hafifçe çevir, farklı ışıklarda dur:
        çeşitlilik = daha sağlam tanıma (augmentation'ın "gerçek" hali!).


GİZLİLİK: Veritabanında sadece embedding (128 sayı) ve isim tutulur. Fotoğraflar veri/yuzler'de
kalır; istersen kayıttan sonra silebilirsin, tanıma çalışmaya devam eder.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2  # noqa: E402


from ortak.arayuz import KareIsleyici, Sonuc, ayar  # noqa: E402
from ortak.cizim import dosya_adi, kutu_ciz, yazi_yaz  # noqa: E402
from ortak.yukle import VERI, modul_yukle  # noqa: E402

yo = modul_yukle("04_yuz/yuz_ortak.py")
DOSYA = "04_yuz/yuz_tanima/kaydet.py"


@dataclass
class ToplaAyarlari:
    isim: str = ayar("ben", "Kaydedilecek kişinin adı (klasör adı olur)")
    aralik_sn: float = ayar(0.7, "İki kayıt arası en az süre (aynı pozu tekrar tekrar kaydetmemek için)", min=0.1, max=5.0, adim=0.1)
    en_fazla: int = ayar(30, "Kişi başına en fazla kaç fotoğraf", min=1, max=500)
    guven: float = ayar(0.8, "YuNet yüz güven eşiği", min=0.3, max=0.99, adim=0.01)


class YuzFotografiTopla(KareIsleyici):
    ISIM = "Yüz fotoğrafı topla (kayıt)"
    BOLUM = "04"
    DOSYA = DOSYA
    ACIKLAMA = ("Karede **tek** yüz varsa hizalanmış yüzü `veri/yuzler/<isim>/` altına kaydeder. Canlı webcam sekmesiyle kullan; "
                "kafanı yavaşça çevir. Sonra **Veritabanı oluştur**.")
    Ayarlar = ToplaAyarlari

    def __init__(self):
        self.son = 0.0

    def isle(self, kare, a: ToplaAyarlari, anlat):
        yuzler = yo.yuzleri_bul(kare, a.guven)
        anlat.adim(f"YuNet: {len(yuzler)} yüz", kod="04_yuz/yuz_ortak.py:yuzleri_bul")
        klasor = VERI / "yuzler" / dosya_adi(a.isim)
        mevcut = len(list(klasor.glob("*.jpg"))) if klasor.exists() else 0
        cikti, ara = kare.copy(), {}
        for y in yuzler:
            kutu_ciz(cikti, yo.kutu_xyxy(y), f"{y[14]:.2f}")
            yo.noktalari_ciz(cikti, y)
        durum = "tek yuz bekleniyor"
        if len(yuzler) == 1:
            hizali = yo.hizala(kare, yuzler[0])
            ara["hizalanmış yüz (112x112)"] = cv2.resize(hizali, (224, 224), interpolation=cv2.INTER_NEAREST)
            anlat.adim("5 noktaya göre hizalandı", kod="04_yuz/yuz_ortak.py:hizala")
            if mevcut >= a.en_fazla:
                durum = "en fazla sayiya ulasildi"
            elif time.time() - self.son >= a.aralik_sn:
                klasor.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(klasor / f"{int(time.time() * 1000)}.jpg"), hizali)
                self.son, mevcut, durum = time.time(), mevcut + 1, "KAYDEDILDI"
                anlat.adim(f"Kaydedildi → {klasor}", kod=f"{DOSYA}:YuzFotografiTopla.isle")
            else:
                durum = "bekle (aralik)"
        yazi_yaz(cikti, [f"{a.isim}: {mevcut}/{a.en_fazla}", durum])
        return Sonuc(goruntu=cikti, ara_gorseller=ara, veri={"kayitli": mevcut}, metin=f"**{a.isim}**: {mevcut} fotoğraf ({durum})")


if __name__ == "__main__":
    from ortak.cli import calistir
    calistir(YuzFotografiTopla)
