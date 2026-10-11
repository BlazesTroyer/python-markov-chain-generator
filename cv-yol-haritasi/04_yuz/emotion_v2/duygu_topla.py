"""
duygu_topla.py — Kendi duygu dataset'ini toplamak (senin 0'dan kurduğun dataset'in düzenli hali).

NASIL KULLANILIR?
    1. Panelde 'etiket'i seç (ör. "mutlu"), Canlı webcam sekmesini aç, o ifadeyi yap.
    2. Karede yüz varsa: yüz bulunur (YuNet), biraz geniş kare olarak kırpılır,
       veri/duygu/<etiket>/<oturum>_<zaman>.jpg olarak kaydedilir.
    3. Etiketi değiştir, tekrarla. Her duygu için 100+ örnek, FARKLI GÜNLERDE / ışıklarda.

ÖNEMLİ — OTURUM (sızıntı yine!):
    Dosya adının başındaki 'oturum' (ör. "sabah1"), egit.py'de train/val bölmesinde GRUP olarak
    kullanılır: Aynı oturumun neredeyse aynı kareleri hem train'de hem val'de olursa doğruluk
    yapay yükselir (01'deki sızıntı dersinin aynısı). Her toplama seansında oturum adını değiştir.
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
DOSYA = "04_yuz/emotion_v2/duygu_topla.py"


@dataclass
class Ayarlar:
    etiket: str = ayar("notr", "Şu an kaydedilen duygu", secenekler=["notr", "mutlu", "uzgun", "sasirmis", "kizgin", "yorgun"])
    oturum: str = ayar("oturum1", "Toplama seansının adı (train/val bölmesinde grup olarak kullanılır)",
                       etki="Her gün/ışık değişiminde yeni ad ver -> dürüst val ölçümü")
    aralik_sn: float = ayar(0.5, "İki kayıt arası en az süre", min=0.1, max=5.0, adim=0.1)
    pay: float = ayar(0.15, "Yüz kutusunun etrafına bırakılan pay (oran)", min=0.0, max=0.5, adim=0.05)


class DuyguTopla(KareIsleyici):
    ISIM = "Duygu dataset'i topla"
    BOLUM = "04"
    DOSYA = DOSYA
    ACIKLAMA = "Webcam'de en büyük yüzü kırpıp `veri/duygu/<etiket>/` altına kaydeder. Oturum adı, eğitimde sızıntıyı önlemek için grup olarak kullanılır."
    Ayarlar = Ayarlar

    def __init__(self):
        self.son = 0.0

    def isle(self, kare, a: Ayarlar, anlat):
        yuzler = yo.yuzleri_bul(kare)
        cikti, ara = kare.copy(), {}
        klasor = VERI / "duygu" / a.etiket
        sayi = len(list(klasor.glob("*.jpg"))) if klasor.exists() else 0
        durum = "yuz yok"
        if len(yuzler):
            en_buyuk = max(yuzler, key=lambda y: y[2] * y[3])          # en büyük (en yakın) yüz
            kirp = yo.yuz_kirp(kare, en_buyuk, a.pay)
            ara["kaydedilecek kırpıntı"] = kirp
            kutu_ciz(cikti, yo.kutu_xyxy(en_buyuk), a.etiket)
            if time.time() - self.son >= a.aralik_sn:
                klasor.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(klasor / f"{dosya_adi(a.oturum)}_{int(time.time() * 1000)}.jpg"), kirp)
                self.son, sayi, durum = time.time(), sayi + 1, "KAYDEDILDI"
                anlat.adim(f"{a.etiket}: kaydedildi ({sayi})", kod=f"{DOSYA}:DuyguTopla.isle")
            else:
                durum = "bekle"
        yazi_yaz(cikti, [f"{a.etiket}: {sayi} ornek", durum])
        return Sonuc(goruntu=cikti, ara_gorseller=ara, veri={"etiket": a.etiket, "sayi": sayi}, metin=f"**{a.etiket}**: {sayi} örnek")


if __name__ == "__main__":
    from ortak.cli import calistir
    calistir(DuyguTopla)
