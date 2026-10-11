"""
jest_kaydet.py — Dinamik jest (hareketli el hareketi) dataset'i kaydet.

STATİK vs DİNAMİK JEST:
    Statik : Tek karede anlaşılır (avuç açık, yumruk, başparmak yukarı). Tek bir (63,) vektör yeter.
    Dinamik: Zamana yayılır (sağa kaydırma, el sallama, daire çizme). Bir DİZİ gerekir: (T kare, F özellik).
    Senin hand controller'ın büyük ihtimalle statik kurallarla çalışıyordu; burada dinamik jestleri
    ÖĞRENEN bir model için veri topluyoruz.

KAYIT DÖNGÜSÜ (durum makinesi — ekranda yazıyor):
    HAZIRLAN (ara_sn) -> KAYIT (T kare boyunca el görünmeli) -> dosyaya yaz -> HAZIRLAN ...
    Kayıt sırasında el kaybolursa o deneme İPTAL edilir (eksik dizi kaydedilmez).

KARE BAŞINA ÖZELLİK (F = 65):
    63 = el_normalize(noktalar)  : elin ŞEKLİ (konum/ölçek/taraf bağımsız)
     2 = bilek (x, y)            : elin görüntüdeki YERİ — kaydırma gibi hareketler bundan anlaşılır
                                   (eğitimde dizinin ilk karesine göre görelileştirilir: jest_egit.dizi_hazirla)

ÇIKTI: veri/jestler/<etiket>/<oturum>_<zaman>.npy   — şekil (T, 65)
ÖNEMLİ: "yok" etiketiyle de kayıt yap (eli rastgele oynat, kaşın, telefona uzan): modelin
"jest değil" demeyi öğrenmesi, yanlış tetiklemeyi en çok azaltan şeydir.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from ortak.arayuz import KareIsleyici, Sonuc, ayar  # noqa: E402
from ortak.cizim import dosya_adi, yazi_yaz  # noqa: E402
from ortak.yukle import VERI, modul_yukle  # noqa: E402

nk = modul_yukle("05_el_poz/noktalar.py")
DOSYA = "05_el_poz/jest_kaydet.py"


def kare_ozelligi(el: dict) -> np.ndarray:
    """Bir elin (65,) özellik vektörü: normalize şekil + ham bilek konumu."""
    return np.concatenate([nk.el_normalize(el["noktalar"], el["taraf"]), el["noktalar"][0, :2]]).astype(np.float32)


@dataclass
class Ayarlar:
    etiket: str = ayar("sag_kaydir", "Kaydedilen jest",
                       secenekler=["sag_kaydir", "sol_kaydir", "el_salla", "avuc", "basparmak", "yumruk", "daire", "yok"])
    oturum: str = ayar("oturum1", "Kayıt seansı adı (eğitimde train/val grubu)")
    uzunluk: int = ayar(30, "Dizi uzunluğu T (kare)", min=8, max=90, etki="30 FPS'te 30 kare ≈ 1 sn")
    ara_sn: float = ayar(1.5, "Kayıtlar arası hazırlanma süresi (sn)", min=0.2, max=10.0, adim=0.1)


class JestKaydet(KareIsleyici):
    ISIM = "Jest kaydet (dinamik dizi)"
    BOLUM = "05"
    DOSYA = DOSYA
    ACIKLAMA = ("Canlı webcam'de HAZIRLAN → KAYIT döngüsüyle jest dizilerini (T×65) `veri/jestler/<etiket>/` altına kaydeder. "
                "Her jest için 30+ deneme; 'yok' sınıfını unutma.")
    GEREKSINIMLER = ("mediapipe",)
    Ayarlar = Ayarlar

    def __init__(self):
        self.sifirla()

    def sifirla(self):
        self.durum, self.zaman, self.tampon = "HAZIRLAN", time.time(), []

    def isle(self, kare, a: Ayarlar, anlat):
        eller = nk.el_noktalari(kare, 1)
        cikti = kare.copy()
        if eller:
            nk.iskelet_ciz(cikti, eller[0]["noktalar"], nk.EL_BAGLANTILARI)
        klasor = VERI / "jestler" / a.etiket
        adet = len(list(klasor.glob("*.npy"))) if klasor.exists() else 0
        if self.durum == "HAZIRLAN":
            kalan = a.ara_sn - (time.time() - self.zaman)
            if kalan <= 0:
                self.durum, self.tampon = "KAYIT", []
                anlat.adim("KAYIT başladı", kod=f"{DOSYA}:JestKaydet.isle")
            yazi_yaz(cikti, [f"HAZIRLAN {max(kalan, 0):.1f}s", f"{a.etiket}: {adet} ornek"], renk=(0, 255, 255))
        else:
            if not eller:
                anlat.uyari("El kayboldu → bu deneme iptal")
                self.durum, self.zaman = "HAZIRLAN", time.time()
            else:
                self.tampon.append(kare_ozelligi(eller[0]))
                if len(self.tampon) >= a.uzunluk:
                    klasor.mkdir(parents=True, exist_ok=True)
                    yol = klasor / f"{dosya_adi(a.oturum)}_{int(time.time() * 1000)}.npy"
                    np.save(yol, np.stack(self.tampon))
                    adet += 1
                    anlat.adim(f"Kaydedildi: {yol.name} şekil={np.stack(self.tampon).shape}", kod=f"{DOSYA}:kare_ozelligi")
                    self.durum, self.zaman = "HAZIRLAN", time.time()
            yazi_yaz(cikti, [f"KAYIT {len(self.tampon)}/{a.uzunluk}", f"{a.etiket}: {adet} ornek"], renk=(0, 0, 255))
        return Sonuc(goruntu=cikti, veri={"durum": self.durum, "adet": adet}, metin=f"{self.durum} — **{a.etiket}**: {adet} örnek")


if __name__ == "__main__":
    from ortak.cli import calistir
    calistir(JestKaydet)
