"""
nesne_topla_sam.py — One-Shot Tarayıcı v2: elinde tuttuğun nesneyi SAM ile maskele, ELİ ÇIKAR, temiz kesit topla.

SENİN v1'İN İLE FARKI:
    v1: eli tarayıp "görmezden gel" mantığı + 6 fotoğraf -> nesnenin ne olduğunu kaba bir şekilde öğren.
    v2: Her karede PİKSEL DÜZEYİNDE maske:
        1. İSTEM   : İlk karede ekranın ortasındaki kutu (nesneyi oraya tut). Sonraki karelerde
                     ÖNCEKİ maskenin kutusu biraz genişletilerek istem yapılır -> nesneyi çevirirken
                     maske onu "takip eder". (Basitleştirilmiş takip; gerçek SAM 2 video belleği daha güçlü.)
        2. SAM     : İstemdeki nesnenin maskesi (nesne + muhtemelen tutan el).
        3. EL ÇIKAR: MediaPipe el noktaları -> el maskesi (dışbükey örtü + genişletme) -> SAM maskesinden çıkar.
        4. TEMİZLE : En büyük bağlı parça bırakılır (el çıkınca kalan kırıntılar atılır).
        5. KAYDET  : Kutuya kırpılmış, ALFA KANALLI (saydam arka planlı) PNG -> veri/nesneler/<isim>/
    Bu kesitler copy_paste_aug.py ile yüzlerce etiketli eğitim görüntüsüne dönüşür → 02'de gerçek YOLO.

ARA GÖRSELLER: istem kutusu + SAM maskesi | el maskesi | son maske | dama tahtası üzerinde kesit
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ortak.arayuz import KareIsleyici, Sonuc, ayar  # noqa: E402
from ortak.cizim import dosya_adi, kutu_ciz, maske_ortu, yazi_yaz  # noqa: E402
from ortak.yukle import VERI, EksikBagimlilik, modul_yukle  # noqa: E402

mo = modul_yukle("06_segmentasyon/maske_ortak.py")
DOSYA = "06_segmentasyon/nesne_topla_sam.py"


@dataclass
class Ayarlar:
    isim: str = ayar("nesnem", "Nesnenin adı (sınıf adı olur) -> veri/nesneler/<isim>/")
    kutu: str = ayar("0.3,0.25,0.7,0.75", "İlk istem kutusu (oran x1,y1,x2,y2): nesneyi bu bölgeye tut")
    takip: bool = ayar(True, "Sonraki karelerde önceki maskenin kutusunu istem olarak kullan",
                       etki="Kapatırsan her karede sabit kutu kullanılır (nesneyi hep ortada tutmalısın)")
    el_cikar: bool = ayar(True, "MediaPipe ile eli bulup maskeden çıkar (v1'deki 'eli görmezden gel'in piksel hali)")
    el_genislet: int = ayar(25, "El maskesini kaç piksel genişlet", min=0, max=80,
                            etki="Artırırsan elin kenarları kesin çıkar ama nesneden de pay kesilebilir")
    min_alan_orani: float = ayar(0.01, "Maske bu orandan küçükse kaydetme (boş/bozuk maske)", min=0.001, max=0.3, adim=0.001)
    aralik_sn: float = ayar(0.5, "İki kayıt arası en az süre", min=0.0, max=10.0, adim=0.1)
    kaydet: bool = ayar(True, "Kesitleri diske kaydet (kapalıysa sadece göster)")
    sam_model: str = ayar("sam2.1_t.pt", "SAM ağırlığı", secenekler=["sam2.1_t.pt", "sam2.1_s.pt", "mobile_sam.pt"])


class NesneToplaSAM(KareIsleyici):
    ISIM = "One-shot tarayıcı v2 (SAM + el çıkarma)"
    BOLUM = "06"
    DOSYA = DOSYA
    ACIKLAMA = ("Elinde tuttuğun nesneyi SAM ile maskeler, MediaPipe el maskesini çıkarır ve saydam arka planlı kesitleri "
                "`veri/nesneler/<isim>/` altına kaydeder. Nesneyi yavaşça çevir. Sonra **Copy-paste augmentation**.")
    GEREKSINIMLER = ("ultralytics", "mediapipe (el çıkarma)")
    Ayarlar = Ayarlar

    def __init__(self):
        self.sifirla()

    def sifirla(self):
        self.onceki_kutu, self.son_kayit = None, 0.0

    def isle(self, kare, a: Ayarlar, anlat):
        y, g = kare.shape[:2]
        if a.takip and self.onceki_kutu is not None:
            x1, y1, x2, y2 = self.onceki_kutu
            pw, ph = (x2 - x1) * 0.1, (y2 - y1) * 0.1                    # %10 genişlet: nesne biraz kaymış olabilir
            istem = np.array([max(0, x1 - pw), max(0, y1 - ph), min(g, x2 + pw), min(y, y2 + ph)])
            kaynak = "önceki maske"
        else:
            istem = np.array([float(v) for v in a.kutu.split(",")]) * [g, y, g, y]
            kaynak = "sabit kutu"
        sam = mo.sam_maske(kare, kutu=istem, model=a.sam_model)
        anlat.adim(f"SAM ({kaynak} istemi): maske alanı %{sam.mean() * 100:.1f}", kod="06_segmentasyon/maske_ortak.py:sam_maske")

        ara = {}
        istem_img = maske_ortu(kare, sam, (255, 0, 0))
        kutu_ciz(istem_img, istem, f"istem ({kaynak})", (0, 255, 255), kesikli=True)
        ara["1_istem + SAM maskesi"] = istem_img
        son = sam.copy()
        if a.el_cikar:
            try:
                nk = modul_yukle("05_el_poz/noktalar.py")
                eller = nk.el_noktalari(kare, 2)
                el = mo.el_maskesi([e["noktalar"] for e in eller], (y, g), a.el_genislet)
                ortak = np.logical_and(son, el).sum()
                son = np.logical_and(son, ~el)
                anlat.adim(f"{len(eller)} el bulundu; SAM maskesinden {ortak} piksel el çıkarıldı", kod="06_segmentasyon/maske_ortak.py:el_maskesi")
                ara["2_el maskesi"] = maske_ortu(kare, el, (0, 0, 255))
            except EksikBagimlilik as e:
                anlat.uyari(f"El çıkarma atlandı: {e}")
        son = mo.en_buyuk_parca(son)
        ara["3_son maske"] = maske_ortu(kare, son, (0, 200, 0))
        alan = son.mean()
        cikti = maske_ortu(kare, son, (0, 200, 0))
        kesit = mo.kesit_rgba(kare, son)
        if kesit.shape[0] > 1:
            ara["4_kesit (saydam)"] = mo.dama_uzerinde(kesit)

        durum = "maske cok kucuk"
        if alan >= a.min_alan_orani:
            self.onceki_kutu = mo.kutu_from_maske(son)
            kutu_ciz(cikti, self.onceki_kutu, a.isim)
            durum = "hazir"
            if a.kaydet and time.time() - self.son_kayit >= a.aralik_sn:
                klasor = VERI / "nesneler" / dosya_adi(a.isim)
                klasor.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(klasor / f"{int(time.time() * 1000)}.png"), kesit)
                self.son_kayit, durum = time.time(), "KAYDEDILDI"
                anlat.adim(f"Kesit kaydedildi → {klasor}", kod="06_segmentasyon/maske_ortak.py:kesit_rgba")
        else:
            self.onceki_kutu = None                                     # takibi kaybettik: sabit kutuya dön
            anlat.uyari(f"Maske alanı %{alan * 100:.2f} < %{a.min_alan_orani * 100:.1f} — takip sıfırlandı")
        klasor = VERI / "nesneler" / dosya_adi(a.isim)
        adet = len(list(klasor.glob("*.png"))) if klasor.exists() else 0
        yazi_yaz(cikti, [f"{a.isim}: {adet} kesit", durum])
        return Sonuc(goruntu=cikti, ara_gorseller=ara, veri={"alan_orani": round(float(alan), 4), "kesit": adet},
                     metin=f"**{a.isim}**: {adet} kesit ({durum})")


if __name__ == "__main__":
    from ortak.cli import calistir
    calistir(NesneToplaSAM)
