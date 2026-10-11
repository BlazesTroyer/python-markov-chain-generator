"""
indeksle.py — Kişisel görsel hafıza, adım 1/2: fotoğraf klasörünü tara, her fotoğrafı vektöre çevir, kaydet.

AKIŞ:
    1. Klasörü (alt klasörler dahil) tara -> görüntü dosyaları
    2. Daha önce indekslenmiş ve DEĞİŞMEMİŞ (aynı değiştirilme zamanı) dosyaları atla (artımlı indeksleme)
    3. Kalanları 'parti' boyutunda gruplar halinde gömücüden geçir (gomucu.py)
    4. veri/indeksler/<isim>.npz olarak kaydet (vektor_indeksi.py)

GİZLİLİK: Fotoğraflar hiçbir yere gönderilmez; vektörler yerel diskte. (CLIP/DINOv2 ağırlıkları bir kez iner.)
SÜRE: CPU'da CLIP ViT-B/32 ile ~10-30 fotoğraf/sn. 10.000 fotoğraf ≈ 10 dk. İş paneldeki "İşler" sekmesinde izlenir.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2  # noqa: E402

from ortak.arayuz import Eylem, Sonuc, ayar  # noqa: E402
from ortak.yukle import modul_yukle, yol_coz  # noqa: E402

gm = modul_yukle("07_embedding/gomucu.py")
vi = modul_yukle("07_embedding/vektor_indeksi.py")
yf = modul_yukle("01_veri_fabrikasi/yolo_format.py")
DOSYA = "07_embedding/foto_hafiza/indeksle.py"


@dataclass
class Ayarlar:
    klasor: str = ayar("", "Fotoğraf klasörü (ör. telefon yedeği). Alt klasörler dahil", tur="klasor")
    gomucu: str = ayar("clip", "Gömücü (metinle arama için clip gerekir)", secenekler=gm.GOMUCULER,
                       etki="renk/yolo_cls/dinov2 sadece 'bu fotoğrafa benzeyenler' aramasını destekler")
    isim: str = ayar("galerim", "İndeks adı -> veri/indeksler/<isim>_<gomucu>.npz")
    parti: int = ayar(32, "Aynı anda kaç fotoğraf gömülsün", min=1, max=256)
    en_fazla: int = ayar(0, "En fazla kaç fotoğraf (0 = hepsi; deneme için küçük tut)", min=0, max=1000000)


class Indeksle(Eylem):
    ISIM = "Fotoğraf hafızası: indeksle"
    BOLUM = "07"
    DOSYA = DOSYA
    ACIKLAMA = "Bir fotoğraf klasörünü seçilen gömücüyle vektörlere çevirip yerel bir indekse kaydeder (artımlı: değişmeyenleri atlar)."
    GEREKSINIMLER = ("open_clip_torch (clip)", "timm (dinov2)", "ultralytics (yolo_cls)")
    UZUN_SURER = True
    Ayarlar = Ayarlar

    def calistir(self, a: Ayarlar, anlat):
        kok = yol_coz(a.klasor)
        if not a.klasor or not kok.exists():
            raise FileNotFoundError(f"Klasör yok: '{a.klasor}'")
        indeks_adi = f"{a.isim}_{a.gomucu}"
        indeks = vi.VektorIndeksi.yukle(indeks_adi) if vi.VektorIndeksi.yol(indeks_adi).exists() else vi.VektorIndeksi(a.gomucu)
        bilinen = dict(zip(indeks.yollar, indeks.zamanlar))
        dosyalar = yf.goruntuleri_listele(kok)
        if a.en_fazla:
            dosyalar = dosyalar[:a.en_fazla]
        yapilacak = [p for p in dosyalar if bilinen.get(str(p)) != p.stat().st_mtime]
        anlat.adim(f"{len(dosyalar)} fotoğraf bulundu, {len(yapilacak)} tanesi yeni/değişmiş (indekste {len(indeks)} kayıt)",
                   kod=f"{DOSYA}:Indeksle.calistir")
        g = gm.gomucu_al(a.gomucu)
        t0, islenen = time.time(), 0
        for i in range(0, len(yapilacak), a.parti):
            parti = yapilacak[i:i + a.parti]
            resimler, gecerli = [], []
            for p in parti:
                img = cv2.imread(str(p))
                if img is not None:
                    resimler.append(img); gecerli.append(p)
            if not resimler:
                continue
            V = g.goruntu(resimler)
            indeks.ekle(V, [str(p) for p in gecerli], [p.stat().st_mtime for p in gecerli])
            islenen += len(gecerli)
            anlat.adim(f"{islenen}/{len(yapilacak)} gömüldü ({islenen / (time.time() - t0):.1f} foto/sn), vektör boyutu {V.shape[1]}",
                       kod=f"07_embedding/gomucu.py:{type(g).__name__}.goruntu")
        yol = indeks.kaydet(indeks_adi)
        return Sonuc(veri={"indeks": str(yol), "kayit": len(indeks), "yeni": islenen},
                     metin=f"✔ `{yol}` — toplam {len(indeks)} fotoğraf ({islenen} yeni). Şimdi **Fotoğraf ara**.")


if __name__ == "__main__":
    from ortak.cli import calistir
    calistir(Indeksle)
