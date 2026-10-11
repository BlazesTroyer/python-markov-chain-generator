"""
copy_paste_aug.py — Saydam nesne kesitlerinden ETİKETLİ YOLO dataset'i sentezle ("6 fotoğraftan gerçek YOLO'ya").

COPY-PASTE AUGMENTATION NEDİR?
    Maskelenmiş nesneyi alıp FARKLI arka planlara, farklı boyut/açı/konumda yapıştırmak.
    Maske elimizde olduğu için etiket kutusu PİKSEL KESİNDİR (kutu_from_maske) — elle etiketleme yok!
    Literatürde (Ghiasi ve ark. 2021) instance segmentasyonda çok etkili bulunmuş basit bir yöntem.

HER SENTEZ GÖRÜNTÜDE:
    1. Rastgele bir arka plan seç (kendi odanın nesnesiz fotoğrafları en iyisi; yoksa sentetik üretilir).
    2. 1..en_fazla_nesne kadar kesit seç; her birine: ölçek, döndürme, yatay çevirme, parlaklık.
    3. Alfa kanalıyla karıştırarak yapıştır:  piksel = alfa * nesne + (1 - alfa) * arka_plan
    4. Yapıştırılan alfa maskesinden kutuyu hesapla; önceki nesnelerin üstüne binen kısmı onların
       maskesinden düş (gerçekçi kapanma) ve çok kapanan nesnenin etiketini at.
    5. 01'deki YOLO formatında yaz: veri/ham/<isim>/images|labels + siniflar.txt

SONRA: 01/etiket_kontrol.py → 01/dataset_bol.py → 02/egit.py   (aynı hat!)

UYARI: Sentetik veri gerçek veriyi TAMAMEN yerine koymaz. En iyi sonuç: sentetik + az sayıda gerçek
etiketli fotoğraf (özellikle val/test GERÇEK fotoğraflardan olmalı).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ortak.arayuz import Eylem, Sonuc, ayar  # noqa: E402
from ortak.cizim import id_rengi, izgara, kutu_ciz  # noqa: E402
from ortak.yukle import VERI, modul_yukle, yol_coz  # noqa: E402

mo = modul_yukle("06_segmentasyon/maske_ortak.py")
yf = modul_yukle("01_veri_fabrikasi/yolo_format.py")
od = modul_yukle("01_veri_fabrikasi/ornek_dataset.py")
DOSYA = "06_segmentasyon/copy_paste_aug.py"


@dataclass
class Ayarlar:
    kesit_klasoru: str = ayar("veri/nesneler", "Alt klasör = sınıf; içinde saydam PNG kesitler", tur="klasor")
    arka_plan_klasoru: str = ayar("", "Arka plan fotoğrafları (boş = sentetik arka plan üret)", tur="klasor",
                                  etki="Kendi odanın NESNESİZ fotoğrafları çok daha iyi sonuç verir")
    isim: str = ayar("kopyala_yapistir", "Çıktı adı -> veri/ham/<isim>/")
    goruntu_sayisi: int = ayar(300, "Kaç sentez görüntü", min=10, max=10000, adim=10)
    boyut: int = ayar(640, "Görüntü boyutu (kare)", min=160, max=1280, adim=32)
    en_fazla_nesne: int = ayar(3, "Görüntü başına en fazla nesne", min=1, max=10)
    olcek_min: float = ayar(0.15, "Nesne boyu / görüntü boyu (en az)", min=0.03, max=0.9, adim=0.01)
    olcek_max: float = ayar(0.45, "Nesne boyu / görüntü boyu (en çok)", min=0.05, max=1.0, adim=0.01)
    aci_max: float = ayar(30.0, "± döndürme (derece)", min=0.0, max=180.0, adim=5.0)
    bos_orani: float = ayar(0.1, "Nesnesiz (negatif) görüntü oranı", min=0.0, max=0.5, adim=0.05)
    tohum: int = ayar(0, "Tohum", min=0, max=9999)


def kesit_donustur(bgra: np.ndarray, olcek_px: int, aci: float, cevir: bool, parlaklik: float) -> np.ndarray:
    """Kesiti en uzun kenarı olcek_px olacak şekilde ölçekle, döndür, çevir, parlaklığını değiştir."""
    y, g = bgra.shape[:2]
    o = olcek_px / max(y, g)
    k = cv2.resize(bgra, (max(1, int(g * o)), max(1, int(y * o))), interpolation=cv2.INTER_AREA)
    if cevir:
        k = k[:, ::-1]
    if aci:
        y, g = k.shape[:2]
        M = cv2.getRotationMatrix2D((g / 2, y / 2), aci, 1.0)
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        yeni_g, yeni_y = int(y * sin + g * cos), int(y * cos + g * sin)   # dönen görüntü kesilmesin diye tuvali büyüt
        M[0, 2] += yeni_g / 2 - g / 2
        M[1, 2] += yeni_y / 2 - y / 2
        k = cv2.warpAffine(k, M, (yeni_g, yeni_y), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    k = k.copy()
    k[..., :3] = np.clip(k[..., :3].astype(np.float32) * parlaklik, 0, 255).astype(np.uint8)
    return k


def yapistir(tuval: np.ndarray, kesit: np.ndarray, x: int, y: int) -> np.ndarray:
    """Alfa karıştırma ile yapıştır; tuvaldeki NESNE MASKESİNİ (Y,G bool) döndür."""
    h, w = kesit.shape[:2]
    alfa = kesit[..., 3:4].astype(np.float32) / 255.0
    bolge = tuval[y:y + h, x:x + w].astype(np.float32)
    tuval[y:y + h, x:x + w] = (alfa * kesit[..., :3] + (1 - alfa) * bolge).astype(np.uint8)
    m = np.zeros(tuval.shape[:2], bool)
    m[y:y + h, x:x + w] = kesit[..., 3] > 127
    return m


class CopyPasteAug(Eylem):
    ISIM = "Copy-paste augmentation → YOLO dataset"
    BOLUM = "06"
    DOSYA = DOSYA
    ACIKLAMA = ("Saydam kesitleri farklı arka planlara rastgele yapıştırıp **etiketleri maskeden kesin olarak hesaplayarak** YOLO "
                "dataset'i üretir. Çıktı `veri/ham/<isim>/` → 01'deki kontrol/bölme → 02'de eğitim.")
    UZUN_SURER = True
    Ayarlar = Ayarlar

    def calistir(self, a: Ayarlar, anlat):
        rng = np.random.default_rng(a.tohum)
        kok = yol_coz(a.kesit_klasoru)
        siniflar = sorted(p.name for p in kok.iterdir() if p.is_dir() and any(p.glob("*.png")))
        kesitler = {i: [cv2.imread(str(f), cv2.IMREAD_UNCHANGED) for f in sorted((kok / s).glob("*.png"))] for i, s in enumerate(siniflar)}
        kesitler = {i: [k for k in ks if k is not None and k.ndim == 3 and k.shape[2] == 4] for i, ks in kesitler.items()}
        if not siniflar:
            raise FileNotFoundError(f"{kok} altında PNG kesit içeren sınıf klasörü yok (önce nesne_topla_sam.py)")
        anlat.adim(f"Sınıflar ve kesit sayıları: { {s: len(kesitler[i]) for i, s in enumerate(siniflar)} }", kod=f"{DOSYA}:CopyPasteAug.calistir")
        arka_planlar = []
        if a.arka_plan_klasoru:
            arka_planlar = [cv2.imread(str(p)) for p in yf.goruntuleri_listele(yol_coz(a.arka_plan_klasoru))]
            arka_planlar = [b for b in arka_planlar if b is not None]
        if not arka_planlar:
            anlat.uyari("Arka plan klasörü yok/boş → sentetik arka planlar kullanılıyor (gerçek oda fotoğrafları çok daha iyi).")

        hedef = VERI / "ham" / a.isim
        (hedef / "images").mkdir(parents=True, exist_ok=True)
        (hedef / "labels").mkdir(parents=True, exist_ok=True)
        (hedef / "siniflar.txt").write_text("\n".join(siniflar) + "\n", encoding="utf-8")
        B = a.boyut
        onizleme, atilan = [], 0
        for n in range(a.goruntu_sayisi):
            if arka_planlar:
                bg = arka_planlar[rng.integers(len(arka_planlar))]
                tuval = cv2.resize(bg, (B, B))
            else:
                tuval = od._arka_plan(rng, B, rng.uniform(40, 200, 3))
            maskeler, sinif_nolari = [], []
            if rng.random() >= a.bos_orani:
                for _ in range(rng.integers(1, a.en_fazla_nesne + 1)):
                    s = int(rng.integers(len(siniflar)))
                    if not kesitler[s]:
                        continue
                    k = kesit_donustur(kesitler[s][rng.integers(len(kesitler[s]))], int(B * rng.uniform(a.olcek_min, a.olcek_max)),
                                       rng.uniform(-a.aci_max, a.aci_max), rng.random() < 0.5, rng.uniform(0.7, 1.3))
                    h, w = k.shape[:2]
                    if h >= B or w >= B:
                        continue
                    yeni = yapistir(tuval, k, int(rng.integers(0, B - w)), int(rng.integers(0, B - h)))
                    maskeler = [m & ~yeni for m in maskeler]     # yeni nesne eskilerin üstünü kapatır
                    maskeler.append(yeni)
                    sinif_nolari.append(s)
            kutular, tutulan = [], []
            for m, s in zip(maskeler, sinif_nolari):
                kutu = mo.kutu_from_maske(m)
                if kutu is None or m.sum() < 0.002 * B * B:      # neredeyse tamamen kapanmış -> etiketleme
                    atilan += 1
                    continue
                kutular.append(kutu); tutulan.append(s)
            ad = f"cp_{n:05d}"
            cv2.imwrite(str(hedef / "images" / f"{ad}.jpg"), tuval)
            yf.etiket_yaz(hedef / "labels" / f"{ad}.txt", kutular, tutulan, B, B)
            if len(onizleme) < 8:
                g = tuval.copy()
                for kk, s in zip(kutular, tutulan):
                    kutu_ciz(g, kk, siniflar[s], id_rengi(s))
                onizleme.append(g)
        anlat.adim(f"{a.goruntu_sayisi} görüntü yazıldı → {hedef}; {atilan} nesne kapanma yüzünden etiketlenmedi",
                   kod="01_veri_fabrikasi/yolo_format.py:etiket_yaz")
        izg = izgara(onizleme, sutun=4)
        return Sonuc(goruntu=izg, ara_gorseller={"örnek sentez görüntüler (kutular maskeden)": izg},
                     veri={"klasor": str(hedef), "siniflar": siniflar, "atilan": atilan},
                     metin=f"✔ `{hedef}` — sınıflar: {siniflar}.\n\nSıradaki: 01 **Etiket kontrol** → **Dataset böl** (kaynak `veri/ham/{a.isim}`) → 02 **Eğit**.")


if __name__ == "__main__":
    from ortak.cli import calistir
    calistir(CopyPasteAug)
