"""
augment_uret.py — Bölünmüş bir dataset'in SADECE train kısmını çevrimdışı (offline) çoğaltır.

ÖNEMLİ: ÇEVRİMİÇİ vs ÇEVRİMDIŞI AUGMENTATION
    Çevrimiçi (online): Ultralytics eğitimde her epoch'ta görüntüleri anında rastgele değiştirir
        (mosaic, hsv, flip, scale...). Diske bir şey yazılmaz. 02'deki deneyler bunu açıp kapatır.
    Çevrimdışı (offline): Bu dosya. Değiştirilmiş kopyalar DİSKE yazılır.
        Ne zaman faydalı?  - Çok az verin varken (ör. one-shot'taki 6 fotoğraf)
                           - Kütüphanenin yapmadığı özel bir dönüşüm istiyorsan (ör. copy-paste, 06)
                           - Augmentation'ın etkisini GÖZLE incelemek istiyorsan

KURALLAR:
    - val ve test'e ASLA augmentation uygulanmaz (gerçek dünyayı ölçmeliler).
    - Orijinaller korunur; kopyalar "<ad>_aug<k>.jpg" olarak eklenir.

ÇIKTI: veri/datasetler/<kaynak_adı>_<politika>x<kopya>/   (val/test kaynakla aynı, train çoğaltılmış)
       Böylece 02'de "yok / hafif / ağır" versiyonlarını yan yana eğitip karşılaştırabilirsin.
"""
from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ortak.arayuz import Eylem, Sonuc, ayar  # noqa: E402
from ortak.cizim import id_rengi, izgara, kutu_ciz  # noqa: E402
from ortak.yukle import modul_yukle  # noqa: E402

yf = modul_yukle("01_veri_fabrikasi/yolo_format.py")
oe = modul_yukle("01_veri_fabrikasi/otomatik_etiketle.py")
aug = modul_yukle("00_temeller/augment_sifirdan.py")
DOSYA = "01_veri_fabrikasi/augment_uret.py"


@dataclass
class Ayarlar:
    dataset: str = ayar("veri/datasetler/sekiller", "Bölünmüş dataset (dataset_bol.py çıktısı)", tur="klasor")
    politika: str = ayar("hafif", "Augmentation seviyesi (00_temeller/augment_sifirdan.py:POLITIKALAR)",
                         secenekler=["yok", "hafif", "agir"])
    kopya: int = ayar(2, "Her train görüntüsü için kaç augment'li kopya", min=0, max=20,
                      etki="Artırırsan train büyür; ama aynı görüntünün kopyaları gerçek yeni veri kadar değerli değildir")
    tohum: int = ayar(0, "Rastgelelik tohumu", min=0, max=9999)


class AugmentUret(Eylem):
    ISIM = "Çevrimdışı augmentation ile train'i çoğalt"
    BOLUM = "01"
    DOSYA = DOSYA
    ACIKLAMA = ("Sadece **train** bölmesini seçili politikayla çoğaltır, val/test'e dokunmaz. Çıktı ayrı bir dataset "
                "klasörüdür → 02'de 'yok / hafif / ağır' versiyonlarını karşılaştırabilirsin.")
    Ayarlar = Ayarlar

    def calistir(self, ayarlar: Ayarlar, anlat):
        kaynak = oe.klasoru_coz(ayarlar.dataset)
        if not (kaynak / "data.yaml").exists():
            raise FileNotFoundError(f"{kaynak}/data.yaml yok — önce dataset_bol.py çalıştır")
        hedef = kaynak.parent / f"{kaynak.name}_{ayarlar.politika}x{ayarlar.kopya}"
        if hedef.exists():
            shutil.rmtree(hedef)
        shutil.copytree(kaynak, hedef)   # val/test birebir aynı kalır
        meta = yf.data_yaml_oku(kaynak / "data.yaml")
        names = meta["names"]
        isimler = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
        yf.data_yaml_yaz(hedef, isimler, test_var=(hedef / "images" / "test").exists())
        anlat.adim(f"{kaynak.name} → {hedef.name} kopyalandı (val/test değişmeden)", kod=f"{DOSYA}:AugmentUret.calistir")

        rng = np.random.default_rng(ayarlar.tohum)
        train = yf.goruntuleri_listele(hedef / "images" / "train")
        uretilen, ornekler, basliklar = 0, [], []
        for gp in train:
            img = cv2.imread(str(gp))
            y, g = img.shape[:2]
            kutular, siniflar, _ = yf.etiket_oku(yf.etiket_yolu(gp), g, y)
            for k in range(ayarlar.kopya):
                yeni, yk, ys, adimlar = aug.uygula(img.copy(), kutular, siniflar, ayarlar.politika, rng)
                ad = f"{gp.stem}_aug{k}"
                cv2.imwrite(str(gp.parent / f"{ad}.jpg"), yeni)
                yf.etiket_yaz(yf.etiket_yolu(gp.parent / f"{ad}.jpg"), yk, ys, g, y)
                uretilen += 1
                if len(ornekler) < 12:
                    goster = yeni.copy()
                    for kk, ss in zip(yk, ys):
                        kutu_ciz(goster, kk, isimler[ss], id_rengi(ss))
                    ornekler.append(goster); basliklar.append(ad)
                    anlat.bilgi(f"{ad}: {' → '.join(adimlar) or '(dönüşüm yok)'}", kod="00_temeller/augment_sifirdan.py:uygula")
        anlat.adim(f"{len(train)} train görüntüsünden {uretilen} kopya üretildi", veri={"train_yeni_toplam": len(train) + uretilen})
        izg = izgara(ornekler, basliklar) if ornekler else None
        return Sonuc(goruntu=izg, veri={"dataset": str(hedef), "uretilen": uretilen, "train_once": len(train)},
                     metin=f"✔ `{hedef}` — train: {len(train)} → {len(train) + uretilen}. data.yaml: `{hedef / 'data.yaml'}`")


if __name__ == "__main__":
    from ortak.cli import calistir
    calistir(AugmentUret)
