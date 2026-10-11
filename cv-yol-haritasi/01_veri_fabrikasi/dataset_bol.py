"""
dataset_bol.py — Ham dataset'i train / val / test olarak böler ve data.yaml yazar.

ÜÇ BÖLME NEDEN?
    train : model buradan öğrenir.
    val   : eğitim SIRASINDA "ne kadar iyi?" ölçümü; erken durdurma ve en iyi ağırlık seçimi buna göre.
    test  : EN SONDA, bir kez, dürüst ölçüm. Ayar yaparken val'e bakarsın; val'e göre ayar
            yapa yapa val'e de "dolaylı" uyum sağlarsın. test bu yüzden el değmemiş kalmalı.

EN SİNSİ HATA: VERİ SIZINTISI (leakage)
    Aynı videodan çıkmış 2 ardışık kare neredeyse aynıdır. Biri train'e, diğeri val'e düşerse
    model val'de "ezberden" bilir -> mAP 0.98 görürsün, gerçek hayatta 0.6 çıkar.
    ÇÖZÜM: GRUPLU BÖLME. Dosya adının grup kısmı (son '_' öncesi: "vid03_00012" -> "vid03")
    aynı olan bütün görüntüler AYNI bölmeye gider. kare_cikar.py adları buna göre verir.

ADIMLAR:
    1. images/ altındaki görüntüleri listele, her birinin grubunu bul.
    2. Grupları karıştır (tohumla — tekrarlanabilir) ve oranlara göre dağıt.
    3. Görüntü + etiketi hedef klasörlere KOPYALA (ham veri bozulmaz).
    4. data.yaml yaz (Ultralytics bunu okur).
"""
from __future__ import annotations

import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from ortak.arayuz import Eylem, Sonuc, ayar  # noqa: E402
from ortak.yukle import VERI, modul_yukle  # noqa: E402

yf = modul_yukle("01_veri_fabrikasi/yolo_format.py")
oe = modul_yukle("01_veri_fabrikasi/otomatik_etiketle.py")
ek = modul_yukle("01_veri_fabrikasi/etiket_kontrol.py")
DOSYA = "01_veri_fabrikasi/dataset_bol.py"


@dataclass
class Ayarlar:
    kaynak: str = ayar("veri/ham/sekiller", "Ham dataset klasörü (images/ + labels/ + siniflar.txt)", tur="klasor")
    isim: str = ayar("sekiller", "Çıktı adı -> veri/datasetler/<isim>/")
    val_orani: float = ayar(0.2, "Doğrulama (val) oranı", min=0.05, max=0.5, adim=0.05)
    test_orani: float = ayar(0.1, "Test oranı (0 = test bölmesi yok)", min=0.0, max=0.4, adim=0.05)
    grupla: bool = ayar(True, "Aynı gruptaki (aynı video) görüntüler aynı bölmeye gitsin",
                        etki="Kapatırsan sızıntı olur, val mAP'i yapay olarak yükselir — deneyip farkı gör!")
    tohum: int = ayar(0, "Karıştırma tohumu", min=0, max=9999)


def grup_adi(dosya: Path) -> str:
    """'vid03_00012.jpg' -> 'vid03'.  Alt çizgi yoksa dosyanın kendisi bir gruptur."""
    return dosya.stem.rsplit("_", 1)[0] if "_" in dosya.stem else dosya.stem


def gruplari_bol(gruplar: list[str], val_orani: float, test_orani: float, rng) -> dict[str, str]:
    """Grup -> bölme adı eşlemesi. Oranlar grup SAYISI üzerinden uygulanır."""
    gruplar = list(gruplar)
    rng.shuffle(gruplar)
    n = len(gruplar)
    n_test = int(round(n * test_orani))
    n_val = max(1, int(round(n * val_orani))) if n > 1 else 0
    eslesme = {}
    for i, gr in enumerate(gruplar):
        eslesme[gr] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")
    return eslesme


class DatasetBol(Eylem):
    ISIM = "Dataset böl (train/val/test, sızıntısız)"
    BOLUM = "01"
    DOSYA = DOSYA
    ACIKLAMA = ("Ham dataset'i **gruplu** (aynı videonun kareleri aynı bölmede) olarak train/val/test'e böler ve "
                "`data.yaml` yazar. `grupla` kapatılıp açılarak veri sızıntısının etkisi 02'de deneyle görülebilir.")
    Ayarlar = Ayarlar

    def calistir(self, ayarlar: Ayarlar, anlat):
        kaynak = oe.klasoru_coz(ayarlar.kaynak)
        hedef = VERI / "datasetler" / ayarlar.isim
        isimler = ek.sinif_isimleri(kaynak)
        if not isimler:
            raise FileNotFoundError(f"{kaynak}/siniflar.txt bulunamadı (sınıf adları gerekli)")
        goruntuler = yf.goruntuleri_listele(kaynak / "images")
        anlat.adim(f"{len(goruntuler)} görüntü bulundu", kod=f"{DOSYA}:DatasetBol.calistir")

        rng = np.random.default_rng(ayarlar.tohum)
        if ayarlar.grupla:
            gruplar = defaultdict(list)
            for g in goruntuler:
                gruplar[grup_adi(g)].append(g)
            anlat.adim(f"{len(gruplar)} grup bulundu (ör. {list(gruplar)[:4]})", kod=f"{DOSYA}:grup_adi")
        else:
            gruplar = {g.stem: [g] for g in goruntuler}   # her görüntü kendi grubu = rastgele bölme
            anlat.uyari("Gruplama KAPALI: ardışık kareler farklı bölmelere düşebilir (sızıntı).")
        eslesme = gruplari_bol(sorted(gruplar), ayarlar.val_orani, ayarlar.test_orani, rng)

        if hedef.exists():
            shutil.rmtree(hedef)   # eski bölmeyi temizle (ham veri ayrı yerde, güvende)
        sayac = Counter()
        sinif_bolme = defaultdict(Counter)
        for gr, dosyalar in gruplar.items():
            bolme = eslesme[gr]
            for gp in dosyalar:
                (hedef / "images" / bolme).mkdir(parents=True, exist_ok=True)
                (hedef / "labels" / bolme).mkdir(parents=True, exist_ok=True)
                shutil.copy2(gp, hedef / "images" / bolme / gp.name)
                kaynak_txt = kaynak / "labels" / (gp.stem + ".txt")
                hedef_txt = hedef / "labels" / bolme / (gp.stem + ".txt")
                if kaynak_txt.exists():
                    shutil.copy2(kaynak_txt, hedef_txt)
                    for satir in kaynak_txt.read_text().splitlines():
                        if satir.strip():
                            sinif_bolme[bolme][isimler[int(float(satir.split()[0]))]] += 1
                else:
                    hedef_txt.write_text("")   # etiketsiz = negatif örnek kabul edilir
                sayac[bolme] += 1
        yaml = yf.data_yaml_yaz(hedef, isimler, test_var=ayarlar.test_orani > 0)
        anlat.adim("Kopyalandı ve data.yaml yazıldı", kod="01_veri_fabrikasi/yolo_format.py:data_yaml_yaz",
                   veri={b: sayac[b] for b in ("train", "val", "test")})
        for b in ("train", "val", "test"):
            anlat.bilgi(f"{b}: {dict(sinif_bolme[b])}")

        ara = {f"{b} sınıf dağılımı": ek.sutun_grafigi(dict(sinif_bolme[b]), f"{b} ({sayac[b]} goruntu)") for b in ("train", "val", "test") if sayac[b]}
        return Sonuc(goruntu=next(iter(ara.values()), None), ara_gorseller=ara, veri={
            "data_yaml": str(yaml), "bolmeler": dict(sayac), "siniflar": isimler,
            "grup_eslesmesi": eslesme if len(eslesme) <= 100 else f"{len(eslesme)} grup"},
            metin=(f"✔ `{hedef}`\n\n| bölme | görüntü |\n|---|---|\n" +
                   "\n".join(f"| {b} | {sayac[b]} |" for b in ("train", "val", "test")) +
                   f"\n\n`data.yaml`: `{yaml}` → 02'de eğitimde bunu ver."))


if __name__ == "__main__":
    from ortak.cli import calistir
    calistir(DatasetBol)
