"""
kayit.py — Panelde hangi projelerin, hangi sırayla görüneceğinin TEK listesi.

Yeni bir proje eklemek için:
    1. Proje dosyasında KareIsleyici ya da Eylem'den türeyen bir sınıf yaz.
    2. Aşağıdaki PROJELER listesine (dosya yolu, sınıf adı) ekle.
    Panel geri kalan her şeyi (sekme, form, butonlar) kendisi üretir.

Bir modül yüklenemezse (ör. sözdizimi hatası) panel çökmez; o proje sekmesinde hatayı gösterir.
"""
from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ortak.yukle import modul_yukle  # noqa: E402

BOLUMLER = {
    "00": "00 · Temeller",
    "01": "01 · Veri Fabrikası",
    "02": "02 · YOLO Fine-Tune",
    "03": "03 · Takip",
    "04": "04 · Yüz",
    "05": "05 · El / Poz / Jest",
    "06": "06 · Segmentasyon",
    "07": "07 · Embedding",
    "08": "08 · OCR",
    "09": "09 · Agent",
    "10": "10 · Dağıtım",
}

# (dosya yolu — cv-yol-haritasi/ köküne göre, sınıf adı)
PROJELER: list[tuple[str, str]] = [
    # 00 · Temeller
    ("00_temeller/augmentation_gorsellestir.py", "AugmentationGorsellestir"),
    ("00_temeller/metrikler.py", "MetrikOyunAlani"),
    ("00_temeller/hareket_kapisi.py", "HareketKapisi"),
    ("00_temeller/klasik_islemler.py", "KlasikIslemler"),
]


@dataclass
class YuklenenProje:
    dosya: str
    sinif_adi: str
    sinif: type | None     # yüklenemediyse None
    hata: str | None       # yüklenemediyse nedeni

    @property
    def bolum(self) -> str:
        return self.dosya[:2]


def projeleri_yukle() -> list[YuklenenProje]:
    sonuc = []
    for dosya, sinif_adi in PROJELER:
        try:
            modul = modul_yukle(dosya)
            sonuc.append(YuklenenProje(dosya, sinif_adi, getattr(modul, sinif_adi), None))
        except Exception:  # panel açılsın, hatayı sekmede göster
            sonuc.append(YuklenenProje(dosya, sinif_adi, None, traceback.format_exc()))
    return sonuc
