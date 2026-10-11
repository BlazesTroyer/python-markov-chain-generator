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
    # 01 · Veri Fabrikası (sırası = önerilen iş akışı)
    ("01_veri_fabrikasi/ornek_dataset.py", "OrnekDataset"),
    ("01_veri_fabrikasi/kare_cikar.py", "KareCikar"),
    ("01_veri_fabrikasi/otomatik_etiketle.py", "OtomatikEtiketle"),
    ("01_veri_fabrikasi/etiket_kontrol.py", "EtiketKontrol"),
    ("01_veri_fabrikasi/dataset_bol.py", "DatasetBol"),
    ("01_veri_fabrikasi/augment_uret.py", "AugmentUret"),
    # 02 · YOLO Fine-Tune
    ("02_yolo_finetune/egit.py", "Egit"),
    ("02_yolo_finetune/deney_karsilastir.py", "DeneyKarsilastir"),
    ("02_yolo_finetune/tahmin.py", "Tahmin"),
    ("02_yolo_finetune/telefon_dedektoru_v2.py", "TelefonDedektoruV2"),
    ("02_yolo_finetune/esya_hafizasi.py", "EsyaHafizasi"),
    ("02_yolo_finetune/esya_hafizasi.py", "NeredeSorgu"),
    # 03 · Takip
    ("03_takip/sentetik_test.py", "SentetikTest"),
    ("03_takip/iou_takipci.py", "IouTakipDemo"),
    ("03_takip/sort_sifirdan.py", "SortDemo"),
    ("03_takip/karsilastir_bytetrack.py", "KarsilastirByteTrack"),
    ("03_takip/ekran_suresi.py", "EkranSuresi"),
    ("03_takip/ekran_suresi.py", "EkranSuresiRaporu"),
    ("03_takip/bolge_sayaci.py", "BolgeSayaci"),
    # 04 · Yüz
    ("04_yuz/yuz_tanima/kaydet.py", "YuzFotografiTopla"),
    ("04_yuz/yuz_tanima/veritabani.py", "VeritabaniOlustur"),
    ("04_yuz/yuz_tanima/tani.py", "YuzTani"),
    ("04_yuz/emotion_v2/duygu_topla.py", "DuyguTopla"),
    ("04_yuz/emotion_v2/egit.py", "DuyguEgit"),
    ("04_yuz/emotion_v2/canli.py", "DuyguCanli"),
    # 05 · El / Poz / Jest
    ("05_el_poz/jest_canli.py", "JestCanli"),
    ("05_el_poz/jest_kaydet.py", "JestKaydet"),
    ("05_el_poz/jest_egit.py", "JestEgit"),
    ("05_el_poz/tekrar_sayaci.py", "TekrarSayaciDemo"),
    ("05_el_poz/ergonomi_kocu.py", "ErgonomiKocu"),
    # 06 · Segmentasyon
    ("06_segmentasyon/nesne_topla_sam.py", "NesneToplaSAM"),
    ("06_segmentasyon/copy_paste_aug.py", "CopyPasteAug"),
    ("06_segmentasyon/seg_karsilastir.py", "SegKarsilastir"),
    # 07 · Embedding
    ("07_embedding/foto_hafiza/indeksle.py", "Indeksle"),
    ("07_embedding/foto_hafiza/ara.py", "FotografAra"),
    ("07_embedding/oneshot_karsilastir.py", "OneShotKarsilastir"),
    # 08 · OCR
    ("08_ocr/belge_duzlestir.py", "BelgeDuzlestir"),
    ("08_ocr/fis_okuyucu.py", "FisOkuyucu"),
    ("08_ocr/fis_okuyucu.py", "GiderRaporu"),
    # 09 · Agent
    ("09_agent/olay_ureteci.py", "OlayUreteci"),
    ("09_agent/asistan.py", "Asistan"),
    ("09_agent/vlm_yorumla.py", "VlmYorumla"),
    # 10 · Dağıtım  (kamera_7x24.py panel için değil, sürekli çalışan betik — README'ye bak)
    ("10_dagitim/export.py", "Export"),
    ("10_dagitim/benchmark.py", "Benchmark"),
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
