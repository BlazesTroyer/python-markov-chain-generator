"""
deney_karsilastir.py — "Bu değişiklik gerçekten işe yaradı mı?" sorusunu SAYIYLA cevaplar.

BİLİMSEL YÖNTEM (bütün ML işinde aynı):
    1. Her şeyi sabit tut (aynı veri, aynı tohum, aynı epoch, aynı model).
    2. SADECE BİR şeyi değiştir (ör. augmentation).
    3. Aynı ölçütle (val ve test mAP) karşılaştır.
    Farklar küçükse (±0.01) bu gürültü olabilir; farklı tohumlarla tekrarlamadan kesin konuşma.

HAZIR DENEYLER:
    augmentation  : kapali / varsayilan / agir             -> augmentation'ın değeri
    freeze        : 0 / 10                                  -> backbone'u dondurmanın etkisi
    model         : yolo11n / yolo11s                       -> model boyutu vs doğruluk
    veri          : data / data2 (ör. gruplu vs gruplamasız) -> sızıntı deneyi (01 README, Deney 1)
                    ya da augment_uret.py çıktısı vs orijinal -> offline augmentation'ın değeri

ÇIKTI: Karşılaştırma tablosu + çubuk grafik + veri/egitimler/deney_<ad>.md raporu
"""
from __future__ import annotations

import dataclasses
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ortak.arayuz import Eylem, Sonuc, ayar  # noqa: E402
from ortak.yukle import VERI, modul_yukle  # noqa: E402

eg = modul_yukle("02_yolo_finetune/egit.py")
ek = modul_yukle("01_veri_fabrikasi/etiket_kontrol.py")
DOSYA = "02_yolo_finetune/deney_karsilastir.py"

DENEYLER = {
    "augmentation": ("augmentation", ["kapali", "varsayilan", "agir"]),
    "freeze": ("freeze", [0, 10]),
    "model": ("model", ["yolo11n.pt", "yolo11s.pt"]),
    "veri": ("data", None),   # değerler: data ve data2 ayarlarından
}


@dataclass
class Ayarlar:
    deney: str = ayar("augmentation", "Hangi tek şey değiştirilsin?", secenekler=list(DENEYLER))
    data: str = ayar("veri/datasetler/sekiller/data.yaml", "Dataset (bütün varyantlarda aynı)", tur="dosya")
    data2: str = ayar("veri/datasetler/sekiller_hafifx2/data.yaml", "Sadece 'veri' deneyinde: ikinci dataset", tur="dosya")
    model: str = ayar("yolo11n.pt", "Model (model deneyi dışında sabit)")
    epochs: int = ayar(20, "Her varyant kaç epoch (kısa tut: karşılaştırma için yeter)", min=1, max=300)
    imgsz: int = ayar(416, "Görüntü boyutu", min=160, max=1280, adim=32)
    batch: int = ayar(16, "Batch", min=-1, max=128)
    cihaz: str = ayar("auto", "auto | cpu | 0 | mps")
    tohum: int = ayar(0, "Bütün varyantlarda aynı tohum", min=0, max=9999)


class DeneyKarsilastir(Eylem):
    ISIM = "Deney karşılaştır (tek değişken)"
    BOLUM = "02"
    DOSYA = DOSYA
    ACIKLAMA = ("Aynı veri/tohum/epoch ile **tek bir şeyi** değiştirerek birden çok eğitim yapar ve val/test mAP'lerini "
                "tablo + grafikle karşılaştırır. Augmentation'ın, freeze'in, model boyutunun ve veri sızıntısının etkisini ölç.")
    GEREKSINIMLER = ("ultralytics",)
    UZUN_SURER = True
    Ayarlar = Ayarlar

    def calistir(self, ayarlar: Ayarlar, anlat):
        alan, degerler = DENEYLER[ayarlar.deney]
        if ayarlar.deney == "veri":
            degerler = [ayarlar.data, ayarlar.data2]
        taban = eg.Ayarlar(data=ayarlar.data, model=ayarlar.model, epochs=ayarlar.epochs, imgsz=ayarlar.imgsz,
                           batch=ayarlar.batch, cihaz=ayarlar.cihaz, tohum=ayarlar.tohum, patience=0)
        anlat.adim(f"Deney: '{alan}' ∈ {degerler}; geri kalan her şey sabit", kod=f"{DOSYA}:DeneyKarsilastir.calistir")

        sonuclar = []
        for d in degerler:
            etiket = Path(str(d)).parent.name if ayarlar.deney == "veri" else str(d)
            isim = f"deney_{ayarlar.deney}_{etiket}".replace(".pt", "")
            varyant = dataclasses.replace(taban, isim=isim, **{alan: d})
            anlat.adim(f"Varyant başlıyor: {alan}={d}", kod="02_yolo_finetune/egit.py:egitim_calistir")
            t0 = time.time()
            ozet = eg.egitim_calistir(varyant, anlat)
            ozet["sure_dk"] = (time.time() - t0) / 60
            ozet["varyant"] = etiket
            sonuclar.append(ozet)

        tablo = "| varyant | val mAP50 | val mAP50-95 | test mAP50-95 | en iyi epoch | overfitting | süre (dk) |\n|---|---|---|---|---|---|---|\n"
        tablo += "\n".join(
            f"| {s['varyant']} | {s['mAP50']:.3f} | {s['mAP50_95']:.3f} | {s.get('test_mAP50_95', float('nan')):.3f} | "
            f"{s['en_iyi_epoch']}/{s['toplam_epoch']} | {'⚠' if s['overfitting'] else '—'} | {s['sure_dk']:.1f} |" for s in sonuclar)
        rapor = VERI / "egitimler" / f"deney_{ayarlar.deney}.md"
        rapor.parent.mkdir(parents=True, exist_ok=True)
        rapor.write_text(f"# Deney: {ayarlar.deney}\n\nSabit: {taban}\n\n{tablo}\n", encoding="utf-8")
        anlat.adim(f"Rapor yazıldı: {rapor}")
        grafik = ek.sutun_grafigi({s["varyant"]: int(round(s["mAP50_95"] * 1000)) for s in sonuclar}, "val mAP50-95 (x1000)")
        return Sonuc(goruntu=grafik, ara_gorseller={"karşılaştırma": grafik}, veri={"sonuclar": sonuclar, "rapor": str(rapor)},
                     metin=tablo + "\n\n> Fark ±0.01 civarındaysa gürültü olabilir: farklı `tohum` ile tekrarla.")


if __name__ == "__main__":
    from ortak.cli import calistir
    calistir(DeneyKarsilastir)
