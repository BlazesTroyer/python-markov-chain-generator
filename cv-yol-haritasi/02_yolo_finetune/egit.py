"""
egit.py — YOLO'yu KENDİ dataset'inle fine-tune et; her hiperparametrenin ne yaptığı yanında yazılı.

FINE-TUNING NEDİR?
    Sıfırdan eğitim: milyonlarca görüntü + günlerce GPU. Fine-tuning: COCO'da (80 sınıf, 118k
    görüntü) zaten "kenar, köşe, doku, nesne parçası" görmeyi öğrenmiş ağırlıklardan başlayıp,
    sadece SENİN sınıflarını ayırt etmeyi öğretmek. 100–300 etiketli görüntü bile iş görür.

    YOLO'nun yapısı kabaca:
        backbone (omurga)  -> görüntüden özellik çıkarır      (genel bilgi — korunmaya değer)
        neck (boyun)       -> farklı ölçekleri birleştirir
        head (kafa)        -> kutu + sınıf tahmini             (senin sınıflarına göre YENİDEN kurulur)
    `freeze=10` ilk 10 katmanı (≈ backbone) dondurur: daha hızlı, az veride daha kararlı;
    ama nesnelerin COCO'dakilerden çok farklıysa (ör. özel parçalar) tam eğitim daha iyi olabilir.

ADIMLAR (bu dosyada):
    1. Ayarları Ultralytics argümanlarına çevir ve HEPSİNİ log'a yaz (gizli ayar kalmasın).
    2. model.train(...)  -> veri/egitimler/<isim>/ altına ağırlıklar, grafikler, results.csv
    3. results.csv'yi oku, egitim_yorumla() ile "overfitting var mı, en iyi epoch hangisi?" yorumla.
    4. (opsiyonel) test bölmesinde dürüst değerlendirme.
    5. Ultralytics'in ürettiği öğretici görselleri topla:
         train_batch0.jpg  -> MODELİN GERÇEKTE GÖRDÜĞÜ eğitim görüntüleri (mosaic + augmentation!)
         results.png       -> loss ve mAP eğrileri
         confusion_matrix_normalized.png -> hangi sınıf hangisiyle karışıyor
         val_batch0_pred.jpg -> doğrulama görüntülerinde tahminler

ÇALIŞTIR:
    python 02_yolo_finetune/egit.py --data veri/datasetler/sekiller/data.yaml --epochs 30 --anlat
"""
from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from ortak.arayuz import Eylem, Sonuc, ayar  # noqa: E402
from ortak.yukle import VERI, gerekli, modul_yukle, yol_coz  # noqa: E402

yy = modul_yukle("02_yolo_finetune/yolo_yardimci.py")
DOSYA = "02_yolo_finetune/egit.py"


@dataclass
class Ayarlar:
    data: str = ayar("veri/datasetler/sekiller/data.yaml", "Dataset tanımı (01/dataset_bol.py üretir)", tur="dosya")
    model: str = ayar("yolo11n.pt", "Başlangıç ağırlığı (n<s<m<l<x: büyüdükçe daha doğru ama yavaş)",
                      secenekler=["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolov8n.pt"],
                      etki="CPU'da / kişisel projede 'n' ile başla; yetmezse 's'")
    isim: str = ayar("deneme", "Çalıştırma adı -> veri/egitimler/<isim>/")
    epochs: int = ayar(30, "Kaç tur (bütün train verisinin modelden kaç kez geçeceği)", min=1, max=500,
                       etki="Az: yetersiz öğrenme. Çok: overfitting (patience bunu durdurur)")
    imgsz: int = ayar(640, "Eğitim görüntü boyutu (kare)", min=160, max=1280, adim=32,
                      etki="Büyütürsen küçük nesneler daha iyi ama ~karesel yavaşlar")
    batch: int = ayar(16, "Bir adımda kaç görüntü (-1: GPU belleğine göre otomatik)", min=-1, max=128,
                      etki="Bellek hatası alırsan düşür")
    lr0: float = ayar(0.01, "Başlangıç öğrenme oranı", min=0.0001, max=0.1, adim=0.0005,
                      etki="Fine-tune'da çok yüksek: önceden öğrenilen bozulur; çok düşük: yavaş öğrenir")
    freeze: int = ayar(0, "İlk kaç katman dondurulsun (0 = hepsi eğitilir, 10 ≈ backbone)", min=0, max=22,
                       etki="Az veri + COCO'ya benzer nesneler: 10 dene (hızlı, kararlı)")
    patience: int = ayar(20, "Val mAP bu kadar epoch iyileşmezse erken dur", min=0, max=200)
    augmentation: str = ayar("varsayilan", "Çevrimiçi augmentation profili (yolo_yardimci.AUG_ON_AYARLARI)",
                             secenekler=["kapali", "varsayilan", "agir"],
                             etki="Deneyde 'kapali' ile karşılaştır: val mAP farkı augmentation'ın değeridir")
    cihaz: str = ayar("auto", "auto | cpu | 0 (ilk GPU) | mps (Apple)")
    tohum: int = ayar(0, "Tekrarlanabilirlik tohumu", min=0, max=9999)
    test_degerlendir: bool = ayar(True, "Eğitim sonunda test bölmesinde de ölç (varsa)")


def egitim_yorumla(results_csv: Path, pencere: int = 5) -> dict:
    """results.csv'yi okuyup insan için yorum üret (saf fonksiyon — test edilir).

    Overfitting işareti: Son `pencere` epoch'ta train box_loss DÜŞERKEN val box_loss YÜKSELİYORSA.
    """
    with open(results_csv, newline="") as f:
        satirlar = [{k.strip(): float(v) for k, v in r.items() if v and v.strip()} for r in csv.DictReader(f)]
    if not satirlar:
        return {"yorum": ["results.csv boş"]}
    mk = next(k for k in satirlar[0] if k.startswith("metrics/mAP50-95"))
    m50 = next(k for k in satirlar[0] if k.startswith("metrics/mAP50(") or k == "metrics/mAP50")
    en_iyi = max(range(len(satirlar)), key=lambda i: satirlar[i][mk])
    yorum = [f"En iyi epoch: {en_iyi + 1}/{len(satirlar)} (mAP50-95={satirlar[en_iyi][mk]:.3f}, mAP50={satirlar[en_iyi][m50]:.3f})"]
    asiri = False
    if len(satirlar) >= pencere * 2:
        son, onceki = satirlar[-pencere:], satirlar[-2 * pencere:-pencere]
        ort = lambda rs, k: sum(r[k] for r in rs) / len(rs)  # noqa: E731
        tr_dustu = ort(son, "train/box_loss") < ort(onceki, "train/box_loss")
        val_artti = ort(son, "val/box_loss") > ort(onceki, "val/box_loss") * 1.02
        asiri = tr_dustu and val_artti
        if asiri:
            yorum.append("⚠ Overfitting işareti: train loss düşüyor, val loss yükseliyor. Daha çok veri / augmentation / daha az epoch.")
    if en_iyi < len(satirlar) - 1 and len(satirlar) > 3:
        yorum.append(f"Son epoch en iyi değil — best.pt {en_iyi + 1}. epoch'un ağırlıklarıdır (last.pt değil, best.pt kullan).")
    if en_iyi == len(satirlar) - 1 and len(satirlar) > 3:
        yorum.append("Hâlâ iyileşiyordu → epoch sayısını artırmak fayda edebilir.")
    return {"en_iyi_epoch": en_iyi + 1, "toplam_epoch": len(satirlar), "mAP50": satirlar[en_iyi][m50],
            "mAP50_95": satirlar[en_iyi][mk], "overfitting": asiri, "yorum": yorum}


def egitim_calistir(ayarlar: Ayarlar, anlat) -> dict:
    """Bir eğitimi çalıştır, özet sözlüğü döndür. deney_karsilastir.py de bunu çağırır."""
    gerekli("ultralytics", "pip install ultralytics")
    data = yol_coz(ayarlar.data)
    if not data.exists():
        raise FileNotFoundError(f"{data} yok — önce 01_veri_fabrikasi/dataset_bol.py")
    argumanlar = dict(
        data=str(data), epochs=ayarlar.epochs, imgsz=ayarlar.imgsz, batch=ayarlar.batch, lr0=ayarlar.lr0,
        patience=ayarlar.patience, seed=ayarlar.tohum, deterministic=True,
        project=str(VERI / "egitimler"), name=ayarlar.isim, exist_ok=True, plots=True, verbose=False,
        **yy.AUG_ON_AYARLARI[ayarlar.augmentation],
    )
    if ayarlar.freeze:
        argumanlar["freeze"] = ayarlar.freeze
    if ayarlar.cihaz != "auto":
        argumanlar["device"] = ayarlar.cihaz
    anlat.adim("Ultralytics'e giden TÜM argümanlar", kod=f"{DOSYA}:egitim_calistir", veri=argumanlar)

    model = yy.model_yukle(ayarlar.model)
    model.train(**argumanlar)
    kayit = Path(model.trainer.save_dir)
    anlat.adim(f"Eğitim bitti → {kayit}", kod="ultralytics:YOLO.train")

    ozet = egitim_yorumla(kayit / "results.csv")
    for y in ozet["yorum"]:
        anlat.bilgi(y, kod=f"{DOSYA}:egitim_yorumla")
    ozet["agirlik"] = str(kayit / "weights" / "best.pt")
    ozet["klasor"] = str(kayit)

    if ayarlar.test_degerlendir and "test:" in data.read_text():
        en_iyi = gerekli("ultralytics").YOLO(ozet["agirlik"])
        m = en_iyi.val(data=str(data), split="test", imgsz=ayarlar.imgsz, verbose=False, plots=False,
                       project=str(kayit), name="test", exist_ok=True)
        ozet["test_mAP50"], ozet["test_mAP50_95"] = float(m.box.map50), float(m.box.map)
        anlat.adim(f"TEST bölmesi (hiç görülmemiş): mAP50={ozet['test_mAP50']:.3f} mAP50-95={ozet['test_mAP50_95']:.3f}",
                   kod="ultralytics:YOLO.val(split='test')")
    return ozet


class Egit(Eylem):
    ISIM = "YOLO fine-tune (eğit)"
    BOLUM = "02"
    DOSYA = DOSYA
    ACIKLAMA = ("Kendi dataset'inle YOLO eğitir. Bütün argümanlar log'a yazılır; sonunda overfitting yorumu, en iyi epoch, "
                "test mAP'i ve **modelin gerçekte gördüğü (augment'li) eğitim görüntüleri** gösterilir.")
    GEREKSINIMLER = ("ultralytics",)
    UZUN_SURER = True
    Ayarlar = Ayarlar

    def calistir(self, ayarlar: Ayarlar, anlat):
        ozet = egitim_calistir(ayarlar, anlat)
        kayit = Path(ozet["klasor"])
        gorseller = {}
        for ad, aciklama in [("results.png", "loss ve mAP eğrileri"),
                             ("train_batch0.jpg", "modelin gördüğü eğitim görüntüleri (augment'li)"),
                             ("confusion_matrix_normalized.png", "karışıklık matrisi"),
                             ("val_batch0_pred.jpg", "val tahminleri"), ("val_batch0_labels.jpg", "val gerçek etiketler")]:
            if (kayit / ad).exists():
                gorseller[f"{ad} — {aciklama}"] = cv2.imread(str(kayit / ad))
        metin = "\n".join(f"- {y}" for y in ozet["yorum"])
        if "test_mAP50" in ozet:
            metin += f"\n- **Test**: mAP50={ozet['test_mAP50']:.3f}, mAP50-95={ozet['test_mAP50_95']:.3f}"
        metin += f"\n\nAğırlık: `{ozet['agirlik']}` → `tahmin.py --model` ile kullan."
        return Sonuc(goruntu=next(iter(gorseller.values()), None), ara_gorseller=gorseller, metin=metin, veri=ozet)


if __name__ == "__main__":
    from ortak.cli import calistir
    calistir(Egit)
