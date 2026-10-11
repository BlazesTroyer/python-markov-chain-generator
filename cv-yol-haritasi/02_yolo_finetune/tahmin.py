"""
tahmin.py — Eğittiğin (ya da hazır) YOLO modeliyle tespit + NMS'in içini açıp göstermek.

TESPİT ÇIKTISI NASIL OLUŞUR?
    1. Model görüntüyü imgsz'e küçültür (oranı koruyup kenarları gri doldurur = "letterbox").
    2. Ağ, binlerce ADAY kutu üretir (her biri: konum + her sınıf için skor).
    3. Güven eşiği (conf): skoru düşük adaylar atılır.
    4. NMS (Non-Maximum Suppression): AYNI nesne için üst üste binen adaylardan sadece en
       güvenlisi kalır. Kural: en yüksek skorlu kutuyu tut; onunla IoU'su > iou_esigi olan
       (aynı sınıftan) diğerlerini sil; kalanlarla tekrarla.
    5. Kutular orijinal görüntü boyutuna geri ölçeklenir.

    nms_goster=True iken:
        - Ultralytics'ten NMS'i neredeyse kapalı (iou=0.99) sonuçlar alınır  -> "ham adaylar"
        - Bunlara BİZİM nms_sifirdan() fonksiyonumuz uygulanır               -> "bizim NMS"
        - Ultralytics'in kendi NMS'li sonucuyla sayı karşılaştırılır          -> aynı olmalı (≈)
      Ara görsellerde üçünü yan yana görürsün.

ÇALIŞTIR:
    python 02_yolo_finetune/tahmin.py --kaynak 0 --model veri/egitimler/deneme/weights/best.pt --goster
    python 02_yolo_finetune/tahmin.py --kaynak foto.jpg --nms-goster --guven 0.1 --anlat
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from ortak.arayuz import KareIsleyici, Sonuc, ayar  # noqa: E402
from ortak.cizim import id_rengi, kutu_ciz, yazi_yaz  # noqa: E402
from ortak.yukle import modul_yukle  # noqa: E402

yy = modul_yukle("02_yolo_finetune/yolo_yardimci.py")
met = modul_yukle("00_temeller/metrikler.py")
DOSYA = "02_yolo_finetune/tahmin.py"


@dataclass
class Ayarlar:
    model: str = ayar("yolo11n.pt", "Model: hazır (yolo11n.pt = COCO 80 sınıf) ya da kendi best.pt yolun", tur="dosya")
    guven: float = ayar(0.25, "Güven eşiği (conf)", min=0.01, max=0.95, adim=0.01,
                        etki="Düşürürsen daha çok tespit + daha çok yanlış pozitif")
    iou: float = ayar(0.7, "NMS IoU eşiği", min=0.1, max=0.95, adim=0.05,
                      etki="Düşürürsen yan yana duran aynı sınıf nesneler birbirini silebilir; artırırsan çift kutular kalır")
    imgsz: int = ayar(640, "Çıkarım boyutu", min=160, max=1280, adim=32, etki="Küçültürsen hızlanır, küçük nesneleri kaçırır")
    siniflar: str = ayar("", "Sadece bu sınıflar (virgülle ad; boş = hepsi). Ör: 'cell phone,cup'")
    nms_goster: bool = ayar(False, "NMS öncesi/sonrası ve sıfırdan NMS karşılaştırmasını göster")


def nms_sifirdan(kutular: np.ndarray, skorlar: np.ndarray, siniflar: np.ndarray, iou_esigi: float) -> list[int]:
    """Sınıf bazlı NMS. Tutulan kutuların indekslerini döndürür."""
    tutulan = []
    for s in np.unique(siniflar):
        idx = np.where(siniflar == s)[0]
        sira = idx[np.argsort(-skorlar[idx])]           # bu sınıfın adayları, güvene göre azalan
        while len(sira):
            en_iyi = sira[0]
            tutulan.append(int(en_iyi))
            if len(sira) == 1:
                break
            ious = met.iou_matrisi(kutular[en_iyi:en_iyi + 1], kutular[sira[1:]])[0]
            sira = sira[1:][ious <= iou_esigi]          # en iyiyle çok örtüşenler silinir
    return sorted(tutulan, key=lambda i: -skorlar[i])


def sinif_filtresi(metin: str, isimler: dict) -> list[int] | None:
    if not metin.strip():
        return None
    istenen = {s.strip().lower() for s in metin.split(",")}
    return [no for no, ad in isimler.items() if ad.lower() in istenen]


class Tahmin(KareIsleyici):
    ISIM = "Tespit (tahmin) + NMS'in içi"
    BOLUM = "02"
    DOSYA = DOSYA
    ACIKLAMA = ("Hazır ya da kendi eğittiğin YOLO modeliyle tespit. `nms_goster` açıkken ham adayları, sıfırdan yazılmış "
                "NMS'i ve Ultralytics'in sonucunu yan yana gösterir.")
    GEREKSINIMLER = ("ultralytics",)
    Ayarlar = Ayarlar

    def isle(self, kare, ayarlar: Ayarlar, anlat):
        model = yy.model_yukle(ayarlar.model)
        secili = sinif_filtresi(ayarlar.siniflar, model.names)
        anlat.adim(f"Model: {ayarlar.model} ({len(model.names)} sınıf)", kod="02_yolo_finetune/yolo_yardimci.py:model_yukle",
                   veri={"filtre": secili})
        r = model.predict(kare, conf=ayarlar.guven, iou=ayarlar.iou, imgsz=ayarlar.imgsz, classes=secili, verbose=False)[0]
        kutular, siniflar, guvenler, isimler = yy.sonucu_ayikla(r)
        hiz = r.speed  # {'preprocess': ms, 'inference': ms, 'postprocess': ms}
        anlat.adim(f"{len(kutular)} tespit", kod="ultralytics:YOLO.predict",
                   veri={"on_isleme_ms": hiz["preprocess"], "ag_ms": hiz["inference"], "nms_ms": hiz["postprocess"]})

        cikti = kare.copy()
        for k, s, c in zip(kutular, siniflar, guvenler):
            kutu_ciz(cikti, k, f"{isimler[s]} {c:.2f}", id_rengi(s))
        yazi_yaz(cikti, [f"{len(kutular)} tespit", f"ag: {hiz['inference']:.0f} ms"])
        ara, veri = {}, {"tespit": [{"sinif": isimler[s], "guven": round(float(c), 3), "kutu": k.round().tolist()}
                                    for k, s, c in zip(kutular, siniflar, guvenler)]}

        if ayarlar.nms_goster:
            ham = model.predict(kare, conf=ayarlar.guven, iou=0.99, imgsz=ayarlar.imgsz, classes=secili, verbose=False)[0]
            hk, hs, hg, _ = yy.sonucu_ayikla(ham)
            tut = nms_sifirdan(hk, hg, hs, ayarlar.iou)
            anlat.adim(f"NMS: {len(hk)} ham aday → bizim NMS {len(tut)} kutu, Ultralytics {len(kutular)} kutu",
                       kod=f"{DOSYA}:nms_sifirdan", veri={"iou_esigi": ayarlar.iou})
            ham_img, biz_img = kare.copy(), kare.copy()
            for k, s in zip(hk, hs):
                kutu_ciz(ham_img, k, "", id_rengi(s), 1)
            for i in tut:
                kutu_ciz(biz_img, hk[i], f"{isimler[hs[i]]} {hg[i]:.2f}", id_rengi(hs[i]))
            yazi_yaz(ham_img, f"Ham adaylar (NMS ~kapali): {len(hk)}")
            yazi_yaz(biz_img, f"Sifirdan NMS (iou={ayarlar.iou}): {len(tut)}")
            ara = {"1_ham_adaylar": ham_img, "2_bizim_nms": biz_img, "3_ultralytics": cikti}
            veri["nms"] = {"ham": len(hk), "bizim": len(tut), "ultralytics": len(kutular)}

        return Sonuc(goruntu=cikti, ara_gorseller=ara, veri=veri,
                     metin=f"**{len(kutular)} tespit**: " + ", ".join(sorted({isimler[s] for s in siniflar})) if len(kutular) else "Tespit yok.")


if __name__ == "__main__":
    from ortak.cli import calistir
    calistir(Tahmin)
