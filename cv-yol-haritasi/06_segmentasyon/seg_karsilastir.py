"""
seg_karsilastir.py — Instance segmentasyon (YOLO-seg) vs istemli segmentasyon (SAM), maske metrikleriyle.

NE YAPAR?
    1. YOLO-seg: Her nesne için sınıf + kutu + MASKE (instance segmentasyon). Hızlı, ama sadece
       eğitildiği sınıfları bilir (COCO: 80 sınıf).
    2. Aynı nesnenin kutusu SAM'e istem olarak verilir -> SAM maskesi. SAM sınıf bilmez ama
       sınırları genelde daha ince çizer.
    3. İki maske arasında IoU ve Dice hesaplanır (maske_ortak.py — sıfırdan).
       Yüksek uyum = iki model de nesnenin sınırında hemfikir. Düşük uyum -> ara görsele bak:
       hangisi daha doğru? (Genelde ince yapılar: parmaklar, kulplar, saç.)

KULLANIM ALANI: 06'nın asıl dersi "hangi segmentasyon ne zaman?":
    - Sınıflar sabit ve hız önemli -> YOLO-seg (fine-tune edilebilir: yolo11n-seg.pt + seg dataset)
    - "Şu şey"in maskesi lazım, sınıfı önemsiz (one-shot) -> SAM
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from ortak.arayuz import KareIsleyici, Sonuc, ayar  # noqa: E402
from ortak.cizim import id_rengi, kutu_ciz, maske_ortu, yazi_yaz  # noqa: E402
from ortak.yukle import modul_yukle  # noqa: E402

mo = modul_yukle("06_segmentasyon/maske_ortak.py")
yy = modul_yukle("02_yolo_finetune/yolo_yardimci.py")
DOSYA = "06_segmentasyon/seg_karsilastir.py"


@dataclass
class Ayarlar:
    model: str = ayar("yolo11n-seg.pt", "YOLO segmentasyon modeli", tur="dosya")
    sam_model: str = ayar("sam2.1_t.pt", "SAM ağırlığı", secenekler=["sam2.1_t.pt", "sam2.1_s.pt", "mobile_sam.pt"])
    guven: float = ayar(0.4, "YOLO güven eşiği", min=0.05, max=0.95, adim=0.05)
    en_fazla: int = ayar(4, "SAM ile karşılaştırılacak en fazla nesne (SAM yavaştır)", min=1, max=20)


class SegKarsilastir(KareIsleyici):
    ISIM = "YOLO-seg vs SAM (IoU / Dice)"
    BOLUM = "06"
    DOSYA = DOSYA
    ACIKLAMA = "YOLO-seg instance maskeleriyle, aynı kutuları istem olarak alan SAM maskelerini karşılaştırır; IoU ve Dice hesaplar."
    GEREKSINIMLER = ("ultralytics",)
    Ayarlar = Ayarlar

    def isle(self, kare, a: Ayarlar, anlat):
        r = yy.model_yukle(a.model).predict(kare, conf=a.guven, verbose=False, retina_masks=True)[0]
        kutular, siniflar, guvenler, isimler = yy.sonucu_ayikla(r)
        anlat.adim(f"YOLO-seg: {len(kutular)} nesne", kod="ultralytics:YOLO.predict (seg)")
        yolo_img, sam_img, tablo = kare.copy(), kare.copy(), []
        if r.masks is None:
            return Sonuc(goruntu=kare, metin="Nesne yok.")
        maskeler = r.masks.data.cpu().numpy().astype(bool)
        for i, (k, s, m) in enumerate(zip(kutular[:a.en_fazla], siniflar, maskeler)):
            sm = mo.sam_maske(kare, kutu=k, model=a.sam_model)
            iou, dc = mo.maske_iou(m, sm), mo.dice(m, sm)
            tablo.append({"nesne": f"{isimler[s]}#{i}", "IoU": round(iou, 3), "Dice": round(dc, 3),
                          "yolo_px": int(m.sum()), "sam_px": int(sm.sum())})
            anlat.adim(f"{isimler[s]}#{i}: IoU={iou:.3f} Dice={dc:.3f} (YOLO {m.sum()} px, SAM {sm.sum()} px)",
                       kod="06_segmentasyon/maske_ortak.py:maske_iou / dice")
            yolo_img = maske_ortu(yolo_img, m, id_rengi(i))
            sam_img = maske_ortu(sam_img, sm, id_rengi(i))
            kutu_ciz(yolo_img, k, f"{isimler[s]} {guvenler[i]:.2f}", id_rengi(i))
            kutu_ciz(sam_img, k, f"IoU {iou:.2f}", id_rengi(i))
        yazi_yaz(yolo_img, "YOLO-seg"); yazi_yaz(sam_img, "SAM (kutu istemi)")
        ozet = "| nesne | IoU | Dice | YOLO px | SAM px |\n|---|---|---|---|---|\n" + "\n".join(
            f"| {t['nesne']} | {t['IoU']} | {t['Dice']} | {t['yolo_px']} | {t['sam_px']} |" for t in tablo)
        return Sonuc(goruntu=np.hstack([yolo_img, sam_img]), veri={"karsilastirma": tablo}, metin=ozet)


if __name__ == "__main__":
    from ortak.cli import calistir
    calistir(SegKarsilastir)
