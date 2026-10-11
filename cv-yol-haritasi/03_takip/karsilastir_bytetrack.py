"""
karsilastir_bytetrack.py — Bizim sıfırdan SORT'umuz vs Ultralytics'in ByteTrack / BoT-SORT'u, AYNI videoda yan yana.

BYTETRACK NE FARK EDER?
    SORT, güven eşiğinin ALTINDAKİ tespitleri hiç görmez. Ama bir kişi kısmen kapandığında
    dedektör onu düşük güvenle (ör. 0.2) bulur — SORT bu bilgiyi çöpe atar, ID kopabilir.
    ByteTrack iki turda eşler:
        1. tur: YÜKSEK güvenli tespitler <-> izler (SORT gibi)
        2. tur: Eşleşmeyen izler <-> DÜŞÜK güvenli tespitler  ("her kutu değerlidir")
    BoT-SORT bunlara kamera hareketi telafisi (GMC) ve opsiyonel görünüm (ReID) ekler.

BU DOSYA:
    Sol panel : Ultralytics model.track(persist=True, tracker="bytetrack.yaml" | "botsort.yaml")
    Sağ panel : Aynı karedeki tespitler (güven >= guven) -> bizim Sort sınıfımız
    Üstte     : her birinin şimdiye kadar verdiği toplam ID sayısı. Aynı kişi sayısı için
                DAHA AZ ID = daha az kopma (genelde daha iyi takip).

NOT: Ultralytics takip durumunu modelin içinde tutar; bu yüzden bu demo kendi model örneğini
oluşturur (önbellekteki ortak modeli kirletmemek için). sifirla() modeli yeniden kurar.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from ortak.arayuz import KareIsleyici, Sonuc, ayar  # noqa: E402
from ortak.cizim import id_rengi, kutu_ciz, yazi_yaz  # noqa: E402
from ortak.yukle import gerekli, modul_yukle  # noqa: E402

st = modul_yukle("03_takip/sort_sifirdan.py")
to = modul_yukle("03_takip/takip_ortak.py")
th = modul_yukle("02_yolo_finetune/tahmin.py")
yy = modul_yukle("02_yolo_finetune/yolo_yardimci.py")
DOSYA = "03_takip/karsilastir_bytetrack.py"


@dataclass
class Ayarlar:
    model: str = ayar("yolo11n.pt", "YOLO modeli", tur="dosya")
    takipci: str = ayar("bytetrack", "Ultralytics takipçisi", secenekler=["bytetrack", "botsort"])
    guven: float = ayar(0.3, "Güven eşiği (SORT bunun altını hiç görmez; ByteTrack 2. turda kullanır)", min=0.05, max=0.9, adim=0.05)
    siniflar: str = ayar("person", "Takip edilecek sınıflar")
    max_yas: int = ayar(30, "SORT max_yas (ByteTrack'in track_buffer'ı varsayılan 30)", min=0, max=120)


class KarsilastirByteTrack(KareIsleyici):
    ISIM = "SORT (bizim) vs ByteTrack/BoT-SORT"
    BOLUM = "03"
    DOSYA = DOSYA
    ACIKLAMA = ("Aynı videoda sol: Ultralytics ByteTrack/BoT-SORT, sağ: sıfırdan SORT'umuz. Toplam verilen ID sayısını karşılaştır "
                "(aynı kişi sayısında daha az ID = daha az kopma). **Video sekmesiyle** kullan.")
    GEREKSINIMLER = ("ultralytics",)
    Ayarlar = Ayarlar

    def __init__(self):
        self.model = None
        self.sort = None
        self.ultra_idler: set[int] = set()

    def sifirla(self):
        self.model, self.sort, self.ultra_idler = None, None, set()

    def isle(self, kare, a: Ayarlar, anlat):
        if self.model is None:
            self.model = gerekli("ultralytics").YOLO(a.model)   # kendi örneği: takip durumu burada yaşar
            self.sort = st.Sort(a.max_yas, 3, 0.3)
        secili = th.sinif_filtresi(a.siniflar, self.model.names)

        # Sol: Ultralytics takip
        r = self.model.track(kare, persist=True, tracker=f"{a.takipci}.yaml", conf=0.1, classes=secili, verbose=False)[0]
        sol = kare.copy()
        if r.boxes is not None and r.boxes.id is not None:
            for k, i in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.id.cpu().numpy().astype(int)):
                self.ultra_idler.add(int(i))
                kutu_ciz(sol, k, f"ID {i}", id_rengi(int(i)))
        anlat.adim(f"{a.takipci}: toplam {len(self.ultra_idler)} ID", kod=f"ultralytics:YOLO.track(tracker={a.takipci}.yaml)")

        # Sağ: aynı ağırlıklarla AYRI bir model örneği üzerinden tespit + bizim SORT.
        # (Takip yapan örnekte predict çağırmak, içine kayıtlı takipçiyi de ilerletirdi -> karşılaştırma bozulurdu.)
        p = yy.model_yukle(a.model).predict(kare, conf=a.guven, classes=secili, verbose=False)[0]
        tespit = np.hstack([p.boxes.xyxy.cpu().numpy(), p.boxes.conf.cpu().numpy()[:, None], p.boxes.cls.cpu().numpy()[:, None]]).astype(np.float32)
        izler = self.sort.guncelle(tespit)
        sag = to.izleri_ciz(kare.copy(), izler, self.model.names)
        anlat.adim(f"SORT (sıfırdan): toplam {self.sort.sonraki_id - 1} ID", kod="03_takip/sort_sifirdan.py:Sort.guncelle")

        yazi_yaz(sol, [f"{a.takipci} (Ultralytics)", f"toplam ID: {len(self.ultra_idler)}"])
        yazi_yaz(sag, ["SORT (sifirdan)", f"toplam ID: {self.sort.sonraki_id - 1}"])
        return Sonuc(goruntu=np.hstack([sol, sag]),
                     veri={a.takipci: len(self.ultra_idler), "sort": self.sort.sonraki_id - 1},
                     metin=f"Toplam verilen ID — {a.takipci}: **{len(self.ultra_idler)}**, SORT: **{self.sort.sonraki_id - 1}**")


if __name__ == "__main__":
    from ortak.cli import calistir
    calistir(KarsilastirByteTrack)
