"""
iou_takipci.py — EN BASİT takipçi: hareket modeli yok, sadece "önceki kutuyla en çok örtüşen".

ALGORİTMA (her karede):
    1. Mevcut izlerin SON kutuları ile yeni tespitler arasında IoU matrisi kur.   (N tespit × M iz)
    2. Eşleştir: Macar algoritması (scipy.optimize.linear_sum_assignment) toplam IoU'yu en
       büyükleyen bire-bir eşleşmeyi bulur. IoU < iou_esigi olan çiftler reddedilir.
    3. Eşleşen iz: kutusu tespitle değişir.  Eşleşmeyen tespit: YENİ ID.
       Eşleşmeyen iz: 'kayip' sayacı artar; max_kayip'i aşınca silinir.

ZAYIFLIKLARI (SORT bunları çözer — sentetik_test.py'de sayıyla görürsün):
    - Hızlı nesne: Bir karede kutu kendi genişliğinden fazla kayarsa IoU=0 -> yeni ID (ID kopması).
    - Kaçırma: Dedektör 2-3 kare kaçırırsa, nesne o sırada hareket ettiyse eski kutuyla örtüşmez.
    - Kesişen yollar: İki kişi yan yana geçerken ID'ler yer değiştirebilir (ID switch).

Neden yine de var? Algoritmanın iskeletini en saf haliyle görmek için. SORT = bu + Kalman tahmini.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from ortak.arayuz import KareIsleyici, Sonuc, ayar  # noqa: E402
from ortak.cizim import yazi_yaz  # noqa: E402
from ortak.yukle import gerekli, modul_yukle  # noqa: E402

met = modul_yukle("00_temeller/metrikler.py")
to = modul_yukle("03_takip/takip_ortak.py")
DOSYA = "03_takip/iou_takipci.py"


def macar_eslestir(iou: np.ndarray, esik: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """IoU matrisi (N tespit × M iz) -> (eşleşmeler, eşleşmeyen tespitler, eşleşmeyen izler).

    Macar algoritması MALİYETİ en aza indirir; IoU'yu en çoklamak için maliyet = -IoU.
    """
    n, m = iou.shape
    if n == 0 or m == 0:
        return [], list(range(n)), list(range(m))
    linear_sum_assignment = gerekli("scipy.optimize").linear_sum_assignment
    satir, sutun = linear_sum_assignment(-iou)
    eslesme = [(int(i), int(j)) for i, j in zip(satir, sutun) if iou[i, j] >= esik]
    et = {i for i, _ in eslesme}
    ei = {j for _, j in eslesme}
    return eslesme, [i for i in range(n) if i not in et], [j for j in range(m) if j not in ei]


class IouTakipci:
    """Algoritmanın kendisi (görüntüden bağımsız; testlerde sentetik kutularla çalışır)."""

    def __init__(self, iou_esigi: float = 0.3, max_kayip: int = 5):
        self.iou_esigi, self.max_kayip = iou_esigi, max_kayip
        self.izler: list[dict] = []
        self.sonraki_id = 1
        self.son_adim: dict = {}

    def guncelle(self, tespitler: np.ndarray) -> list[dict]:
        tespitler = np.asarray(tespitler, np.float32).reshape(-1, 6)
        iz_kutulari = np.array([iz["kutu"] for iz in self.izler], np.float32).reshape(-1, 4)
        iou = met.iou_matrisi(tespitler[:, :4], iz_kutulari)
        # Farklı sınıflar asla eşleşmesin
        for j, iz in enumerate(self.izler):
            iou[tespitler[:, 5] != iz["sinif"], j] = 0.0
        eslesme, yeni, kayip = macar_eslestir(iou, self.iou_esigi)
        for i, j in eslesme:
            iz = self.izler[j]
            iz["kutu"], iz["kayip"] = tespitler[i, :4].copy(), 0
            iz["gecmis"].append(((iz["kutu"][0] + iz["kutu"][2]) / 2, (iz["kutu"][1] + iz["kutu"][3]) / 2))
            del iz["gecmis"][:-30]
        for j in kayip:
            self.izler[j]["kayip"] += 1
        for i in yeni:
            k = tespitler[i, :4].copy()
            self.izler.append({"id": self.sonraki_id, "kutu": k, "sinif": int(tespitler[i, 5]), "kayip": 0,
                               "gecmis": [((k[0] + k[2]) / 2, (k[1] + k[3]) / 2)]})
            self.sonraki_id += 1
        silinen = [iz["id"] for iz in self.izler if iz["kayip"] > self.max_kayip]
        self.izler = [iz for iz in self.izler if iz["kayip"] <= self.max_kayip]
        self.son_adim = {"iou": iou, "eslesme": eslesme, "yeni": yeni, "kayip": kayip, "silinen": silinen}
        return [iz for iz in self.izler if iz["kayip"] == 0]


@dataclass
class Ayarlar(to.TespitAyarlari):
    iou_esigi: float = ayar(0.3, "Eşleşme için en düşük IoU", min=0.05, max=0.9, adim=0.05,
                            etki="Artırırsan hızlı nesnelerde ID kopar; azaltırsan yakın nesneler karışır")
    max_kayip: int = ayar(5, "Kaç kare görülmezse iz silinsin", min=0, max=100)


class IouTakipDemo(KareIsleyici):
    ISIM = "IoU takipçi (en basit)"
    BOLUM = "03"
    DOSYA = DOSYA
    ACIKLAMA = ("Hareket modeli olmadan, sadece önceki kutuyla örtüşmeye bakan takipçi. Ara görselde **IoU maliyet matrisi** "
                "ve Macar algoritmasının seçtiği eşleşmeler görünür. Hızlı harekette ID'lerin koptuğunu gözle.")
    Ayarlar = Ayarlar

    def __init__(self):
        self.takipci = None

    def sifirla(self):
        self.takipci = None

    def isle(self, kare, ayarlar: Ayarlar, anlat):
        if self.takipci is None:
            self.takipci = IouTakipci(ayarlar.iou_esigi, ayarlar.max_kayip)
        self.takipci.iou_esigi, self.takipci.max_kayip = ayarlar.iou_esigi, ayarlar.max_kayip
        tespitler, isimler = to.tespit_al(kare, ayarlar, anlat)
        izler = self.takipci.guncelle(tespitler)
        a = self.takipci.son_adim
        anlat.adim(f"IoU matrisi {a['iou'].shape} → Macar: {len(a['eslesme'])} eşleşme, {len(a['yeni'])} yeni ID, "
                   f"{len(a['kayip'])} kayıp iz, silinen {a['silinen']}", kod=f"{DOSYA}:IouTakipci.guncelle",
                   veri={"eslesmeler(tespit,iz)": a["eslesme"]})
        cikti = to.izleri_ciz(kare.copy(), izler, isimler, tespitler)
        yazi_yaz(cikti, [f"aktif iz: {len(izler)}", f"toplam ID: {self.takipci.sonraki_id - 1}"])
        return Sonuc(goruntu=cikti, ara_gorseller={"IoU maliyet matrisi (satır=tespit, sütun=iz)": to.maliyet_matrisi_gorsel(a["iou"], a["eslesme"])},
                     veri={"aktif": [iz["id"] for iz in izler], "toplam_id": self.takipci.sonraki_id - 1},
                     metin=f"Aktif iz: {len(izler)}, şimdiye kadar verilen ID: {self.takipci.sonraki_id - 1}")


if __name__ == "__main__":
    from ortak.cli import calistir
    calistir(IouTakipDemo)
