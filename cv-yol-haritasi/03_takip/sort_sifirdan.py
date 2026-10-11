"""
sort_sifirdan.py — SORT (Simple Online and Realtime Tracking, Bewley 2016) SIFIRDAN.

SORT = IoU takipçi (iou_takipci.py) + Kalman tahmini (kalman.py). O kadar.
Ama bu küçük ekleme ID kopmalarını ciddi azaltır. Nedenini her karede ara görsellerde görürsün:
kesikli kutular = Kalman'ın TAHMİNİ, düz kutular = tespit. Eşleştirme tahminle yapılır.

HER KAREDE 6 ADIM (guncelle fonksiyonu, satır satır):
    1. TAHMİN    : Her izin Kalman'ı bir kare ileri tahmin yapar (sabit hız varsayımı).
    2. MALİYET   : Tespitler × tahminler IoU matrisi (farklı sınıflar 0).
    3. EŞLEŞTİR  : Macar algoritması; IoU < iou_esigi çiftler reddedilir.
    4. GÜNCELLE  : Eşleşen izin Kalman'ı tespitle düzeltilir (vurus++, kayip=0).
    5. DOĞUM/ÖLÜM: Eşleşmeyen tespit -> yeni iz (henüz "deneme" durumunda).
                   Eşleşmeyen iz -> kayip++ ; kayip > max_yas ise silinir.
    6. ÇIKTI     : Sadece "onaylı" izler raporlanır: en az min_vurus kez eşleşmiş ve bu karede görülmüş.
                   (min_vurus: tek karelik hayalet tespitlerin ID almasını engeller.)

PARAMETRELER:
    max_yas   : Görülmeden kaç kare yaşasın. Büyük = kısa kapanmalarda (biri önünden geçti) ID korunur.
    min_vurus : Kaç eşleşmeden sonra "gerçek" sayılsın.
    iou_esigi : Tahmin-tespit eşleşmesi için en düşük IoU.

BİLİNEN SINIR — İLK HIZ:
    Yeni bir izin Kalman hızı 0'dır (henüz bilinmez). Nesne İLK karelerden itibaren kare başına
    kendi boyunun büyük kısmı kadar hareket ediyorsa ilk tahmin tespitle örtüşmez ve iz hiç
    "tutunamaz" (her karede yeni ID). Çözümler: daha yüksek FPS, daha düşük iou_esigi, ya da
    ByteTrack/OC-SORT gibi gelişmiş ilişkilendirme. test_03_takip.py bu sınırı da gösterir.

SORT'UN SINIRI -> DeepSORT / ByteTrack:
    SORT sadece konuma bakar. İki kişi uzun süre üst üste binip ayrılırsa ID'ler karışabilir.
    DeepSORT görünüm (ReID embedding) ekler; ByteTrack düşük güvenli tespitleri ikinci turda eşler.
    karsilastir_bytetrack.py ikisini yan yana gösterir.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from ortak.arayuz import KareIsleyici, Sonuc, ayar  # noqa: E402
from ortak.cizim import id_rengi, kutu_ciz, yazi_yaz  # noqa: E402
from ortak.yukle import modul_yukle  # noqa: E402

met = modul_yukle("00_temeller/metrikler.py")
kal = modul_yukle("03_takip/kalman.py")
it = modul_yukle("03_takip/iou_takipci.py")
to = modul_yukle("03_takip/takip_ortak.py")
DOSYA = "03_takip/sort_sifirdan.py"


@dataclass
class Iz:
    id: int
    kalman: object
    sinif: int
    vurus: int = 1          # toplam eşleşme sayısı
    kayip: int = 0          # art arda kaç karedir görülmedi
    yas: int = 0            # kaç karedir var
    gecmis: list = field(default_factory=list)

    def sozluk(self) -> dict:
        k = self.kalman.kutu()
        return {"id": self.id, "kutu": k, "sinif": self.sinif, "gecmis": list(self.gecmis), "hiz": self.kalman.hiz(),
                "vurus": self.vurus, "kayip": self.kayip}


class Sort:
    def __init__(self, max_yas: int = 10, min_vurus: int = 3, iou_esigi: float = 0.3):
        self.max_yas, self.min_vurus, self.iou_esigi = max_yas, min_vurus, iou_esigi
        self.izler: list[Iz] = []
        self.sonraki_id = 1
        self.kare_no = 0
        self.son_adim: dict = {}

    def guncelle(self, tespitler: np.ndarray) -> list[dict]:
        tespitler = np.asarray(tespitler, np.float32).reshape(-1, 6)
        self.kare_no += 1

        # 1. TAHMİN
        tahminler = np.array([iz.kalman.tahmin() for iz in self.izler], np.float32).reshape(-1, 4)
        for iz in self.izler:
            iz.yas += 1

        # 2. MALİYET (IoU, farklı sınıf = 0)
        iou = met.iou_matrisi(tespitler[:, :4], tahminler)
        for j, iz in enumerate(self.izler):
            iou[tespitler[:, 5] != iz.sinif, j] = 0.0

        # 3. EŞLEŞTİR (Macar)
        eslesme, yeni, kayip = it.macar_eslestir(iou, self.iou_esigi)

        # 4. GÜNCELLE
        for i, j in eslesme:
            iz = self.izler[j]
            k = iz.kalman.guncelle(tespitler[i, :4])
            iz.vurus, iz.kayip = iz.vurus + 1, 0
            iz.gecmis.append(((k[0] + k[2]) / 2, (k[1] + k[3]) / 2))
            del iz.gecmis[:-30]

        # 5. DOĞUM / ÖLÜM
        for j in kayip:
            self.izler[j].kayip += 1
        for i in yeni:
            k = tespitler[i, :4]
            self.izler.append(Iz(self.sonraki_id, kal.KalmanKutu(k), int(tespitler[i, 5]),
                                 gecmis=[((k[0] + k[2]) / 2, (k[1] + k[3]) / 2)]))
            self.sonraki_id += 1
        silinen = [iz.id for iz in self.izler if iz.kayip > self.max_yas]
        self.izler = [iz for iz in self.izler if iz.kayip <= self.max_yas]

        # 6. ÇIKTI: onaylı ve bu karede görülen
        onayli = [iz.sozluk() for iz in self.izler
                  if iz.kayip == 0 and (iz.vurus >= self.min_vurus or self.kare_no <= self.min_vurus)]
        self.son_adim = {"tahminler": tahminler, "iou": iou, "eslesme": eslesme, "yeni": yeni, "kayip": kayip,
                         "silinen": silinen, "deneme": [iz.id for iz in self.izler if iz.vurus < self.min_vurus],
                         "gorunmeyen": [(iz.id, iz.kayip) for iz in self.izler if iz.kayip > 0]}
        return onayli

    def tum_izler(self) -> list[dict]:
        """Görülmeyenler dahil hepsi (çizimde 'hayalet' izleri göstermek için)."""
        return [iz.sozluk() for iz in self.izler]


@dataclass
class Ayarlar(to.TespitAyarlari):
    max_yas: int = ayar(15, "Görülmeden kaç kare yaşasın", min=0, max=120,
                        etki="Artırırsan kısa kapanmalarda ID korunur; çok artırırsan giden kişinin izi uzun süre 'hayalet' kalır")
    min_vurus: int = ayar(3, "Kaç eşleşmeden sonra iz onaylansın", min=1, max=20,
                          etki="Artırırsan tek kare hayalet tespitler ID almaz ama yeni gelenler geç görünür")
    iou_esigi: float = ayar(0.3, "Tahmin–tespit eşleşmesi için en düşük IoU", min=0.05, max=0.9, adim=0.05)
    hayaletleri_goster: bool = ayar(True, "Görülmeyen izlerin Kalman tahminini kesikli çiz")


class SortDemo(KareIsleyici):
    ISIM = "SORT (Kalman + Macar), sıfırdan"
    BOLUM = "03"
    DOSYA = DOSYA
    ACIKLAMA = ("Sıfırdan SORT. Ara görsellerde: **Kalman tahminleri (kesikli) vs tespitler (beyaz)**, IoU maliyet matrisi ve "
                "eşleşmeler. Log'da her karede doğan/ölen/kayıp izler. Kişi önünden biri geçince ID korunuyor mu, izle.")
    Ayarlar = Ayarlar

    def __init__(self):
        self.sort = None

    def sifirla(self):
        self.sort = None

    def isle(self, kare, ayarlar: Ayarlar, anlat):
        if self.sort is None:
            self.sort = Sort(ayarlar.max_yas, ayarlar.min_vurus, ayarlar.iou_esigi)
        self.sort.max_yas, self.sort.min_vurus, self.sort.iou_esigi = ayarlar.max_yas, ayarlar.min_vurus, ayarlar.iou_esigi
        tespitler, isimler = to.tespit_al(kare, ayarlar, anlat)
        izler = self.sort.guncelle(tespitler)
        a = self.sort.son_adim
        anlat.adim(f"1-TAHMİN: {len(a['tahminler'])} iz bir kare ileri tahmin edildi", kod="03_takip/kalman.py:KalmanKutu.tahmin")
        anlat.adim(f"2/3-EŞLEŞTİR: IoU {a['iou'].shape} → {len(a['eslesme'])} eşleşme", kod="03_takip/iou_takipci.py:macar_eslestir",
                   veri={"eslesmeler(tespit,iz)": a["eslesme"]})
        anlat.adim(f"4/5: yeni={len(a['yeni'])} kayıp={a['gorunmeyen']} silinen={a['silinen']} deneme={a['deneme']}",
                   kod=f"{DOSYA}:Sort.guncelle")

        tahmin_img = kare.copy()
        for t in tespitler:
            kutu_ciz(tahmin_img, t[:4], "tespit", (255, 255, 255), 1)
        for j, k in enumerate(a["tahminler"]):
            kutu_ciz(tahmin_img, k, f"tahmin iz{j}", (0, 255, 255), 2, kesikli=True)

        cikti = to.izleri_ciz(kare.copy(), izler, isimler, tespitler)
        if ayarlar.hayaletleri_goster:
            for iz in self.sort.tum_izler():
                if iz["kayip"] > 0:
                    kutu_ciz(cikti, iz["kutu"], f"ID {iz['id']} (kayip {iz['kayip']})", id_rengi(iz["id"]), 1, kesikli=True)
        yazi_yaz(cikti, [f"onayli iz: {len(izler)}", f"toplam ID: {self.sort.sonraki_id - 1}", f"kare: {self.sort.kare_no}"])
        return Sonuc(goruntu=cikti,
                     ara_gorseller={"Kalman tahmini (kesikli) vs tespit (beyaz)": tahmin_img,
                                    "IoU maliyet matrisi": to.maliyet_matrisi_gorsel(a["iou"], a["eslesme"])},
                     veri={"izler": [{"id": i["id"], "kutu": np.round(i["kutu"]).tolist(), "hiz": np.round(i["hiz"], 1).tolist()} for i in izler],
                           "toplam_id": self.sort.sonraki_id - 1},
                     metin=f"Onaylı iz: {len(izler)} — toplam verilen ID: {self.sort.sonraki_id - 1}")


if __name__ == "__main__":
    from ortak.cli import calistir
    calistir(SortDemo)
