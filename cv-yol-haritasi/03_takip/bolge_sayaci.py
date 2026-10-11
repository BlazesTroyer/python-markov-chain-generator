"""
bolge_sayaci.py — Çizgi geçişi sayacı: "Odaya kaç kişi girdi / çıktı?" (takibin klasik uygulaması)

NEDEN TAKİP ŞART?
    Tespit tek başına "çizginin solunda 1 kişi var" der. Bir kişinin çizgiyi GEÇTİĞİNİ
    anlamak için aynı kişinin önceki karede hangi tarafta olduğunu bilmek gerekir -> ID.

GEOMETRİ (taraf testi — çapraz çarpım):
    Çizgi A->B, nokta P.  taraf = sign( (B-A) × (P-A) ) = sign( (Bx-Ax)(Py-Ay) - (By-Ay)(Px-Ax) )
    + ise çizginin bir yanında, - ise diğer yanında. Bir ID'nin tarafı +'dan -'ye dönerse "giriş",
    -'den +'ya dönerse "çıkış" sayılır. Görüntüde y aşağı doğru arttığı için varsayılan dikey çizgide
    (yukarıdan aşağı çizilmiş) SOL taraf +, SAĞ taraf -'dir: soldan sağa geçiş = giriş.
    Kapın ters yöndeyse ayarlardaki 'yon_ters' ile çevir.

    Kullanılan nokta: kutunun ALT ORTASI (ayakların yeri) — perspektifte zemin çizgisiyle daha tutarlı.

OLAY: {"tur": "cizgi_gecildi", "kimlik": id, "ek": {"yon": "giris"|"cikis"}}
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ortak.arayuz import KareIsleyici, Sonuc, ayar  # noqa: E402
from ortak.cizim import yazi_yaz  # noqa: E402
from ortak.yukle import modul_yukle  # noqa: E402

st = modul_yukle("03_takip/sort_sifirdan.py")
to = modul_yukle("03_takip/takip_ortak.py")
DOSYA = "03_takip/bolge_sayaci.py"


def taraf(a, b, p) -> int:
    """Noktanın A->B çizgisine göre tarafı: +1, -1 ya da 0 (tam üstünde)."""
    capraz = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    return int(np.sign(capraz))


def segment_icinde(a, b, p) -> bool:
    """Nokta, çizginin uçları arasındaki şeritte mi? (Sonsuz doğru değil, SEGMENT sayalım.)"""
    ab = np.subtract(b, a)
    t = np.dot(np.subtract(p, a), ab) / max(np.dot(ab, ab), 1e-9)
    return 0.0 <= t <= 1.0


@dataclass
class Ayarlar(to.TespitAyarlari):
    cizgi: str = ayar("0.5,0.0,0.5,1.0", "Çizgi (oran): x1,y1,x2,y2. Varsayılan: dikey orta çizgi")
    yon_ters: bool = ayar(False, "Giriş/çıkış yönünü ters çevir")


class BolgeSayaci(KareIsleyici):
    ISIM = "Çizgi geçişi sayacı (giriş/çıkış)"
    BOLUM = "03"
    DOSYA = DOSYA
    ACIKLAMA = "SORT izlerinin çizgiyi hangi yönde geçtiğini çapraz çarpımla bulur; giriş/çıkış sayar ve olay üretir."
    Ayarlar = Ayarlar

    def __init__(self):
        self.sifirla()

    def sifirla(self):
        self.sort = st.Sort(max_yas=20, min_vurus=3, iou_esigi=0.3)
        self.son_taraf: dict[int, int] = {}
        self.sayac = {"giris": 0, "cikis": 0}

    def isle(self, kare, a: Ayarlar, anlat):
        y, g = kare.shape[:2]
        x1, y1, x2, y2 = (float(v) for v in a.cizgi.split(","))
        A, B = (x1 * g, y1 * y), (x2 * g, y2 * y)
        tespitler, isimler = to.tespit_al(kare, a, anlat)
        izler = self.sort.guncelle(tespitler)
        olaylar = []
        for iz in izler:
            k = iz["kutu"]
            ayak = ((k[0] + k[2]) / 2, k[3])
            t = taraf(A, B, ayak)
            onceki = self.son_taraf.get(iz["id"])
            if t != 0:
                if onceki is not None and onceki != t and segment_icinde(A, B, ayak):
                    yon = "giris" if (onceki > t) != a.yon_ters else "cikis"   # + -> - = giriş
                    self.sayac[yon] += 1
                    olaylar.append({"tur": "cizgi_gecildi", "kaynak": DOSYA, "nesne": isimler.get(int(iz["sinif"]), ""),
                                    "kimlik": iz["id"], "ek": {"yon": yon}})
                    anlat.adim(f"ID {iz['id']} çizgiyi geçti: taraf {onceki:+d} → {t:+d} = {yon}", kod=f"{DOSYA}:taraf")
                self.son_taraf[iz["id"]] = t
        cikti = to.izleri_ciz(kare.copy(), izler, isimler)
        cv2.line(cikti, tuple(map(int, A)), tuple(map(int, B)), (0, 255, 255), 3)
        yazi_yaz(cikti, [f"GIRIS: {self.sayac['giris']}", f"CIKIS: {self.sayac['cikis']}", f"icerde (tahmini): {self.sayac['giris'] - self.sayac['cikis']}"])
        return Sonuc(goruntu=cikti, olaylar=olaylar, veri=dict(self.sayac),
                     metin=f"Giriş **{self.sayac['giris']}**, çıkış **{self.sayac['cikis']}**")


if __name__ == "__main__":
    from ortak.cli import calistir
    calistir(BolgeSayaci)
