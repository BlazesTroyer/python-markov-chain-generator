"""
ara.py — Kişisel görsel hafıza, adım 2/2: doğal dille ("dog on the beach") ya da örnek fotoğrafla ara.

İKİ SORGU TÜRÜ:
    metin   : CLIP metin kodlayıcısı -> metin vektörü -> görüntü vektörleriyle kosinüs (sadece 'clip' indeksinde)
              "Sıfır atış" (zero-shot): hiç etiketlemeden, hiç eğitmeden anlam bazlı arama.
    goruntu : Örnek fotoğrafın vektörü -> en benzer fotoğraflar (her gömücüyle çalışır)
              Farklı gömücülerin "benzerlik" anlayışını karşılaştır: renk mi, nesne türü mü, aynı nesne mi?

09_agent/mcp_server.py'deki search_photos() aracı fotograf_ara() fonksiyonunu kullanır.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2  # noqa: E402

from ortak.arayuz import Eylem, Sonuc, ayar  # noqa: E402
from ortak.cizim import izgara  # noqa: E402
from ortak.yukle import modul_yukle, yol_coz  # noqa: E402

gm = modul_yukle("07_embedding/gomucu.py")
vi = modul_yukle("07_embedding/vektor_indeksi.py")
DOSYA = "07_embedding/foto_hafiza/ara.py"


def fotograf_ara(sorgu: str, indeks_adi: str = "galerim_clip", k: int = 8, ornek_goruntu: str = "") -> list[tuple[str, float]]:
    """Metin ya da örnek görüntüyle ara -> [(yol, benzerlik)]. (09 agent bunu çağırır.)"""
    indeks = vi.VektorIndeksi.yukle(indeks_adi)
    g = gm.gomucu_al(indeks.gomucu)
    if ornek_goruntu:
        q = g.goruntu([cv2.imread(str(yol_coz(ornek_goruntu)))])[0]
    else:
        if not g.metin_destekli:
            raise ValueError(f"'{indeks.gomucu}' indeksi metinle aranamaz; clip ile indeksle ya da örnek görüntü ver")
        q = g.metin([sorgu])[0]
    return indeks.ara(q, k)


@dataclass
class Ayarlar:
    sorgu: str = ayar("a dog on the beach", "Metin sorgusu (CLIP İngilizce'de daha iyi)")
    ornek_goruntu: str = ayar("", "Doluysa metin yerine bu fotoğrafa benzeyenleri ara", tur="dosya")
    indeks: str = ayar("galerim_clip", "İndeks adı (indeksle.py: <isim>_<gomucu>)")
    k: int = ayar(8, "Kaç sonuç", min=1, max=48)


class FotografAra(Eylem):
    ISIM = "Fotoğraf hafızası: ara"
    BOLUM = "07"
    DOSYA = DOSYA
    ACIKLAMA = "Doğal dille (CLIP) ya da örnek fotoğrafla indekste arar; en benzer fotoğrafları benzerlik puanlarıyla gösterir."
    Ayarlar = Ayarlar

    def calistir(self, a: Ayarlar, anlat):
        sonuclar = fotograf_ara(a.sorgu, a.indeks, a.k, a.ornek_goruntu)
        anlat.adim(f"Sorgu {'görüntü: ' + a.ornek_goruntu if a.ornek_goruntu else 'metin: ' + repr(a.sorgu)} → {len(sonuclar)} sonuç",
                   kod="07_embedding/vektor_indeksi.py:VektorIndeksi.ara", veri={Path(y).name: round(s, 3) for y, s in sonuclar})
        resimler = [cv2.imread(y) for y, _ in sonuclar]
        izg = izgara([r for r in resimler if r is not None], [f"{s:.3f} {Path(y).name}" for (y, s), r in zip(sonuclar, resimler) if r is not None])
        return Sonuc(goruntu=izg, veri={"sonuclar": sonuclar},
                     metin="\n".join(f"{i + 1}. `{y}` — {s:.3f}" for i, (y, s) in enumerate(sonuclar)) +
                           "\n\n> CLIP benzerlikleri genelde 0.2–0.35 arasıdır; mutlak değer değil SIRALAMA önemlidir.")


if __name__ == "__main__":
    from ortak.cli import calistir
    calistir(FotografAra)
