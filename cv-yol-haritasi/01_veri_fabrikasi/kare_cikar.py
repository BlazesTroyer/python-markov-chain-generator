"""
kare_cikar.py — Telefonla çektiğin videodan, etiketlemeye değer kareleri seçip çıkarır.

NEDEN SADECE "HER N KAREDE BİR" DEĞİL?
    30 FPS'lik bir videoda ardışık kareler neredeyse aynıdır. Hepsini etiketlemek:
      - zaman kaybı (aynı şeyi 30 kez etiketlersin),
      - YANILTICI (train ve val'e neredeyse aynı kareler düşer -> sahte yüksek mAP).
    Bu yüzden 3 filtre uygularız ve HER ATLANAN KARENİN NEDENİNİ log'a yazarız:

      1. Aralık         : her `her_n_kare` karede bir aday al.
      2. Bulanıklık     : Laplacian varyansı < esik ise at. (Laplacian = 2. türev = kenar
                          keskinliği. Bulanık görüntüde kenarlar yumuşak -> varyans düşük.)
      3. Tekrar (dHash) : Son KAYDEDİLEN kareyle "fark hash'i" çok benzerse at.
                          dHash: 9x8'e küçült, her pikseli sağ komşusuyla karşılaştır -> 64 bit.
                          İki hash arasındaki farklı bit sayısı (Hamming) küçükse görüntüler benzer.

ÇIKTI: veri/ham/<isim>/images/<video_adı>_<kare_no>.jpg
       (Dosya adındaki video adı, dataset_bol.py'nin GRUP bilgisidir — sızıntıyı önler.)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ortak.arayuz import Eylem, Sonuc, ayar  # noqa: E402
from ortak.cizim import izgara  # noqa: E402
from ortak.kaynak import kareleri_oku  # noqa: E402
from ortak.yukle import VERI  # noqa: E402

DOSYA = "01_veri_fabrikasi/kare_cikar.py"


@dataclass
class Ayarlar:
    video: str = ayar("", "Video dosyası ya da video klasörü (klasörse içindeki tüm videolar)", tur="dosya")
    isim: str = ayar("esyalarim", "Dataset adı (veri/ham/<isim>/images/ altına yazılır)")
    her_n_kare: int = ayar(10, "Her kaç karede bir aday alınsın", min=1, max=120,
                           etki="Küçültürsen daha çok ama birbirine benzer kare")
    bulaniklik_esigi: float = ayar(60.0, "Laplacian varyansı bunun altındaysa 'bulanık' say ve at", min=0, max=500, adim=5,
                                   etki="Artırırsan sadece çok keskin kareler kalır (hareketli videoda az kare)")
    benzerlik_esigi: int = ayar(6, "dHash Hamming mesafesi bunun altındaysa 'tekrar' say ve at (0-64)", min=0, max=30,
                                etki="Artırırsan daha çeşitli ama daha az kare")
    en_fazla: int = ayar(300, "Video başına en fazla kaç kare kaydedilsin", min=1, max=5000)


def keskinlik(gri: np.ndarray) -> float:
    """Laplacian varyansı: yüksek = keskin, düşük = bulanık."""
    return float(cv2.Laplacian(gri, cv2.CV_64F).var())


def dhash(gri: np.ndarray) -> int:
    """64 bitlik fark hash'i (algısal benzerlik için)."""
    kucuk = cv2.resize(gri, (9, 8), interpolation=cv2.INTER_AREA)
    bitler = (kucuk[:, 1:] > kucuk[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bitler), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


class KareCikar(Eylem):
    ISIM = "Videodan kare çıkar (bulanık/tekrar filtreli)"
    BOLUM = "01"
    DOSYA = DOSYA
    ACIKLAMA = ("Video(lar)dan etiketlemeye değer kareleri seçer: aralık + bulanıklık (Laplacian) + tekrar (dHash) filtresi. "
                "Atlanan her karenin nedeni log'da.")
    Ayarlar = Ayarlar

    def calistir(self, ayarlar: Ayarlar, anlat):
        kaynak = Path(ayarlar.video)
        if not ayarlar.video or not kaynak.exists():
            raise FileNotFoundError(f"Video bulunamadı: '{ayarlar.video}'")
        videolar = sorted(p for p in kaynak.iterdir() if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}) if kaynak.is_dir() else [kaynak]
        hedef = VERI / "ham" / ayarlar.isim / "images"
        hedef.mkdir(parents=True, exist_ok=True)
        anlat.adim(f"{len(videolar)} video → {hedef}", kod=f"{DOSYA}:KareCikar.calistir")

        istatistik = {"aday": 0, "bulanik": 0, "tekrar": 0, "kaydedilen": 0}
        onizleme, basliklar = [], []
        for v in videolar:
            son_hash, kaydedilen = None, 0
            for no, kare in kareleri_oku(str(v), atla=ayarlar.her_n_kare):
                istatistik["aday"] += 1
                gri = cv2.cvtColor(kare, cv2.COLOR_BGR2GRAY)
                k = keskinlik(gri)
                if k < ayarlar.bulaniklik_esigi:
                    istatistik["bulanik"] += 1
                    anlat.bilgi(f"{v.name} aday#{no}: ATLA bulanık (keskinlik={k:.0f} < {ayarlar.bulaniklik_esigi})", kod=f"{DOSYA}:keskinlik")
                    continue
                h = dhash(gri)
                if son_hash is not None and hamming(h, son_hash) < ayarlar.benzerlik_esigi:
                    istatistik["tekrar"] += 1
                    anlat.bilgi(f"{v.name} aday#{no}: ATLA tekrar (hamming={hamming(h, son_hash)})", kod=f"{DOSYA}:dhash")
                    continue
                ad = f"{v.stem}_{no * ayarlar.her_n_kare:06d}.jpg"
                cv2.imwrite(str(hedef / ad), kare)
                son_hash, kaydedilen = h, kaydedilen + 1
                istatistik["kaydedilen"] += 1
                if len(onizleme) < 12:
                    onizleme.append(kare); basliklar.append(f"{ad} k={k:.0f}")
                if kaydedilen >= ayarlar.en_fazla:
                    break
            anlat.adim(f"{v.name}: {kaydedilen} kare kaydedildi", kod=f"{DOSYA}:KareCikar.calistir")
        anlat.adim("Özet", veri=istatistik)
        izg = izgara(onizleme, basliklar) if onizleme else None
        return Sonuc(goruntu=izg, metin=(f"Aday {istatistik['aday']} → bulanık {istatistik['bulanik']}, tekrar {istatistik['tekrar']}, "
                                         f"**kaydedilen {istatistik['kaydedilen']}**.\n\nSıradaki: **Otomatik etiketle** (klasör: `veri/ham/{ayarlar.isim}`)"),
                     veri=istatistik)


if __name__ == "__main__":
    from ortak.cli import calistir
    calistir(KareCikar)
