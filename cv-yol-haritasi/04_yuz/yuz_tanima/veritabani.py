"""
veritabani.py — Yüz tanıma, adım 2/3: embedding veritabanı kur ve EŞİĞİ ÖLÇ.

    veri/yuzler/<isim>/*.jpg  ->  her fotoğrafın 128 boyutlu embedding'i  ->  veri/yuz_db.npz
                                  (isimler: (N,), vektorler: (N,128))

EŞİK ANALİZİ (esik_analizi fonksiyonu):
    Bütün fotoğraf çiftleri için kosinüs benzerliği hesaplanır ve iki gruba ayrılır:
        "aynı kişi" çiftleri   (yeşil histogram)
        "farklı kişi" çiftleri (kırmızı histogram)
    Her olası eşik için:
        Yanlış Kabul (FAR): farklı kişiyi "aynı" sanma oranı   -> güvenlik riski
        Yanlış Ret  (FRR): aynı kişiyi "farklı" sanma oranı    -> kullanım rahatsızlığı
    FAR + FRR'yi en küçük yapan eşik önerilir. İki histogram ne kadar ayrıksa tanıma o kadar güvenilir.

GİZLİLİK: Veritabanında sadece embedding (128 sayı) ve isim tutulur.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ortak.arayuz import Eylem, Sonuc, ayar  # noqa: E402
from ortak.cizim import izgara  # noqa: E402
from ortak.yukle import VERI, modul_yukle, yol_coz  # noqa: E402

yo = modul_yukle("04_yuz/yuz_ortak.py")
DOSYA = "04_yuz/yuz_tanima/veritabani.py"
DB_YOLU = VERI / "yuz_db.npz"


def esik_analizi(isimler: np.ndarray, vektorler: np.ndarray) -> dict:
    """Aynı/farklı kişi benzerlik dağılımları ve eşik taraması (saf fonksiyon — test edilir)."""
    benzerlik = vektorler @ vektorler.T          # normalize vektörlerde iç çarpım = kosinüs
    i, j = np.triu_indices(len(isimler), k=1)    # her çifti bir kez al
    ayni = benzerlik[i, j][isimler[i] == isimler[j]]
    farkli = benzerlik[i, j][isimler[i] != isimler[j]]
    tarama = []
    for e in np.round(np.arange(0.1, 0.9, 0.025), 3):
        far = float((farkli >= e).mean()) if len(farkli) else 0.0
        frr = float((ayni < e).mean()) if len(ayni) else 0.0
        tarama.append({"esik": float(e), "FAR": far, "FRR": frr})
    en_iyi = None
    if len(ayni) and len(farkli):
        # FAR+FRR'si en küçük olan eşikler arasından ORTADAKİNİ seç: iki dağılımın arasındaki boşluğun
        # kenarı yerine ortası, yeni (görülmemiş) fotoğraflarda daha güvenli bir paydır.
        en_kucuk = min(t["FAR"] + t["FRR"] for t in tarama)
        adaylar = [t for t in tarama if t["FAR"] + t["FRR"] == en_kucuk]
        en_iyi = adaylar[len(adaylar) // 2]
    return {"ayni": ayni, "farkli": farkli, "tarama": tarama, "onerilen": en_iyi}


def histogram_gorsel(ayni: np.ndarray, farkli: np.ndarray, esik: float | None, G=560, Y=260) -> np.ndarray:
    """Yeşil = aynı kişi çiftleri, kırmızı = farklı kişi çiftleri. Dikey çizgi = önerilen eşik."""
    img = np.full((Y, G, 3), 255, np.uint8)
    kutular = np.linspace(-0.2, 1.0, 49)
    ha = np.histogram(ayni, kutular)[0] if len(ayni) else np.zeros(48)
    hf = np.histogram(farkli, kutular)[0] if len(farkli) else np.zeros(48)
    tavan = max(ha.max(), hf.max(), 1)
    gen = (G - 40) / 48
    for k in range(48):
        x = int(20 + k * gen)
        for h, renk, kay in ((hf, (60, 60, 220), 0), (ha, (60, 170, 60), int(gen / 2))):
            boy = int((Y - 50) * h[k] / tavan)
            cv2.rectangle(img, (x + kay, Y - 30 - boy), (x + kay + int(gen / 2) - 1, Y - 30), renk, -1)
    for v in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = int(20 + (v + 0.2) / 1.2 * (G - 40))
        cv2.putText(img, f"{v:.2f}", (x - 12, Y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    if esik is not None:
        x = int(20 + (esik + 0.2) / 1.2 * (G - 40))
        cv2.line(img, (x, 10), (x, Y - 30), (0, 0, 0), 2)
        cv2.putText(img, f"esik {esik:.2f}", (x + 4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.putText(img, "yesil: ayni kisi  kirmizi: farkli kisi", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    return img


@dataclass
class DbAyarlari:
    klasor: str = ayar("veri/yuzler", "Kişi klasörlerinin olduğu yer (her alt klasör bir kişi)", tur="klasor")
    hizali_mi: bool = ayar(True, "Fotoğraflar zaten hizalı mı? (YuzFotografiTopla çıktısı: evet; normal fotoğraf: hayır)",
                           etki="Hayır seçersen her fotoğrafta önce yüz bulunur ve hizalanır")


class VeritabaniOlustur(Eylem):
    ISIM = "Yüz veritabanı oluştur + eşik analizi"
    BOLUM = "04"
    DOSYA = DOSYA
    ACIKLAMA = ("Her kişinin fotoğraflarından embedding çıkarır, `veri/yuz_db.npz`'ye yazar. Sonra aynı/farklı kişi benzerlik "
                "dağılımını çizip **yanlış kabul / yanlış ret** oranlarına göre eşik önerir.")
    Ayarlar = DbAyarlari

    def calistir(self, a: DbAyarlari, anlat):
        kok = yol_coz(a.klasor)
        isimler, vektorler, ornekler, basliklar = [], [], [], []
        for kisi in sorted(p for p in kok.iterdir() if p.is_dir()):
            n = 0
            for f in sorted(kisi.glob("*.jpg")) + sorted(kisi.glob("*.png")):
                img = cv2.imread(str(f))
                if img is None:
                    continue
                if not a.hizali_mi:
                    yuzler = yo.yuzleri_bul(img)
                    if len(yuzler) != 1:
                        anlat.uyari(f"{f.name}: {len(yuzler)} yüz (1 olmalı) — atlandı")
                        continue
                    img = yo.hizala(img, yuzler[0])
                vektorler.append(yo.embedding(img))
                isimler.append(kisi.name)
                n += 1
                if n <= 3:
                    ornekler.append(img); basliklar.append(kisi.name)
            anlat.adim(f"{kisi.name}: {n} embedding", kod="04_yuz/yuz_ortak.py:embedding")
        if not vektorler:
            raise FileNotFoundError(f"{kok} altında kişi klasörü/fotoğraf yok")
        isimler, vektorler = np.array(isimler), np.stack(vektorler)
        DB_YOLU.parent.mkdir(parents=True, exist_ok=True)
        np.savez(DB_YOLU, isimler=isimler, vektorler=vektorler)
        anlat.adim(f"Veritabanı yazıldı: {DB_YOLU} ({len(isimler)} vektör, {len(set(isimler))} kişi)")

        ea = esik_analizi(isimler, vektorler)
        on = ea["onerilen"]
        anlat.adim(f"Eşik analizi: {len(ea['ayni'])} aynı-kişi, {len(ea['farkli'])} farklı-kişi çifti", kod=f"{DOSYA}:esik_analizi",
                   veri={"onerilen": on})
        tablo = "| eşik | yanlış kabul (FAR) | yanlış ret (FRR) |\n|---|---|---|\n" + "\n".join(
            f"| {t['esik']:.3f} | %{t['FAR'] * 100:.1f} | %{t['FRR'] * 100:.1f} |" for t in ea["tarama"][::4])
        hist = histogram_gorsel(ea["ayni"], ea["farkli"], on["esik"] if on else None)
        metin = f"✔ {len(set(isimler))} kişi, {len(isimler)} vektör.\n\n"
        metin += (f"**Önerilen eşik: {on['esik']:.3f}** (FAR %{on['FAR'] * 100:.1f}, FRR %{on['FRR'] * 100:.1f})\n\n" if on
                  else "Eşik analizi için en az 2 kişi ve kişi başına 2+ fotoğraf gerekir.\n\n")
        return Sonuc(goruntu=hist, ara_gorseller={"benzerlik dağılımı": hist, "örnek kayıtlar": izgara(ornekler, basliklar, hucre=112)},
                     metin=metin + tablo, veri={"kisiler": sorted(set(isimler.tolist())), "onerilen": on})


if __name__ == "__main__":
    from ortak.cli import calistir
    calistir(VeritabaniOlustur)
