"""
ornek_dataset.py — Kamerasız, internetsiz, 1 dakikada bir "oyuncak" tespit dataset'i üretir.

NEDEN?
    01 → 02 hattını (etiket kontrol → bölme → augmentation → YOLO eğitimi → mAP) kendi
    fotoğraflarını çekmeden ÖNCE uçtan uca denemek için. Her şeyin nasıl aktığını küçük ve
    hızlı bir veriyle gör; sonra aynı adımları kendi eşyalarınla tekrarla.

NE ÜRETİR?
    Rastgele arka planlar üzerine 3 sınıf şekil: 0=daire, 1=kare, 2=ucgen
    Dosya adları "vid<grup>_<no>.jpg" -> sanki farklı videolardan alınmış kareler gibi.
    (Bu, dataset_bol.py'deki "aynı videonun kareleri aynı bölmeye" kuralını denemek için.)
    Bazı görüntüler bilerek BOŞ (negatif örnek): modele "burada nesne yok" demeyi öğretir.

ÇIKTI:
    veri/ham/<isim>/images/*.jpg
    veri/ham/<isim>/labels/*.txt     (YOLO formatı)
    veri/ham/<isim>/siniflar.txt     (her satır bir sınıf adı, sırası = sınıf no)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ortak.arayuz import Eylem, Sonuc, ayar  # noqa: E402
from ortak.cizim import izgara, kutu_ciz  # noqa: E402
from ortak.yukle import VERI, modul_yukle  # noqa: E402

yf = modul_yukle("01_veri_fabrikasi/yolo_format.py")
DOSYA = "01_veri_fabrikasi/ornek_dataset.py"
SINIFLAR = ["daire", "kare", "ucgen"]


@dataclass
class Ayarlar:
    isim: str = ayar("sekiller", "Dataset adı (veri/ham/<isim>/ altına yazılır)")
    goruntu_sayisi: int = ayar(240, "Kaç görüntü", min=20, max=5000, adim=10,
                               etki="Fazlası daha iyi model ama daha uzun eğitim")
    grup_sayisi: int = ayar(12, "Kaç farklı 'video' (grup)", min=2, max=100)
    boyut: int = ayar(320, "Görüntü kenar uzunluğu (px)", min=128, max=1024, adim=32)
    bos_orani: float = ayar(0.1, "Nesnesiz (negatif) görüntü oranı", min=0.0, max=0.5, adim=0.05)
    tohum: int = ayar(0, "Rastgelelik tohumu", min=0, max=9999)


def _arka_plan(rng, b, grup_rengi):
    """Her 'video' grubunun kendine özgü bir arka plan tonu olsun (gerçek videolar gibi)."""
    img = np.zeros((b, b, 3), np.float32)
    yon = np.linspace(0, 1, b)[None, :, None] if rng.random() < 0.5 else np.linspace(0, 1, b)[:, None, None]
    img += grup_rengi * (0.6 + 0.4 * yon)
    img += rng.normal(0, 12, img.shape)
    for _ in range(rng.integers(2, 8)):    # dikkat dağıtıcı çizgiler
        p1, p2 = rng.integers(0, b, 2), rng.integers(0, b, 2)
        cv2.line(img, tuple(int(v) for v in p1), tuple(int(v) for v in p2), tuple(float(v) for v in rng.uniform(0, 255, 3)), int(rng.integers(1, 4)))
    return np.clip(img, 0, 255).astype(np.uint8)


def _sekil_ciz(img, sinif, rng):
    b = img.shape[0]
    r = int(rng.uniform(0.06, 0.18) * b)
    cx, cy = int(rng.uniform(r, b - r)), int(rng.uniform(r, b - r))
    renk = tuple(int(v) for v in rng.uniform(30, 255, 3))
    if sinif == 0:
        cv2.circle(img, (cx, cy), r, renk, -1)
        return [cx - r, cy - r, cx + r, cy + r]
    if sinif == 1:
        cv2.rectangle(img, (cx - r, cy - r), (cx + r, cy + r), renk, -1)
        return [cx - r, cy - r, cx + r, cy + r]
    pts = np.array([[cx, cy - r], [cx - r, cy + r], [cx + r, cy + r]], np.int32)
    cv2.fillPoly(img, [pts], renk)
    return [cx - r, cy - r, cx + r, cy + r]


class OrnekDataset(Eylem):
    ISIM = "Örnek (sentetik) dataset üret"
    BOLUM = "01"
    DOSYA = DOSYA
    ACIKLAMA = ("Kamerasız ve internetsiz, YOLO formatında küçük bir şekil dataset'i üretir (daire/kare/üçgen). "
                "01→02 hattını uçtan uca hızlıca denemek için. Sonra aynı adımları kendi fotoğraflarınla yap.")
    Ayarlar = Ayarlar

    def calistir(self, ayarlar: Ayarlar, anlat):
        rng = np.random.default_rng(ayarlar.tohum)
        hedef = VERI / "ham" / ayarlar.isim
        (hedef / "images").mkdir(parents=True, exist_ok=True)
        (hedef / "labels").mkdir(parents=True, exist_ok=True)
        (hedef / "siniflar.txt").write_text("\n".join(SINIFLAR) + "\n")
        anlat.adim(f"Hedef klasör: {hedef}", kod=f"{DOSYA}:OrnekDataset.calistir")

        grup_renkleri = rng.uniform(40, 200, (ayarlar.grup_sayisi, 3))
        sayac = np.zeros(len(SINIFLAR), int)
        onizleme, basliklar = [], []
        for i in range(ayarlar.goruntu_sayisi):
            grup = i % ayarlar.grup_sayisi
            img = _arka_plan(rng, ayarlar.boyut, grup_renkleri[grup])
            kutular, siniflar = [], []
            if rng.random() >= ayarlar.bos_orani:
                for _ in range(rng.integers(1, 4)):
                    s = int(rng.integers(0, len(SINIFLAR)))
                    kutular.append(_sekil_ciz(img, s, rng))
                    siniflar.append(s)
                    sayac[s] += 1
            ad = f"vid{grup:02d}_{i:05d}"
            cv2.imwrite(str(hedef / "images" / f"{ad}.jpg"), img)
            yf.etiket_yaz(hedef / "labels" / f"{ad}.txt", kutular, siniflar, ayarlar.boyut, ayarlar.boyut)
            if len(onizleme) < 8:
                g = img.copy()
                for k, s in zip(kutular, siniflar):
                    kutu_ciz(g, k, SINIFLAR[s])
                onizleme.append(g)
                basliklar.append(ad)
        anlat.adim(f"{ayarlar.goruntu_sayisi} görüntü + etiket yazıldı", kod=f"{DOSYA}:_sekil_ciz / yolo_format.etiket_yaz",
                   veri={s: int(n) for s, n in zip(SINIFLAR, sayac)})
        ornek_txt = next((hedef / "labels").glob("*.txt")).read_text().strip() or "(boş = negatif örnek)"
        anlat.bilgi(f"Örnek etiket satırları:\n{ornek_txt}", kod="01_veri_fabrikasi/yolo_format.py:etiket_yaz")
        izg = izgara(onizleme, basliklar, sutun=4)
        return Sonuc(goruntu=izg, ara_gorseller={"önizleme": izg},
                     metin=(f"✔ `{hedef}` hazır.\n\nSıradaki adım: **Etiket kontrol** → **Dataset böl** "
                            f"(kaynak: `veri/ham/{ayarlar.isim}`)."),
                     veri={"klasor": str(hedef), "sinif_sayilari": dict(zip(SINIFLAR, sayac.tolist()))})


if __name__ == "__main__":
    from ortak.cli import calistir
    calistir(OrnekDataset)
