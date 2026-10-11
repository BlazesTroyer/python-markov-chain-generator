"""
sentetik_test.py — Takipçileri GERÇEK CEVAPLARI bildiğimiz sentetik bir sahnede yarıştır ve ÖLÇ.

NEDEN SENTETİK?
    Gerçek videoda "doğru ID" bilgisi yoktur (etiketlemek çok zahmetli). Sentetik sahnede her
    nesnenin gerçek kimliğini biliriz; böylece takipçinin hatalarını SAYABİLİRİZ.

SAHNE:
    - n_nesne kutu, sabit hızla hareket eder, duvarlardan seker (yollar kesişir).
    - Dedektör simülasyonu: her kutuya konum gürültüsü eklenir, `kacirma` olasılığıyla tespit
      kaybolur, iki kutu çok örtüşürse arkadaki görünmez (kapanma/occlusion), ara sıra hayali tespit (FP).

METRİKLER (takip_metrikleri fonksiyonu):
    Her karede gerçek kutular (GT) ile takipçi çıktısı IoU>=0.5 ile eşleştirilir.
      FN   : eşleşmeyen gerçek nesne (takipçi göstermedi)
      FP   : eşleşmeyen takipçi kutusu
      IDSW : (ID switch) bir gerçek nesne, son eşleştiği ID'den FARKLI bir ID ile eşleşti
      MOTA = 1 - (FN + FP + IDSW) / toplam_GT       (1 = mükemmel; negatif olabilir)
      parçalanma: bir gerçek nesneye ömrü boyunca ortalama kaç farklı ID verildi (1 = ideal)
    Daha kapsamlı metrikler (IDF1, HOTA) için: TrackEval kütüphanesi (README'de).

DENEYİN AMACI:
    hiz'ı artır -> IoU takipçinin ID'leri kopar, SORT (Kalman tahmini sayesinde) dayanır.
    kacirma'yı artır -> max_yas'ın neden önemli olduğunu gör.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from ortak.arayuz import Eylem, Sonuc, ayar  # noqa: E402
from ortak.cizim import id_rengi, izgara, kutu_ciz, yazi_yaz  # noqa: E402
from ortak.kaynak import VideoYazici  # noqa: E402
from ortak.yukle import VERI, modul_yukle  # noqa: E402

met = modul_yukle("00_temeller/metrikler.py")
it = modul_yukle("03_takip/iou_takipci.py")
st = modul_yukle("03_takip/sort_sifirdan.py")
DOSYA = "03_takip/sentetik_test.py"


@dataclass
class Ayarlar:
    n_nesne: int = ayar(5, "Nesne sayısı", min=1, max=20)
    kare_sayisi: int = ayar(150, "Kare sayısı", min=20, max=1000, adim=10)
    hiz: float = ayar(8.0, "Nesnelerin ortalama hızı (piksel/kare)", min=1.0, max=40.0, adim=1.0,
                      etki="Artırınca IoU takipçinin ID'leri kopar; SORT'un Kalman tahmini fark yaratır")
    kacirma: float = ayar(0.1, "Dedektörün bir nesneyi kaçırma olasılığı", min=0.0, max=0.7, adim=0.05)
    gurultu: float = ayar(3.0, "Tespit kutusu konum gürültüsü (piksel)", min=0.0, max=20.0, adim=0.5)
    yanlis_pozitif: float = ayar(0.1, "Kare başına ortalama hayali tespit", min=0.0, max=3.0, adim=0.1)
    max_yas: int = ayar(10, "İki takipçi için de: görülmeden kaç kare yaşasın", min=0, max=60)
    tohum: int = ayar(0, "Rastgelelik tohumu", min=0, max=9999)


def sahne_uret(a: Ayarlar, G=640, Y=480):
    """(gt_kareleri, tespit_kareleri) üret. gt: [(gt_id, kutu)], tespit: (N,6) dizi."""
    rng = np.random.default_rng(a.tohum)
    boyut = rng.uniform(40, 90, (a.n_nesne, 2))
    konum = rng.uniform([0, 0], [G - 100, Y - 100], (a.n_nesne, 2))
    aci = rng.uniform(0, 2 * np.pi, a.n_nesne)
    hiz = np.stack([np.cos(aci), np.sin(aci)], 1) * a.hiz * rng.uniform(0.6, 1.4, (a.n_nesne, 1))
    gt_kareler, tespit_kareler = [], []
    for _ in range(a.kare_sayisi):
        konum += hiz
        for d, sinir in ((0, G), (1, Y)):         # duvardan sekme
            tasan = (konum[:, d] < 0) | (konum[:, d] + boyut[:, d] > sinir)
            hiz[tasan, d] *= -1
            konum[:, d] = np.clip(konum[:, d], 0, sinir - boyut[:, d])
        kutular = np.hstack([konum, konum + boyut])
        gt_kareler.append([(i + 1, kutular[i].copy()) for i in range(a.n_nesne)])
        iou = met.iou_matrisi(kutular, kutular)
        tespitler = []
        for i in range(a.n_nesne):
            kapandi = any(iou[i, j] > 0.4 and j > i for j in range(a.n_nesne))   # j önde, i arkada kalıyor
            if kapandi or rng.random() < a.kacirma:
                continue
            k = kutular[i] + rng.normal(0, a.gurultu, 4)
            tespitler.append([*k, 0.9, 0])
        for _ in range(rng.poisson(a.yanlis_pozitif)):
            x, y = rng.uniform(0, G - 60), rng.uniform(0, Y - 60)
            tespitler.append([x, y, x + 50, y + 50, 0.5, 0])
        tespit_kareler.append(np.array(tespitler, np.float32).reshape(-1, 6))
    return gt_kareler, tespit_kareler


def takip_metrikleri(gt_kareler, iz_kareler) -> dict:
    """gt_kareler: [[(gt_id, kutu)]], iz_kareler: [[(iz_id, kutu)]] -> MOTA, IDSW, FN, FP, parçalanma."""
    fn = fp = idsw = toplam = 0
    son_eslesme: dict[int, int] = {}
    gt_idleri = defaultdict(set)
    for gtler, izler in zip(gt_kareler, iz_kareler):
        toplam += len(gtler)
        iou = met.iou_matrisi([k for _, k in gtler], [k for _, k in izler])
        eslesme, kalan_gt, kalan_iz = it.macar_eslestir(iou, 0.5)
        fn += len(kalan_gt)
        fp += len(kalan_iz)
        for i, j in eslesme:
            g, t = gtler[i][0], izler[j][0]
            if g in son_eslesme and son_eslesme[g] != t:
                idsw += 1
            son_eslesme[g] = t
            gt_idleri[g].add(t)
    return {"MOTA": 1 - (fn + fp + idsw) / max(toplam, 1), "IDSW": idsw, "FN": fn, "FP": fp,
            "parcalanma": float(np.mean([len(v) for v in gt_idleri.values()])) if gt_idleri else 0.0}


class SentetikTest(Eylem):
    ISIM = "Sentetik yarış: IoU takipçi vs SORT (ölçümlü)"
    BOLUM = "03"
    DOSYA = DOSYA
    ACIKLAMA = ("Gerçek kimlikleri bilinen sentetik sahnede iki takipçiyi aynı tespitlerle çalıştırır; **MOTA, ID switch, "
                "parçalanma** ölçer ve yan yana video üretir. `hiz` ve `kacirma` ile oynayıp farkı sayıyla gör.")
    Ayarlar = Ayarlar

    def calistir(self, a: Ayarlar, anlat):
        gt_kareler, tespit_kareler = sahne_uret(a)
        anlat.adim(f"{a.kare_sayisi} karelik sahne, {a.n_nesne} nesne, toplam {sum(map(len, tespit_kareler))} tespit",
                   kod=f"{DOSYA}:sahne_uret")
        takipciler = {"IoU": it.IouTakipci(0.3, a.max_yas), "SORT": st.Sort(a.max_yas, 2, 0.3)}
        ciktilar = {ad: [] for ad in takipciler}
        video_yolu = VERI / "takip_sentetik.mp4"
        video_yolu.parent.mkdir(parents=True, exist_ok=True)
        video = VideoYazici(video_yolu, 15)
        ornekler, basliklar = [], []
        for no, (gtler, tespitler) in enumerate(zip(gt_kareler, tespit_kareler)):
            paneller = []
            for ad, tk in takipciler.items():
                izler = tk.guncelle(tespitler)
                ciktilar[ad].append([(iz["id"], np.asarray(iz["kutu"])) for iz in izler])
                img = np.full((480, 640, 3), 35, np.uint8)
                for gid, k in gtler:
                    kutu_ciz(img, k, "", (150, 150, 150), 1)   # gri = gerçek konum (GT)
                for iz in izler:
                    kutu_ciz(img, iz["kutu"], f"ID {iz['id']}", id_rengi(iz["id"]))
                yazi_yaz(img, [f"{ad}  kare {no}", f"verilen ID: {tk.sonraki_id - 1}"])
                paneller.append(img)
            birlesik = np.hstack(paneller)
            video.yaz(birlesik)
            if no % max(1, a.kare_sayisi // 6) == 0 and len(ornekler) < 6:
                ornekler.append(birlesik); basliklar.append(f"kare {no}")
        video.kapat()

        sonuc = {ad: takip_metrikleri(gt_kareler, c) for ad, c in ciktilar.items()}
        for ad, m in sonuc.items():
            anlat.adim(f"{ad}: MOTA={m['MOTA']:.3f} IDSW={m['IDSW']} FN={m['FN']} FP={m['FP']} parçalanma={m['parcalanma']:.2f}",
                       kod=f"{DOSYA}:takip_metrikleri")
        tablo = "| takipçi | MOTA ↑ | ID switch ↓ | FN ↓ | FP ↓ | parçalanma (1=ideal) |\n|---|---|---|---|---|---|\n"
        tablo += "\n".join(f"| {ad} | {m['MOTA']:.3f} | {m['IDSW']} | {m['FN']} | {m['FP']} | {m['parcalanma']:.2f} |" for ad, m in sonuc.items())
        izg = izgara(ornekler, basliklar, sutun=2, hucre=520)
        return Sonuc(goruntu=izg, ara_gorseller={"örnek kareler (sol IoU, sağ SORT; gri = gerçek)": izg}, veri=sonuc,
                     metin=f"{tablo}\n\nYan yana video: `{video_yolu}`")


if __name__ == "__main__":
    from ortak.cli import calistir
    calistir(SentetikTest)
