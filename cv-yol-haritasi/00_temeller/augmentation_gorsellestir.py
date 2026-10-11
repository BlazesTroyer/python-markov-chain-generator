"""
augmentation_gorsellestir.py — Augmentation'ı GÖZLE görmek: hangi dönüşüm, kutuya ne yaptı?

NE YAPAR?
    Bir görüntü + bir (veya birkaç) kutu alır. İki modda çalışır:
      mod="tek_tek"  : Her dönüşümü AYRI AYRI uygular -> her birinin etkisini yan yana görürsün.
      mod="politika" : Seçili politikayı (hafif/agir) N kez rastgele uygular -> eğitimde modelin
                       göreceği çeşitliliği görürsün.
    İki uygulama vardır:
      yontem="sifirdan"        : 00_temeller/augment_sifirdan.py (her satırı okunabilir)
      yontem="albumentations"  : Endüstri standardı kütüphane (ReplayCompose ile hangi
                                 dönüşümün hangi parametreyle uygulandığı log'a yazılır)

GİRDİ : herhangi bir görüntü (webcam karesi, fotoğraf)
ÇIKTI : ızgara görüntü (her hücre bir örnek; yeşil kutu = dönüştürülmüş etiket)
        log: her örnekte uygulanan dönüşümler + kutunun önce/sonra koordinatları

ÇALIŞTIR:
    python 00_temeller/augmentation_gorsellestir.py --kaynak foto.jpg --mod tek_tek --anlat --debug-klasor dbg/
    python 00_temeller/augmentation_gorsellestir.py --kaynak foto.jpg --mod politika --politika agir

NEYE BAKMALISIN?
    - Döndürmede kutu nesneyi hâlâ sarıyor mu? (Köşe dönüşümü yüzünden biraz şişer — normal.)
    - 'agir' politikada görüntüler hâlâ GERÇEKÇİ mi? Gerçek hayatta asla görmeyeceğin
      bozulmalar modele zarar verir. Augmentation'ın sanatı bu dengedir.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from ortak.arayuz import KareIsleyici, Sonuc, ayar  # noqa: E402
from ortak.cizim import izgara, kutu_ciz  # noqa: E402
from ortak.yukle import gerekli, modul_yukle  # noqa: E402

aug = modul_yukle("00_temeller/augment_sifirdan.py")
DOSYA = "00_temeller/augmentation_gorsellestir.py"


@dataclass
class Ayarlar:
    mod: str = ayar("tek_tek", "tek_tek: her dönüşüm ayrı | politika: rastgele karışım",
                    secenekler=["tek_tek", "politika"])
    yontem: str = ayar("sifirdan", "Uygulama: kendi kodumuz mu, albumentations mı?",
                       secenekler=["sifirdan", "albumentations"])
    politika: str = ayar("agir", "politika modunda kullanılacak seviye", secenekler=["hafif", "agir"],
                         etki="agir: az veride çeşitlilik artar ama gerçek dışı örnek riski de artar")
    ornek_sayisi: int = ayar(8, "politika modunda kaç örnek üretilsin", min=1, max=24, adim=1)
    kutu: str = ayar("0.30,0.30,0.70,0.70", "Örnek etiket kutusu (oran: x1,y1,x2,y2; 0..1). ';' ile birden çok kutu",
                     etki="Kendi nesnenin yerini yazarsan dönüşümlerin onu nasıl etkilediğini görürsün")
    tohum: int = ayar(0, "Rastgelelik tohumu (0 = her seferinde farklı)", min=0, max=9999, adim=1,
                      etki="Aynı tohum = aynı sonuç; karşılaştırma yaparken sabitle")


def _kutulari_coz(metin: str, g: int, y: int) -> np.ndarray:
    """'0.3,0.3,0.7,0.7;0.1,0.1,0.2,0.2' -> piksel kutuları (N,4)."""
    kutular = []
    for parca in metin.split(";"):
        x1, y1, x2, y2 = (float(v) for v in parca.split(","))
        kutular.append([x1 * g, y1 * y, x2 * g, y2 * y])
    return np.array(kutular, np.float32)


def _albumentations_politikasi(seviye: str):
    """augment_sifirdan.POLITIKALAR'ın albumentations karşılığı (aynı fikir, kütüphane sürümü)."""
    A = gerekli("albumentations")
    guc = 1.0 if seviye == "agir" else 0.4
    p = aug.POLITIKALAR[seviye]["olasilik"]
    donusumler = [
        A.RandomBrightnessContrast(brightness_limit=0.2 * guc, contrast_limit=0.4 * guc, p=p.get("parlaklik_kontrast", 0)),
        A.HueSaturationValue(hue_shift_limit=int(10 * guc), sat_shift_limit=int(60 * guc), val_shift_limit=int(40 * guc), p=p.get("hsv_kaydir", 0)),
        A.MotionBlur(blur_limit=(3, max(3, int(3 + 12 * guc)) | 1), p=p.get("bulaniklik", 0)),
        A.HorizontalFlip(p=p.get("yatay_cevir", 0)),
        A.Affine(rotate=(-20 * guc, 20 * guc), p=p.get("dondur", 0)),
        A.Affine(scale=(1 - 0.25 * guc, 1 + 0.25 * guc), translate_percent=(-0.1 * guc, 0.1 * guc), p=p.get("olcekle_kaydir", 0)),
    ]
    try:  # albumentations 2.x parametre adları
        donusumler.append(A.GaussNoise(std_range=(0.02, 0.1 * guc), p=p.get("gurultu", 0)))
        donusumler.append(A.CoarseDropout(num_holes_range=(1, 3), hole_height_range=(0.08, 0.25),
                                          hole_width_range=(0.08, 0.25), fill=114, p=p.get("kesip_cikar", 0)))
    except TypeError:  # 1.x
        donusumler.append(A.GaussNoise(p=p.get("gurultu", 0)))
        donusumler.append(A.CoarseDropout(max_holes=3, p=p.get("kesip_cikar", 0)))
    # ReplayCompose: "hangi dönüşüm gerçekten uygulandı, hangi parametreyle?" bilgisini saklar
    return A.ReplayCompose(donusumler, bbox_params=A.BboxParams(format="pascal_voc", label_fields=["etiketler"], min_visibility=0.2))


def _uygulananlari_oku(replay: dict) -> list[str]:
    """ReplayCompose kaydından uygulanmış dönüşümlerin adını+parametresini çıkar."""
    cikti = []
    for t in replay.get("transforms", []):
        if t.get("applied"):
            ad = t["__class_fullname__"].split(".")[-1]
            parametreler = {k: v for k, v in (t.get("params") or {}).items()
                            if isinstance(v, (int, float, str, bool)) and k not in ("rows", "cols")}
            cikti.append(f"{ad}({', '.join(f'{k}={v:.2f}' if isinstance(v, float) else f'{k}={v}' for k, v in list(parametreler.items())[:4])})")
    return cikti


class AugmentationGorsellestir(KareIsleyici):
    ISIM = "Augmentation görselleştirici"
    BOLUM = "00"
    DOSYA = DOSYA
    ACIKLAMA = ("Bir görüntüye augmentation dönüşümlerini uygular; **kutuların da birlikte dönüştüğünü** gösterir. "
                "`tek_tek` modunda her dönüşümün etkisini ayrı ayrı, `politika` modunda eğitimde modelin göreceği çeşitliliği görürsün.")
    GEREKSINIMLER = ("albumentations (opsiyonel)",)
    Ayarlar = Ayarlar

    def isle(self, kare, ayarlar: Ayarlar, anlat):
        y, g = kare.shape[:2]
        rng = np.random.default_rng(ayarlar.tohum or None)
        kutular = _kutulari_coz(ayarlar.kutu, g, y)
        etiketler = ["nesne"] * len(kutular)
        anlat.adim(f"Girdi {g}x{y}, {len(kutular)} kutu", kod=f"{DOSYA}:_kutulari_coz",
                   veri={"kutular": kutular.round().tolist()})

        orijinal = kare.copy()
        for k in kutular:
            kutu_ciz(orijinal, k, "orijinal", (0, 220, 0))
        hucreler, basliklar = [orijinal], ["ORIJINAL"]

        if ayarlar.mod == "tek_tek":
            # Her dönüşümü guc=1 ile tek başına uygula -> etkisi net görünsün
            for isim, fonk in aug.DONUSUMLER.items():
                yeni, yk, aciklama = fonk(kare.copy(), kutular.copy(), rng, 1.0)
                anlat.adim(aciklama, kod=f"00_temeller/augment_sifirdan.py:{fonk.__name__}",
                           veri={"kutu_once": kutular[0].round().tolist(), "kutu_sonra": yk[0].round().tolist()})
                for k in yk:
                    kutu_ciz(yeni, k, "", (0, 220, 0))
                hucreler.append(yeni)
                basliklar.append(isim)
        else:
            if ayarlar.yontem == "albumentations":
                tf = _albumentations_politikasi(ayarlar.politika)
                anlat.bilgi("albumentations ReplayCompose kuruldu", kod=f"{DOSYA}:_albumentations_politikasi")
            for i in range(ayarlar.ornek_sayisi):
                if ayarlar.yontem == "sifirdan":
                    yeni, yk, ye, adimlar = aug.uygula(kare.copy(), kutular, etiketler, ayarlar.politika, rng)
                    kod = "00_temeller/augment_sifirdan.py:uygula"
                else:
                    out = tf(image=kare, bboxes=kutular.tolist(), etiketler=etiketler)
                    yeni, yk = out["image"].copy(), np.array(out["bboxes"], np.float32).reshape(-1, 4)
                    adimlar = _uygulananlari_oku(out["replay"])
                    kod = f"{DOSYA}:_uygulananlari_oku"
                anlat.adim(f"Örnek {i + 1}: {' → '.join(adimlar) or '(hiçbir dönüşüm tutmadı)'}", kod=kod,
                           veri={"kalan_kutu": len(yk)})
                for k in yk:
                    kutu_ciz(yeni, k, "", (0, 220, 0))
                hucreler.append(yeni)
                basliklar.append(f"#{i + 1}: {len(adimlar)} donusum")

        izg = izgara(hucreler, basliklar, sutun=4)
        return Sonuc(
            goruntu=izg,
            ara_gorseller={b: h for b, h in zip(basliklar, hucreler)},
            metin=(f"**{len(hucreler) - 1} örnek üretildi** ({ayarlar.mod}, {ayarlar.yontem}).\n\n"
                   "Yeşil kutular dönüştürülmüş etiketlerdir. Geometrik dönüşümlerde kutunun nesneyle "
                   "birlikte hareket ettiğine dikkat et; fotometriklerde kutu yerinde kalır."),
            veri={"ornek": len(hucreler) - 1, "mod": ayarlar.mod, "yontem": ayarlar.yontem},
        )


if __name__ == "__main__":
    from ortak.cli import calistir
    calistir(AugmentationGorsellestir)
