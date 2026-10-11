"""
telefon_dedektoru_v2.py — Phone call detector'ın yeniden yapılmış hali: kural tabanlı VE öğrenilmiş, yan yana.

İKİ MOD:
    mod="kural"  (taban çizgisi — senin v1'ine benzer mantık, ama ölçülebilir)
        1. COCO dedektörü 'cell phone' kutularını bulur.               (yolo11n.pt)
        2. Poz modeli kişinin 17 anahtar noktasını bulur.              (yolo11n-pose.pt)
           Kullandıklarımız: 3/4 = sol/sağ kulak, 9/10 = sol/sağ bilek, 5/6 = omuzlar.
        3. Telefonun merkezi ile en yakın KULAK ve BİLEK arasındaki mesafe ölçülür.
           Mesafe, kişinin omuz genişliğine bölünür -> kameraya yakın/uzak olmaktan bağımsız olur.
        4. Kural: kulak_mesafe < esik_kulak -> "kulakta";  bilek_mesafe < esik_bilek -> "elde";  değilse "ortada"
    mod="model"  (02'de kendi verinle eğittiğin model)
        Sınıf adlarında "ear"/"kulak" geçen -> "kulakta", "hand"/"el" geçen -> "elde".
        Kuralların kaçırdığı durumları (telefon yüze yakın ama kulakta değil vb.) veriyle öğrenir.

ZAMANSAL KARARLILIK (durum makinesi):
    Tek karelik hatalar yüzünden durum titremesin diye: yeni bir durum `kararlilik` kadar ART ARDA
    görülmeden kabul edilmez. ("Debounce" — 05'teki jest ve tekrar sayacında da aynı fikir.)
        durum: "yok" --(N kare 'kulakta')--> "kulakta" --(N kare 'yok')--> "yok"
    Durum DEĞİŞTİĞİNDE bir olay üretilir: {"tur": "telefon_durumu", "ek": {"once": ..., "sonra": ...}}
    03_takip/ekran_suresi.py ve 09_agent bu olaylardan süre hesaplar.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ortak.arayuz import KareIsleyici, Sonuc, ayar  # noqa: E402
from ortak.cizim import kutu_ciz, yazi_yaz  # noqa: E402
from ortak.yukle import modul_yukle  # noqa: E402

yy = modul_yukle("02_yolo_finetune/yolo_yardimci.py")
DOSYA = "02_yolo_finetune/telefon_dedektoru_v2.py"
KULAKLAR, BILEKLER, OMUZLAR = (3, 4), (9, 10), (5, 6)


@dataclass
class Ayarlar:
    mod: str = ayar("kural", "kural: dedektör + poz + mesafe kuralı | model: kendi eğittiğin sınıflar", secenekler=["kural", "model"])
    model: str = ayar("yolo11n.pt", "kural: COCO dedektörü | model: kendi best.pt yolun", tur="dosya")
    poz_modeli: str = ayar("yolo11n-pose.pt", "Sadece kural modu: anahtar nokta modeli")
    guven: float = ayar(0.3, "Tespit güven eşiği", min=0.05, max=0.9, adim=0.05)
    esik_kulak: float = ayar(0.6, "Telefon–kulak mesafesi / omuz genişliği bunun altındaysa 'kulakta'", min=0.1, max=2.0, adim=0.05,
                             etki="Artırırsan telefon yüze yaklaşınca bile 'kulakta' der (yanlış pozitif)")
    esik_bilek: float = ayar(0.7, "Telefon–bilek mesafesi / omuz genişliği bunun altındaysa 'elde'", min=0.1, max=2.0, adim=0.05)
    kararlilik: int = ayar(5, "Durum değişimi için gereken art arda kare sayısı", min=1, max=60,
                           etki="Artırırsan titreme azalır ama tepki gecikir (30 FPS'te 5 kare ≈ 0.17 sn)")


class DurumMakinesi:
    """Genel 'debounce' durum makinesi: aday durum N kare boyunca görülürse kabul edilir."""

    def __init__(self, baslangic: str = "yok"):
        self.durum, self.aday, self.sayac = baslangic, baslangic, 0

    def guncelle(self, gozlem: str, gereken: int) -> tuple[bool, str]:
        """Döndürür: (durum_degisti_mi, onceki_durum)."""
        if gozlem == self.durum:
            self.aday, self.sayac = gozlem, 0
            return False, self.durum
        if gozlem == self.aday:
            self.sayac += 1
        else:
            self.aday, self.sayac = gozlem, 1
        if self.sayac >= gereken:
            onceki, self.durum, self.sayac = self.durum, gozlem, 0
            return True, onceki
        return False, self.durum


def en_yakin_mesafe(nokta, kps, kp_conf, indeksler, olcek) -> float:
    """Noktaya en yakın (görünür) anahtar noktanın mesafesi / olcek. Görünür nokta yoksa sonsuz."""
    mesafeler = [np.linalg.norm(kps[i] - nokta) for i in indeksler if kp_conf[i] > 0.3]
    return min(mesafeler) / max(olcek, 1.0) if mesafeler else float("inf")


class TelefonDedektoruV2(KareIsleyici):
    ISIM = "Telefon dedektörü v2 (kural vs model)"
    BOLUM = "02"
    DOSYA = DOSYA
    ACIKLAMA = ("Telefon **kulakta / elde / yok** durumunu bulur. `kural` modu: COCO dedektörü + poz noktaları + "
                "ölçek bağımsız mesafe kuralı; `model` modu: kendi eğittiğin sınıflar. Durum makinesi titremeyi engeller; "
                "durum değişince olay üretir.")
    GEREKSINIMLER = ("ultralytics",)
    Ayarlar = Ayarlar

    def __init__(self):
        self.makine = DurumMakinesi()

    def sifirla(self):
        self.makine = DurumMakinesi()

    def _kural_gozlem(self, kare, ayarlar, anlat, cikti, ara):
        det = yy.model_yukle(ayarlar.model)
        telefon_no = [k for k, v in det.names.items() if v == "cell phone"]
        r = det.predict(kare, conf=ayarlar.guven, classes=telefon_no or None, verbose=False)[0]
        tk, _, tg, _ = yy.sonucu_ayikla(r)
        poz = yy.model_yukle(ayarlar.poz_modeli).predict(kare, conf=0.3, verbose=False)[0]
        kisiler = []
        if poz.keypoints is not None and len(poz.keypoints):
            kisiler = list(zip(poz.keypoints.xy.cpu().numpy(), poz.keypoints.conf.cpu().numpy()))
        anlat.adim(f"{len(tk)} telefon, {len(kisiler)} kişi (poz)", kod=f"{DOSYA}:_kural_gozlem",
                   veri={"telefon_guven": [round(float(g), 2) for g in tg]})

        gozlem = "yok"
        poz_img = kare.copy()
        for kps, kc in kisiler:
            for i, (x, y) in enumerate(kps):
                if kc[i] > 0.3:
                    renk = (0, 0, 255) if i in KULAKLAR else (255, 0, 0) if i in BILEKLER else (0, 255, 0)
                    cv2.circle(poz_img, (int(x), int(y)), 5, renk, -1)
        for k, g in zip(tk, tg):
            merkez = np.array([(k[0] + k[2]) / 2, (k[1] + k[3]) / 2])
            en_iyi = ("ortada", float("inf"), float("inf"))
            for kps, kc in kisiler:
                omuz = np.linalg.norm(kps[OMUZLAR[0]] - kps[OMUZLAR[1]]) if min(kc[OMUZLAR[0]], kc[OMUZLAR[1]]) > 0.3 else (k[3] - k[1]) * 3
                d_kulak = en_yakin_mesafe(merkez, kps, kc, KULAKLAR, omuz)
                d_bilek = en_yakin_mesafe(merkez, kps, kc, BILEKLER, omuz)
                durum = "kulakta" if d_kulak < ayarlar.esik_kulak else "elde" if d_bilek < ayarlar.esik_bilek else "ortada"
                if min(d_kulak, d_bilek) < min(en_iyi[1], en_iyi[2]):
                    en_iyi = (durum, d_kulak, d_bilek)
            anlat.bilgi(f"telefon {g:.2f}: kulak_mesafe={en_iyi[1]:.2f} (eşik {ayarlar.esik_kulak}), "
                        f"bilek_mesafe={en_iyi[2]:.2f} (eşik {ayarlar.esik_bilek}) → {en_iyi[0]}", kod=f"{DOSYA}:en_yakin_mesafe")
            kutu_ciz(cikti, k, f"telefon {en_iyi[0]} {g:.2f}", (0, 200, 255))
            kutu_ciz(poz_img, k, f"k={en_iyi[1]:.2f} b={en_iyi[2]:.2f}", (0, 200, 255))
            oncelik = {"kulakta": 3, "elde": 2, "ortada": 1, "yok": 0}
            if oncelik[en_iyi[0]] > oncelik[gozlem]:
                gozlem = en_iyi[0]
        ara["poz noktaları (kırmızı=kulak, mavi=bilek)"] = poz_img
        return "yok" if gozlem == "ortada" else gozlem

    def _model_gozlem(self, kare, ayarlar, anlat, cikti):
        m = yy.model_yukle(ayarlar.model)
        r = m.predict(kare, conf=ayarlar.guven, verbose=False)[0]
        kutular, siniflar, guvenler, isimler = yy.sonucu_ayikla(r)
        gozlem = "yok"
        for k, s, g in zip(kutular, siniflar, guvenler):
            ad = isimler[s].lower()
            durum = "kulakta" if ("ear" in ad or "kulak" in ad) else "elde" if ("hand" in ad or "el" in ad.split("_")) else "yok"
            kutu_ciz(cikti, k, f"{isimler[s]} {g:.2f}", (0, 200, 255))
            if durum == "kulakta" or (durum == "elde" and gozlem == "yok"):
                gozlem = durum
        anlat.adim(f"Model sınıfları → gözlem: {gozlem}", kod=f"{DOSYA}:_model_gozlem",
                   veri={"tespitler": [isimler[s] for s in siniflar]})
        return gozlem

    def isle(self, kare, ayarlar: Ayarlar, anlat):
        cikti, ara = kare.copy(), {}
        gozlem = (self._kural_gozlem(kare, ayarlar, anlat, cikti, ara) if ayarlar.mod == "kural"
                  else self._model_gozlem(kare, ayarlar, anlat, cikti))
        degisti, onceki = self.makine.guncelle(gozlem, ayarlar.kararlilik)
        anlat.adim(f"Bu kare: '{gozlem}' | kararlı durum: '{self.makine.durum}' (aday '{self.makine.aday}' "
                   f"{self.makine.sayac}/{ayarlar.kararlilik})", kod=f"{DOSYA}:DurumMakinesi.guncelle")
        olaylar = []
        if degisti:
            olaylar.append({"tur": "telefon_durumu", "kaynak": DOSYA, "nesne": "telefon",
                            "ek": {"once": onceki, "sonra": self.makine.durum, "mod": ayarlar.mod}})
            anlat.adim(f"DURUM DEĞİŞTİ: {onceki} → {self.makine.durum} (olay üretildi)")
        renk = {"kulakta": (0, 0, 255), "elde": (0, 200, 255), "yok": (180, 180, 180)}[self.makine.durum]
        yazi_yaz(cikti, [f"TELEFON: {self.makine.durum.upper()}", f"kare: {gozlem}  ({ayarlar.mod})"], renk=renk)
        return Sonuc(goruntu=cikti, ara_gorseller=ara, olaylar=olaylar,
                     metin=f"Kararlı durum: **{self.makine.durum}** (bu karedeki gözlem: {gozlem})",
                     veri={"durum": self.makine.durum, "gozlem": gozlem})


if __name__ == "__main__":
    from ortak.cli import calistir
    calistir(TelefonDedektoruV2)
