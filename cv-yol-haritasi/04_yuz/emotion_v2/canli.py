"""
canli.py — Eğittiğin duygu modelini canlı çalıştır + ZAMANSAL YUMUŞATMA.

SORUN: Kare kare sınıflandırma TİTREK'tir. Göz kırpınca "yorgun", ağız kıpırdayınca "şaşırmış"...
       v1'de muhtemelen bunu yaşadın. Çözüm: kararı birkaç karenin bilgisini birleştirerek ver.

YUMUŞATMA YÖNTEMLERİ (Yumusatici sınıfı):
    yok        : her karenin olasılığı aynen kullanılır (karşılaştırma için)
    ema        : Üstel hareketli ortalama:   p_yumuşak = alfa * p_yeni + (1 - alfa) * p_yumuşak
                 alfa küçük -> çok yumuşak ama geç tepki; alfa büyük -> hızlı ama titrek.
                 alfa=0.2 yaklaşık son ~5 karenin ağırlıklı ortalaması gibidir.
    pencere_oy : Son N karenin en olası sınıfları arasında çoğunluk oyu.

Ardından bir de DurumMakinesi (02'deki debounce) var: Yumuşatılmış sınıf `kararlilik` kare
boyunca değişmeden kalırsa "duygu değişti" olayı üretilir.

ARA GÖRSEL: Ham (kare) ve yumuşatılmış olasılıklar yan yana çubuk grafik -> farkı GÖZLE gör.
"""
from __future__ import annotations

import sys
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ortak.arayuz import KareIsleyici, Sonuc, ayar  # noqa: E402
from ortak.cizim import kutu_ciz, yazi_yaz  # noqa: E402
from ortak.yukle import gerekli, modul_yukle, yol_coz  # noqa: E402

yo = modul_yukle("04_yuz/yuz_ortak.py")
eg = modul_yukle("04_yuz/emotion_v2/egit.py")
tel = modul_yukle("02_yolo_finetune/telefon_dedektoru_v2.py")
DOSYA = "04_yuz/emotion_v2/canli.py"


class Yumusatici:
    """Olasılık vektörlerini zaman içinde yumuşatır (saf numpy — test edilir)."""

    def __init__(self, mod: str = "ema", alfa: float = 0.3, pencere: int = 10):
        self.mod, self.alfa = mod, alfa
        self.p: np.ndarray | None = None
        self.gecmis: deque = deque(maxlen=pencere)

    def guncelle(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, np.float32)
        if self.mod == "yok":
            return p
        if self.mod == "ema":
            self.p = p.copy() if self.p is None else self.alfa * p + (1 - self.alfa) * self.p
            return self.p
        # pencere_oy: son N karenin argmax'larının oy oranları
        self.gecmis.append(int(p.argmax()))
        oy = np.zeros_like(p)
        for sinif, n in Counter(self.gecmis).items():
            oy[sinif] = n / len(self.gecmis)
        return oy


def olasilik_cubuklari(siniflar, ham, yumusak, G=360) -> np.ndarray:
    Y = 30 + 26 * len(siniflar)
    img = np.full((Y, G, 3), 255, np.uint8)
    cv2.putText(img, "gri: bu kare  |  mavi: yumusatilmis", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
    for i, s in enumerate(siniflar):
        y0 = 26 + i * 26
        cv2.putText(img, s[:9], (4, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
        cv2.rectangle(img, (90, y0), (90 + int(ham[i] * 250), y0 + 9), (170, 170, 170), -1)
        cv2.rectangle(img, (90, y0 + 11), (90 + int(yumusak[i] * 250), y0 + 20), (200, 120, 30), -1)
    return img


@dataclass
class Ayarlar:
    model: str = ayar("veri/modeller/duygu_v2/model.pt", "egit.py çıktısı", tur="dosya")
    yumusatma: str = ayar("ema", "Zamansal yumuşatma", secenekler=["yok", "ema", "pencere_oy"],
                          etki="'yok' ile karşılaştır: etiketin ne kadar titrediğini gör")
    alfa: float = ayar(0.3, "EMA katsayısı (büyük = hızlı/titrek)", min=0.02, max=1.0, adim=0.02)
    pencere: int = ayar(10, "pencere_oy için kare sayısı", min=2, max=60)
    kararlilik: int = ayar(6, "Duygu değişimi olayı için art arda kare", min=1, max=60)


class DuyguCanli(KareIsleyici):
    ISIM = "Emotion v2 canlı (EMA yumuşatmalı)"
    BOLUM = "04"
    DOSYA = DOSYA
    ACIKLAMA = ("En büyük yüzü bulur, eğittiğin modelle sınıflandırır, olasılıkları **zamansal olarak yumuşatır** ve kararlı değişimde "
                "`duygu_degisti` olayı üretir. Ara görselde ham vs yumuşatılmış olasılıklar.")
    GEREKSINIMLER = ("torch", "torchvision", "timm")
    Ayarlar = Ayarlar

    def __init__(self):
        self.sifirla()
        self._model = None

    def sifirla(self):
        self.yum, self.makine, self._ayar_imzasi = None, tel.DurumMakinesi("?"), None

    def _yukle(self, yol):
        if self._model is None or self._model[0] != yol:
            torch, timm = gerekli("torch"), gerekli("timm")
            kayit = torch.load(yol_coz(yol), map_location="cpu", weights_only=False)
            m = timm.create_model(kayit["model"], pretrained=False, num_classes=len(kayit["siniflar"]))
            m.load_state_dict(kayit["durum"])
            m.eval()
            self._model = (yol, m, kayit["siniflar"], eg.donusumler(kayit["boyut"], False, False))
        return self._model[1:]

    def isle(self, kare, a: Ayarlar, anlat):
        torch = gerekli("torch")
        model, siniflar, tf = self._yukle(a.model)
        imza = (a.yumusatma, a.alfa, a.pencere)
        if self.yum is None or imza != self._ayar_imzasi:
            self.yum, self._ayar_imzasi = Yumusatici(*imza), imza
        yuzler = yo.yuzleri_bul(kare)
        cikti, ara, olaylar = kare.copy(), {}, []
        if not len(yuzler):
            yazi_yaz(cikti, "yuz yok")
            return Sonuc(goruntu=cikti, metin="Yüz yok.")
        y = max(yuzler, key=lambda r: r[2] * r[3])
        kirp = yo.yuz_kirp(kare, y)
        with torch.no_grad():
            p = torch.softmax(model(tf(cv2.cvtColor(kirp, cv2.COLOR_BGR2RGB))[None]), 1)[0].numpy()
        ys = self.yum.guncelle(p)
        anlat.adim(f"Ham: {siniflar[int(p.argmax())]} ({p.max():.2f}) → yumuşatılmış ({a.yumusatma}): {siniflar[int(ys.argmax())]} ({ys.max():.2f})",
                   kod=f"{DOSYA}:Yumusatici.guncelle", veri={s: round(float(v), 2) for s, v in zip(siniflar, ys)})
        degisti, onceki = self.makine.guncelle(siniflar[int(ys.argmax())], a.kararlilik)
        if degisti:
            olaylar.append({"tur": "duygu_degisti", "kaynak": DOSYA, "nesne": self.makine.durum, "guven": float(ys.max()), "ek": {"once": onceki}})
            anlat.adim(f"DUYGU DEĞİŞTİ: {onceki} → {self.makine.durum}")
        kutu_ciz(cikti, yo.kutu_xyxy(y), f"{self.makine.durum} ({siniflar[int(p.argmax())]})")
        ara["olasılıklar"] = olasilik_cubuklari(siniflar, p, ys)
        ara["modele giren yüz"] = kirp
        yazi_yaz(cikti, [f"kararli: {self.makine.durum}", f"bu kare: {siniflar[int(p.argmax())]}"])
        return Sonuc(goruntu=cikti, ara_gorseller=ara, olaylar=olaylar,
                     veri={"kararli": self.makine.durum, "ham": siniflar[int(p.argmax())]},
                     metin=f"Kararlı duygu: **{self.makine.durum}** (bu kare: {siniflar[int(p.argmax())]})")


if __name__ == "__main__":
    from ortak.cli import calistir
    calistir(DuyguCanli)
