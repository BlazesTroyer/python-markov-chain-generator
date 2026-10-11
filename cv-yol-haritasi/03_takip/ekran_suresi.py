"""
ekran_suresi.py — "Bugün telefonla kaç dakika geçirdim?" — takip + durum makinesi + süre hesabı.

NEDEN TAKİP GEREKLİ?
    Tespit tek başına "bu karede telefonlu biri var" der. Ama SÜRE ölçmek için "aynı kişi,
    aynı oturum" bilgisi gerekir: Kişi 1 telefonu bıraktı, kişi 2 aldı -> iki ayrı oturum.
    SORT her kişiye bir ID verir; her ID'nin kendi durum makinesi ve kronometresi olur.

AKIŞ (her kare):
    1. YOLO: 'person' ve 'cell phone' tespitleri.
    2. SORT: sadece kişiler takip edilir (kişi ID'leri).
    3. Her kişi için gözlem: telefon kutusunun merkezi, kişinin (biraz genişletilmiş) kutusunun
       İÇİNDE mi?  -> "telefonla" / "yok"
    4. Her kişi ID'sinin DurumMakinesi'ne gözlem verilir (02'deki debounce, aynı sınıf).
    5. Durum "telefonla" iken kronometre işler. Durum değişince olay:
         telefon_alindi     (yok -> telefonla)
         telefon_kullanimi  (telefonla -> yok, ek.sure_sn ile)
    6. gunluk_ozet(): olay deposundan bugünkü toplam süreyi hesaplar (09'daki screen_time_today bunu kullanır).

ZAMAN KAYNAĞI:
    fps=0  -> gerçek saat (canlı webcam için doğru)
    fps>0  -> her kare 1/fps saniye sayılır (kayıtlı video işlerken doğru; işlem hızından bağımsız)
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ortak.arayuz import Eylem, KareIsleyici, Sonuc, ayar  # noqa: E402
from ortak.cizim import id_rengi, kutu_ciz, yazi_yaz  # noqa: E402
from ortak.olay_deposu import OlayDeposu  # noqa: E402
from ortak.yukle import modul_yukle  # noqa: E402

st = modul_yukle("03_takip/sort_sifirdan.py")
tel = modul_yukle("02_yolo_finetune/telefon_dedektoru_v2.py")
yy = modul_yukle("02_yolo_finetune/yolo_yardimci.py")
ek = modul_yukle("01_veri_fabrikasi/etiket_kontrol.py")
DOSYA = "03_takip/ekran_suresi.py"


@dataclass
class Ayarlar:
    model: str = ayar("yolo11n.pt", "YOLO modeli (person + cell phone sınıfları olmalı)", tur="dosya")
    guven: float = ayar(0.3, "Güven eşiği", min=0.05, max=0.9, adim=0.05)
    genisletme: float = ayar(0.15, "Kişi kutusu her yönde bu oranda genişletilir (eldeki telefon kutunun biraz dışında kalabilir)",
                             min=0.0, max=0.6, adim=0.05)
    kararlilik: int = ayar(8, "Durum değişimi için art arda kare", min=1, max=60,
                           etki="Düşükse telefon bir an görünmeyince oturum bölünür; yüksekse kısa kullanımlar kaçar")
    fps: float = ayar(0.0, "0 = gerçek saat (canlı); >0 = kayıtlı videonun FPS'i", min=0.0, max=120.0, adim=1.0)


class EkranSuresi(KareIsleyici):
    ISIM = "Ekran süresi takipçisi (kişi ID bazlı)"
    BOLUM = "03"
    DOSYA = DOSYA
    ACIKLAMA = ("Kişileri SORT ile takip eder; her kişi ID'si için telefonla geçen süreyi ölçer. Durum makinesi kısa titremeleri yutar. "
                "Oturum bitince `telefon_kullanimi` olayı (süreli) üretir → '🗄️ Olay Deposu' ve 09'daki agent.")
    GEREKSINIMLER = ("ultralytics",)
    Ayarlar = Ayarlar

    def __init__(self):
        self.sifirla()

    def sifirla(self):
        self.sort = st.Sort(max_yas=30, min_vurus=3, iou_esigi=0.3)
        self.makineler: dict[int, object] = {}
        self.baslangic: dict[int, float] = {}    # kişi id -> oturum başlangıç zamanı
        self.toplam: dict[int, float] = {}       # kişi id -> toplam telefonlu süre (sn)
        self.kare_no = 0

    def _zaman(self, a: Ayarlar) -> float:
        return self.kare_no / a.fps if a.fps > 0 else time.time()

    def isle(self, kare, a: Ayarlar, anlat):
        self.kare_no += 1
        simdi = self._zaman(a)
        model = yy.model_yukle(a.model)
        ad_to_no = {v: k for k, v in model.names.items()}
        r = model.predict(kare, conf=a.guven, classes=[ad_to_no["person"], ad_to_no["cell phone"]], verbose=False)[0]
        kutular, siniflar, guvenler, _ = yy.sonucu_ayikla(r)
        kisi_m = siniflar == ad_to_no["person"]
        telefonlar = kutular[~kisi_m]
        tespit = np.hstack([kutular[kisi_m], guvenler[kisi_m, None], siniflar[kisi_m, None]]).astype(np.float32)
        izler = self.sort.guncelle(tespit)
        anlat.adim(f"{int(kisi_m.sum())} kişi, {len(telefonlar)} telefon → {len(izler)} kişi izi", kod=f"{DOSYA}:EkranSuresi.isle")

        cikti, olaylar = kare.copy(), []
        for t in telefonlar:
            kutu_ciz(cikti, t, "telefon", (0, 200, 255))
        for iz in izler:
            kid, k = iz["id"], iz["kutu"]
            w, h = k[2] - k[0], k[3] - k[1]
            gen = np.array([k[0] - w * a.genisletme, k[1] - h * a.genisletme, k[2] + w * a.genisletme, k[3] + h * a.genisletme])
            merkezler = (telefonlar[:, :2] + telefonlar[:, 2:]) / 2 if len(telefonlar) else np.zeros((0, 2))
            icinde = bool(((merkezler[:, 0] >= gen[0]) & (merkezler[:, 0] <= gen[2]) & (merkezler[:, 1] >= gen[1]) & (merkezler[:, 1] <= gen[3])).any())
            gozlem = "telefonla" if icinde else "yok"
            m = self.makineler.setdefault(kid, tel.DurumMakinesi())
            degisti, onceki = m.guncelle(gozlem, a.kararlilik)
            if degisti and m.durum == "telefonla":
                self.baslangic[kid] = simdi
                olaylar.append({"tur": "telefon_alindi", "kaynak": DOSYA, "nesne": "telefon", "kimlik": kid})
                anlat.adim(f"Kişi {kid}: telefon ALINDI", kod="02_yolo_finetune/telefon_dedektoru_v2.py:DurumMakinesi.guncelle")
            elif degisti and onceki == "telefonla":
                sure = simdi - self.baslangic.pop(kid, simdi)
                self.toplam[kid] = self.toplam.get(kid, 0) + sure
                olaylar.append({"tur": "telefon_kullanimi", "kaynak": DOSYA, "nesne": "telefon", "kimlik": kid, "ek": {"sure_sn": round(sure, 1)}})
                anlat.adim(f"Kişi {kid}: telefon BIRAKILDI, oturum {sure:.1f} sn", kod=f"{DOSYA}:EkranSuresi.isle")
            aktif = simdi - self.baslangic[kid] if kid in self.baslangic else 0.0
            toplam = self.toplam.get(kid, 0) + aktif
            renk = (0, 0, 255) if m.durum == "telefonla" else id_rengi(kid)
            kutu_ciz(cikti, k, f"Kisi {kid} {'TELEFONDA' if m.durum == 'telefonla' else ''} {toplam:.0f}s", renk)
            cv2.rectangle(cikti, tuple(gen[:2].astype(int)), tuple(gen[2:].astype(int)), (120, 120, 120), 1)
            anlat.bilgi(f"Kişi {kid}: gözlem={gozlem} durum={m.durum} toplam={toplam:.1f}s")
        yazi_yaz(cikti, [f"kisi: {len(izler)}", "gri cerceve = genisletilmis kisi alani"])
        return Sonuc(goruntu=cikti, olaylar=olaylar, veri={"toplam_sn": {k: round(v, 1) for k, v in self.toplam.items()}},
                     metin="Kişi bazlı toplam (tamamlanan oturumlar): " + (", ".join(f"Kişi {k}: {v:.0f} sn" for k, v in self.toplam.items()) or "henüz yok"))


def gunluk_ozet(depo: OlayDeposu | None = None, gun_baslangici: float | None = None) -> dict:
    """Olay deposundan bugünkü telefon kullanım özetini hesapla (09 agent screen_time_today bunu kullanır)."""
    depo = depo or OlayDeposu()
    if gun_baslangici is None:
        t = time.localtime()
        gun_baslangici = time.mktime((t.tm_year, t.tm_mon, t.tm_mday, 0, 0, 0, 0, 0, -1))
    oturumlar = depo.sorgula(tur="telefon_kullanimi", baslangic=gun_baslangici, bitis=gun_baslangici + 86400, limit=10000)
    sureler = [o["ek"].get("sure_sn", 0) for o in oturumlar]
    saatlik = {f"{h:02d}": 0.0 for h in range(24)}
    for o in oturumlar:
        saatlik[time.strftime("%H", time.localtime(o["zaman"]))] += o["ek"].get("sure_sn", 0) / 60
    return {"toplam_dk": round(sum(sureler) / 60, 1), "oturum": len(sureler),
            "en_uzun_dk": round(max(sureler, default=0) / 60, 1), "saatlik_dk": saatlik}


@dataclass
class RaporAyarlari:
    gun_once: int = ayar(0, "Kaç gün önce (0 = bugün)", min=0, max=30)


class EkranSuresiRaporu(Eylem):
    ISIM = "Ekran süresi raporu (günlük)"
    BOLUM = "03"
    DOSYA = DOSYA
    ACIKLAMA = "Olay deposundaki `telefon_kullanimi` olaylarından günlük toplam, oturum sayısı ve saatlik dağılım çıkarır."
    Ayarlar = RaporAyarlari

    def calistir(self, a: RaporAyarlari, anlat):
        t = time.localtime(time.time() - a.gun_once * 86400)
        bas = time.mktime((t.tm_year, t.tm_mon, t.tm_mday, 0, 0, 0, 0, 0, -1))
        oz = gunluk_ozet(OlayDeposu(), bas)
        anlat.adim(f"{time.strftime('%Y-%m-%d', t)}: {oz['oturum']} oturum, {oz['toplam_dk']} dk", kod=f"{DOSYA}:gunluk_ozet")
        dolu = {h: int(round(v)) for h, v in oz["saatlik_dk"].items() if v > 0} or {"-": 0}
        grafik = ek.sutun_grafigi(dolu, "Saatlik telefon kullanimi (dk)")
        return Sonuc(goruntu=grafik, veri=oz,
                     metin=f"**{time.strftime('%d.%m.%Y', t)}**: toplam **{oz['toplam_dk']} dk**, {oz['oturum']} oturum, en uzun {oz['en_uzun_dk']} dk")


if __name__ == "__main__":
    from ortak.cli import calistir
    calistir(EkranSuresi)
