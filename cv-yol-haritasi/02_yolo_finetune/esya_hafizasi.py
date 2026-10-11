"""
esya_hafizasi.py — "Anahtarlarımı en son nerede gördün?" sorusunun altyapısı.

NE YAPAR?
    Her karede dedektörü çalıştırır; gördüğü eşyaları OLAY DEPOSU'na (SQLite) yazar.
    Ama her kareyi değil! 30 FPS'te her kareyi yazmak saatte 100.000 satır demek.
    Sadece ANLAMLI anları yazarız (throttle / seyreltme):
        - Bu eşya `tekrar_sn` saniyedir kaydedilmediyse, VEYA
        - Son kaydedilen yerinden belirgin şekilde hareket ettiyse (merkez kayması > hareket_orani)
    Her kayıtta:
        nesne    : "anahtar"
        kutu     : piksel konumu
        ek.konum : insan dilinde konum ("sol üst", "orta", "sağ alt"...) — görüntü 3x3'e bölünür
        kare_yolu: eşyanın kırpılmış küçük görüntüsü (veri/kareler/...) -> agent "kare ekte" diyebilir

    nerede(nesne) fonksiyonu ve NeredeSorgu eylemi bu kayıtları insan cümlesine çevirir.
    09_agent/mcp_server.py'deki where_is() aracı AYNI fonksiyonu kullanır.

KULLANIM:
    python 02_yolo_finetune/esya_hafizasi.py --kaynak 0 --model veri/egitimler/esyalarim/weights/best.pt --olaylari-kaydet
    (Hazır COCO modeliyle denemek için: --model yolo11n.pt --isim-esleme "cell phone=telefon, cup=kupa, book=kitap")
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
from ortak.cizim import dosya_adi, id_rengi, kutu_ciz, yazi_yaz  # noqa: E402
from ortak.olay_deposu import OlayDeposu  # noqa: E402
from ortak.yukle import VERI, modul_yukle  # noqa: E402

yy = modul_yukle("02_yolo_finetune/yolo_yardimci.py")
DOSYA = "02_yolo_finetune/esya_hafizasi.py"


@dataclass
class Ayarlar:
    model: str = ayar("yolo11n.pt", "Dedektör (kendi best.pt'n ya da COCO)", tur="dosya")
    guven: float = ayar(0.4, "Güven eşiği", min=0.05, max=0.95, adim=0.05)
    isim_esleme: str = ayar("cell phone=telefon, cup=kupa, book=kitap, remote=kumanda, bottle=şişe",
                            "Model sınıf adı = kayıtta kullanılacak ad. Boşsa model adları aynen kullanılır", tur="uzun_metin")
    sadece_eslesenler: bool = ayar(True, "Sadece isim_esleme'deki sınıfları kaydet (kişi, sandalye gibi gürültüyü atla)")
    tekrar_sn: float = ayar(30.0, "Aynı eşya en fazla bu kadar saniyede bir kaydedilsin (yerinde duruyorsa)", min=1, max=600,
                            etki="Düşürürsen veritabanı hızla büyür; artırırsan 'son görülme' zamanı daha kaba olur")
    hareket_orani: float = ayar(0.08, "Merkez bu kadar kaydıysa (görüntü köşegenine oran) hemen kaydet", min=0.01, max=0.5, adim=0.01)
    kirpinti_kaydet: bool = ayar(True, "Eşyanın kırpılmış görüntüsünü veri/kareler/ altına kaydet")


def esleme_coz(metin: str) -> dict[str, str]:
    """'cell phone=telefon, cup=kupa' -> {'cell phone': 'telefon', 'cup': 'kupa'}"""
    sonuc = {}
    for parca in metin.replace("\n", ",").split(","):
        if "=" in parca:
            a, b = parca.split("=", 1)
            sonuc[a.strip()] = b.strip()
    return sonuc


def konum_adi(kutu, g: int, y: int) -> str:
    """Kutu merkezini 3x3 ızgarada insan diline çevir: 'sol üst', 'orta', 'sağ alt'..."""
    cx, cy = (kutu[0] + kutu[2]) / 2 / g, (kutu[1] + kutu[3]) / 2 / y
    yatay = ["sol", "orta", "sağ"][min(int(cx * 3), 2)]
    dikey = ["üst", "orta", "alt"][min(int(cy * 3), 2)]
    if yatay == "orta" and dikey == "orta":
        return "orta"
    return f"{yatay} {dikey}" if yatay != "orta" else f"orta {dikey}"


def nerede(nesne: str, depo: OlayDeposu | None = None) -> dict:
    """Bir eşyanın son görülmesini insan cümlesiyle döndür (09_agent where_is() bunu kullanır)."""
    depo = depo or OlayDeposu()
    son = depo.son_gorulme(nesne)
    if not son:
        return {"bulundu": False, "cumle": f"'{nesne}' hiç görülmedi."}
    gecen = time.time() - son["zaman"]
    ne_zaman = f"{gecen:.0f} sn önce" if gecen < 120 else f"{gecen / 60:.0f} dk önce" if gecen < 7200 else f"{gecen / 3600:.1f} saat önce"
    konum = son["ek"].get("konum", "?")
    return {"bulundu": True, "cumle": f"'{son['nesne']}' en son {son['saat']} ({ne_zaman}) kameranın {konum} bölgesinde görüldü.",
            "kayit": son}


class EsyaHafizasi(KareIsleyici):
    ISIM = "Eşya hafızası (nerede gördüm?)"
    BOLUM = "02"
    DOSYA = DOSYA
    ACIKLAMA = ("Dedektörün gördüğü eşyaları **seyrelterek** olay deposuna yazar (konum adı + kırpıntı ile). "
                "Panelde 'Olayları depoya yaz' kutusunu işaretle; sonra 'Nerede?' sekmesinden ya da 09'daki agent'tan sor.")
    GEREKSINIMLER = ("ultralytics",)
    Ayarlar = Ayarlar

    def __init__(self):
        # nesne adı -> bilinen örneklerin listesi [(son_kayit_zamani, merkez), ...]
        # Aynı sınıftan birden çok nesne olabilir (2 kupa). Her tespit, AYNI ADLI en yakın bilinen
        # örnekle eşleştirilir. (Bu, 03'teki takibin çok basit bir hali; orada kimlik numaraları da var.)
        self.son_kayit: dict[str, list[list]] = {}

    def sifirla(self):
        self.son_kayit.clear()

    def isle(self, kare, ayarlar: Ayarlar, anlat):
        model = yy.model_yukle(ayarlar.model)
        r = model.predict(kare, conf=ayarlar.guven, verbose=False)[0]
        kutular, siniflar, guvenler, isimler = yy.sonucu_ayikla(r)
        anlat.adim(f"{len(kutular)} tespit", kod="ultralytics:YOLO.predict")
        return self.tespitleri_isle(kare, kutular, siniflar, guvenler, isimler, ayarlar, anlat)

    def tespitleri_isle(self, kare, kutular, siniflar, guvenler, isimler, ayarlar: Ayarlar, anlat, cikti=None):
        """Hazır tespitlerden kayıt kararlarını ver. (09_agent/olay_ureteci.py aynı YOLO sonucunu
        hem kişi takibi hem eşya hafızası için kullanmak üzere bunu doğrudan çağırır — çift çıkarım olmaz.)"""
        esleme = esleme_coz(ayarlar.isim_esleme)
        y, g = kare.shape[:2]
        kosegen = float(np.hypot(g, y))
        cikti = kare.copy() if cikti is None else cikti
        olaylar, simdi = [], time.time()
        for k, s, c in zip(kutular, siniflar, guvenler):
            model_adi = isimler[s]
            if ayarlar.sadece_eslesenler and esleme and model_adi not in esleme:
                continue
            ad = esleme.get(model_adi, model_adi)
            merkez = np.array([(k[0] + k[2]) / 2, (k[1] + k[3]) / 2])
            konum = konum_adi(k, g, y)
            bilinenler = self.son_kayit.setdefault(ad, [])
            # Aynı adlı bilinen örneklerden en yakınını bul
            mesafeler = [np.linalg.norm(merkez - b[1]) / kosegen for b in bilinenler]
            en_yakin = int(np.argmin(mesafeler)) if mesafeler else -1
            neden = None
            if en_yakin < 0:
                neden = "ilk kez görüldü"
            elif mesafeler[en_yakin] > ayarlar.hareket_orani:
                neden = f"bilinen yerlerinden uzakta: yeni örnek ya da yer değiştirdi ({mesafeler[en_yakin]:.2f} > {ayarlar.hareket_orani})"
                en_yakin = -1   # yeni bir örnek gibi ekle
            elif simdi - bilinenler[en_yakin][0] > ayarlar.tekrar_sn:
                neden = f"{ayarlar.tekrar_sn:.0f} sn geçti"
            kutu_ciz(cikti, k, f"{ad} {c:.2f} [{konum}]", id_rengi(s))
            if neden is None:
                bilinenler[en_yakin][1] = merkez   # küçük kıpırtıları izle ama kaydetme
                anlat.bilgi(f"{ad}: kaydedilmedi (yerinde, {simdi - bilinenler[en_yakin][0]:.0f} sn önce kaydedilmişti)",
                            kod=f"{DOSYA}:EsyaHafizasi.isle")
                continue
            kare_yolu = ""
            if ayarlar.kirpinti_kaydet:
                klasor = VERI / "kareler"
                klasor.mkdir(parents=True, exist_ok=True)
                x1, y1, x2, y2 = k.astype(int)
                pay = 20  # bağlam görünsün diye kutunun biraz dışını da al
                kirpinti = kare[max(0, y1 - pay):y2 + pay, max(0, x1 - pay):x2 + pay]
                kare_yolu = str(klasor / f"{dosya_adi(ad)}_{int(simdi * 1000)}.jpg")
                cv2.imwrite(kare_yolu, kirpinti)
            olaylar.append({"tur": "nesne_goruldu", "kaynak": DOSYA, "nesne": ad, "kutu": k.tolist(), "guven": float(c),
                            "ek": {"konum": konum, "model_sinifi": model_adi, "kare_boyutu": [g, y]}, "kare_yolu": kare_yolu})
            if en_yakin >= 0:
                bilinenler[en_yakin] = [simdi, merkez]
            else:
                bilinenler.append([simdi, merkez])
                del bilinenler[:-10]   # sınıf başına en fazla 10 bilinen örnek tut (eski yerler unutulur)
            anlat.adim(f"{ad} KAYIT → {konum} ({neden})", kod=f"{DOSYA}:konum_adi")
        yazi_yaz(cikti, f"{len(olaylar)} yeni kayit | hafizada {sum(map(len, self.son_kayit.values()))} esya ornegi")
        return Sonuc(goruntu=cikti, olaylar=olaylar, veri={"kayitlar": [o["nesne"] for o in olaylar]},
                     metin=f"Bu karede {len(olaylar)} kayıt üretildi. (Depoya yazılması için panelde 'Olayları depoya yaz' açık olmalı.)")


@dataclass
class SorguAyarlari:
    nesne: str = ayar("telefon", "Hangi eşya?")
    gecmis: int = ayar(10, "Son kaç görülme listelensin", min=1, max=100)


class NeredeSorgu(Eylem):
    ISIM = "Nerede? (eşya sorgula)"
    BOLUM = "02"
    DOSYA = DOSYA
    ACIKLAMA = "Olay deposundan bir eşyanın son görülmesini ve geçmişini getirir. 09'daki agent'ın `where_is` aracıyla aynı fonksiyon."
    Ayarlar = SorguAyarlari

    def calistir(self, ayarlar: SorguAyarlari, anlat):
        depo = OlayDeposu()
        cevap = nerede(ayarlar.nesne, depo)
        anlat.adim(f"Sorgu: son_gorulme('{ayarlar.nesne}')", kod="ortak/olay_deposu.py:son_gorulme")
        gecmis = depo.sorgula(tur="nesne_goruldu", nesne=ayarlar.nesne, limit=ayarlar.gecmis)
        img = None
        if cevap["bulundu"] and cevap["kayit"]["kare_yolu"] and Path(cevap["kayit"]["kare_yolu"]).exists():
            img = cv2.imread(cevap["kayit"]["kare_yolu"])
        tablo = "\n".join(f"| {k['saat']} | {k['ek'].get('konum')} | {k['guven']:.2f} |" for k in gecmis)
        return Sonuc(goruntu=img, veri=cevap, metin=f"**{cevap['cumle']}**\n\n| zaman | konum | güven |\n|---|---|---|\n{tablo}")


if __name__ == "__main__":
    from ortak.cli import calistir
    calistir(EsyaHafizasi)
